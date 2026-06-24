# -*- coding: utf-8 -*-
"""子 Agent 偏好注入单测（task_dispatch 服务端注入）。

背景：偏好原来只注入主 Agent 上下文，子 Agent 拿不到，只能靠主 Agent 把偏好
抄进 demands——这是**软约束**，模型漏抄就静默降级成无偏好推荐，不报错也无告警。
现在由 task_dispatch 从 ShoppingContext 取 buyer_id 直接注入，去掉这个软约束。

要钉死的四件事：
    1. search_agent 确实收到 <buyer-preferences>；
    2. trade_agent 不注入（它只按给定 product_id 下单，偏好影响不了它）；
    3. 无偏好时不塞空块；
    4. 读记忆失败只跳过注入，不能让派发挂掉。
"""
from agentscope.message import AssistantMsg

from app.application.memory.preference_selector import PreferenceSelector
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore

SNAPSHOT = ShoppingContextSnapshot(
    shopping_session_id="s1", buyer_id="buyer-001", locale="zh-CN", currency="CNY",
)


class RecordingWorker:
    """记录收到的输入消息，返回固定结论。"""

    def __init__(self, seen: list) -> None:
        self._seen = seen

    async def reply(self, inputs):
        messages = inputs if isinstance(inputs, list) else [inputs]
        self._seen.extend(messages)
        return AssistantMsg("worker", '{"hits": []}')


class RecordingFactory:
    def __init__(self) -> None:
        self.seen: list = []

    def build(self) -> RecordingWorker:
        return RecordingWorker(self.seen)


def _texts(messages) -> str:
    return "\n".join(m.get_text_content() or "" for m in messages)


async def _seeded_store(tmp_path, *preferences) -> JsonFilePreferenceStore:
    store = JsonFilePreferenceStore(tmp_path)
    for kind, statement in preferences:
        await store.append(
            BuyerPreference(buyer_id="buyer-001", kind=kind, statement=statement),
        )
    return store


def _tool(search_factory, trade_factory, store, **kwargs):
    return build_task_dispatch_tool(
        search_factory, trade_factory, TradeEventBus(),
        preference_store=store,
        preference_selector=PreferenceSelector(),
        **kwargs,
    )


class TestSearchAgentInjection:
    async def test_search_agent_receives_preferences(self, tmp_path):
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(tmp_path, ("dislike", "不要塑料材质"))
        tool = _tool(search, trade, store)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找个 300 元内的旅行三件套")
        finally:
            ShoppingContext.reset(token)

        received = _texts(search.seen)
        assert "<buyer-preferences>" in received
        assert "不要塑料材质" in received
        assert "找个 300 元内的旅行三件套" in received, "原始 demands 不能被顶掉"

    async def test_hint_comes_before_demands(self, tmp_path):
        """偏好块要排在 demands 之前，读起来是"先给背景再下指令"。"""
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(tmp_path, ("dislike", "不要塑料材质"))
        tool = _tool(search, trade, store)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        assert "<buyer-preferences>" in (search.seen[0].get_text_content() or "")

    async def test_no_preferences_means_no_empty_block(self, tmp_path):
        search, trade = RecordingFactory(), RecordingFactory()
        store = JsonFilePreferenceStore(tmp_path)
        tool = _tool(search, trade, store)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        assert len(search.seen) == 1
        assert "<buyer-preferences>" not in _texts(search.seen)


class TestTradeAgentNotInjected:
    async def test_trade_agent_gets_no_preferences(self, tmp_path):
        """交易子 Agent 只按给定 product_id 执行，注入偏好纯属白花 token。"""
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(tmp_path, ("dislike", "不要塑料材质"))
        tool = _tool(search, trade, store)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="trade_agent", demands="为 P1001 下单")
        finally:
            ShoppingContext.reset(token)

        assert len(trade.seen) == 1
        assert "<buyer-preferences>" not in _texts(trade.seen)


class TestGuards:
    async def test_switch_off_disables_injection(self, tmp_path):
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(tmp_path, ("dislike", "不要塑料材质"))
        tool = _tool(search, trade, store, subagent_inject=False)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        assert "<buyer-preferences>" not in _texts(search.seen)

    async def test_no_store_is_tolerated(self, tmp_path):
        """未注入 preference_store 时（旧装配）不能炸。"""
        search, trade = RecordingFactory(), RecordingFactory()
        tool = build_task_dispatch_tool(search, trade, TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        assert len(search.seen) == 1

    async def test_store_failure_skips_injection_not_dispatch(self, tmp_path):
        """读记忆失败只是少了个性化，派发本身必须照常完成。"""

        class BrokenStore(JsonFilePreferenceStore):
            async def list_by_buyer(self, buyer_id: str):
                raise RuntimeError("存储不可用")

        search, trade = RecordingFactory(), RecordingFactory()
        tool = _tool(search, trade, BrokenStore(tmp_path))

        token = ShoppingContext.set(SNAPSHOT)
        try:
            chunk = await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        assert len(search.seen) == 1, "派发要照常发生"
        assert chunk is not None

    async def test_missing_context_skips_injection(self, tmp_path):
        """没有 ShoppingContext（拿不到 buyer_id）时不注入，也不报错。"""
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(tmp_path, ("dislike", "不要塑料材质"))
        tool = _tool(search, trade, store)

        await tool(subagent_type="search_agent", demands="找旅行三件套")

        assert "<buyer-preferences>" not in _texts(search.seen)

    async def test_dislike_survives_top_k_zero(self, tmp_path):
        """top_k=0 也要保住黑名单——安全底线在子 Agent 侧同样成立。"""
        search, trade = RecordingFactory(), RecordingFactory()
        store = await _seeded_store(
            tmp_path, ("dislike", "不要塑料材质"), ("like", "喜欢小众设计"),
        )
        tool = _tool(search, trade, store, preference_top_k=0)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await tool(subagent_type="search_agent", demands="找旅行三件套")
        finally:
            ShoppingContext.reset(token)

        received = _texts(search.seen)
        assert "不要塑料材质" in received
        assert "喜欢小众设计" not in received
