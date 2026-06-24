# -*- coding: utf-8 -*-
"""长期记忆写/撤回两个工具的单测。

两条纪律必须钉死：
    1. buyer_id 只能来自 ShoppingContext，绝不能让模型入参决定改谁的记忆；
    2. 撤回未命中时不能假装成功——要把现存偏好回给模型让它用原文重试，
       否则模型会告诉买家"已经帮你撤回了"，而记忆里那条还在。
"""
from app.application.tools.forget_preference_tool import build_forget_preference_tool
from app.application.tools.remember_preference_tool import build_remember_preference_tool
from app.domain.buyer.preference import BuyerPreference
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore

SNAPSHOT = ShoppingContextSnapshot(
    shopping_session_id="s1", buyer_id="buyer-001", locale="zh-CN", currency="CNY",
)


def _text(chunk) -> str:
    return "".join(
        getattr(block, "text", "") or "" for block in chunk.content
    )


class TestForgetPreferenceTool:
    async def test_deletes_existing_preference(self, tmp_path):
        store = JsonFilePreferenceStore(tmp_path)
        await store.append(
            BuyerPreference(buyer_id="buyer-001", kind="dislike", statement="不要塑料材质"),
        )
        tool = build_forget_preference_tool(store, TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)
        try:
            chunk = await tool(statement="不要塑料材质")
        finally:
            ShoppingContext.reset(token)

        assert "已撤回" in _text(chunk)
        assert await store.list_by_buyer("buyer-001") == []

    async def test_miss_lists_remaining_and_deletes_nothing(self, tmp_path):
        """未命中不算错误，但必须如实说没删，并把现存偏好列出来。"""
        store = JsonFilePreferenceStore(tmp_path)
        await store.append(
            BuyerPreference(buyer_id="buyer-001", kind="dislike", statement="不要塑料材质"),
        )
        tool = build_forget_preference_tool(store, TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)
        try:
            chunk = await tool(statement="不要塑料")
        finally:
            ShoppingContext.reset(token)

        text = _text(chunk)
        assert "未找到" in text
        assert "不要塑料材质" in text, "要把现存偏好回给模型，供它用原文重试"
        assert len(await store.list_by_buyer("buyer-001")) == 1, "未命中不得删任何东西"

    async def test_miss_on_empty_store_is_explicit(self, tmp_path):
        store = JsonFilePreferenceStore(tmp_path)
        tool = build_forget_preference_tool(store, TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)
        try:
            chunk = await tool(statement="不要塑料材质")
        finally:
            ShoppingContext.reset(token)

        assert "没有任何长期偏好" in _text(chunk)

    async def test_buyer_id_comes_from_context_not_model(self, tmp_path):
        """工具签名里没有 buyer_id：模型无从指定改谁的记忆。"""
        store = JsonFilePreferenceStore(tmp_path)
        await store.append(
            BuyerPreference(buyer_id="buyer-999", kind="dislike", statement="不要塑料材质"),
        )
        tool = build_forget_preference_tool(store, TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)  # 当前买家是 buyer-001
        try:
            chunk = await tool(statement="不要塑料材质")
        finally:
            ShoppingContext.reset(token)

        assert "未找到" in _text(chunk)
        assert len(await store.list_by_buyer("buyer-999")) == 1, "不得删到别人的记忆"

    async def test_store_failure_reported_not_swallowed(self, tmp_path):
        class BrokenStore(JsonFilePreferenceStore):
            async def delete(self, buyer_id: str, statement: str) -> bool:
                raise RuntimeError("数据库连接断了")

        tool = build_forget_preference_tool(BrokenStore(tmp_path), TradeEventBus())

        token = ShoppingContext.set(SNAPSHOT)
        try:
            chunk = await tool(statement="不要塑料材质")
        finally:
            ShoppingContext.reset(token)

        assert "[error]" in _text(chunk), "写失败必须如实报错，不能假装撤回成功"


class TestRememberForgetRoundTrip:
    async def test_remember_then_forget(self, tmp_path):
        store = JsonFilePreferenceStore(tmp_path)
        bus = TradeEventBus()
        remember = build_remember_preference_tool(store, bus)
        forget = build_forget_preference_tool(store, bus)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await remember(kind="dislike", statement="不要塑料材质")
            assert len(await store.list_by_buyer("buyer-001")) == 1

            await forget(statement="不要塑料材质")
            assert await store.list_by_buyer("buyer-001") == []
        finally:
            ShoppingContext.reset(token)

    async def test_forget_then_remember_again(self, tmp_path):
        """买家撤回后又改主意，应能重新记住。"""
        store = JsonFilePreferenceStore(tmp_path)
        bus = TradeEventBus()
        remember = build_remember_preference_tool(store, bus)
        forget = build_forget_preference_tool(store, bus)

        token = ShoppingContext.set(SNAPSHOT)
        try:
            await remember(kind="dislike", statement="不要塑料材质")
            await forget(statement="不要塑料材质")
            await remember(kind="dislike", statement="不要塑料材质")
        finally:
            ShoppingContext.reset(token)

        assert len(await store.list_by_buyer("buyer-001")) == 1
