# -*- coding: utf-8 -*-
"""MainAgentOrchestrator

应用层编排入口：
    1. 把会话快照写入 ShoppingContext（ContextVar，工具与子 Agent 透明可读）；
    2. 长期记忆读路径：买家偏好经 PreferenceSelector 按本轮 query 相关性挑选后，
       有变化时随本轮输入注入一条 <buyer-preferences> hint 消息（dislike 不参与截断）；
    3. 消费 MainAgent 的 reply_stream 类型化事件流并映射到 TradeEventBus：
       TextBlockDeltaEvent → token.delta
       Task* 工具结果      → plan.update（从 AgentState.tasks_context 快照）
       （业务工具的 tool.invoke / tool.result 与 agent.dispatch 由工具自身发布）
    4. 上下文压缩检测：本轮结束后 AgentState.summary 发生变化即发布 context.compressed；
    5. 上游瞬时错误（限流/并发/5xx）有界重试；
    6. 结束后发布 final.result / error，落盘 AgentState，返回最终文本。

为何重试要放在这一层：2.0 模型层只对"建流阶段"的异常重试，而 OpenAI 兼容网关常把
限流错误写在 SSE 流中间（报 openai.APIError），此时已经走出模型层重试范围，不兜底就会
整轮失败。重试期间前端可能看到重复的流式片段，final.result 到达时会被覆盖。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from agentscope.agent import Agent
from agentscope.event import (
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
)
from agentscope.message import Msg, UserMsg

from app.application.agents.main_agent import SessionRegistry
from app.application.harness.drift_detector import DriftDetector
from app.application.harness.loop_detector import LoopDetector
from app.application.memory.preference_selector import (
    PreferenceSelector,
    render_preference_hint,
    render_preference_lines,
)
from app.domain.buyer.preference import PreferenceStore
from app.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from app.domain.session.ports.session_store import SessionStore  # noqa: F401 —— 保留类型引用
from app.infrastructure.cache.semantic_cache import SemanticCache
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.budget import init_budget
from app.infrastructure.security.output_guard import audit_output
from app.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)

# 内置 Task 计划工具名，其结果落地后向前端推送 plan.update 快照
_TASK_TOOL_NAMES = {"TaskCreate", "TaskUpdate", "TaskList", "TaskGet"}

# 上游瞬时故障判据与模型层共用同一份（app/infrastructure/transient.py），避免两处标记表漂移。
# 模型层已做一轮退避重试 + 备用模型回退，这里是最外层兜底：覆盖模型层之外
# （工具、子 Agent 调度、事件消费）招致的瞬时失败。
_MAX_TURN_RETRIES = 2
_RETRY_BASE_SECONDS = 6.0


@dataclass(frozen=True)
class SubmitIntentInput:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str


@dataclass(frozen=True)
class SubmitIntentOutput:
    shopping_session_id: str
    final_text: str


def _tasks_snapshot(agent: Agent) -> dict:
    tasks = agent.state.tasks_context.tasks
    return {
        "tasks": [
            {"id": task.id, "subject": task.subject, "state": str(task.state)}
            for task in tasks
        ],
    }


class MainAgentOrchestrator:
    def __init__(
        self,
        sessions: SessionRegistry,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        conversation_store: Optional[ConversationStore] = None,
        semantic_cache: Optional[SemanticCache] = None,
        output_guard_enabled: bool = True,
        loop_detector: Optional[LoopDetector] = None,
        token_budget_total: int = 0,
        drift_detector: Optional[DriftDetector] = None,
        preference_selector: Optional[PreferenceSelector] = None,
        preference_top_k: int = 5,
    ) -> None:
        self._sessions = sessions
        self._bus = bus
        self._preference_store = preference_store
        self._conversation_store = conversation_store
        self._semantic_cache = semantic_cache
        self._output_guard_enabled = output_guard_enabled
        self._loop_detector = loop_detector
        self._token_budget_total = token_budget_total
        self._drift_detector = drift_detector
        # 默认 selector 不带 embedder，退化为“按时间倒序取 top_k”，单测与无凭据环境可直接跑
        self._preference_selector = preference_selector or PreferenceSelector()
        self._preference_top_k = preference_top_k
        # 会话内已注入的偏好快照，变化时才重新注入，避免每轮重复填充上下文
        self._injected_preferences: dict[str, str] = {}

    def _guard_final_text(self, session_id: str, text: str) -> str:
        """L4 输出审核：最终回复推给买家前脱敏内部信息。

        命中只脱敏并发一条告警事件，**不阻断回复**：
        一条误判不应让整轮对话失败。
        """
        if not self._output_guard_enabled or not text:
            return text
        safe, cleaned = audit_output(text)
        if not safe:
            logger.warning("L4 输出审核命中敏感内容，已脱敏（会话 %s）", session_id)
            self._bus.publish(session_id, "error", {"message": "输出审核命中内部信息，已脱敏后下发"})
        return cleaned

    async def handle_intent(self, intent: SubmitIntentInput) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        snapshot = ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
        )
        token = ShoppingContext.set(snapshot)
        started_at = time.monotonic()
        # 本轮 Token 预算（TOKEN_BUDGET_TOTAL=0时为 None，不启用四档降级）
        init_budget(self._token_budget_total)
        if self._drift_detector is not None:
            self._drift_detector.start_turn(session_id, intent.raw_query)
        # 开始录事件轨迹（本轮结束后批量入库）
        trace = self._bus.subscribe(session_id) if self._conversation_store else None
        final_text = ""
        try:
            agent = await self._sessions.get_or_create(session_id)
            summary_before = agent.state.summary
            # 语义缓存：仅首轮（无历史上下文）且非写操作意图时尝试命中，命中则零模型调用
            has_history = bool(agent.state.context)
            cached = await self._lookup_cache(intent, has_history)
            if cached is not None:
                final_text = self._guard_final_text(session_id, cached)
                self._bus.publish(session_id, "final.result", {"text": final_text})
                return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)

            inputs = await self._build_inputs(intent, session_id)

            final_text = await self._reply_with_retry(session_id, agent, inputs)
            final_text = self._guard_final_text(session_id, final_text)
            await self._check_drift(session_id)

            self._publish_compression(session_id, agent, summary_before)
            self._bus.publish(session_id, "final.result", {"text": final_text})
            await self._remember_cache(intent, final_text, has_history)
            return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)
        except Exception as err:  # noqa: BLE001 —— 兜底转事件，避免长任务静默失败
            logger.exception("MainAgent 异常")
            self._bus.publish(session_id, "error", {"message": str(err)})
            final_text = f"[error] {err}"
            return SubmitIntentOutput(shopping_session_id=session_id, final_text=final_text)
        finally:
            # 无论成功失败都落盘会话状态（失败内部仅告警）
            await self._sessions.persist(session_id)
            await self._record_conversation(
                intent, final_text, int((time.monotonic() - started_at) * 1000), trace,
            )
            # 循环检测是"本轮内是不是在打转"的判定，轮末必须清零；
            # 阶段无关的顺序记录（SequencingTracker）则按会话保留，
            # 否则第 1 轮检索、第 3 轮下单会被误判为"未检索就下单"。
            if self._loop_detector is not None:
                self._loop_detector.reset(session_id)
            if self._drift_detector is not None:
                self._drift_detector.reset(session_id)
            ShoppingContext.reset(token)

    async def _preference_scope(self, buyer_id: str) -> str:
        """买家当前偏好的指纹，作为语义缓存的分桶维度。

        用**全量偏好**而不是本轮选中的子集：选中子集随 query 变，拿它做 key
        会让缓存碎成一盘沙。全量偏好只在真正新增/撤回时变，恰好是正确的失效时机。
        """
        try:
            preferences = await self._preference_store.list_by_buyer(buyer_id)
        except Exception as err:  # noqa: BLE001 —— 读不到就当无偏好，不阻断对话
            logger.warning("读取偏好指纹失败，缓存按无偏好分桶：%s", err)
            return ""
        if not preferences:
            return ""
        return hashlib.sha256(
            render_preference_lines(preferences).encode(),
        ).hexdigest()[:16]

    async def _lookup_cache(self, intent: SubmitIntentInput, has_history: bool) -> Optional[str]:
        """语义缓存查询；命中时发 cache.hit 事件让过程可见（不静默复用）。"""
        if self._semantic_cache is None:
            return None
        hit = await self._semantic_cache.lookup(
            intent.buyer_id,
            intent.raw_query,
            has_history,
            scope=await self._preference_scope(intent.buyer_id),
        )
        if hit is None:
            return None
        logger.info("语义缓存命中（%.4f）：%s", hit.similarity, intent.raw_query)
        self._bus.publish(
            intent.shopping_session_id,
            "cache.hit",
            {"similarity": hit.similarity, "matched_query": hit.matched_query},
        )
        return hit.reply

    async def _remember_cache(
        self, intent: SubmitIntentInput, final_text: str, has_history: bool,
    ) -> None:
        if self._semantic_cache is None:
            return
        await self._semantic_cache.remember(
            intent.buyer_id,
            intent.raw_query,
            final_text,
            has_history,
            scope=await self._preference_scope(intent.buyer_id),
        )

    async def _record_conversation(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        latency_ms: int,
        trace: Optional[asyncio.Queue],
    ) -> None:
        """对话流水 + 事件轨迹入库。写库失败只告警，不影响已经返回给买家的结果。"""
        if self._conversation_store is None:
            return
        session_id = intent.shopping_session_id
        events: list[ConversationEventRecord] = []
        if trace is not None:
            self._bus.unsubscribe(session_id, trace)
            while not trace.empty():
                event = trace.get_nowait()
                # token.delta 量大且已被 final.result 汇总，不入库
                if event.type == "token.delta":
                    continue
                events.append(
                    ConversationEventRecord(
                        session_id=session_id,
                        type=event.type,
                        payload=event.payload if isinstance(event.payload, dict) else {"value": event.payload},
                        occurred_at=event.occurred_at,
                    ),
                )
        try:
            await self._conversation_store.touch_session(
                session_id, intent.buyer_id, intent.locale, intent.currency,
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="buyer",
                    content=intent.raw_query,
                ),
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="agent",
                    content=final_text,
                    latency_ms=latency_ms,
                ),
            )
            await self._conversation_store.append_events(events)
        except Exception as err:  # noqa: BLE001
            logger.warning("对话记录写入失败：%s（%s）", session_id, err)

    async def _reply_with_retry(self, session_id: str, agent: Agent, inputs: list[Msg]) -> str:
        """跑一轮 Agent 并映射事件流；上游瞬时错误按指数退避重试。"""
        last_error: Exception | None = None
        for attempt in range(_MAX_TURN_RETRIES + 1):
            try:
                return await self._consume_reply(session_id, agent, inputs)
            except Exception as err:  # noqa: BLE001
                if not is_transient_error(err) or attempt >= _MAX_TURN_RETRIES:
                    raise
                last_error = err
                # 指数退避：网关速率类限流对固定间隔重试不敏感
                delay = _RETRY_BASE_SECONDS * (3**attempt)
                logger.warning(
                    "上游瞬时故障，%.0fs 后重试（第 %d/%d 次）：%s",
                    delay,
                    attempt + 1,
                    _MAX_TURN_RETRIES,
                    err,
                )
                self._bus.publish(
                    session_id,
                    "error",
                    {"message": f"上游瞬时故障，正在重试：{err}", "retrying": True},
                )
                # 重试时不再重复送入 inputs，避免上下文里出现两次买家发言
                inputs = []
                await asyncio.sleep(delay)
        raise last_error if last_error else RuntimeError("reply 重试耗尽")

    async def _consume_reply(self, session_id: str, agent: Agent, inputs: list[Msg]) -> str:
        final_text = ""
        # tool_call_id → 工具名，用于把 ToolResultEndEvent 关联回 Task 工具
        call_names: dict[str, str] = {}
        async for event in agent.reply_stream(inputs or None, yield_final_msg=True):
            if isinstance(event, Msg):
                final_text = event.get_text_content() or ""
            elif isinstance(event, TextBlockDeltaEvent):
                if event.delta:
                    self._bus.publish(
                        session_id,
                        "token.delta",
                        {"name": agent.name, "token": event.delta},
                    )
            elif isinstance(event, ToolCallStartEvent):
                call_names[event.tool_call_id] = event.tool_call_name
            elif isinstance(event, ToolResultEndEvent):
                tool_name = call_names.get(event.tool_call_id)
                if tool_name in _TASK_TOOL_NAMES:
                    self._bus.publish(session_id, "plan.update", _tasks_snapshot(agent))
                self._observe_for_drift(session_id, tool_name, event)
        return final_text

    def _observe_for_drift(self, session_id: str, tool_name: Optional[str], event: Any) -> None:
        """把一次工具结果记进漂移轨迹（开关关时零开销）。"""
        if self._drift_detector is None or not tool_name:
            return
        text = ""
        try:
            blocks = getattr(event, "output", None) or []
            text = "\n".join(
                str(getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else ""))
                for block in blocks
            )
        except Exception:  # noqa: BLE001 —— 观测不能影响主链路
            text = ""
        # “无候选”的判据：检索类工具返回的 hits 为空
        result_empty = bool(text) and ('"hits": []' in text or '"hits":[]' in text)
        self._drift_detector.observe_action(
            session_id, f"{tool_name} {text[:200]}", result_empty=result_empty,
        )

    async def _check_drift(self, session_id: str) -> None:
        """轮末漂移判定：命中只发事件 + 记日志。

        不在这里改写回复——本轮已结束，注入纠正提示已经来不及了；
        漂移信号的价值在于**被看见**（进事件流与 trace，供 bad case 回收）。
        """
        if self._drift_detector is None:
            return
        try:
            report = await self._drift_detector.check(session_id)
        except Exception as err:  # noqa: BLE001
            logger.warning("漂移检测异常，忽略：%s", err)
            return
        if report.drifted:
            logger.warning("检测到静默漂移（会话 %s）：%s", session_id, report.reasons or report.verdict)
            self._bus.publish(
                session_id,
                "error",
                {
                    "message": "检测到可能的目标漂移",
                    "reasons": report.reasons,
                    "verdict": report.verdict,
                },
            )

    def _publish_compression(self, session_id: str, agent: Agent, summary_before: str | None) -> None:
        """上下文压缩发生时，2.0 会把早期消息压成摘要写入 AgentState.summary，
        比对本轮前后的 summary 即可判定并上报。"""
        summary_after = agent.state.summary
        if not summary_after or summary_after == summary_before:
            return
        self._bus.publish(
            session_id,
            "context.compressed",
            {
                "summary_length": len(summary_after),
                "context_messages": len(agent.state.context),
            },
        )

    async def _build_inputs(self, intent: SubmitIntentInput, session_id: str) -> list[Msg]:
        """长期记忆读路径：偏好有变化时随本轮输入注入 hint 消息。"""
        user_msg = UserMsg(intent.buyer_id, intent.raw_query)
        try:
            preferences = await self._preference_store.list_by_buyer(intent.buyer_id)
        except Exception as err:  # noqa: BLE001 —— 记忆读取失败不阻断对话
            logger.warning("读取买家偏好失败：%s", err)
            preferences = []
        if not preferences:
            return [user_msg]

        # 按与本轮 query 的相关性挑选：偏好越攒越多时，全量铺进去会把真正相关的那几条稀释。
        # dislike 不参与截断（见 PreferenceSelector 文档字符串）。
        selected = await self._preference_selector.select(
            preferences, query=intent.raw_query, top_k=self._preference_top_k,
        )
        if not selected:
            return [user_msg]

        rendered = render_preference_lines(selected)
        if self._injected_preferences.get(session_id) == rendered:
            return [user_msg]
        self._injected_preferences[session_id] = rendered
        hint_msg = UserMsg("memory_hint", render_preference_hint(selected))
        return [hint_msg, user_msg]
