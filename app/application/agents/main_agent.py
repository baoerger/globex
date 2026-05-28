# -*- coding: utf-8 -*-
"""MainAgent（CommerceConcierge）

跨境电商超级框总调度。基于 AgentScope 2.0 Agent，工具集分四层：
    1. 全部业务工具（product_search / create_order / query_order / cancel_order / [web_search]）
       ——与子 Agent 持有同一批工具，简单任务主 Agent 直接单干；
    2. 内置 Task 计划四件套（TaskCreate / TaskUpdate / TaskList / TaskGet）
       ——2.0 官方计划管理，挂在 AgentState.tasks_context 上；
    3. task_dispatch——满足"可并行 / 上下文隔离 / 链深"任一条件时派发专家子 Agent；
    4. remember_preference_tool——长期记忆写路径（读路径由 orchestrator 注入 hint）。

每个 shopping_session_id 对应一个 MainAgent 实例，由 SessionRegistry 缓存；
AgentState 每轮落盘 DATA_DIR/sessions/，服务重启后恢复多轮对话；
子 Agent 则每次调度新建（上下文隔离）。
"""
from __future__ import annotations

import logging
from typing import Optional

from agentscope.agent import Agent, ReActConfig
from agentscope.state import AgentState
from agentscope.tool import (
    FunctionTool,
    TaskCreate,
    TaskGet,
    TaskList,
    TaskUpdate,
    Toolkit,
)

from app.application.agents.context_policy import build_context_config
from app.application.agents.permissions import allow_business_tools
from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.application.harness.assertions import SequencingTracker
from app.application.harness.loop_detector import LoopDetector
from app.application.memory.preference_selector import PreferenceSelector
from app.application.prompts.loader import load_prompts
from app.application.tools.forget_preference_tool import build_forget_preference_tool
from app.application.tools.remember_preference_tool import build_remember_preference_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.domain.buyer.preference import PreferenceStore
from app.domain.session.ports.session_store import SessionStore
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.harness_middleware import HarnessToolMiddleware
from app.infrastructure.llm import create_chat_model
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)
from app.infrastructure.settings import Settings
from app.infrastructure.tracing import build_agent_middlewares

logger = logging.getLogger(__name__)


class MainAgentFactory:
    def __init__(
        self,
        settings: Settings,
        search_factory: SearchAgentFactory,
        trade_factory: TradeAgentFactory,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        circuit_registry: CircuitBreakerRegistry,
        throttle: GatewayThrottle,
        sequencing: Optional[SequencingTracker] = None,
        loop_detector: Optional[LoopDetector] = None,
        preference_selector: Optional[PreferenceSelector] = None,
    ) -> None:
        self._settings = settings
        self._search_factory = search_factory
        self._trade_factory = trade_factory
        self._bus = bus
        self._preference_store = preference_store
        self._circuit_registry = circuit_registry
        self._throttle = throttle
        # 与 orchestrator 共用同一个 selector，保证主/子 Agent 的偏好选取口径一致
        self._preference_selector = preference_selector or PreferenceSelector()
        # 护栏判定器按会话累积状态，须跨 Agent 实例共享（与熔断注册表同理）
        self._sequencing = sequencing or SequencingTracker()
        self._loop_detector = loop_detector or LoopDetector(
            repeat_threshold=settings.loop_repeat_threshold,
        )

    def _resilience(self) -> list:
        """工具中间件链。

        洋葱顺序：Harness 在外、Resilience 在内——先做准入判定（顺序/循环），
        再进超时与熔断保护；这样被硬拒的调用不会白白占用一次熔断名额。
        """
        chain: list = []
        if self._settings.harness_enabled:
            chain.append(
                HarnessToolMiddleware(
                    sequencing=self._sequencing,
                    loop_detector=self._loop_detector,
                    bus=self._bus,
                ),
            )
        chain.append(ToolResilienceMiddleware(self._circuit_registry, self._bus))
        return chain

    def build(self, restored_state: Optional[AgentState] = None) -> Agent:
        prompts = load_prompts()["main_agent"]

        tools = [
            # 1. 业务工具：与子 Agent 同一批，主 Agent 可单干
            *self._search_factory.build_tools(),
            *self._trade_factory.build_tools(),
            # 2. 内置 Task 计划工具（is_state_injected，挂 AgentState.tasks_context）
            TaskCreate(),
            TaskUpdate(),
            TaskList(),
            TaskGet(),
            # 3. SubAgent as Tool 调度（is_concurrency_safe 默认为 True，
            #    主 Agent 同一轮发起的多个派发会被 2.0 并发批执行）
            FunctionTool(
                build_task_dispatch_tool(
                    self._search_factory,
                    self._trade_factory,
                    self._bus,
                    preference_store=self._preference_store,
                    preference_selector=self._preference_selector,
                    preference_top_k=self._settings.preference_top_k,
                    subagent_inject=self._settings.preference_subagent_inject,
                ),
                is_concurrency_safe=True,
                middlewares=self._resilience(),
            ),
            # 4. 长期记忆写路径
            FunctionTool(
                build_remember_preference_tool(self._preference_store, self._bus),
                middlewares=self._resilience(),
            ),
            # 5. 长期记忆撤回路径（买家说“以后不用避开塑料了”）
            FunctionTool(
                build_forget_preference_tool(self._preference_store, self._bus),
                middlewares=self._resilience(),
            ),
        ]

        return allow_business_tools(
            Agent(
                name=prompts["name"],
                system_prompt=prompts["system_prompt"],
                model=create_chat_model(self._settings, throttle=self._throttle, bus=self._bus),
                toolkit=Toolkit(tools=tools),
                middlewares=build_agent_middlewares(self._settings),
                context_config=build_context_config(
                    self._settings.context_size,
                    self._settings.tool_result_limit,
                ),
                state=restored_state,
                react_config=ReActConfig(max_iters=15),
            ),
        )


class SessionRegistry:
    """按 shopping_session_id 缓存 MainAgent 实例，支撑多轮对话；
    AgentState 经 SessionStore 端口落盘（SQLite 或文件），服务重启后恢复。"""

    def __init__(self, main_factory: MainAgentFactory, session_store: SessionStore) -> None:
        self._main_factory = main_factory
        self._session_store = session_store
        self._agents: dict[str, Agent] = {}

    async def get_or_create(self, shopping_session_id: str) -> Agent:
        if shopping_session_id not in self._agents:
            restored_state = await self._try_restore(shopping_session_id)
            self._agents[shopping_session_id] = self._main_factory.build(restored_state)
        return self._agents[shopping_session_id]

    async def persist(self, shopping_session_id: str) -> None:
        """每轮对话结束后落盘 AgentState 快照；失败仅告警不影响主链路。"""
        agent = self._agents.get(shopping_session_id)
        if agent is None:
            return
        try:
            await self._session_store.save(shopping_session_id, agent.state.model_dump_json())
        except Exception as err:  # noqa: BLE001
            logger.warning("会话状态落盘失败：%s（%s）", shopping_session_id, err)

    async def _try_restore(self, shopping_session_id: str) -> Optional[AgentState]:
        try:
            raw = await self._session_store.load(shopping_session_id)
        except Exception as err:  # noqa: BLE001 —— 存储不可用时按新会话继续，不阻断对话
            logger.warning("会话状态读取失败，按新会话处理：%s（%s）", shopping_session_id, err)
            return None
        if raw is None:
            return None
        try:
            state = AgentState.model_validate_json(raw)
            logger.info("会话状态已恢复：%s（%d 条上下文）", shopping_session_id, len(state.context))
            return state
        except Exception as err:  # noqa: BLE001 —— 快照损坏按新会话处理
            logger.warning("会话状态恢复失败，按新会话处理：%s（%s）", shopping_session_id, err)
            return None
