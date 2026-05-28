# -*- coding: utf-8 -*-
"""SearchAgent

跨境商品检索专家。基于 AgentScope 2.0 Agent：
    工具集：product_search_tool（embedding+rerank 二阶段召回）
          category_insight_tool（品类洞察 RAG，选购常识）
          web_search_tool（可选，跨境政策兜底）

对外通过 task_dispatch 工具被 MainAgent 调度（SubAgent as Tool 模式）。
每次调度新建独立实例：2.0 的对话上下文内建于 AgentState，独立实例天然上下文隔离。
`build_tools()` 同时供 MainAgent 复用——主 Agent 持有同一批业务工具，可以不派发自己单干。
"""
from __future__ import annotations

from agentscope.agent import Agent, ReActConfig
from agentscope.rag import KnowledgeBase
from agentscope.tool import FunctionTool, Toolkit

from app.application.agents.context_policy import build_context_config
from app.application.prompts.loader import load_prompts
from app.application.tools.category_insight_tool import build_category_insight_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.web_search_tool import build_web_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import create_chat_model
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)
from app.infrastructure.settings import Settings
from app.infrastructure.tracing import build_agent_middlewares


class SearchAgentFactory:
    def __init__(
        self,
        settings: Settings,
        catalog_search: CatalogSearchUseCase,
        bus: TradeEventBus,
        knowledge_base: KnowledgeBase,
        circuit_registry: CircuitBreakerRegistry,
        throttle: GatewayThrottle,
    ) -> None:
        self._settings = settings
        self._catalog_search = catalog_search
        self._bus = bus
        self._knowledge_base = knowledge_base
        self._circuit_registry = circuit_registry
        # 闸门由组装根下发，三个工厂必须共用同一个，否则各限一份等于没限
        self._throttle = throttle

    def _resilience(self) -> list:
        return [ToolResilienceMiddleware(self._circuit_registry, self._bus)]

    def build_tools(self) -> list[FunctionTool]:
        """SearchAgent 的业务工具集，MainAgent 单干时持有同一批（均带超时+熔断保护）。

        web_search_tool 按"有 TAVILY_API_KEY 才注册"设计，未配置时 Agent 看不到它。
        """
        tools = [
            FunctionTool(
                build_product_search_tool(self._catalog_search, self._bus),
                is_read_only=True,
                middlewares=self._resilience(),
            ),
            FunctionTool(
                build_category_insight_tool(self._knowledge_base, self._bus),
                is_read_only=True,
                middlewares=self._resilience(),
            ),
        ]
        if self._settings.tavily_api_key:
            tools.append(
                FunctionTool(
                    build_web_search_tool(self._settings, self._bus),
                    is_read_only=True,
                    middlewares=self._resilience(),
                ),
            )
        return tools

    def build(self) -> Agent:
        prompts = load_prompts()["sub_agents"]["search"]
        return Agent(
            name=prompts["name"],
            system_prompt=prompts["system_prompt"],
            model=create_chat_model(self._settings, throttle=self._throttle, bus=self._bus),
            toolkit=Toolkit(tools=list(self.build_tools())),
            middlewares=build_agent_middlewares(self._settings),
            context_config=build_context_config(
                self._settings.context_size,
                self._settings.tool_result_limit,
            ),
            react_config=ReActConfig(max_iters=6),
        )
