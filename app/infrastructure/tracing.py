# -*- coding: utf-8 -*-
"""tracing / middleware 装配

可观测装配：OTEL_EXPORTER_OTLP_ENDPOINT 配置时初始化全局 OpenTelemetry
TracerProvider（OTLP/HTTP 导出）；未配置时不初始化——AgentScope 的
TracingMiddleware 在全局 tracer 未配置时逐 hook 短路，近零开销。

Agent 中间件统一在 build_agent_middlewares 装配：
    TracingMiddleware               全链路 Trace
    ReplyBudgetControlMiddleware    Token 预算护栏（REPLY_TOKEN_BUDGET > 0 才挂）
"""
from __future__ import annotations

import logging

from agentscope.middleware import ReplyBudgetControlMiddleware, TracingMiddleware

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

_initialized = False

_BUDGET_HINT = (
    "<system-reminder>本次会话已达到 Token 预算上限。请立即停止调用工具，"
    "基于当前已获得的信息给买家一个明确的收尾回复（如实说明信息可能不完整）。</system-reminder>"
)


def setup_tracing(settings: Settings) -> None:
    """按需初始化全局 OTel TracerProvider（幂等）。"""
    global _initialized
    if _initialized or not settings.otlp_endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "globex-agent"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{settings.otlp_endpoint.rstrip('/')}/v1/traces")),
    )
    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("OTel tracing 已启用：%s", settings.otlp_endpoint)


def build_agent_middlewares(settings: Settings) -> list:
    """全部 Agent 统一的中间件列表（Trace + 可选 Token 预算）。"""
    middlewares: list = [TracingMiddleware()]
    if settings.reply_token_budget > 0:
        middlewares.append(
            ReplyBudgetControlMiddleware(
                token_budget=settings.reply_token_budget,
                hint_message=_BUDGET_HINT,
            ),
        )
    return middlewares
