# -*- coding: utf-8 -*-
"""llm

统一创建 AgentScope 2.0 大模型对象。全项目只从这里拿 model，
主 / 子 Agent 各自持有独立实例。

2.0 的模型接入方式：OpenAICredential（携带 api_key + base_url，天然支持
OpenAI 兼容网关）→ OpenAIChatModel(credential=..., model=...)。

四期在此加两层框架不覆盖的东西：
    1. 配额闸门：闸门必须持有到「流耗尽」。流式调用返回的是异步生成器，
       若在 `async with slot()` 内直接 return，名额会在数据还没读完时释放，
       限流等于没做；
    2. 限流回退：实测网关配额池紧张时主模型单发也会 429，退避重试用尽后换用
       备用模型，并发 model.fallback 事件如实告知，不静默降级。
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, AsyncGenerator, Optional

from agentscope.credential import OpenAICredential
from agentscope.message import Msg
from agentscope.model import ChatResponse, OpenAIChatModel
from agentscope.tool import ToolChoice

from app.infrastructure.budget import current_tier, get_budget, minimal_mode_hint
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.settings import Settings
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)


class ThrottledChatModel(OpenAIChatModel):
    """带配额闸门、退避重试与限流回退的 OpenAIChatModel。"""

    def __init__(
        self,
        *,
        throttle: GatewayThrottle,
        fallback: Optional[OpenAIChatModel] = None,
        max_transient_retries: int = 2,
        retry_base_seconds: float = 6.0,
        bus: Optional[TradeEventBus] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._throttle = throttle
        self._fallback = fallback
        self._max_transient_retries = max_transient_retries
        self._retry_base_seconds = retry_base_seconds
        self._bus = bus

    async def __call__(  # type: ignore[override]
        self,
        messages: list[Msg],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[ToolChoice] = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        # 手动进出上下文而非 async with：流式分支要把名额移交给包装生成器
        slot = self._throttle.slot()
        await slot.__aenter__()
        try:
            result = await self._call_with_fallback(messages, tools, tool_choice, **kwargs)
        except BaseException:
            await slot.__aexit__(*sys.exc_info())
            raise

        if not hasattr(result, "__aiter__"):
            await slot.__aexit__(None, None, None)
            _charge_budget(result)
            return result
        return self._release_after_stream(slot, result)

    @staticmethod
    async def _release_after_stream(slot: Any, stream: Any) -> AsyncGenerator[ChatResponse, None]:
        """把闸门名额持有到流真正读完（含调用方提前中断的情况）。"""
        last: Any = None
        try:
            async for chunk in stream:
                last = chunk
                yield chunk
        finally:
            # 流式的 usage 在最后一个 chunk 上，读完再记账
            _charge_budget(last)
            await slot.__aexit__(None, None, None)

    async def _invoke_upstream(
        self,
        messages: list[Msg],
        tools: Optional[list[dict]],
        tool_choice: Optional[ToolChoice],
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """真正打上游的一跳。抽成方法便于替换与测试。

        同时是四档预算降级（16-4 章）的作用点：
        预算充足（main）走主模型；剩余不足时切到更便宜的备用模型，
        并发 model.fallback 事件如实告知——**降级不静默**。
        minimal 档额外注入简洁模式提示，压住 Think 长度。
        """
        tier = current_tier()
        if tier != "main":
            hint = minimal_mode_hint()
            if hint:
                messages = [*messages, Msg(name="system", content=hint, role="system")]
            if self._fallback is not None:
                logger.info("Token 预算档位 %s，切用备用模型 %s", tier, self._fallback.model)
                self._publish_budget_tier(tier)
                return await self._fallback(messages, tools, tool_choice, **kwargs)
        return await super().__call__(messages, tools, tool_choice, **kwargs)

    def _publish_budget_tier(self, tier: str) -> None:
        if self._bus is None or self._fallback is None:
            return
        self._bus.publish(
            ShoppingContext.current_session_id(),
            "model.fallback",
            {
                "from": self.model,
                "to": self._fallback.model,
                "reason": f"Token 预算档位 {tier}",
                "budget_tier": tier,
            },
        )

    async def _call_with_fallback(
        self,
        messages: list[Msg],
        tools: Optional[list[dict]],
        tool_choice: Optional[ToolChoice],
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        last_error: Optional[BaseException] = None
        for attempt in range(self._max_transient_retries + 1):
            try:
                return await self._invoke_upstream(messages, tools, tool_choice, **kwargs)
            except BaseException as err:
                if not is_transient_error(err):
                    raise
                last_error = err
                if attempt < self._max_transient_retries:
                    # 指数退避：网关速率类限流对固定间隔重试不敏感
                    delay = self._retry_base_seconds * (3**attempt)
                    logger.warning(
                        "模型 %s 遇上游瞬时故障，%.0fs 后重试（第 %d/%d 次）：%s",
                        self.model, delay, attempt + 1, self._max_transient_retries, err,
                    )
                    await asyncio.sleep(delay)

        if self._fallback is None:
            raise last_error  # type: ignore[misc]

        logger.warning("模型 %s 重试用尽，回退到 %s：%s", self.model, self._fallback.model, last_error)
        self._publish_fallback(str(last_error))
        return await self._fallback(messages, tools, tool_choice, **kwargs)

    def _publish_fallback(self, reason: str) -> None:
        if self._bus is None or self._fallback is None:
            return
        session_id = ShoppingContext.current_session_id()
        self._bus.publish(
            session_id,
            "model.fallback",
            {"from": self.model, "to": self._fallback.model, "reason": reason},
        )


def _safe_field(source: Any, name: str) -> Any:
    """宽容取字段。

    坑：ChatResponse.usage 的 `__getattr__` 对缺失字段抛 **KeyError**，
    而 `getattr(obj, name, default)` 只吃 AttributeError——直接用 getattr 带默认值
    依旧会把 KeyError 抛到主链路，把整轮对话搞成 [error]。实测踩过。
    """
    if isinstance(source, dict):
        return source.get(name)
    try:
        return getattr(source, name, None)
    except Exception:  # noqa: BLE001 —— 包含 KeyError 等非标准实现
        return None


def _charge_budget(response: Any) -> None:
    """把一次模型调用的 token 记进当前意图的预算账本。

    未启用预算（TOKEN_BUDGET_TOTAL=0）时直接返回，零开销。
    usage 字段各网关存在差异，取不到就不计——**记账失败绝不能影响主链路**。
    """
    try:
        budget = get_budget()
        if budget is None or response is None:
            return
        usage = _safe_field(response, "usage")
        if usage is None:
            return
        total = _safe_field(usage, "total_tokens")
        if total is None:
            prompt = _safe_field(usage, "input_tokens") or _safe_field(usage, "prompt_tokens") or 0
            completion = (
                _safe_field(usage, "output_tokens")
                or _safe_field(usage, "completion_tokens")
                or 0
            )
            total = int(prompt) + int(completion)
        budget.charge("llm", int(total))
    except Exception as err:  # noqa: BLE001
        logger.debug("Token 记账跳过（不影响主链路）：%s", err)


def create_chat_model(
    settings: Settings,
    stream: bool = True,
    throttle: Optional[GatewayThrottle] = None,
    bus: Optional[TradeEventBus] = None,
) -> OpenAIChatModel:
    credential = OpenAICredential(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    common = {
        "credential": credential,
        "stream": stream,
        # 上下文窗口：压缩触发阈值（ContextConfig.trigger_ratio）按此值比例计算
        "context_size": settings.context_size,
    }
    fallback = (
        OpenAIChatModel(model=settings.llm_fallback_model, **common)
        if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model
        else None
    )
    return ThrottledChatModel(
        model=settings.llm_model,
        throttle=throttle or GatewayThrottle(settings.llm_max_concurrency, settings.llm_min_interval_seconds),
        fallback=fallback,
        max_transient_retries=settings.llm_max_retries,
        bus=bus,
        **common,
    )
