# -*- coding: utf-8 -*-
"""ToolResilienceMiddleware

工具韧性中间件（复用 2.0 ToolMiddlewareBase 的洋葱式拦截）：

    超时：单次工具执行超过阈值即中断，返回结构化 error chunk，不把异常抛穿到 Agent 循环
    熔断：按工具名统计连续失败，达阈值后打开熔断，reset_seconds 内直接短路返回降级提示；
          冷却期满转半开，放一次探测请求，成功即闭合、失败即重新打开

设计取舍：熔断状态按 (工具名) 维度进程内共享（CircuitBreakerRegistry），
让同一工具在不同 Agent 实例间共享故障视图；降级返回始终是 ToolChunk(ERROR)，
Agent 能看到失败原因并如实告知买家，不会误以为工具成功。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Optional

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolBase, ToolChunk, ToolMiddlewareBase

from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus

logger = logging.getLogger(__name__)

# 工具超时分级（秒）：检索/知识库偏长，订单要快，子代理调度最宽松
DEFAULT_TIMEOUTS: dict[str, float] = {
    "product_search_tool": 15.0,
    "category_insight_tool": 15.0,
    "web_search_tool": 20.0,
    "create_order_tool": 10.0,
    "query_order_tool": 10.0,
    "cancel_order_tool": 10.0,
    "remember_preference_tool": 10.0,
    "task_dispatch": 180.0,
}
_FALLBACK_TIMEOUT = 30.0


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: Optional[float] = None
    half_open_probing: bool = False

    @property
    def status(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.half_open_probing else "open"


@dataclass
class CircuitBreakerRegistry:
    """进程内共享的熔断状态表（按工具名）。"""

    failure_threshold: int = 3
    reset_seconds: float = 60.0
    _states: dict[str, _CircuitState] = field(default_factory=dict)

    def _state(self, tool_name: str) -> _CircuitState:
        return self._states.setdefault(tool_name, _CircuitState())

    def status(self, tool_name: str) -> str:
        return self._state(tool_name).status

    def allow(self, tool_name: str, now: Optional[float] = None) -> bool:
        """是否放行本次调用；冷却期满自动转半开并放行一次探测。"""
        state = self._state(tool_name)
        if state.opened_at is None:
            return True
        elapsed = (now or time.monotonic()) - state.opened_at
        if elapsed < self.reset_seconds:
            return False
        state.half_open_probing = True
        return True

    def record_success(self, tool_name: str) -> None:
        self._states[tool_name] = _CircuitState()

    def record_failure(self, tool_name: str, now: Optional[float] = None) -> None:
        state = self._state(tool_name)
        if state.half_open_probing:
            # 半开探测再次失败：重新打开并重置冷却计时
            state.opened_at = now or time.monotonic()
            state.half_open_probing = False
            return
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold:
            state.opened_at = now or time.monotonic()


class ToolResilienceMiddleware(ToolMiddlewareBase):
    """工具超时 + 熔断中间件。

    熔断注册表可以是进程内的 `CircuitBreakerRegistry`，
    也可以是 Redis 支撑的 `SharedCircuitBreakerRegistry`（跨实例共享）。
    后者的读写是异步的，故这里统一走 `_allow` / `_record_*` 三个适配函数：
    注册表提供 `*_async` 时优先 await 它，否则回落同步方法。
    """

    def __init__(
        self,
        registry: CircuitBreakerRegistry,
        bus: Optional[TradeEventBus] = None,
        timeouts: Optional[dict[str, float]] = None,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._timeouts = timeouts or DEFAULT_TIMEOUTS

    def _timeout_for(self, tool_name: str) -> float:
        return self._timeouts.get(tool_name, _FALLBACK_TIMEOUT)

    def _publish_circuit(self, tool_name: str, circuit: str, detail: str) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            ShoppingContext.current_session_id(),
            "tool.result",
            {"tool": tool_name, "circuit": circuit, "error": detail},
        )

    async def on_tool_call(
        self,
        tool: ToolBase,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[ToolChunk, None]],
    ) -> AsyncGenerator[ToolChunk, None]:
        tool_name = tool.name

        if not await _allow(self._registry, tool_name):
            detail = f"{tool_name} 连续失败已熔断，暂不可用，请稍后再试或改用其他方式"
            logger.warning("工具熔断短路：%s", tool_name)
            self._publish_circuit(tool_name, "open", detail)
            yield ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {detail}")],
                state=ToolResultState.ERROR,
            )
            return

        timeout = self._timeout_for(tool_name)
        chunks: list[ToolChunk] = []
        try:
            # 先在超时保护内收集全部 chunk，再对外 yield：
            # 保证超时能被拦在中间件内部，不会把半截流交给 Agent
            async def _collect() -> list[ToolChunk]:
                collected: list[ToolChunk] = []
                async for chunk in next_handler(**input_kwargs):
                    collected.append(chunk)
                return collected

            chunks = await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            await _record_failure(self._registry, tool_name)
            detail = f"{tool_name} 执行超过 {timeout:.0f} 秒已中断"
            logger.warning("工具超时：%s（%.0fs）", tool_name, timeout)
            self._publish_circuit(tool_name, await _status(self._registry, tool_name), detail)
            yield ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {detail}")],
                state=ToolResultState.ERROR,
            )
            return
        except Exception as err:  # noqa: BLE001 —— 未捕获异常也计入失败并降级
            await _record_failure(self._registry, tool_name)
            detail = f"{tool_name} 执行异常：{err}"
            logger.warning("工具异常：%s（%s）", tool_name, err)
            self._publish_circuit(tool_name, await _status(self._registry, tool_name), detail)
            yield ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {detail}")],
                state=ToolResultState.ERROR,
            )
            return

        # 工具自身返回 ERROR 也计入连续失败（如下游 5xx 持续报错）
        if chunks and chunks[-1].state == ToolResultState.ERROR:
            await _record_failure(self._registry, tool_name)
        else:
            await _record_success(self._registry, tool_name)

        for chunk in chunks:
            yield chunk


# ---- 注册表适配：共享实现的读写是异步的，本地实现是同步的 ----


async def _allow(registry: Any, tool_name: str) -> bool:
    if hasattr(registry, "allow_async"):
        return await registry.allow_async(tool_name)
    return registry.allow(tool_name)


async def _record_failure(registry: Any, tool_name: str) -> None:
    if hasattr(registry, "record_failure_async"):
        await registry.record_failure_async(tool_name)
        return
    registry.record_failure(tool_name)


async def _record_success(registry: Any, tool_name: str) -> None:
    if hasattr(registry, "record_success_async"):
        await registry.record_success_async(tool_name)
        return
    registry.record_success(tool_name)


async def _status(registry: Any, tool_name: str) -> str:
    if hasattr(registry, "status_async"):
        return await registry.status_async(tool_name)
    return registry.status(tool_name)
