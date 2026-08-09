# -*- coding: utf-8 -*-
"""shared_breaker —— 熔断状态跨实例共享（16-5 章）

问题：`CircuitBreakerRegistry` 是进程内的。水平扩容成多个 API / worker 副本后，
每个进程各统计一份失败视图——A 副本已熔断某工具，B 副本还在继续打它，
下游依旧被打满，熔断等于只生效了 1/N。

方案：把「连续失败数」与「熔断打开时刻」放 Redis，所有副本读写同一份状态。
沿用 `CircuitBreakerRegistry` 的方法签名（`allow` / `record_success` /
`record_failure` / `status`），装配层按 `BREAKER_SHARED` 决定注入哪个实现，
`ToolResilienceMiddleware` 无需改动。

**降级原则**：Redis 不可用时一律「放行」并回落到本地状态，
绝不能因为熔断存储故障把正常工具调用全部拒掉——那是把可用性问题放大成事故。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from app.infrastructure.cache.redis_cache import RedisCache
from app.infrastructure.resilience import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

_KEY_PREFIX = "globex:breaker:"
# 状态 TTL 需明显大于冷却期，避免冷却还没到就被 Redis 过期清掉
_TTL_MULTIPLIER = 10


class SharedCircuitBreakerRegistry(CircuitBreakerRegistry):
    """Redis 支撑的熔断注册表。

    继承本地实现：Redis 不可用时自动退化为父类的进程内行为。
    """

    def __init__(
        self,
        cache: RedisCache,
        failure_threshold: int = 3,
        reset_seconds: float = 60.0,
    ) -> None:
        super().__init__(failure_threshold=failure_threshold, reset_seconds=reset_seconds)
        self._cache = cache

    @property
    def shared(self) -> bool:
        return self._cache.enabled

    def _key(self, tool_name: str) -> str:
        return f"{_KEY_PREFIX}{tool_name}"

    @property
    def _ttl(self) -> int:
        return max(60, int(self.reset_seconds * _TTL_MULTIPLIER))

    # ---- 以下四个方法保持与父类一致的签名，供中间件无感调用 ----

    async def allow_async(self, tool_name: str, now: Optional[float] = None) -> bool:
        if not self.shared:
            return self.allow(tool_name, now)
        state = await self._load(tool_name)
        if state is None:
            return True
        opened_at = state.get("opened_at")
        if not opened_at:
            return True
        elapsed = (now or time.time()) - float(opened_at)
        if elapsed < self.reset_seconds:
            return False
        # 冷却期满：标记半开，放一次探测
        state["half_open"] = 1
        await self._save(tool_name, state)
        return True

    async def record_success_async(self, tool_name: str) -> None:
        if not self.shared:
            self.record_success(tool_name)
            return
        await self._cache.delete(self._key(tool_name))

    async def record_failure_async(self, tool_name: str, now: Optional[float] = None) -> None:
        if not self.shared:
            self.record_failure(tool_name, now)
            return
        state = await self._load(tool_name) or {"failures": 0, "opened_at": 0, "half_open": 0}
        stamp = now or time.time()
        if state.get("half_open"):
            # 半开探测再次失败：重新打开并重置冷却计时
            state.update({"opened_at": stamp, "half_open": 0})
        else:
            state["failures"] = int(state.get("failures", 0)) + 1
            if state["failures"] >= self.failure_threshold:
                state["opened_at"] = stamp
        await self._save(tool_name, state)

    async def status_async(self, tool_name: str) -> str:
        if not self.shared:
            return self.status(tool_name)
        state = await self._load(tool_name)
        if state is None or not state.get("opened_at"):
            return "closed"
        return "half_open" if state.get("half_open") else "open"

    # ---- Redis 读写：任何异常都退化为「无状态」，即放行 ----

    async def _load(self, tool_name: str) -> Optional[dict]:
        try:
            return await self._cache.get_json(self._key(tool_name))
        except Exception as err:  # noqa: BLE001
            logger.warning("共享熔断状态读取失败，按放行处理：%s", err)
            return None

    async def _save(self, tool_name: str, state: dict) -> None:
        try:
            await self._cache.set_json(self._key(tool_name), state, self._ttl)
        except Exception as err:  # noqa: BLE001
            logger.warning("共享熔断状态写入失败（本次仅本地生效）：%s", err)
