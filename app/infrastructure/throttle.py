# -*- coding: utf-8 -*-
"""GatewayThrottle

大模型网关配额闸门：同时约束「并发数」与「请求起点间隔」。

为什么两个都要治：实测网关既报 "Too many concurrent requests"（同时在飞的请求超限），
也报 "Request rate increased too quickly"（速率爬升过快）。只限并发挡不住一批请求
同时起跑，还要把起跑时刻摊开；只限速率又挡不住长请求堆积。

进程内实现，多实例部署时需换成 Redis 令牌桶（见模块二），端口语义保持一致。
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


class GatewayThrottle:
    def __init__(self, max_concurrency: int, min_interval_seconds: float) -> None:
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._interval_lock = asyncio.Lock()
        self._min_interval = max(0.0, min_interval_seconds)
        self._last_start = 0.0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """占用一个请求名额。

        注意调用方：流式请求必须把本上下文持有到流耗尽，否则名额会在数据还没
        读完时就被释放，限流等于没做（见 ThrottledChatModel）。
        """
        async with self._semaphore:
            await self._space_out()
            yield

    async def _space_out(self) -> None:
        if self._min_interval <= 0:
            return
        # 串行化起跑时刻，保证相邻两次请求的间隔不小于 min_interval
        async with self._interval_lock:
            wait = self._min_interval - (time.monotonic() - self._last_start)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_start = time.monotonic()
