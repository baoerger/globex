# -*- coding: utf-8 -*-
"""四期第一部分：网关配额闸门、瞬时故障判定、限流回退

不依赖真实网关：用桩模型（StubChatModel）制造限流与流式行为。
"""
from __future__ import annotations

import asyncio
import time

import pytest

from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.transient import is_transient_error

pytestmark = pytest.mark.asyncio


class TestTransientJudgement:
    @pytest.mark.parametrize(
        "message",
        [
            "Too many concurrent requests.",
            "Request rate increased too quickly.",
            "Throttling.Concurrency",
            "Error code: 429",
            "Read timeout",
            "502 Bad Gateway",
            "Connection reset by peer",
        ],
    )
    async def test_gateway_failures_are_transient(self, message):
        assert is_transient_error(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "The model `qwen3-plus` does not exist or you do not have access to it.",
            "invalid_request_error: messages must not be empty",
            "库存不足",
        ],
    )
    async def test_business_failures_are_not_transient(self, message):
        """业务错误不能被当成瞬时故障重试，否则会掩盖真实问题。"""
        assert is_transient_error(RuntimeError(message)) is False


class TestGatewayThrottle:
    async def test_concurrency_cap_is_enforced(self):
        throttle = GatewayThrottle(max_concurrency=2, min_interval_seconds=0)
        in_flight = 0
        peak = 0

        async def worker():
            nonlocal in_flight, peak
            async with throttle.slot():
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(6)))
        assert peak == 2, f"同时在飞的请求应被压到 2，实际 {peak}"

    async def test_min_interval_spaces_out_starts(self):
        throttle = GatewayThrottle(max_concurrency=8, min_interval_seconds=0.05)
        starts: list[float] = []

        async def worker():
            async with throttle.slot():
                starts.append(time.monotonic())

        await asyncio.gather(*(worker() for _ in range(4)))
        starts.sort()
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert all(gap >= 0.04 for gap in gaps), f"起跑间隔未被摊开：{gaps}"

    async def test_slot_released_when_body_raises(self):
        """异常路径也必须归还名额，否则一次报错就会永久占死一个并发位。"""
        throttle = GatewayThrottle(max_concurrency=1, min_interval_seconds=0)
        with pytest.raises(RuntimeError):
            async with throttle.slot():
                raise RuntimeError("boom")

        # 还能再拿到名额说明已归还
        await asyncio.wait_for(throttle.slot().__aenter__(), timeout=0.5)
