# -*- coding: utf-8 -*-
"""P3 基础设施单测：熔断跨实例共享 + 队列双流优先级。

用内存假 Redis 替代真实服务，验证行为契约；真实 Redis 的连通性
在端到端环节用 docker 起的实例验证。
"""
import json

import pytest

from app.domain.queue.ports.task_queue import IntentTask
from app.infrastructure.queue.redis_stream_queue import (
    _LARGE_STREAM,
    _STREAM,
    RedisStreamTaskQueue,
)
from app.infrastructure.shared_breaker import SharedCircuitBreakerRegistry


class FakeCache:
    """最小 RedisCache 替身：只实现共享熔断用到的三个方法。"""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self.store: dict = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def get_json(self, key: str):
        return self.store.get(key)

    async def set_json(self, key: str, value, ttl_seconds: int) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


class BrokenCache(FakeCache):
    async def get_json(self, key: str):
        raise RuntimeError("redis down")

    async def set_json(self, key: str, value, ttl_seconds: int) -> None:
        raise RuntimeError("redis down")


class TestSharedCircuitBreaker:
    async def test_opens_after_threshold_and_shares_state(self):
        cache = FakeCache()
        a = SharedCircuitBreakerRegistry(cache, failure_threshold=2, reset_seconds=60)
        b = SharedCircuitBreakerRegistry(cache, failure_threshold=2, reset_seconds=60)

        await a.record_failure_async("product_search_tool")
        await a.record_failure_async("product_search_tool")

        assert await a.status_async("product_search_tool") == "open"
        # 另一个"副本"共享同一份状态：这正是本模块存在的意义
        assert await b.status_async("product_search_tool") == "open"
        assert await b.allow_async("product_search_tool") is False

    async def test_success_clears_state(self):
        cache = FakeCache()
        reg = SharedCircuitBreakerRegistry(cache, failure_threshold=1, reset_seconds=60)
        await reg.record_failure_async("web_search_tool")
        assert await reg.status_async("web_search_tool") == "open"

        await reg.record_success_async("web_search_tool")
        assert await reg.status_async("web_search_tool") == "closed"
        assert await reg.allow_async("web_search_tool") is True

    async def test_half_open_after_cooldown_then_reopen(self):
        cache = FakeCache()
        reg = SharedCircuitBreakerRegistry(cache, failure_threshold=1, reset_seconds=10)
        await reg.record_failure_async("web_search_tool", now=1000.0)
        assert await reg.allow_async("web_search_tool", now=1005.0) is False, "冷却期内不放行"

        # 冷却期满：放一次探测并转半开
        assert await reg.allow_async("web_search_tool", now=1011.0) is True
        assert await reg.status_async("web_search_tool") == "half_open"

        # 探测再失败 → 重新打开且冷却计时重置
        await reg.record_failure_async("web_search_tool", now=1012.0)
        assert await reg.status_async("web_search_tool") == "open"
        assert await reg.allow_async("web_search_tool", now=1015.0) is False

    async def test_falls_back_to_local_when_cache_disabled(self):
        reg = SharedCircuitBreakerRegistry(FakeCache(enabled=False), failure_threshold=1)
        assert reg.shared is False
        await reg.record_failure_async("t")
        assert await reg.status_async("t") == "open", "无 Redis 时退化为进程内熔断，仍要生效"

    async def test_redis_failure_allows_traffic(self):
        """熔断存储故障绝不能把正常调用全拒掉。"""
        reg = SharedCircuitBreakerRegistry(BrokenCache(), failure_threshold=1)
        await reg.record_failure_async("t")  # 写失败被吞掉
        assert await reg.allow_async("t") is True
        assert await reg.status_async("t") == "closed"


class FakeRedisClient:
    """记录 xadd 目标流，够用来验证路由。"""

    def __init__(self) -> None:
        self.added: list[tuple[str, dict]] = []
        self.groups: list[tuple[str, str]] = []
        self.acked: list[tuple[str, str]] = []

    async def xgroup_create(self, stream, group, id="0", mkstream=True):
        self.groups.append((stream, group))

    async def xadd(self, stream, fields):
        self.added.append((stream, fields))

    async def xack(self, stream, group, message_id):
        self.acked.append((stream, message_id))

    async def xinfo_groups(self, stream):
        return [{"name": "globex-workers", "lag": 2}]


def _task(priority: int) -> IntentTask:
    return IntentTask(
        task_id="t1",
        shopping_session_id="s1",
        buyer_id="b1",
        locale="zh-CN",
        currency="CNY",
        raw_query="找个登机箱",
        priority=priority,
    )


class TestQueuePriorityRouting:
    async def test_normal_task_goes_to_normal_stream(self):
        client = FakeRedisClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task(priority=0))
        assert client.added[0][0] == _STREAM

    async def test_large_task_goes_to_large_stream(self):
        client = FakeRedisClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task(priority=1))
        assert client.added[0][0] == _LARGE_STREAM

    async def test_priority_survives_serialization(self):
        payload = _task(priority=1).to_dict()
        restored = IntentTask.from_dict(json.loads(json.dumps(payload)))
        assert restored.priority == 1

    async def test_priority_defaults_to_zero_for_legacy_payload(self):
        """老消息没有 priority 字段，反序列化不能炸。"""
        legacy = {
            "task_id": "t1",
            "shopping_session_id": "s1",
            "buyer_id": "b1",
            "locale": "zh-CN",
            "currency": "CNY",
            "raw_query": "q",
            "enqueued_at": "",
        }
        assert IntentTask.from_dict(legacy).priority == 0

    async def test_ensure_group_creates_both_streams(self):
        client = FakeRedisClient()
        await RedisStreamTaskQueue(client).ensure_group()
        assert {stream for stream, _ in client.groups} == {_STREAM, _LARGE_STREAM}

    async def test_depth_sums_both_streams(self):
        client = FakeRedisClient()
        assert await RedisStreamTaskQueue(client).depth() == 4  # 两条流各 lag=2

    async def test_handle_one_acks_originating_stream(self):
        """ack 必须回到消息所属的流，否则大请求流的消息永远不被确认。"""
        client = FakeRedisClient()
        queue = RedisStreamTaskQueue(client)
        handled: list[IntentTask] = []

        async def handler(task: IntentTask) -> None:
            handled.append(task)

        payload = json.dumps(_task(priority=1).to_dict(), ensure_ascii=False)
        await queue._handle_one(  # noqa: SLF001
            _LARGE_STREAM, "1-0", {"payload": payload}, handler, max_deliveries=3,
        )

        assert len(handled) == 1
        assert client.acked == [(_LARGE_STREAM, "1-0")]
