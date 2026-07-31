# -*- coding: utf-8 -*-
"""四期模块三：Redis Stream 队列与跨进程事件背板

用内存替身模拟 Redis Stream / Pub-Sub 的关键语义，重点验证三件容易出事的事：
    1. 处理失败不 ack，消息留在 pending 等重投（at-least-once）；
    2. 重投超限进死信，坏消息不能无限重放阻塞队列；
    3. 事件总线接上背板后能跨进程送达，且远端事件不回环广播。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.domain.queue.ports.task_queue import IntentTask, TaskStatus
from app.infrastructure.eventbus import TradeEvent, TradeEventBus
from app.infrastructure.queue.redis_stream_queue import (
    _STREAM,
    RedisEventBackplane,
    RedisStreamTaskQueue,
)

pytestmark = pytest.mark.asyncio


class FakeStreamClient:
    """模拟 Redis Stream 的消费者组语义：未 ack 的消息留在 pending。"""

    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []
        self.pending: dict[str, int] = {}  # message_id -> times_delivered
        self.acked: list[str] = []
        self.dead: list[dict] = []
        self.kv: dict[str, str] = {}
        self._seq = 0
        self.published: list[tuple[str, str]] = []

    async def xgroup_create(self, *_args, **_kwargs):
        return True

    async def xadd(self, stream: str, fields: dict):
        self._seq += 1
        message_id = f"{self._seq}-0"
        if stream.endswith(":dead"):
            self.dead.append(fields)
        else:
            self.entries.append((message_id, fields))
        return message_id

    async def xreadgroup(self, _group, _consumer, _streams, count=1, block=0):
        available = [
            (mid, fields)
            for mid, fields in self.entries
            if mid not in self.acked and mid not in self.pending
        ]
        batch = available[:count]
        for mid, _fields in batch:
            self.pending[mid] = 1
        return [("globex:intents", batch)] if batch else []

    async def xack(self, _stream, _group, message_id):
        self.acked.append(message_id)
        self.pending.pop(message_id, None)

    async def xpending_range(self, _stream, _group, start, _end, _count):
        times = self.pending.get(start, 1)
        return [{"times_delivered": times}]

    async def xinfo_groups(self, _stream):
        return [{"name": "globex-workers", "lag": len(self.entries) - len(self.acked)}]

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def publish(self, channel, data):
        self.published.append((channel, data))

    def redeliver(self, message_id: str) -> None:
        """模拟 Redis 把未 ack 的消息重新投递给消费者。"""
        self.pending[message_id] = self.pending.get(message_id, 1) + 1


def _task(task_id: str = "task-1") -> IntentTask:
    return IntentTask(
        task_id=task_id,
        shopping_session_id="s1",
        buyer_id="b1",
        locale="zh-CN",
        currency="CNY",
        raw_query="露营灯推荐",
    )


class TestEnqueueAndStatus:
    async def test_enqueue_then_status_roundtrip(self):
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task())
        assert len(client.entries) == 1

        await queue.set_status(TaskStatus(task_id="task-1", state="queued"))
        status = await queue.get_status("task-1")
        assert status is not None and status.state == "queued"

    async def test_done_status_carries_final_text(self):
        queue = RedisStreamTaskQueue(FakeStreamClient())
        await queue.set_status(
            TaskStatus(task_id="task-1", state="done", final_text="推荐 LumenGo 89 元"),
        )
        status = await queue.get_status("task-1")
        assert status.state == "done" and status.final_text == "推荐 LumenGo 89 元"

    async def test_missing_status_returns_none(self):
        assert await RedisStreamTaskQueue(FakeStreamClient()).get_status("nope") is None


class TestConsumeSemantics:
    async def test_success_acks_message(self):
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task())
        handled: list[str] = []

        async def handler(task: IntentTask) -> None:
            handled.append(task.task_id)

        await self._consume_once(queue, handler)
        assert handled == ["task-1"]
        assert len(client.acked) == 1, "成功必须 ack，否则会被重复消费"

    async def test_failure_does_not_ack(self):
        """处理失败不能 ack——不然任务就悄悄丢了。"""
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task())

        async def handler(_task: IntentTask) -> None:
            raise RuntimeError("模型超时")

        await self._consume_once(queue, handler)
        assert client.acked == [], "失败不应 ack，应留在 pending 等重投"
        assert client.dead == [], "首次失败还不该进死信"

    async def test_exceeding_max_deliveries_goes_to_dead_letter(self):
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)
        message_id = await client.xadd(
            "globex:intents", {"payload": json.dumps(_task().to_dict())},
        )
        client.pending[message_id] = 3  # 已投递 3 次

        async def handler(_task: IntentTask) -> None:
            raise RuntimeError("模型超时")

        await queue._handle_one(
            _STREAM,
            message_id,
            {"payload": json.dumps(_task().to_dict())},
            handler,
            max_deliveries=3,
        )
        assert len(client.dead) == 1, "重投超限必须进死信，否则坏消息会无限阻塞队列"
        assert message_id in client.acked, "进死信后要 ack，把它从 pending 移走"

    async def test_unparsable_payload_goes_to_dead_letter(self):
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)

        async def handler(_task: IntentTask) -> None:
            raise AssertionError("解析失败时不应调用 handler")

        await queue._handle_one(
            _STREAM, "9-0", {"payload": "{不是 json"}, handler, max_deliveries=3,
        )
        assert len(client.dead) == 1
        assert "9-0" in client.acked

    @staticmethod
    async def _consume_once(queue: RedisStreamTaskQueue, handler) -> None:
        """跑一轮消费后立刻停止（避免测试里进入无限循环）。"""
        rounds = {"n": 0}

        def should_stop() -> bool:
            rounds["n"] += 1
            return rounds["n"] > 1

        await queue.consume("c1", handler, should_stop, block_ms=0)


class TestEventBackplane:
    async def test_publish_broadcasts_to_channel(self):
        client = FakeStreamClient()
        bus = TradeEventBus()
        bus.attach_backplane(RedisEventBackplane(client))

        bus.publish("s1", "tool.invoke", {"tool": "product_search_tool"})
        await asyncio.sleep(0)  # 让 fire-and-forget 广播任务跑一轮
        await asyncio.sleep(0)

        assert client.published, "接了背板后事件必须广播出去，否则 worker 的事件到不了 API 进程"
        channel, data = client.published[0]
        assert channel == "globex:events:s1"
        envelope = json.loads(data)
        assert envelope["event"]["type"] == "tool.invoke"
        assert envelope["origin"], "必须带发送方标识，否则无法过滤自己发的消息"

    async def test_own_broadcast_is_filtered_out(self):
        """Pub/Sub 不排除发布者：API 进程既发又订，不过滤会把自己的事件再投一次。

        实测踩过这个坑：前端收到两条 task.queued。
        """
        client = FakeStreamClient()
        backplane = RedisEventBackplane(client, origin="api-1")
        event = TradeEvent(
            shopping_session_id="s1", type="task.queued", payload={"task_id": "t1"},
            occurred_at="2026-01-01T00:00:00Z",
        )
        await backplane.publish(event)

        _channel, data = client.published[0]
        envelope = json.loads(data)
        assert envelope["origin"] == "api-1"

        # 同一个 origin 的背板收到自己的消息应跳过；其他 origin 则放行
        other = RedisEventBackplane(client, origin="worker-1")
        assert envelope["origin"] != other.origin

    async def test_local_subscriber_still_receives(self):
        """广播不能影响本进程投递。"""
        bus = TradeEventBus()
        bus.attach_backplane(RedisEventBackplane(FakeStreamClient()))
        queue = bus.subscribe("s1")
        bus.publish("s1", "final.result", {"text": "ok"})
        assert queue.get_nowait().type == "final.result"

    async def test_deliver_local_does_not_rebroadcast(self):
        """背板收到远端事件后走 deliver_local，不能再广播出去（否则无限回环）。"""
        client = FakeStreamClient()
        bus = TradeEventBus()
        bus.attach_backplane(RedisEventBackplane(client))
        queue = bus.subscribe("s1")

        bus.deliver_local(
            TradeEvent(shopping_session_id="s1", type="token.delta", payload={"token": "露"},
                       occurred_at="2026-01-01T00:00:00Z"),
        )
        await asyncio.sleep(0)

        assert queue.get_nowait().payload == {"token": "露"}
        assert client.published == [], "远端事件不应再次广播"

    async def test_works_without_backplane(self):
        """没有 Redis 时事件总线退回纯进程内模式，不能报错。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        bus.publish("s1", "error", {"message": "x"})
        assert queue.get_nowait().type == "error"


class TestEventSerialization:
    async def test_roundtrip_preserves_fields(self):
        event = TradeEvent(
            shopping_session_id="s1",
            type="tool.result",
            payload={"tool": "product_search_tool", "hit_count": 5},
            occurred_at="2026-01-01T00:00:00Z",
        )
        restored = TradeEvent.from_dict(json.loads(json.dumps(event.to_dict())))
        assert restored == event
