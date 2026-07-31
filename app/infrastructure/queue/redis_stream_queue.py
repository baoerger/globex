# -*- coding: utf-8 -*-
"""RedisStreamTaskQueue + RedisEventBackplane

削峰用 Redis Stream（消费者组 + ack + pending 重投 + 死信），
跨进程事件广播用 Redis Pub/Sub。两者共用一个 Redis 连接。

为什么用 Stream 而不是 List：Stream 有消费者组与未确认（pending）列表，
worker 崩溃后未 ack 的消息能被重新领取，List 做不到这点。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Callable, Optional

from app.domain.queue.ports.task_queue import IntentTask, TaskQueue, TaskStatus
from app.infrastructure.eventbus import TradeEvent

logger = logging.getLogger(__name__)

_STREAM = "globex:intents"
# 大请求（长会话）单独一条流：与正常流同一消费者组名，
# 但 xreadgroup 里正常流排在前面——Redis 按传入顺序返回，天然形成优先级。
_LARGE_STREAM = "globex:intents:large"
_DEAD_STREAM = "globex:intents:dead"
_GROUP = "globex-workers"
_STATUS_PREFIX = "globex:task:"
_STATUS_TTL = 3600
_EVENT_CHANNEL_PREFIX = "globex:events:"


class RedisStreamTaskQueue(TaskQueue):
    def __init__(self, client: Any) -> None:
        self._client = client

    async def ensure_group(self) -> None:
        """幂等创建消费者组（两条流各一个）。组已存在时 Redis 抛 BUSYGROUP，属正常情况。"""
        for stream in (_STREAM, _LARGE_STREAM):
            try:
                await self._client.xgroup_create(stream, _GROUP, id="0", mkstream=True)
            except Exception as err:  # noqa: BLE001
                if "BUSYGROUP" not in str(err):
                    raise

    async def enqueue(self, task: IntentTask) -> None:
        stream = _LARGE_STREAM if task.priority > 0 else _STREAM
        await self._client.xadd(stream, {"payload": json.dumps(task.to_dict(), ensure_ascii=False)})

    async def set_status(self, status: TaskStatus) -> None:
        await self._client.set(
            f"{_STATUS_PREFIX}{status.task_id}",
            json.dumps(
                {
                    "task_id": status.task_id,
                    "state": status.state,
                    "final_text": status.final_text,
                    "error": status.error,
                },
                ensure_ascii=False,
            ),
            ex=_STATUS_TTL,
        )

    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        raw = await self._client.get(f"{_STATUS_PREFIX}{task_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        position = await self.depth() if data.get("state") == "queued" else 0
        return TaskStatus(
            task_id=data["task_id"],
            state=data["state"],
            final_text=data.get("final_text", ""),
            error=data.get("error", ""),
            queue_position=position,
        )

    async def depth(self) -> int:
        """两条流未被消费者组读取的消息数之和（lag）。取不到时退回 0，不让观测拖垮主链路。"""
        total = 0
        for stream in (_STREAM, _LARGE_STREAM):
            total += await self._stream_depth(stream)
        return total

    async def _stream_depth(self, stream: str) -> int:
        try:
            groups = await self._client.xinfo_groups(stream)
        except Exception:  # noqa: BLE001 —— stream 还没创建
            return 0
        for group in groups:
            if group.get("name") == _GROUP:
                lag = group.get("lag")
                if lag is not None:
                    return int(lag)
                return int(group.get("pending", 0))
        return 0

    async def consume(
        self,
        consumer_name: str,
        handler: Callable[[IntentTask], Any],
        should_stop: Callable[[], bool],
        block_ms: int = 2000,
        max_deliveries: int = 3,
        concurrency: int = 1,
    ) -> None:
        """消费循环。

        - 处理成功才 ack；失败则不 ack，留给下次重投
        - 投递次数超过 max_deliveries 进死信流，避免坏消息无限重放阻塞队列
        - concurrency > 1 时单次领多条并发跑（削峰的实质就是这个并发度上限）
        - 退出前等在途任务跑完；未 ack 的由 pending 机制重投，不会丢
        """
        await self.ensure_group()
        in_flight: set[asyncio.Task] = set()
        while not should_stop():
            free_slots = max(0, concurrency - len(in_flight))
            if free_slots == 0:
                await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                in_flight = {task for task in in_flight if not task.done()}
                continue
            try:
                # 正常流排前：Redis 按传入顺序返回，短会话不会被长会话堵在后面
                batches = await self._client.xreadgroup(
                    _GROUP,
                    consumer_name,
                    {_STREAM: ">", _LARGE_STREAM: ">"},
                    count=free_slots,
                    block=block_ms,
                )
            except Exception as err:  # noqa: BLE001
                logger.warning("队列读取失败，稍后重试：%s", err)
                await asyncio.sleep(1)
                continue
            if not batches:
                in_flight = {task for task in in_flight if not task.done()}
                continue
            for stream, entries in batches:
                # ack 必须回到消息所属的流，不能写死 _STREAM
                stream_name = stream.decode() if isinstance(stream, bytes) else str(stream)
                for message_id, fields in entries:
                    task = asyncio.create_task(
                        self._handle_one(
                            stream_name, message_id, fields, handler, max_deliveries,
                        ),
                    )
                    in_flight.add(task)

        if in_flight:
            logger.info("等待 %d 个在途任务完成后退出", len(in_flight))
            await asyncio.gather(*in_flight, return_exceptions=True)

    async def _handle_one(
        self,
        stream: str,
        message_id: str,
        fields: dict,
        handler: Callable[[IntentTask], Any],
        max_deliveries: int,
    ) -> None:
        raw = fields.get("payload")
        if not raw:
            await self._client.xack(stream, _GROUP, message_id)
            return
        try:
            task = IntentTask.from_dict(json.loads(raw))
        except Exception as err:  # noqa: BLE001 —— 解不开的消息直接进死信，不能卡住队列
            logger.warning("任务解析失败，进死信：%s（%s）", message_id, err)
            await self._client.xadd(_DEAD_STREAM, {"payload": raw, "reason": str(err)})
            await self._client.xack(stream, _GROUP, message_id)
            return

        try:
            await handler(task)
            await self._client.xack(stream, _GROUP, message_id)
        except Exception as err:  # noqa: BLE001
            deliveries = await self._delivery_count(stream, message_id)
            if deliveries >= max_deliveries:
                logger.error("任务重试超限进死信：%s（%s）", task.task_id, err)
                await self._client.xadd(
                    _DEAD_STREAM, {"payload": raw, "reason": str(err)},
                )
                await self._client.xack(stream, _GROUP, message_id)
            else:
                # 不 ack：留在 pending 里等重投
                logger.warning("任务处理失败（第 %d 次投递）：%s（%s）", deliveries, task.task_id, err)

    async def _delivery_count(self, stream: str, message_id: str) -> int:
        try:
            pending = await self._client.xpending_range(stream, _GROUP, message_id, message_id, 1)
            return int(pending[0]["times_delivered"]) if pending else 1
        except Exception:  # noqa: BLE001
            return 1

    async def claim_stale(self, consumer_name: str, idle_ms: int = 60000) -> list[IntentTask]:
        """把长时间没 ack 的消息领回来（前一个 worker 挂了的情况）。"""
        try:
            result = await self._client.xautoclaim(
                _STREAM, _GROUP, consumer_name, min_idle_time=idle_ms, count=10,
            )
        except Exception:  # noqa: BLE001
            return []
        entries = result[1] if len(result) > 1 else []
        tasks: list[IntentTask] = []
        for _message_id, fields in entries:
            raw = fields.get("payload")
            if raw:
                try:
                    tasks.append(IntentTask.from_dict(json.loads(raw)))
                except Exception:  # noqa: BLE001
                    continue
        return tasks


class RedisEventBackplane:
    """跨进程事件广播：worker 发布 → API 进程订阅 → 转发给本地 WS。

    必须带发送方标识并跳过自己发的消息：Pub/Sub 不会排除发布者，
    API 进程既发布又订阅同一频道，不过滤就会把自己的事件再投递一次
    （实测现象：前端收到两条 task.queued）。
    """

    def __init__(self, client: Any, origin: Optional[str] = None) -> None:
        self._client = client
        self._origin = origin or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

    @property
    def origin(self) -> str:
        return self._origin

    async def publish(self, event: TradeEvent) -> None:
        channel = f"{_EVENT_CHANNEL_PREFIX}{event.shopping_session_id}"
        envelope = {"origin": self._origin, "event": event.to_dict()}
        try:
            await self._client.publish(channel, json.dumps(envelope, ensure_ascii=False))
        except Exception as err:  # noqa: BLE001 —— 广播失败不影响本进程投递
            logger.warning("事件广播失败：%s（%s）", channel, err)

    async def listen(self) -> AsyncIterator[TradeEvent]:
        """订阅所有会话频道，逐条产出**其他进程**的事件。"""
        pubsub = self._client.pubsub()
        await pubsub.psubscribe(f"{_EVENT_CHANNEL_PREFIX}*")
        try:
            async for message in pubsub.listen():
                if message.get("type") != "pmessage":
                    continue
                try:
                    envelope = json.loads(message["data"])
                    if envelope.get("origin") == self._origin:
                        continue  # 自己发的，本地已投递过
                    yield TradeEvent.from_dict(envelope["event"])
                except Exception as err:  # noqa: BLE001
                    logger.warning("远端事件解析失败，跳过：%s", err)
        finally:
            await pubsub.punsubscribe()
            await pubsub.aclose()
