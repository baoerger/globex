# -*- coding: utf-8 -*-
"""TradeEventBus

把 Agent 流式事件、工具调用事件统一汇聚到一个异步发布订阅总线，
Presentation 层按 shopping_session_id 订阅后推送给前端 WebSocket。

事件类型与参考实现语义一一对应：
    agent.dispatch      子 Agent 被调度
    tool.invoke         工具开始执行
    tool.result         工具执行完成
    token.delta         流式 token 增量
    plan.update         Task 计划变更
    context.compressed  上下文压缩发生（三期：Context 工程）
    model.fallback      主模型限流重试用尽，已回退到备用模型（四期）
    cache.hit           语义缓存命中，本轮未调模型（四期）
    task.queued         意图已入队，等待 worker 领取（四期）
    task.started        worker 已开始处理（四期）
    final.result        最终回复
    error               异常
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TradeEventType = str

EVENT_TYPES = (
    "agent.dispatch",
    "tool.invoke",
    "tool.result",
    "token.delta",
    "plan.update",
    "context.compressed",
    "model.fallback",
    "cache.hit",
    "task.queued",
    "task.started",
    "final.result",
    "error",
)


@dataclass(frozen=True)
class TradeEvent:
    shopping_session_id: str
    type: TradeEventType
    payload: Any
    occurred_at: str

    def to_dict(self) -> dict:
        return {
            "shopping_session_id": self.shopping_session_id,
            "type": self.type,
            "payload": self.payload,
            "occurred_at": self.occurred_at,
        }

    @staticmethod
    def from_dict(raw: dict) -> "TradeEvent":
        return TradeEvent(
            shopping_session_id=raw["shopping_session_id"],
            type=raw["type"],
            payload=raw.get("payload"),
            occurred_at=raw.get("occurred_at", ""),
        )


@dataclass
class TradeEventBus:
    """asyncio 版发布订阅：每个订阅者一个独立 Queue，互不阻塞。

    四期加了跨进程背板（backplane）：worker 拆成独立进程后，它产生的
    token.delta / tool.* 事件本来推不到 API 进程的 WS 连接（前端会一片空白）。
    接上背板后：publish 除了派给本进程订阅者，还会广播到 Redis；
    API 进程订阅后转发给本地 WS。

    publish 保持同步签名（十几处调用方不必改），广播用 fire-and-forget 任务发出。
    """

    _subscribers: dict[str, list[asyncio.Queue]] = field(default_factory=dict)
    _backplane: Any = None  # EventBackplane | None，避免 infrastructure 内循环导入
    _pending: set = field(default_factory=set)

    def attach_backplane(self, backplane: Any) -> None:
        self._backplane = backplane

    def subscribe(self, shopping_session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(shopping_session_id, []).append(queue)
        return queue

    def unsubscribe(self, shopping_session_id: str, queue: asyncio.Queue) -> None:
        queues = self._subscribers.get(shopping_session_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._subscribers.pop(shopping_session_id, None)

    def deliver_local(self, event: TradeEvent) -> None:
        """只投递给本进程订阅者（背板收到远端事件后走这里，避免回环广播）。"""
        for queue in self._subscribers.get(event.shopping_session_id, []):
            queue.put_nowait(event)

    def publish(self, shopping_session_id: str, event_type: TradeEventType, payload: Any) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"未知事件类型：{event_type}")
        event = TradeEvent(
            shopping_session_id=shopping_session_id,
            type=event_type,
            payload=payload,
            occurred_at=datetime.now(timezone.utc).isoformat(),
        )
        self.deliver_local(event)
        self._broadcast(event)

    def _broadcast(self, event: TradeEvent) -> None:
        if self._backplane is None:
            return
        try:
            # 强引用挂在集合里，否则任务可能在完成前被 GC 回收
            task = asyncio.create_task(self._backplane.publish(event))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except RuntimeError:
            # 没有运行中的事件循环（例如同步测试里直接调 publish）：跳过广播，本地投递已完成
            pass
