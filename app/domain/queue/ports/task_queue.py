# -*- coding: utf-8 -*-
"""TaskQueue 端口 + 任务值对象

削峰用的任务队列抽象。本期只交 Redis Stream 实现；换 RabbitMQ / Kafka
只需另写一个实现，application 与 domain 不动。

投递语义按 at-least-once 设计（Redis Stream 重投、worker 崩溃重启都会造成
同一任务被消费两次），因此调用方必须自己保证幂等——create_order 是写操作，
重复消费等于重复下单。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IntentTask:
    task_id: str
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str
    enqueued_at: str = field(default_factory=_now_iso)
    # 优先级：0 = 正常请求（1 以上 = 大请求，走低优先队列）。
    # 长会话（轮数多、上下文大）单次耗时明显更长，
    # 把它们分到单独队列，避免把短平快的新会话堵在后面。
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "shopping_session_id": self.shopping_session_id,
            "buyer_id": self.buyer_id,
            "locale": self.locale,
            "currency": self.currency,
            "raw_query": self.raw_query,
            "enqueued_at": self.enqueued_at,
            "priority": self.priority,
        }

    @staticmethod
    def from_dict(raw: dict) -> "IntentTask":
        return IntentTask(
            task_id=raw["task_id"],
            shopping_session_id=raw["shopping_session_id"],
            buyer_id=raw.get("buyer_id", ""),
            locale=raw.get("locale", "zh-CN"),
            currency=raw.get("currency", "CNY"),
            raw_query=raw.get("raw_query", ""),
            enqueued_at=raw.get("enqueued_at", ""),
            priority=int(raw.get("priority", 0)),
        )


@dataclass(frozen=True)
class TaskStatus:
    task_id: str
    state: str  # queued / running / done / failed
    final_text: str = ""
    error: str = ""
    queue_position: int = 0


class TaskQueue(ABC):
    @abstractmethod
    async def enqueue(self, task: IntentTask) -> None:
        ...

    @abstractmethod
    async def set_status(self, status: TaskStatus) -> None:
        ...

    @abstractmethod
    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        ...

    @abstractmethod
    async def depth(self) -> int:
        """待消费任务数，供 /health 与容量观察使用。"""
