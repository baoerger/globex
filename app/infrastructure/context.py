# -*- coding: utf-8 -*-
"""ShoppingContext

用 ContextVar 保存当前任务的会话快照（shopping_session_id / buyer_id / locale / currency），
跨层透明传递：工具与子 Agent 执行时随时读取，无需层层透传参数。
多用户并发任务依赖 asyncio Task 级隔离，不会串台。
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ShoppingContextSnapshot:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str


_current_snapshot: ContextVar[Optional[ShoppingContextSnapshot]] = ContextVar(
    "globex_shopping_context",
    default=None,
)


class ShoppingContext:
    @staticmethod
    def set(snapshot: ShoppingContextSnapshot):
        return _current_snapshot.set(snapshot)

    @staticmethod
    def reset(token) -> None:
        _current_snapshot.reset(token)

    @staticmethod
    def current() -> Optional[ShoppingContextSnapshot]:
        return _current_snapshot.get()

    @staticmethod
    def current_session_id() -> str:
        snapshot = _current_snapshot.get()
        return snapshot.shopping_session_id if snapshot else "anonymous"
