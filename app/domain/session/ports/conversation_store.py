# -*- coding: utf-8 -*-
"""ConversationStore 端口 + 对话记录值对象

四期新增能力：把每轮对话与过程事件结构化落库，供事后追溯、客服排查，
也为五期的自进化飞轮（bad case 采集与回放）预留数据底座。

与 SessionStore 的区别：
    SessionStore      存 AgentState 快照（框架内部结构，只为恢复上下文）
    ConversationStore 存业务可读的对话流水（谁在什么时候说了什么、调了哪些工具）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

VALID_ROLES = ("buyer", "agent")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ConversationTurn:
    """一轮完整问答。turn_index 由存储层按会话自增，写入时可留空。"""

    session_id: str
    buyer_id: str
    role: str  # buyer / agent
    content: str
    model: str = ""
    latency_ms: int = 0
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"ConversationTurn.role 必须是 {VALID_ROLES}：{self.role}")
        if not self.session_id:
            raise ValueError("ConversationTurn.session_id required")


@dataclass(frozen=True)
class ConversationEventRecord:
    """过程事件的持久化形态（对应 TradeEventBus 的一条事件）。"""

    session_id: str
    type: str
    payload: dict[str, Any]
    occurred_at: str = field(default_factory=_now_iso)


class ConversationStore(ABC):
    @abstractmethod
    async def append_turn(self, turn: ConversationTurn) -> None:
        """追加一轮问答。"""

    @abstractmethod
    async def append_events(self, events: list[ConversationEventRecord]) -> None:
        """批量追加过程事件（一轮结束后一次性写，减少往返）。"""

    @abstractmethod
    async def list_turns(self, session_id: str, limit: int = 50) -> list[ConversationTurn]:
        """按 turn_index 升序返回该会话的对话流水。"""

    @abstractmethod
    async def touch_session(
        self,
        session_id: str,
        buyer_id: str,
        locale: str,
        currency: str,
    ) -> None:
        """确保会话主记录存在并刷新活跃时间（upsert）。"""

    @abstractmethod
    async def find_session(self, session_id: str) -> Optional[dict]:
        """会话主记录，不存在返回 None。"""
