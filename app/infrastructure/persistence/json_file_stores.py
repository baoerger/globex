# -*- coding: utf-8 -*-
"""文件持久化实现（零外部依赖的默认形态）

    - 偏好：DATA_DIR/preferences/{buyer_id}.json（追加去重）
    - 会话：DATA_DIR/sessions/{session_id}.json（AgentState 全量快照，重启恢复多轮对话）
    - 对话：DATA_DIR/conversations/{session_id}.jsonl（对话流水 + 事件轨迹）

四期把 session/conversation 的方法改成 async 以对齐端口——文件 IO 本身是同步的，
但端口按数据库实现的需要定义，这样换实现不必改调用方。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from app.domain.buyer.preference import BuyerPreference, PreferenceStore
from app.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from app.domain.session.ports.session_store import SessionStore

logger = logging.getLogger(__name__)


def _safe_name(raw: str) -> str:
    """文件名清洗，避免路径穿越。"""
    safe = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
    return safe or "anonymous"


class JsonFilePreferenceStore(PreferenceStore):
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "preferences"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, buyer_id: str) -> Path:
        return self._dir / f"{_safe_name(buyer_id)}.json"

    async def append(self, preference: BuyerPreference) -> None:
        existing = await self.list_by_buyer(preference.buyer_id)
        if any(p.statement == preference.statement and p.kind == preference.kind for p in existing):
            return  # 幂等去重
        existing.append(preference)
        self._write(preference.buyer_id, existing)

    async def list_by_buyer(self, buyer_id: str) -> list[BuyerPreference]:
        path = self._path(buyer_id)
        if not path.exists():
            return []
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            return [BuyerPreference(**item) for item in items]
        except (ValueError, TypeError) as err:
            logger.warning("偏好文件损坏，按空处理：%s（%s）", path, err)
            return []

    async def delete(self, buyer_id: str, statement: str) -> bool:
        """精确匹配 statement 删除。

        按 statement 而不分 kind：同一句话允许同时存为 like 与 dislike（唯一约束是
        buyer+kind+statement），而买家说“以后别管塑料了”表达的是“忘掉这条说法”，
        而不是“只忘掉它的负向那一面”，所以同 statement 的条目一并清除。
        """
        existing = await self.list_by_buyer(buyer_id)
        remaining = [p for p in existing if p.statement != statement]
        if len(remaining) == len(existing):
            return False
        self._write(buyer_id, remaining)
        return True

    def _write(self, buyer_id: str, preferences: list[BuyerPreference]) -> None:
        payload = [
            {"buyer_id": p.buyer_id, "kind": p.kind, "statement": p.statement, "created_at": p.created_at}
            for p in preferences
        ]
        self._path(buyer_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class JsonFileSessionStore(SessionStore):
    """AgentState 快照的文件存取（AgentState 是 pydantic 模型，直接 JSON round-trip）。"""

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "sessions"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{_safe_name(session_id)}.json"

    async def save(self, session_id: str, state_json: str) -> None:
        self._path(session_id).write_text(state_json, encoding="utf-8")

    async def load(self, session_id: str) -> Optional[str]:
        path = self._path(session_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")


class JsonFileConversationStore(ConversationStore):
    """对话流水的 JSONL 实现：一行一条记录，追加写不覆盖。

    DATABASE_URL=file 时的形态。不做索引与并发控制，仅用于本地开发与排查。
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "conversations"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{_safe_name(session_id)}.jsonl"

    def _append_line(self, session_id: str, record: dict) -> None:
        with self._path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def append_turn(self, turn: ConversationTurn) -> None:
        self._append_line(
            turn.session_id,
            {
                "kind": "turn",
                "buyer_id": turn.buyer_id,
                "role": turn.role,
                "content": turn.content,
                "model": turn.model,
                "latency_ms": turn.latency_ms,
                "created_at": turn.created_at,
            },
        )

    async def append_events(self, events: list[ConversationEventRecord]) -> None:
        for event in events:
            self._append_line(
                event.session_id,
                {
                    "kind": "event",
                    "type": event.type,
                    "payload": event.payload,
                    "occurred_at": event.occurred_at,
                },
            )

    async def list_turns(self, session_id: str, limit: int = 50) -> list[ConversationTurn]:
        path = self._path(session_id)
        if not path.exists():
            return []
        turns: list[ConversationTurn] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # 单行损坏不影响其余记录
            if record.get("kind") != "turn":
                continue
            turns.append(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=record.get("buyer_id", ""),
                    role=record["role"],
                    content=record.get("content", ""),
                    model=record.get("model", ""),
                    latency_ms=record.get("latency_ms", 0),
                    created_at=record.get("created_at", ""),
                ),
            )
        return turns[-limit:]

    async def touch_session(self, session_id: str, buyer_id: str, locale: str, currency: str) -> None:
        # 文件形态没有独立的会话主表，首轮写入时记一条元信息即可
        path = self._path(session_id)
        if path.exists():
            return
        self._append_line(
            session_id,
            {"kind": "session", "buyer_id": buyer_id, "locale": locale, "currency": currency},
        )

    async def find_session(self, session_id: str) -> Optional[dict]:
        path = self._path(session_id)
        if not path.exists():
            return None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("kind") == "session":
                return {"session_id": session_id, **record}
        return {"session_id": session_id}
