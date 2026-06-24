# -*- coding: utf-8 -*-
"""SessionStore 端口

AgentState 快照的存取抽象。四期之前 SessionRegistry 直接依赖了具体的
JsonFileSessionStore，破坏了洋葱架构的依赖方向，这里补上端口。

接口是 async 的：文件实现同步即可完成，但数据库/Redis 实现必须异步，
端口按更严格的一方定义，避免换实现时改调用方。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SessionStore(ABC):
    @abstractmethod
    async def save(self, session_id: str, state_json: str) -> None:
        """保存 AgentState 全量快照（JSON 字符串）。"""

    @abstractmethod
    async def load(self, session_id: str) -> Optional[str]:
        """读取快照；不存在返回 None。"""
