# -*- coding: utf-8 -*-
"""ConnectionManager

WebSocket 连接管理：客户端连上 /commerce/events 后发送
{"shopping_session_id": "..."} 完成订阅，服务端把 TradeEventBus
中对应会话的事件推送给该连接。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

from fastapi import WebSocket, WebSocketDisconnect

from app.infrastructure.eventbus import TradeEventBus

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, bus: TradeEventBus) -> None:
        self._bus = bus

    async def serve(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            subscribe_payload = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        session_id = subscribe_payload.get("shopping_session_id")
        if not session_id:
            await websocket.close(code=4000, reason="缺少 shopping_session_id")
            return

        queue = self._bus.subscribe(session_id)
        logger.info("WebSocket 已订阅会话：%s", session_id)
        try:
            while True:
                event = await queue.get()
                payload = asdict(event)
                payload.pop("shopping_session_id", None)
                await websocket.send_json(payload)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            self._bus.unsubscribe(session_id, queue)
            logger.info("WebSocket 已退订会话：%s", session_id)
