# -*- coding: utf-8 -*-
"""端到端冒烟脚本

用法（先启动服务）：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python scripts/smoke_e2e.py [--query "..."]

流程：WS 订阅会话事件 → POST /commerce/intents → 实时打印事件 → 打印最终回复。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid

import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/commerce/events"


async def listen_events(session_id: str, stop: asyncio.Event) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"shopping_session_id": session_id}))
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1)
            except asyncio.TimeoutError:
                continue
            event = json.loads(raw)
            events.append(event)
            if event["type"] == "token.delta":
                print(event["payload"]["token"], end="", flush=True)
            else:
                print(f"\n[{event['type']}] {json.dumps(event['payload'], ensure_ascii=False)[:300]}")
    return events


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="我想买一套便宜又抗造的旅行三件套，预算 300 块，最好不要塑料的，喜欢小众一点。",
    )
    parser.add_argument("--session", default=None, help="复用已有会话 ID 做多轮对话")
    args = parser.parse_args()

    session_id = args.session or f"smoke-{uuid.uuid4().hex[:8]}"
    print(f"== 会话：{session_id}\n== 意图：{args.query}\n")

    stop = asyncio.Event()
    listener = asyncio.create_task(listen_events(session_id, stop))
    await asyncio.sleep(0.5)  # 等 WS 订阅建立

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(
            f"{BASE_URL}/commerce/intents",
            json={
                "shopping_session_id": session_id,
                "buyer_id": "buyer-001",
                "locale": "zh-CN",
                "currency": "CNY",
                "raw_query": args.query,
            },
        )
        response.raise_for_status()
        body = response.json()

    await asyncio.sleep(1)  # 等尾部事件送达
    stop.set()
    events = await listener

    print("\n\n===== 最终回复 =====")
    print(body["final_text"])
    print("\n===== 事件统计 =====")
    counts: dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
