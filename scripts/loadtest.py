# scripts/loadtest.py
# 依赖：httpx（uv add --group dev httpx）
# 运行：uv run python scripts/loadtest.py --base-url http://localhost:8000 --stages 5,10,20,40
"""Globex 同步意图接口阶梯压测。

与 scripts/verify_parallel.py 的分工：verify_parallel 验「单条意图内部有没有真并行」，
本脚本验「多条意图并发时系统扛不扛得住、端到端多快、什么时候开始排队」。
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

# 混合负载：检索型（只读、快）与交易型（含确认、慢）按 8:2 取样
SEARCH_QUERIES = [
    "预算300元，抗造又不塑料的旅行三件套",
    "找几个适合长途飞行的颈枕，要小众设计",
    "登机箱，铝框、静音轮，1000 以内",
]
TRADE_QUERIES = [
    "刚才那个军绿色的收纳套，帮我下单一件寄到美国",
    "查一下我上一笔订单到哪了",
]


def _pick_query(i: int) -> str:
    return TRADE_QUERIES[i % len(TRADE_QUERIES)] if i % 5 == 0 else SEARCH_QUERIES[i % len(SEARCH_QUERIES)]


async def _one_intent(client: httpx.AsyncClient, i: int) -> tuple[bool, float]:
    payload = {
        "buyer_id": f"load-user-{i:04d}",
        "raw_query": _pick_query(i),
        "locale": "zh-CN",
        "currency": "CNY",
    }
    started = time.monotonic()
    try:
        resp = await client.post("/commerce/intents", json=payload)
        ok = resp.status_code == 200
    except (httpx.HTTPError, asyncio.TimeoutError):
        ok = False
    return ok, time.monotonic() - started


async def _run_stage(base_url: str, concurrency: int) -> None:
    # timeout 要大于单意图端到端最坏耗时，否则测的是客户端超时不是服务端能力
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0, limits=limits) as client:
        wall_start = time.monotonic()
        results = await asyncio.gather(*[_one_intent(client, i) for i in range(concurrency)])
        wall = time.monotonic() - wall_start

    latencies = [d for ok, d in results if ok]
    fails = sum(1 for ok, _ in results if not ok)
    if not latencies:
        print(f"[并发 {concurrency:>3}] 全部失败（{fails} 个），大概率被限流/超时打满")
        return

    latencies.sort()
    p = lambda q: latencies[min(len(latencies) - 1, int(len(latencies) * q))]
    throughput = len(latencies) / wall * 60
    print(
        f"[并发 {concurrency:>3}] 成功 {len(latencies):>3} 失败 {fails:>2} | "
        f"P50 {statistics.median(latencies):5.1f}s P95 {p(0.95):5.1f}s P99 {p(0.99):5.1f}s | "
        f"吞吐 {throughput:5.1f} intents/min"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--stages", default="5,10,20,40", help="逗号分隔的并发阶梯")
    args = parser.parse_args()

    for stage in (int(s) for s in args.stages.split(",")):
        await _run_stage(args.base_url, stage)
        await asyncio.sleep(5)  # 阶梯间歇，让队列排空、限流窗口复位


if __name__ == "__main__":
    asyncio.run(main())
