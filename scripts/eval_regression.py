# -*- coding: utf-8 -*-
"""评测回归脚本

用法（先启动服务）：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python scripts/eval_regression.py [--cases eval/cases.yaml] [--only case_id]

流程：逐 case 顺序打 POST /commerce/intents（同 case 多轮复用会话）→
LLM judge 按 Rubric（P0 数字事实 / P1 行为命中 / P2 表达）逐条打分 →
输出 eval/report-{时间戳}.md。

case 可选 prior_context 字段：告知 judge 本会话之前已成立的事实（如跨会话写入的长期偏好），
否则 judge 只看本会话 transcript，会把"正确应用了历史偏好"误判为"无据添加"。

评分口径：P0 任一不过 = 该 case 直接 FAIL；总分 = P0 0.5 / P1 0.35 / P2 0.15 加权。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.infrastructure.transient import is_transient_error  # noqa: E402

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# judge 不经模型层闸门（直连 httpx），自己退避重试，避免主模型限流时整轮评测报废
_JUDGE_MAX_RETRIES = 3
_JUDGE_RETRY_BASE_SECONDS = 8.0

JUDGE_SYSTEM_PROMPT = """你是严格的电商 Agent 评测员。给你一段"买家多轮提问与 Agent 回复"的对话记录、
商品库事实表（ground truth），以及分级评分细则（P0 数字事实与安全底线 / P1 行为与命中 / P2 表达）。
部分 case 会额外给出"会话前置事实"（如买家在早先会话里已写入的长期偏好）：这些事实真实有效，
即使它不出现在本段对话记录里，Agent 引用或应用它也不算编造。
逐条判断细则是否满足：数字事实类细则以商品库事实表为基准比对（回复中的价格/库存与事实表一致即通过，
运费关税等衍生数字只要金额自洽且未与事实表矛盾即通过）；行为类细则以对话记录与会话前置事实为依据，
拿不准按不通过处理。
每条细则先在 reason 里完成推理，再给出 pass 定论；pass 必须与 reason 的最终结论一致。
只输出 JSON（字段顺序固定：criterion → reason → pass）：
{"p0": [{"criterion": "...", "reason": "...", "pass": true/false}],
 "p1": [...], "p2": [...]}"""


def build_ground_truth() -> str:
    """从种子商品数据与汇率表生成事实表，供 judge 校验数字事实。"""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.domain.catalog.exchange_rate import ExchangeRateTable
    from app.infrastructure.persistence.seed_products import build_seed_products

    lines = ["| product_id | 标题 | 品类 | sku | 价格 | 库存 |", "|---|---|---|---|---|---|"]
    for product in build_seed_products():
        for sku in product.skus:
            lines.append(
                f"| {product.product_id} | {product.title} | {product.category} "
                f"| {sku.sku_id}({sku.spec}) | {sku.price} | {sku.stock} |",
            )
    rates = ", ".join(f"1 {cur} = {rate} CNY" for cur, rate in ExchangeRateTable().rates_to_cny.items())
    lines.append("")
    lines.append(f"系统汇率表（到手价工具按此折算目标币种，折算后的价格属于工具返回，不算自行估算）：{rates}")
    return "\n".join(lines)


async def call_judge(
    client: httpx.AsyncClient,
    transcript: str,
    rubric: dict,
    ground_truth: str,
    prior_context: str = "",
) -> dict:
    prior_block = f"## 会话前置事实\n{prior_context}\n\n" if prior_context else ""
    payload = {
        # judge 可独立指定模型：主模型切新版/被限流时，评分基准不跟着飘
        "model": os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("LLM_MODEL", "qwen-plus"),
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"## 商品库事实表\n{ground_truth}\n\n"
                    f"{prior_block}"
                    f"## 对话记录\n{transcript}\n\n"
                    f"## 评分细则\n{json.dumps(rubric, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    last_error: Exception | None = None
    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            response = await client.post(
                f"{os.environ['LLM_BASE_URL'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as err:  # noqa: BLE001
            if not is_transient_error(err) or attempt == _JUDGE_MAX_RETRIES - 1:
                raise
            last_error = err
            delay = _JUDGE_RETRY_BASE_SECONDS * (2**attempt)
            print(f"   judge 遇限流，{delay:.0f}s 后重试：{err}", flush=True)
            await asyncio.sleep(delay)
    raise last_error if last_error else RuntimeError("judge 重试耗尽")


def score_case(judged: dict) -> tuple[float, bool]:
    """返回 (加权得分, p0全过)。空档位按满分处理。"""

    def ratio(items: list) -> float:
        return sum(1 for item in items if item.get("pass")) / len(items) if items else 1.0

    p0_ratio, p1_ratio, p2_ratio = ratio(judged.get("p0", [])), ratio(judged.get("p1", [])), ratio(judged.get("p2", []))
    weighted = 0.5 * p0_ratio + 0.35 * p1_ratio + 0.15 * p2_ratio
    return round(weighted, 3), p0_ratio == 1.0


async def run_case(client: httpx.AsyncClient, case: dict, ground_truth: str) -> dict:
    session_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    buyer_id = case.get("buyer_id") or f"eval-buyer-{case['id']}"
    transcript_lines: list[str] = []
    for query in case["queries"]:
        response = await client.post(
            f"{BASE_URL}/commerce/intents",
            json={
                "shopping_session_id": session_id,
                "buyer_id": buyer_id,
                "locale": "zh-CN",
                "currency": "CNY",
                "raw_query": query,
            },
            timeout=600,
        )
        response.raise_for_status()
        final_text = response.json()["final_text"]
        transcript_lines.append(f"[买家] {query}\n[Agent] {final_text}")

    transcript = "\n\n".join(transcript_lines)
    judged = await call_judge(client, transcript, case["rubric"], ground_truth, case.get("prior_context", ""))
    score, p0_all_pass = score_case(judged)
    return {
        "id": case["id"],
        "description": case["description"],
        "score": score,
        "p0_pass": p0_all_pass,
        "verdict": "PASS" if p0_all_pass and score >= 0.7 else "FAIL",
        "judged": judged,
        "transcript": transcript,
    }


def render_report(results: list[dict]) -> str:
    lines = [
        f"# Globex 评测回归报告（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"总览：{sum(1 for r in results if r['verdict'] == 'PASS')}/{len(results)} PASS，"
        f"平均分 {sum(r['score'] for r in results) / len(results):.3f}",
        "",
        "| case | 描述 | 得分 | P0 | 结果 |",
        "|------|------|------|-----|------|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['description']} | {r['score']} | "
            f"{'通过' if r['p0_pass'] else '不通过'} | {r['verdict']} |",
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['id']}（{r['verdict']}，{r['score']}）")
        for level in ("p0", "p1", "p2"):
            for item in r["judged"].get(level, []):
                mark = "PASS" if item.get("pass") else "FAIL"
                lines.append(f"- [{level.upper()}][{mark}] {item['criterion']}：{item.get('reason', '')}")
        lines.append("")
        lines.append("<details><summary>对话记录</summary>\n")
        lines.append(r["transcript"])
        lines.append("\n</details>\n")
    return "\n".join(lines)


async def _guard_semantic_cache(allow: bool) -> None:
    """语义缓存开着时拒绝跑回归。

    实测踩过：一条 case 的错误回复进了缓存，之后改 prompt 重跑，回复一字不差——
    评测彻底失去了检验能力。回归必须评 Agent 真实行为。
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            health = (await client.get(f"{BASE_URL}/health")).json()
    except Exception as err:  # noqa: BLE001 —— 拿不到 health 不阻断，后续请求自会报错
        print(f"警告：无法读取 /health（{err}），跳过缓存检查", flush=True)
        return
    if health.get("semantic_cache") and not allow:
        raise SystemExit(
            "拒绝跑回归：服务端语义缓存处于开启状态，评分会变成评缓存。\n"
            "请用 SEMANTIC_CACHE_ENABLED=0 重启服务后重试，例如：\n"
            "  SEMANTIC_CACHE_ENABLED=0 docker compose -f docker/docker-compose.yaml up -d app worker\n"
            "确认要带缓存跑则加 --allow-semantic-cache。",
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.yaml"))
    parser.add_argument("--only", default=None, help="只跑指定 case id")
    parser.add_argument(
        "--allow-semantic-cache",
        action="store_true",
        help="允许在语义缓存开启的环境下跑（不推荐，评的会是缓存而不是 Agent）",
    )
    args = parser.parse_args()

    await _guard_semantic_cache(args.allow_semantic_cache)

    with open(args.cases, encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    results = []
    ground_truth = build_ground_truth()
    async with httpx.AsyncClient() as client:
        for case in cases:  # 顺序执行：memory-recall 依赖 memory-write
            print(f"== 评测 {case['id']} ...", flush=True)
            try:
                result = await run_case(client, case, ground_truth)
            except Exception as err:  # noqa: BLE001 —— 单条失败不中断整轮回归
                result = {
                    "id": case["id"], "description": case["description"],
                    "score": 0.0, "p0_pass": False, "verdict": "ERROR",
                    "judged": {}, "transcript": f"执行异常：{err}",
                }
            print(f"   -> {result['verdict']}（{result['score']}）", flush=True)
            results.append(result)

    report = render_report(results)
    report_path = PROJECT_ROOT / "eval" / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{report_path}")
    print(report.split("\n\n")[1])


if __name__ == "__main__":
    asyncio.run(main())
