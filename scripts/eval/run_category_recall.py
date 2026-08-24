# -*- coding: utf-8 -*-
"""品类知识库（CategoryInsight）召回评测 —— 见教程 13-1 §5。

与商品检索评测共用 `scripts/eval/metrics.py` 的三个指标，区别只在**标注单位**：

    商品检索      标注单位 = product_id
    品类知识库    标注单位 = 知识文档名（如 travel-gear.md）

标注单位为什么是文档名而不是 chunk id：`bootstrap_category_knowledge` 用
`ApproxTokenChunker(chunk_size=512, overlap=50)` 切片，chunk 边界会随 chunk_size
或文档内容变动而漂移，拿 chunk id 当标注会导致「改了一句话就要重标一遍」。
文档名稳定，且「该问题该由哪篇文档回答」本身就是运营能稳定判断的粒度。

用法（项目根目录执行，需 embedding 凭据 + Qdrant）：

    uv run python scripts/eval/run_category_recall.py
    uv run python scripts/eval/run_category_recall.py --top-k 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.infrastructure.rag.category_knowledge import (  # noqa: E402
    bootstrap_category_knowledge,
    build_category_knowledge_base,
)
from app.infrastructure.settings import load_settings  # noqa: E402
from scripts.eval.metrics import (  # noqa: E402
    Aggregate,
    QueryResult,
    Thresholds,
    evaluate,
    gate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

_DATASET = Path("eval/category_recall.jsonl")


def load_dataset(path: Path) -> list[dict]:
    return [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


def source_of(item) -> str:
    """取一条检索结果所属的知识文档名。

    与 `category_insight_tool` 保持同一口径：metadata.source 缺失时退回 document_id，
    否则评测口径和线上口径会不一致。
    """
    metadata = getattr(item.chunk, "metadata", None)
    if metadata:
        return metadata.get("source", item.document_id)
    return item.document_id


async def run_dataset(knowledge_base, cases: list[dict], top_k: int) -> Aggregate:
    results: list[QueryResult] = []
    for case in cases:
        hits = await knowledge_base.search(queries=[case["query"]], top_k=top_k)
        # 同一篇文档可能命中多个 chunk：按首次出现保序去重，落到文档粒度
        retrieved: list[str] = []
        for item in hits:
            src = source_of(item)
            if src not in retrieved:
                retrieved.append(src)

        relevant = case["relevant"]
        results.append(
            QueryResult(
                query=case["query"],
                retrieved=retrieved,
                relevant=relevant,
                recall=recall_at_k(retrieved, relevant, top_k),
                mrr=mrr(retrieved, relevant),
                ndcg=ndcg_at_k(retrieved, relevant, top_k),
                kind=case.get("kind", "knowledge"),
            ),
        )
    return evaluate(results, k=top_k)


def render_report(agg: Aggregate, thresholds: Thresholds) -> str:
    verdict, reasons = gate(agg, thresholds)
    lines = [
        f"# 品类知识库召回评测报告（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）",
        "",
        f"标注集 `{_DATASET}`，K={agg.k}，共 {agg.count} 条。标注单位为知识文档名。",
        "",
        f"| 指标 | 值 | 阈值 |",
        "|---|---|---|",
        f"| Recall@{agg.k} | {agg.recall:.3f} | ≥ {thresholds.recall}（阻断） |",
        f"| MRR | {agg.mrr:.3f} | ≥ {thresholds.mrr}（阻断） |",
        f"| NDCG@{agg.k} | {agg.ndcg:.3f} | ≥ {thresholds.ndcg}（告警） |",
        "",
        f"门禁结论：**{verdict}**",
        "",
    ]
    if reasons:
        lines += ["未达标项：", *[f"- {r}" for r in reasons], ""]
    lines += ["| query | Recall | MRR | NDCG | 召回文档 | 标注文档 |", "|---|---|---|---|---|---|"]
    for r in agg.per_query:
        lines.append(
            f"| {r.query} | {r.recall:.2f} | {r.mrr:.2f} | {r.ndcg:.2f} | "
            f"{','.join(r.retrieved) or '（空）'} | {','.join(r.relevant)} |",
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="品类知识库召回评测")
    parser.add_argument("--dataset", default=str(_DATASET))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.65)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    parser.add_argument("--report-dir", default="eval")
    args = parser.parse_args()

    cases = load_dataset(Path(args.dataset))
    print(f"标注集 {args.dataset}：{len(cases)} 条，K={args.top_k}")

    settings = load_settings()
    knowledge_base = build_category_knowledge_base(settings)
    inserted = await bootstrap_category_knowledge(knowledge_base)
    print(f"知识库就绪（本次新增 {inserted} 篇）")

    agg = await run_dataset(knowledge_base, cases, args.top_k)
    thresholds = Thresholds(recall=args.min_recall, mrr=args.min_mrr, ndcg=args.min_ndcg)
    print(
        f"  Recall@{agg.k}={agg.recall:.3f}  MRR={agg.mrr:.3f}  NDCG@{agg.k}={agg.ndcg:.3f}",
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"category-recall-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(render_report(agg, thresholds), encoding="utf-8")
    print(f"报告已写入 {report_path}")

    verdict, reasons = gate(agg, thresholds)
    print(f"门禁：{verdict}" + (f"（{'；'.join(reasons)}）" if reasons else ""))
    if verdict == "BLOCK":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
