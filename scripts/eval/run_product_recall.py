# -*- coding: utf-8 -*-
"""商品检索（product_search）召回评测 —— 见教程 13-2 章。

直连 `CatalogSearchUseCase`，不过 HTTP、不过 Agent：召回评测的定位是模块级
「日常体检」，改一行权重、换一版 reranker 都该能几秒钟跑一遍，才可能常驻 CI。

用法（项目根目录执行）：

    # 默认档（有 embedding 凭据就走向量+精排，否则自动降级）
    uv run python scripts/eval/run_product_recall.py

    # 三档降级链对比：量化"降级到底损失多少召回质量"
    uv run python scripts/eval/run_product_recall.py --compare-strategies

    # 无凭据也能跑：纯关键词档，适合 CI
    uv run python scripts/eval/run_product_recall.py --strategy keyword_2gram

关于 K 的选择（重要）：`catalog_search._RECALL_TOP_N = 8` 限制了向量召回只取 8 个候选，
因此向量档的 Recall@K 在 K>8 时**不可能再涨**，而关键词档是全库打分无上限。
在 K=10 上对比两档等于系统性地偏袒关键词档，故默认 K=8。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.application.usecases.catalog_search import CatalogSearchUseCase  # noqa: E402
from app.domain.catalog.product_search_spec import ProductSearchSpec  # noqa: E402
from app.infrastructure.embedding.openai_embedding_client import (  # noqa: E402
    OpenAIEmbeddingClient,
)
from app.infrastructure.persistence.in_memory_repositories import (  # noqa: E402
    InMemoryProductRepository,
)
from app.infrastructure.rerank.http_reranker import HttpReranker  # noqa: E402
from app.infrastructure.settings import load_settings  # noqa: E402
from app.infrastructure.vector.index_bootstrap import bootstrap_product_index  # noqa: E402
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex  # noqa: E402
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

_DATASET = Path("eval/product_recall.jsonl")
_STRATEGIES = ("embedding_rerank", "embedding_only", "keyword_2gram")


def load_dataset(path: Path) -> list[dict]:
    cases = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            cases.append(json.loads(raw))
    return cases


async def build_usecase(strategy: str) -> tuple[CatalogSearchUseCase, InMemoryProductRepository, str]:
    """按目标档位装配 UseCase。

    降级档位不是靠开关切换的，而是**靠少注入依赖自然形成**——这正好复用了线上
    真实的降级逻辑（`execute()` 里 embedder/vector_index 为 None 就走关键词），
    评测因此测的是真链路，不是为评测另写的一套。
    """
    repo = InMemoryProductRepository()
    if strategy == "keyword_2gram":
        return CatalogSearchUseCase(repo), repo, "keyword_2gram"

    settings = load_settings()
    embedder = OpenAIEmbeddingClient(settings)
    vector_index = QdrantProductIndex(settings)
    ok = await bootstrap_product_index(repo, embedder, vector_index)
    if not ok:
        print("  [warn] 向量建库失败，本档实际会降级到关键词召回")

    reranker = None
    if strategy == "embedding_rerank":
        if settings.reranker_base_url:
            reranker = HttpReranker(settings)
        else:
            print("  [warn] 未配置 RERANKER_BASE_URL，embedding_rerank 档实际等价于 embedding_only")
    return (
        CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=vector_index, reranker=reranker,
        ),
        repo,
        strategy,
    )


def check_filter(
    case: dict, hits: list[dict], filtered_out: list[dict], repo_products: dict[str, Any],
) -> tuple[Optional[bool], str]:
    """硬约束过滤是否正确。

    只对声明了约束的 query 判定；未声明的返回 None（不参与统计）。

    这里刻意**不重算汇率与关税**——那是 TariffSchedule 的职责，评测重算一遍等于
    把业务逻辑抄两份，抄错了还会误判。改为查两个不依赖换算的事实：
      1. 泄漏：返回结果里有不满足 ship_to 的商品（ships_to 是明确的枚举，无歧义）
      2. 误杀：标注为相关的商品出现在 filtered_out 里
    """
    ship_to = case.get("ship_to")
    price_cap = case.get("price_max_major")
    if not ship_to and price_cap is None:
        return None, ""

    problems = []
    if ship_to:
        for hit in hits:
            product = repo_products.get(hit["product_id"])
            if product is not None and ship_to not in product.ships_to:
                problems.append(f"泄漏 {hit['product_id']}（不可寄 {ship_to}）")

    rejected_ids = {item["product_id"] for item in filtered_out}
    for pid in case["relevant"]:
        if pid in rejected_ids:
            problems.append(f"误杀 {pid}（标注为相关却被硬约束挡掉）")

    return (not problems), "；".join(problems)


async def run_dataset(
    usecase: CatalogSearchUseCase, repo: InMemoryProductRepository, cases: list[dict], top_k: int,
) -> Aggregate:
    products = {p.product_id: p for p in await repo.list_all()}
    results: list[QueryResult] = []

    for case in cases:
        spec = ProductSearchSpec(
            normalized_query=case["query"],
            top_k=top_k,
            price_max_major=case.get("price_max_major"),
            ship_to=case.get("ship_to"),
        )
        payload = await usecase.execute(spec)
        hits = payload.get("hits", [])
        retrieved = [hit["product_id"] for hit in hits]
        relevant = case["relevant"]

        filter_ok, filter_note = check_filter(
            case, hits, payload.get("filtered_out", []), products,
        )
        results.append(
            QueryResult(
                query=case["query"],
                retrieved=retrieved,
                relevant=relevant,
                recall=recall_at_k(retrieved, relevant, top_k),
                mrr=mrr(retrieved, relevant),
                ndcg=ndcg_at_k(retrieved, relevant, top_k),
                filter_ok=filter_ok,
                note=filter_note,
                kind=case.get("kind", "lexical"),
            ),
        )
    return evaluate(results, k=top_k)


def by_kind(agg: Aggregate) -> dict[str, Aggregate]:
    """按 query 类型拆开。

    拆开是必需的：字面类 query 上关键词召回本来就很强（语料描述关键词密集），
    混在一起算总均会把语义类的差距抹平，看不出向量召回到底买到了什么。
    """
    groups: dict[str, list[QueryResult]] = {}
    for r in agg.per_query:
        groups.setdefault(r.kind, []).append(r)
    return {kind: evaluate(rs, k=agg.k) for kind, rs in sorted(groups.items())}


def print_summary(label: str, agg: Aggregate) -> None:
    filt = "n/a" if agg.filter_accuracy is None else f"{agg.filter_accuracy:.3f}"
    print(
        f"  {label:<18} Recall@{agg.k}={agg.recall:.3f}  MRR={agg.mrr:.3f}  "
        f"NDCG@{agg.k}={agg.ndcg:.3f}  过滤准确率={filt}",
    )
    for kind, sub in by_kind(agg).items():
        print(
            f"    └─ {kind:<10}({sub.count:>2} 条) Recall={sub.recall:.3f}  "
            f"MRR={sub.mrr:.3f}  NDCG={sub.ndcg:.3f}",
        )


def render_report(
    per_strategy: dict[str, Aggregate], thresholds: Thresholds, top_k: int,
) -> str:
    lines = [
        f"# 商品检索召回评测报告（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）",
        "",
        f"标注集 `{_DATASET}`，K={top_k}。",
        "",
        "## 指标总览",
        "",
        f"| 档位 | Recall@{top_k} | MRR | NDCG@{top_k} | 过滤准确率 | 门禁 |",
        "|---|---|---|---|---|---|",
    ]
    for name, agg in per_strategy.items():
        verdict, _ = gate(agg, thresholds)
        filt = "n/a" if agg.filter_accuracy is None else f"{agg.filter_accuracy:.3f}"
        lines.append(
            f"| {name} | {agg.recall:.3f} | {agg.mrr:.3f} | {agg.ndcg:.3f} | {filt} | {verdict} |",
        )

    lines += ["", f"门禁阈值：Recall ≥ {thresholds.recall}（阻断）、MRR ≥ {thresholds.mrr}"
              f"（阻断）、NDCG ≥ {thresholds.ndcg}（告警）。", ""]

    lines += ["## 按 query 类型拆分", "",
              "字面类（lexical）query 与语料共享词汇，关键词召回天然占优；"
              "语义类（semantic）query 刻意不含商品字面，是向量召回真正创造价值的地方。", "",
              f"| 档位 | 类型 | 条数 | Recall@{top_k} | MRR | NDCG@{top_k} |",
              "|---|---|---|---|---|---|"]
    for name, agg in per_strategy.items():
        for kind, sub in by_kind(agg).items():
            lines.append(
                f"| {name} | {kind} | {sub.count} | {sub.recall:.3f} | "
                f"{sub.mrr:.3f} | {sub.ndcg:.3f} |",
            )
    lines.append("")

    for name, agg in per_strategy.items():
        verdict, reasons = gate(agg, thresholds)
        lines += [f"## {name}（{verdict}，{agg.count} 条）", ""]
        if reasons:
            lines += ["未达标项：", *[f"- {r}" for r in reasons], ""]
        lines += ["| query | 类型 | Recall | MRR | NDCG | 召回序 | 标注 | 过滤 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in agg.per_query:
            filt = "-" if r.filter_ok is None else ("OK" if r.filter_ok else f"FAIL {r.note}")
            lines.append(
                f"| {r.query} | {r.kind} | {r.recall:.2f} | {r.mrr:.2f} | {r.ndcg:.2f} | "
                f"{','.join(r.retrieved) or '（空）'} | {','.join(r.relevant)} | {filt} |",
            )
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="商品检索召回评测")
    parser.add_argument("--dataset", default=str(_DATASET))
    parser.add_argument("--top-k", type=int, default=8, help="默认 8：向量召回深度上限即 8")
    parser.add_argument("--strategy", choices=_STRATEGIES, default="embedding_rerank")
    parser.add_argument("--compare-strategies", action="store_true", help="三档降级链对比")
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.65)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    parser.add_argument("--report-dir", default="eval")
    args = parser.parse_args()

    cases = load_dataset(Path(args.dataset))
    print(f"标注集 {args.dataset}：{len(cases)} 条，K={args.top_k}")

    targets = list(_STRATEGIES) if args.compare_strategies else [args.strategy]
    thresholds = Thresholds(recall=args.min_recall, mrr=args.min_mrr, ndcg=args.min_ndcg)

    per_strategy: dict[str, Aggregate] = {}
    for strategy in targets:
        print(f"\n[{strategy}] 装配中…")
        usecase, repo, actual = await build_usecase(strategy)
        agg = await run_dataset(usecase, repo, cases, args.top_k)
        per_strategy[strategy] = agg
        print_summary(strategy, agg)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"recall-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(render_report(per_strategy, thresholds, args.top_k), encoding="utf-8")
    print(f"\n报告已写入 {report_path}")

    # 任一档位被阻断即以非零码退出，便于直接接 CI
    verdicts = {name: gate(agg, thresholds)[0] for name, agg in per_strategy.items()}
    print("门禁：" + "，".join(f"{n}={v}" for n, v in verdicts.items()))
    if "BLOCK" in verdicts.values():
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
