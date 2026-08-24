# -*- coding: utf-8 -*-
"""recall metrics —— 召回评测的三个核心指标 + 聚合 + 发版门禁

指标口径与 13-1 章一致：

    Recall@K   Top-K 覆盖了多少标注项      —— 任何召回环节的底线
    MRR        首条命中的倒数排名          —— Top-1 直接影响主 Agent 精挑
    NDCG@K     考虑位置 + 标注序的 gain    —— 不只看命中，还看好的是否靠前

三者都只依赖「召回出来的 id 序列」与「标注的 id 序列」，因此纯函数、零依赖、可单测，
商品检索与品类知识库两条链路共用同一套实现。

标注序即重要性序：`relevant[0]` 最相关。NDCG 用线性 gain（`len(relevant) - i`），
比二元相关更能区分「命中了但排在最后」和「命中且排第一」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    """去重但保序。

    召回结果理论上不该有重复 id，但真实链路里（多路合并、降级重试）可能出现；
    不去重会让 Recall 虚高、MRR 失真，故在指标入口统一清洗。
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Top-K 召回里覆盖了多少标注。

    注意：标注数大于 K 时，本指标天然取不到 1.0（这是 Recall@K 的定义，不是 bug）。
    因此选 K 时应让 K >= 单条 query 的常见标注数。
    """
    if k <= 0:
        return 0.0
    rel = set(relevant)
    if not rel:
        return 0.0
    top_k = set(_dedup_keep_order(retrieved)[:k])
    return len(top_k & rel) / len(rel)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """首条相关项的倒数排名；一条都没命中记 0。"""
    rel = set(relevant)
    if not rel:
        return 0.0
    for index, item in enumerate(_dedup_keep_order(retrieved), start=1):
        if item in rel:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """NDCG@K：按标注序给线性 gain，按位置打折。"""
    if k <= 0 or not relevant:
        return 0.0
    # 标注序越靠前 gain 越大：第 0 位得 len(relevant)，末位得 1
    gain = {item: len(relevant) - i for i, item in enumerate(relevant)}
    ranked = _dedup_keep_order(retrieved)[:k]
    dcg = sum(gain.get(item, 0) / math.log2(i + 2) for i, item in enumerate(ranked))
    ideal = sum(
        gain[item] / math.log2(i + 2) for i, item in enumerate(list(relevant)[:k])
    )
    return dcg / ideal if ideal else 0.0


@dataclass
class QueryResult:
    """单条 query 的评测结果。"""

    query: str
    retrieved: list[str]
    relevant: list[str]
    recall: float
    mrr: float
    ndcg: float
    # 硬约束过滤是否正确：None = 该 query 未声明约束，不参与统计
    filter_ok: bool | None = None
    note: str = ""
    # query 类型（lexical / semantic），用于拆分统计
    kind: str = "lexical"


@dataclass
class Aggregate:
    """整个标注集的平均指标。"""

    k: int
    count: int
    recall: float
    mrr: float
    ndcg: float
    filter_accuracy: float | None = None
    per_query: list[QueryResult] = field(default_factory=list)


def evaluate(results: Sequence[QueryResult], k: int) -> Aggregate:
    """把逐条结果聚合成平均指标（宏平均：每条 query 等权）。"""
    if not results:
        return Aggregate(k=k, count=0, recall=0.0, mrr=0.0, ndcg=0.0, per_query=[])

    count = len(results)
    checked = [r for r in results if r.filter_ok is not None]
    filter_accuracy = (
        sum(1 for r in checked if r.filter_ok) / len(checked) if checked else None
    )
    return Aggregate(
        k=k,
        count=count,
        recall=round(sum(r.recall for r in results) / count, 4),
        mrr=round(sum(r.mrr for r in results) / count, 4),
        ndcg=round(sum(r.ndcg for r in results) / count, 4),
        filter_accuracy=None if filter_accuracy is None else round(filter_accuracy, 4),
        per_query=list(results),
    )


@dataclass(frozen=True)
class Thresholds:
    """发版门禁阈值。

    recall / mrr 低于阈值判 BLOCK（阻断发版）；ndcg 低于阈值判 WARN（告警但不阻断）——
    排序质量退化通常需要人看一眼再决定，不适合无人值守地直接卡死流水线。
    """

    recall: float = 0.75
    mrr: float = 0.65
    ndcg: float = 0.70


def gate(agg: Aggregate, thresholds: Thresholds) -> tuple[str, list[str]]:
    """返回 (verdict, 原因列表)；verdict ∈ PASS / WARN / BLOCK。"""
    blocks: list[str] = []
    warns: list[str] = []
    if agg.count == 0:
        return "BLOCK", ["标注集为空，无法评测"]

    if agg.recall < thresholds.recall:
        blocks.append(f"Recall@{agg.k} {agg.recall} < {thresholds.recall}")
    if agg.mrr < thresholds.mrr:
        blocks.append(f"MRR {agg.mrr} < {thresholds.mrr}")
    if agg.ndcg < thresholds.ndcg:
        warns.append(f"NDCG@{agg.k} {agg.ndcg} < {thresholds.ndcg}")

    if blocks:
        return "BLOCK", blocks + warns
    if warns:
        return "WARN", warns
    return "PASS", []
