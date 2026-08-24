# -*- coding: utf-8 -*-
"""召回评测指标单测（13-2 章脚手架的地基）。

指标本身必须先可信，否则后面所有评测结论都不可信。
重点覆盖：全命中 / 全不中 / 位置敏感 / K 截断 / 空标注 / 标注多于 K / 重复 id。
"""
import pytest

from scripts.eval.metrics import (
    Aggregate,
    QueryResult,
    Thresholds,
    evaluate,
    gate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_all_relevant_hit(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_none_hit(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_partial_hit(self):
        assert recall_at_k(["a", "x"], ["a", "b"], k=2) == 0.5

    def test_k_truncation_excludes_late_hit(self):
        """命中排在 K 之后就不算召回到。"""
        assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_relevant_more_than_k_cannot_reach_one(self):
        """标注数 > K 时天然取不到 1.0——这是定义，不是 bug。"""
        assert recall_at_k(["a", "b"], ["a", "b", "c", "d"], k=2) == 0.5

    def test_empty_relevant_is_zero(self):
        assert recall_at_k(["a"], [], k=3) == 0.0

    def test_non_positive_k_is_zero(self):
        assert recall_at_k(["a"], ["a"], k=0) == 0.0

    def test_duplicates_do_not_inflate(self):
        """重复 id 不应让 Recall 虚高。"""
        assert recall_at_k(["a", "a", "a"], ["a", "b"], k=3) == 0.5


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], ["a"]) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a"], []) == 0.0

    def test_takes_earliest_hit(self):
        assert mrr(["x", "b", "a"], ["a", "b"]) == 0.5

    def test_duplicates_do_not_shift_rank(self):
        """前面的重复项去重后不该把真实命中位置往后推。"""
        assert mrr(["x", "x", "a"], ["a"]) == 0.5


class TestNDCGAtK:
    def test_perfect_order_is_one(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_no_hit_is_zero(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_position_sensitive(self):
        """同样命中同一批，排前面的分必须更高——这是 NDCG 存在的意义。"""
        good = ndcg_at_k(["a", "b", "x"], ["a", "b"], k=3)
        bad = ndcg_at_k(["x", "a", "b"], ["a", "b"], k=3)
        assert good > bad

    def test_annotation_order_matters(self):
        """标注序即重要性序：把更重要的排前面得分更高。"""
        rel = ["a", "b"]  # a 比 b 更相关
        assert ndcg_at_k(["a", "b"], rel, k=2) > ndcg_at_k(["b", "a"], rel, k=2)

    def test_empty_relevant_is_zero(self):
        assert ndcg_at_k(["a"], [], k=3) == 0.0

    def test_non_positive_k_is_zero(self):
        assert ndcg_at_k(["a"], ["a"], k=0) == 0.0

    def test_relevant_more_than_k_uses_truncated_ideal(self):
        """标注多于 K 时，理想 DCG 也只取前 K，避免分母被无法触及的项拉大。"""
        assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d"], k=2) == pytest.approx(1.0)


def _qr(recall: float, m: float, n: float, filter_ok=None) -> QueryResult:
    return QueryResult(
        query="q", retrieved=[], relevant=[], recall=recall, mrr=m, ndcg=n,
        filter_ok=filter_ok,
    )


class TestAggregate:
    def test_empty_results(self):
        agg = evaluate([], k=10)
        assert agg.count == 0 and agg.recall == 0.0

    def test_macro_average(self):
        agg = evaluate([_qr(1.0, 1.0, 1.0), _qr(0.0, 0.0, 0.0)], k=10)
        assert agg.count == 2
        assert agg.recall == 0.5 and agg.mrr == 0.5 and agg.ndcg == 0.5

    def test_filter_accuracy_only_counts_declared(self):
        """未声明硬约束的 query 不参与过滤准确率统计。"""
        agg = evaluate(
            [_qr(1, 1, 1, filter_ok=True), _qr(1, 1, 1, filter_ok=False), _qr(1, 1, 1)],
            k=10,
        )
        assert agg.filter_accuracy == 0.5

    def test_filter_accuracy_none_when_no_constraint(self):
        assert evaluate([_qr(1, 1, 1)], k=10).filter_accuracy is None


class TestGate:
    THRESHOLDS = Thresholds(recall=0.75, mrr=0.65, ndcg=0.70)

    def test_pass(self):
        agg = Aggregate(k=10, count=5, recall=0.9, mrr=0.8, ndcg=0.85)
        assert gate(agg, self.THRESHOLDS) == ("PASS", [])

    def test_block_on_low_recall(self):
        agg = Aggregate(k=10, count=5, recall=0.5, mrr=0.8, ndcg=0.85)
        verdict, reasons = gate(agg, self.THRESHOLDS)
        assert verdict == "BLOCK"
        assert any("Recall@10" in r for r in reasons)

    def test_block_on_low_mrr(self):
        agg = Aggregate(k=10, count=5, recall=0.9, mrr=0.3, ndcg=0.85)
        assert gate(agg, self.THRESHOLDS)[0] == "BLOCK"

    def test_warn_only_on_low_ndcg(self):
        """排序质量退化只告警，不阻断——需要人看一眼再决定。"""
        agg = Aggregate(k=10, count=5, recall=0.9, mrr=0.8, ndcg=0.5)
        verdict, reasons = gate(agg, self.THRESHOLDS)
        assert verdict == "WARN"
        assert any("NDCG@10" in r for r in reasons)

    def test_empty_dataset_blocks(self):
        verdict, reasons = gate(Aggregate(k=10, count=0, recall=0, mrr=0, ndcg=0), self.THRESHOLDS)
        assert verdict == "BLOCK"
        assert "标注集为空" in reasons[0]
