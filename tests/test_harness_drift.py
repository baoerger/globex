# -*- coding: utf-8 -*-
"""P4 漂移检测单测：四类信号 + 周期节流 + 会话隔离 + LLM 终审可选。"""
from app.application.harness.drift_detector import (
    DriftDetector,
    DriftReport,
    extract_keywords,
)


class TestExtractKeywords:
    def test_chinese_bigrams(self):
        assert "旅行" in extract_keywords("旅行三件套")

    def test_ascii_lowercased(self):
        assert "nomadica" in extract_keywords("Nomadica bag")

    def test_empty_is_safe(self):
        assert extract_keywords("") == set()


class TestDriftIntervalAndIsolation:
    async def test_not_due_before_interval(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        det.observe_action("s1", "搜索 旅行三件套")
        assert det.due("s1") is False
        assert (await det.check("s1")).drifted is False

    async def test_due_on_interval(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套 抗造")
        assert det.due("s1") is True

    async def test_sessions_isolated(self):
        """文档示例用模块级 _round_counter 会串台，这里必须按会话隔离。"""
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        det.start_turn("s2", "登机箱")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套")
        assert det.due("s1") is True
        assert det.due("s2") is False, "s2 不应继承 s1 的轮次计数"
        assert det.original_query("s2") == "登机箱"

    def test_reset_clears_session(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        det.observe_action("s1", "x")
        det.reset("s1")
        assert det.original_query("s1") == ""


class TestDriftSignals:
    async def test_goal_forgotten(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套 抗造 无塑料")
        for _ in range(3):
            det.observe_action("s1", "搜索 露营帐篷 睡袋 防潮垫")

        report = await det.check("s1")
        assert report.drifted is True
        assert any("命中率" in r for r in report.reasons)

    async def test_on_target_is_not_drift(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套 抗造 无塑料")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套 抗造 无塑料 帆布")

        report = await det.check("s1")
        assert report.drifted is False, "紧扣原始需求不应被判漂移"

    async def test_exploration_divergence(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套", result_empty=True)

        report = await det.check("s1")
        assert any("无候选" in r for r in report.reasons)

    async def test_empty_streak_resets_on_hit(self):
        det = DriftDetector(check_interval=3)
        det.start_turn("s1", "旅行三件套")
        det.observe_action("s1", "搜索 旅行三件套", result_empty=True)
        det.observe_action("s1", "搜索 旅行三件套", result_empty=True)
        det.observe_action("s1", "搜索 旅行三件套", result_empty=False)

        report = await det.check("s1")
        assert not any("无候选" in r for r in report.reasons)

    async def test_preference_blacklist_hit(self):
        det = DriftDetector(check_interval=1)
        det.start_turn("s1", "旅行三件套")
        det.observe_action("s1", "推荐 旅行三件套")

        report = await det.check("s1", blacklist_hits=["塑料"])
        assert any("黑名单" in r for r in report.reasons)

    async def test_cost_spike(self):
        det = DriftDetector(check_interval=6)
        det.start_turn("s1", "旅行三件套")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套", tokens=100)
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套", tokens=500)

        report = await det.check("s1")
        assert any("两倍" in r for r in report.reasons)


class TestLlmJudge:
    async def test_judge_used_only_when_no_computational_signal(self):
        calls: list[tuple[str, str]] = []

        async def judge(query: str, actions: str) -> str:
            calls.append((query, actions))
            return "严重偏离"

        det = DriftDetector(check_interval=3, judge=judge)
        det.start_turn("s1", "旅行三件套 抗造")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套 抗造")

        report = await det.check("s1")
        assert len(calls) == 1
        assert report.verdict == "严重偏离"
        assert report.drifted is True

    async def test_judge_skipped_when_signal_already_hit(self):
        """纯计算已判定漂移就不必再花钱问模型。"""
        calls: list[str] = []

        async def judge(query: str, actions: str) -> str:
            calls.append(query)
            return "正常"

        det = DriftDetector(check_interval=3, judge=judge)
        det.start_turn("s1", "旅行三件套 抗造 无塑料")
        for _ in range(3):
            det.observe_action("s1", "搜索 露营帐篷")

        await det.check("s1")
        assert calls == [], "已有计算信号时不应再调模型"

    async def test_judge_failure_degrades_to_normal(self):
        async def judge(query: str, actions: str) -> str:
            raise RuntimeError("模型限流")

        det = DriftDetector(check_interval=3, judge=judge)
        det.start_turn("s1", "旅行三件套 抗造")
        for _ in range(3):
            det.observe_action("s1", "搜索 旅行三件套 抗造")

        report = await det.check("s1")
        assert report.drifted is False, "终审失败不能把正常流程判成漂移"


class TestDriftReportHint:
    def test_hint_mentions_original_query(self):
        report = DriftReport(reasons=["连续 3 次检索无候选"])
        hint = report.hint("旅行三件套")
        assert "旅行三件套" in hint
        assert "连续 3 次检索无候选" in hint
