# -*- coding: utf-8 -*-
"""drift_detector —— Silent-Drift 静默漂移检测（17-3 章）

漂移的本质：**每一步都没错，但 5 步之后已经偏离了目标**。
单步断言看得见"这一步格式对不对"，看不见"这一串动作还在不在为原始诉求服务"——
买家要"旅行三件套"，Agent 一路搜到"露营装备"，每步都合法，整体已经跑偏。

四类信号（与文档一致），前三类是**纯计算、零成本**，最后由可选的轻量 LLM 做终审：

    目标遗忘   最近若干轮行为里原始 query 关键词的命中率 < 20%
    探索发散   连续 3 次检索返回空候选
    偏好丢失   候选属性命中长期偏好黑名单
    成本失控   最近几轮平均 token > 历史均值 × 2

两处刻意的实现差异：

    1. **按会话隔离**。文档示例用模块级 `_round_counter` 全局计数，
       并发多会话会互相干扰，这里所有状态按 shopping_session_id 分桶。
    2. **LLM 终审是可选注入**。默认只跑纯计算信号（零成本、可单测）；
       需要语义终审时由装配层注入一个 async judge，避免本模块硬依赖模型层。

默认不启用（`DRIFT_DETECT_ENABLED=0`）：即便纯计算部分零成本，
注入纠正提示也会改变模型行为，应当是显式开启的选择。
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# 每 N 轮检测一次，控制成本
CHECK_INTERVAL = 3
# 目标遗忘：关键词命中率下界
KEYWORD_HIT_FLOOR = 0.2
# 探索发散：连续空结果次数
EMPTY_RESULT_LIMIT = 3
# 成本失控：相对历史均值的倍数
COST_SPIKE_MULTIPLIER = 2.0

CORRECTION_HINT = (
    "检测到你的动作可能已偏离买家的原始诉求（{reasons}）。"
    "请回到买家最初的需求：{original_query}。"
    "先复述你理解的需求要点，再据此选择下一步动作。"
)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}")


def extract_keywords(text: str) -> set[str]:
    """极简关键词抽取：英文/数字整词 + 中文 2-gram。

    与 catalog_search.tokenize 同思路，但这里只做漂移判定，不参与召回。
    """
    keywords: set[str] = set()
    for token in _WORD_RE.findall(text or ""):
        if token.isascii():
            keywords.add(token.lower())
            continue
        keywords.update(token[i : i + 2] for i in range(len(token) - 1))
    return keywords


@dataclass
class _SessionTrace:
    original_query: str = ""
    keywords: set[str] = field(default_factory=set)
    rounds: int = 0
    recent_actions: list[str] = field(default_factory=list)
    consecutive_empty: int = 0
    token_history: list[int] = field(default_factory=list)


@dataclass
class DriftReport:
    """一次漂移判定的结论。"""

    reasons: list[str] = field(default_factory=list)
    verdict: str = "正常"

    @property
    def drifted(self) -> bool:
        return bool(self.reasons) or self.verdict != "正常"

    def hint(self, original_query: str) -> str:
        return CORRECTION_HINT.format(
            reasons="；".join(self.reasons) or self.verdict,
            original_query=original_query,
        )


LlmJudge = Callable[[str, str], Awaitable[str]]


@dataclass
class DriftDetector:
    """按会话累积行为轨迹，周期性判定是否漂移。"""

    check_interval: int = CHECK_INTERVAL
    judge: Optional[LlmJudge] = None
    _traces: dict[str, _SessionTrace] = field(default_factory=lambda: defaultdict(_SessionTrace))

    def start_turn(self, session_id: str, original_query: str) -> None:
        trace = self._traces[session_id]
        if not trace.original_query:
            trace.original_query = original_query
            trace.keywords = extract_keywords(original_query)

    def observe_action(
        self,
        session_id: str,
        summary: str,
        *,
        result_empty: bool = False,
        tokens: int = 0,
    ) -> None:
        """记录一次 Act 的摘要与结果特征。"""
        trace = self._traces[session_id]
        trace.rounds += 1
        trace.recent_actions.append(summary)
        del trace.recent_actions[:-3]  # 只留最近 3 轮
        trace.consecutive_empty = trace.consecutive_empty + 1 if result_empty else 0
        if tokens > 0:
            trace.token_history.append(tokens)

    def due(self, session_id: str) -> bool:
        """是否到了检测轮次。"""
        trace = self._traces[session_id]
        return trace.rounds > 0 and trace.rounds % self.check_interval == 0

    def computational_signals(
        self,
        session_id: str,
        blacklist_hits: Optional[list[str]] = None,
    ) -> list[str]:
        """三类零成本信号 + 成本失控，返回命中的原因列表。"""
        trace = self._traces[session_id]
        reasons: list[str] = []

        # 1. 目标遗忘
        if trace.keywords and trace.recent_actions:
            recent = extract_keywords(" ".join(trace.recent_actions))
            hit_ratio = len(trace.keywords & recent) / len(trace.keywords)
            if hit_ratio < KEYWORD_HIT_FLOOR:
                reasons.append(f"最近动作与原始需求关键词命中率仅 {hit_ratio:.0%}")

        # 2. 探索发散
        if trace.consecutive_empty >= EMPTY_RESULT_LIMIT:
            reasons.append(f"连续 {trace.consecutive_empty} 次检索无候选")

        # 3. 偏好丢失
        if blacklist_hits:
            reasons.append(f"候选命中买家黑名单属性：{', '.join(blacklist_hits)}")

        # 4. 成本失控
        history = trace.token_history
        if len(history) >= 6:
            recent_avg = sum(history[-3:]) / 3
            baseline = sum(history[:-3]) / len(history[:-3])
            if baseline > 0 and recent_avg > baseline * COST_SPIKE_MULTIPLIER:
                reasons.append(
                    f"最近 3 轮平均 token（{recent_avg:.0f}）超历史均值（{baseline:.0f}）两倍",
                )
        return reasons

    async def check(
        self,
        session_id: str,
        blacklist_hits: Optional[list[str]] = None,
    ) -> DriftReport:
        """周期性判定。未到轮次或无轨迹时返回"正常"。"""
        if not self.due(session_id):
            return DriftReport()

        report = DriftReport(reasons=self.computational_signals(session_id, blacklist_hits))

        # 纯计算已判定漂移就不必再花钱问模型
        if report.reasons or self.judge is None:
            return report

        trace = self._traces[session_id]
        if not trace.original_query or not trace.recent_actions:
            return report
        try:
            verdict = await self.judge(trace.original_query, "\n".join(trace.recent_actions))
            report.verdict = (verdict or "正常").strip()
        except Exception as err:  # noqa: BLE001 —— 终审失败不能影响主链路
            logger.warning("漂移终审失败，按正常处理：%s", err)
        return report

    def original_query(self, session_id: str) -> str:
        return self._traces[session_id].original_query

    def reset(self, session_id: str) -> None:
        self._traces.pop(session_id, None)
