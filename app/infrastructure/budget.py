# -*- coding: utf-8 -*-
"""budget —— 请求级 Token 预算与四档路由降级（16-4 章）

为什么 Agent 必须做请求级预算：普通 LLM 应用一次请求就是一次调用，
Agent 一次意图是多轮 Think/Act 加多次往返，单条请求就能打穿成本。
`ReplyBudgetControlMiddleware`（已挂）只管**单轮回复**的长度；
本模块管的是**整条意图**的累计消耗，两者互补。

四档降级（按剩余预算比例，与文档一致）：

    > 50%      main      主模型，无限制
    20% - 50%  lite      切轻量模型
    5% - 20%   minimal   轻量模型 + 注入简洁模式 hint，压 Think 长度
    < 5%       fallback  不再调 LLM，用已有中间结果做规则兜底

预算通过 ContextVar 挂在当前意图上：一次意图一个实例，
子 Agent 派发时协程继承 ContextVar 快照，天然共享同一份预算账本。

默认不启用（`TOKEN_BUDGET_TOTAL=0`）：它会改变模型选择，必须是显式开启的选择。
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 档位阈值（剩余比例下界）
TIER_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.50, "main"),
    (0.20, "lite"),
    (0.05, "minimal"),
    (0.0, "fallback"),
)

MINIMAL_MODE_HINT = (
    "当前对话 Token 预算已接近上限，请进入简洁模式："
    "不要展开推理过程，直接给出结论与最关键的 2-3 条依据。"
)

FALLBACK_NOTICE = (
    "本轮 Token 预算已耗尽，未再调用模型。以下是基于已获取信息的整理结果。"
)


@dataclass
class TokenBudget:
    """一条意图的 Token 账本。"""

    total_limit: int
    used: int = 0
    # 记录每次消耗的来源，便于事后定位是哪一步烧的
    entries: list[tuple[str, int]] = field(default_factory=list)

    def charge(self, source: str, tokens: int) -> None:
        if tokens <= 0:
            return
        self.used += tokens
        self.entries.append((source, tokens))

    @property
    def remaining(self) -> int:
        return max(0, self.total_limit - self.used)

    @property
    def remaining_ratio(self) -> float:
        if self.total_limit <= 0:
            return 1.0
        return self.remaining / self.total_limit

    @property
    def tier(self) -> str:
        """当前档位。"""
        ratio = self.remaining_ratio
        for lower_bound, tier in TIER_THRESHOLDS:
            if ratio > lower_bound:
                return tier
        return "fallback"

    @property
    def exhausted(self) -> bool:
        return self.tier == "fallback"


_budget_var: ContextVar[Optional[TokenBudget]] = ContextVar("token_budget", default=None)


def init_budget(total_limit: int) -> Optional[TokenBudget]:
    """在意图入口初始化预算；total_limit <= 0 表示不启用。"""
    if total_limit <= 0:
        _budget_var.set(None)
        return None
    budget = TokenBudget(total_limit=total_limit)
    _budget_var.set(budget)
    return budget


def get_budget() -> Optional[TokenBudget]:
    return _budget_var.get()


def current_tier() -> str:
    """未启用预算时恒为 main，调用方无需分支判断。"""
    budget = get_budget()
    return budget.tier if budget is not None else "main"


def resolve_model(main_model: str, lite_model: str) -> str:
    """按当前档位选模型。

    fallback 档不该走到这里（调用方应先判 exhausted 直接规则兜底），
    真走到了也返回 lite，避免因为选不出模型把整轮打挂。
    """
    tier = current_tier()
    if tier == "main":
        return main_model
    return lite_model or main_model


def minimal_mode_hint() -> Optional[str]:
    """minimal 档需要注入的简洁模式提示。"""
    return MINIMAL_MODE_HINT if current_tier() == "minimal" else None
