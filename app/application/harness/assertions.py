# -*- coding: utf-8 -*-
"""assertions —— 三类单步断言（17-3 章）

    Schema      工具返回是否是结构完整的合法 JSON        <1ms  纯校验
    Sequencing  工具调用顺序是否满足前置条件            <1ms  规则判断
    Semantic    工具返回与买家诉求是否语义对齐          ~50ms 轻量 LLM（默认不启用）

核心约定（与文档一致）：断言失败**一律不 raise**，只记进 `assertions_failed`，
让主 Agent 在下一轮 Think 时自己看到并纠正。护栏的目的是让模型自愈，
不是把一次格式抖动升级成整轮失败。

两处刻意的实现差异，均为修正文档示例里的缺陷：

    1. **调用记录按会话隔离**。文档示例是模块级 `_called_tools: list`，
       并发多会话会互相污染，这里按 shopping_session_id 分桶。
    2. **写路径前置校验采用"有证据才硬拒"**。文档说 create_order 必须硬拒绝，
       但会话可能从 AgentState 快照恢复（进程重启后内存记录为空），
       此时硬拒会把合法下单误杀。故规则收紧为：
       本会话已观测到工具调用、却没有一次检索 → 硬拒；
       完全没有观测记录（刚恢复/首次调用）→ 降级为警告。
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# 工具名 → 期望返回结构里必须存在的关键字段
# （逐个对过真实工具实现：product_search 回 hits/recall_strategy，
#   category_insight 回 insights，订单三工具回 Order.snapshot()）
TOOL_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "product_search_tool": ("hits", "recall_strategy"),
    "category_insight_tool": ("insights",),
    "create_order_tool": ("order_id", "status"),
    "query_order_tool": ("order_id", "status"),
    "cancel_order_tool": ("order_id", "status"),
}

# 工具名 → 必须在它之前调用过的工具
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "create_order_tool": ("product_search_tool",),
    "cancel_order_tool": ("query_order_tool",),
}

# 写路径：前置不满足时硬拒（前提是有观测证据，见模块 docstring）
HARD_REJECT_TOOLS = frozenset({"create_order_tool"})


@dataclass
class AssertionOutcome:
    """一次断言的结论。"""

    failures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reject_reason: Optional[str] = None

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None


def check_schema(tool_name: str, tool_result: Any) -> AssertionOutcome:
    """Schema 断言：返回是否是含关键字段的合法 JSON。"""
    outcome = AssertionOutcome()
    required = TOOL_REQUIRED_FIELDS.get(tool_name)
    if not required:
        return outcome  # 不在检查范围

    data: Any = tool_result
    if isinstance(tool_result, str):
        # 工具错误返回是 "[error] ..." 文本，不是 JSON——不算 schema 违约
        if tool_result.lstrip().startswith("[error]"):
            return outcome
        try:
            data = json.loads(tool_result)
        except (json.JSONDecodeError, TypeError):
            outcome.failures.append(
                {"type": "schema", "tool": tool_name, "reason": "工具返回不是合法 JSON"},
            )
            return outcome

    if not isinstance(data, dict):
        outcome.failures.append(
            {"type": "schema", "tool": tool_name, "reason": "工具返回不是 JSON 对象"},
        )
        return outcome

    missing = [key for key in required if key not in data]
    if missing:
        outcome.failures.append(
            {
                "type": "schema",
                "tool": tool_name,
                "reason": f"缺少必需字段：{', '.join(missing)}",
            },
        )
    return outcome


@dataclass
class SequencingTracker:
    """按会话记录已调用工具，做顺序断言。"""

    _called: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def record(self, session_id: str, tool_name: str) -> None:
        self._called[session_id].append(tool_name)

    def called(self, session_id: str) -> list[str]:
        return list(self._called.get(session_id, []))

    def reset(self, session_id: str) -> None:
        self._called.pop(session_id, None)

    def check(self, session_id: str, tool_name: str) -> AssertionOutcome:
        """Sequencing 断言：前置工具是否已调用过。"""
        outcome = AssertionOutcome()
        prerequisites = PREREQUISITES.get(tool_name)
        if not prerequisites:
            return outcome

        history = self._called.get(session_id, [])
        for prereq in prerequisites:
            if prereq in history:
                continue

            if tool_name in HARD_REJECT_TOOLS and history:
                # 有观测证据（本会话调过工具）却没检索过 → 硬拒，钱不能错扣
                outcome.reject_reason = (
                    f"{tool_name} 需要先执行 {prereq}：本会话尚未检索过商品，"
                    f"请先确认买家要买的具体商品再下单。"
                )
            else:
                outcome.warnings.append(
                    f"注意：{tool_name} 通常在 {prereq} 之后调用，但当前 {prereq} 尚未执行。",
                )
            break
        return outcome
