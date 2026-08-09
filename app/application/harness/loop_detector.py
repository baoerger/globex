# -*- coding: utf-8 -*-
"""loop_detector —— 循环不收敛检测（14 章第四道护栏的补充）

`ReActConfig(max_iters)` 只保证「不会无限跑」，但它是硬上限：模型在两个工具间
反复横跳时，token 会一路烧到上限才停。本模块在**远早于上限**的位置发现打转，
给模型一次自愈机会。

设计取舍：

    1. **不强制终止**，只回一条收敛提示，由调用方注入 observation。
       硬终止会把「本来第 4 次调用就能成功」的正常重试也杀掉；
    2. **按会话隔离**。文档 17-3 的示例用了模块级 `_called_tools: list`，
       多会话并发时会互相污染（A 会话的调用记录算到 B 头上），
       这里改为按 shopping_session_id 分桶；
    3. 只看**最近 window 次**调用，避免长对话里早期的正常重复被反复计入。
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Optional

# 同一工具在窗口内连续出现达此次数即判定打转
DEFAULT_REPEAT_THRESHOLD = 3
# 滑动窗口长度
DEFAULT_WINDOW = 6

CONVERGE_HINT = (
    "检测到你已连续 {count} 次调用 {tool}，但仍未推进。"
    "请不要重复同一动作：要么换检索词或换工具，要么基于现有信息直接给买家结论；"
    "如果确实拿不到数据，就如实说明并给出替代建议。"
)


@dataclass
class LoopDetector:
    """按会话统计工具调用序列，判断是否在打转。"""

    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD
    window: int = DEFAULT_WINDOW
    _calls: dict[str, Deque[str]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=DEFAULT_WINDOW)),
    )

    def _bucket(self, session_id: str) -> Deque[str]:
        bucket = self._calls[session_id]
        # window 可被构造参数改写，defaultdict 的 maxlen 固定，故按需重建
        if bucket.maxlen != self.window:
            rebuilt: Deque[str] = deque(bucket, maxlen=self.window)
            self._calls[session_id] = rebuilt
            return rebuilt
        return bucket

    def record(self, session_id: str, tool_name: str) -> None:
        self._bucket(session_id).append(tool_name)

    def trailing_repeat(self, session_id: str, tool_name: str) -> int:
        """返回该工具在序列尾部连续出现的次数。"""
        count = 0
        for name in reversed(self._bucket(session_id)):
            if name != tool_name:
                break
            count += 1
        return count

    def check(self, session_id: str, tool_name: str) -> Optional[str]:
        """记录本次调用并判断是否需要收敛提示。

        Returns:
            需要提示时返回提示文本，否则 None。
        """
        self.record(session_id, tool_name)
        count = self.trailing_repeat(session_id, tool_name)
        if count >= self.repeat_threshold:
            return CONVERGE_HINT.format(count=count, tool=tool_name)
        return None

    def reset(self, session_id: str) -> None:
        """一轮意图结束后清理，避免跨轮误判。"""
        self._calls.pop(session_id, None)
