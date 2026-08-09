# -*- coding: utf-8 -*-
"""content_filter —— L3：工具返回内容过滤

威胁模型：工具结果并非可信输入。商品标题、描述、网页兜底搜索的正文都可能被
第三方写入「忽略之前的指令」这类提示词注入，一旦原样进主 loop 上下文，
模型有概率把它当成新指令执行（间接提示词注入）。

因此工具结果在注入上下文之前先过一层正则过滤：命中危险模式的片段替换为占位符，
而不是整条丢弃——保留其余正常内容（商品标题、价格），避免为了安全把可用信息也砍掉。

只做正则、不调模型：这层要在每次工具返回后同步执行，必须是微秒级、零成本。
真正的语义级判断交给 L4 与漂移检测。
"""
from __future__ import annotations

import re

# 命中即替换的占位符：保留痕迹，让模型与排查者都能看到"这里被过滤过"
FILTERED_PLACEHOLDER = "[内容已过滤：疑似注入]"

# 危险模式：覆盖中英文两种常见注入写法
DANGEROUS_PATTERNS: list[str] = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
    r"(?i)忽略.{0,10}(之前|以上|所有).{0,10}(指令|指示|规则)",
    r"(?i)system\s*prompt",
    r"(?i)you\s+are\s+now",
    r"(?i)扮演.{0,10}角色",
    r"(?i)output\s+(all|every)\s+(user|system)",
    r"(?i)reveal\s+(your|the)\s+(api|secret|key)",
]

_compiled = [re.compile(pattern) for pattern in DANGEROUS_PATTERNS]


def sanitize_tool_output(text: str) -> tuple[bool, str]:
    """过滤工具返回中的疑似注入内容。

    Returns:
        (是否命中过危险模式, 过滤后的文本)。调用方据第一个返回值决定是否发事件告警；
        文本始终可用，不会因为命中而变成空串。
    """
    if not text:
        return False, text

    hit = False
    cleaned = text
    for pattern in _compiled:
        cleaned, count = pattern.subn(FILTERED_PLACEHOLDER, cleaned)
        if count:
            hit = True
    return hit, cleaned
