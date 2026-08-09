# -*- coding: utf-8 -*-
"""output_guard —— L4：输出审核

最后一道闸：Agent 的最终回复推给买家之前，检查是否夹带了内部实现信息。
命中即脱敏，并把 (是否安全) 回给调用方去发告警事件——脱敏动作本身不阻断回复，
否则一次误判就等于整轮对话失败。

**刻意收窄的范围**（与 16-6 章文档的差异，如实记录）：

    1. 文档示例里有 `item_id`，本项目 schema 中不存在该字段，故不纳入；
    2. `product_id`（如 P1001）**不脱敏**——它本就随商品卡通过 tool.result 事件
       下发给前端渲染，属于对外契约的一部分，脱敏反而会破坏正常回复；
    3. 内部工具名按真实工具集逐个列出，不用 `\\w+_tool` 这类宽泛模式，
       避免把买家可见的正常措辞误伤。

判据只有一条：**脱敏对象必须是买家侧无需知道、且泄露有害的东西**。
"""
from __future__ import annotations

import re

REDACTED = "[已脱敏]"

# 真实存在的内部工具名（与 app/application/tools/ 一致）
_INTERNAL_TOOLS = (
    "product_search_tool",
    "category_insight_tool",
    "web_search_tool",
    "create_order_tool",
    "query_order_tool",
    "cancel_order_tool",
    "remember_preference_tool",
    "task_dispatch",
)

SENSITIVE_PATTERNS: list[str] = [
    # 会话内部 ID
    r"shopping_session_id\s*[:=]\s*[\w-]+",
    # API Key 形态
    r"sk-[a-zA-Z0-9]{20,}",
    # 内部服务地址（容器网络内主机名）
    r"https?://(?:vllm|reranker|qdrant|redis|opensearch)(?::\d+)?(?:/\S*)?",
    # 内部工具名
    r"\b(?:" + "|".join(_INTERNAL_TOOLS) + r")\b",
]

_compiled = [re.compile(pattern) for pattern in SENSITIVE_PATTERNS]


def audit_output(text: str) -> tuple[bool, str]:
    """审核最终输出。

    Returns:
        (是否安全, 处理后的文本)。安全 = 未命中任何敏感模式；
        命中时文本已就地脱敏，可直接下发。
    """
    if not text:
        return True, text

    safe = True
    cleaned = text
    for pattern in _compiled:
        cleaned, count = pattern.subn(REDACTED, cleaned)
        if count:
            safe = False
    return safe, cleaned
