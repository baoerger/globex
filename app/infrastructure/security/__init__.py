# -*- coding: utf-8 -*-
"""security

安全护栏的纯函数层（16-6 章 L3/L4）：

    content_filter  L3 工具返回内容过滤——工具结果注入主 loop 上下文之前拦注入
    output_guard    L4 输出审核——最终回复推给买家之前拦内部信息泄露

L1（工具白名单）由 app/application/agents/permissions.py 承担，
L2（System/User 角色隔离 + 边界声明）在 app/application/prompts/globex.yml 里，
本模块只补 L3/L4 两层。四层各管一段、互为兜底。
"""
from app.infrastructure.security.content_filter import (
    DANGEROUS_PATTERNS,
    FILTERED_PLACEHOLDER,
    sanitize_tool_output,
)
from app.infrastructure.security.output_guard import (
    SENSITIVE_PATTERNS,
    audit_output,
)

__all__ = [
    "DANGEROUS_PATTERNS",
    "FILTERED_PLACEHOLDER",
    "sanitize_tool_output",
    "SENSITIVE_PATTERNS",
    "audit_output",
]
