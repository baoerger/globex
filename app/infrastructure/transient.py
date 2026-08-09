# -*- coding: utf-8 -*-
"""transient

上游瞬时故障判定。模型层与编排层共用同一份判据，避免两处标记表各自漂移。

为什么按 message 特征匹配而不是按异常类型：OpenAI 兼容网关把限流写在 SSE 流中间时，
抛出的是笼统的 openai.APIError，类型上无法与真实业务错误区分，只能看文案；
同一套判据还要覆盖 httpx 超时与网关 5xx。
"""
from __future__ import annotations

# 全小写匹配。"throttling" 来自实测：网关限流返回 code=Throttling.Concurrency
_TRANSIENT_ERROR_MARKERS = (
    "too many concurrent",
    "rate limit",
    "request rate",
    "too many requests",
    "throttling",
    "429",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "connection reset",
    "connection error",
)


def is_transient_error(error: BaseException) -> bool:
    """判断异常是否属于可重试的上游瞬时故障。"""
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)
