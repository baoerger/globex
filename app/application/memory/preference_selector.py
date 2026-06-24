# -*- coding: utf-8 -*-
"""PreferenceSelector —— 决定本轮注入哪几条买家偏好。

买家偏好会越攒越多。全量注入的问题不是"贵"，是**上下文被无关偏好挤占**：
50 条偏好铺进去，本轮真正相关的那两条反而被稀释。所以要按与本轮 query 的
相关性挑选。

但有一条**安全底线**：`dislike`（忌口 / 黑名单）**全量保留，不参与 top_k 截断**。

为什么这条不能按相关性裁：
    query「推荐个咖啡杯」与偏好「不要塑料材质」的向量相似度很低，
    纯 top_k 会把它排到末尾丢掉，于是推出一只塑料杯。
漏掉黑名单是**安全问题**（推了用户明确拒绝的东西），不是相关性问题。
只有 `like`（正向偏好）走相关性排序——漏掉一条"喜欢小众设计"最多是推荐不够贴，
不会造成"推了我明确说不要的东西"这种硬伤。
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

from app.domain.buyer.preference import BuyerPreference
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient

logger = logging.getLogger(__name__)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def render_preference_lines(preferences: Sequence[BuyerPreference]) -> str:
    """渲染偏好正文行。主 Agent 与子 Agent 共用同一个实现，避免两处格式漂。"""
    return "\n".join(f"- [{p.kind}] {p.statement}" for p in preferences)


def render_preference_hint(preferences: Sequence[BuyerPreference]) -> str:
    """渲染完整的 <buyer-preferences> 注入块。

    不写进 system prompt 而作为一条 hint 消息注入：system prompt 是 Prompt Cache
    的稳定前缀，把因人而异的偏好拼进去等于每个买家的前缀都不一样，命中率直接崩。
    """
    return (
        "<buyer-preferences>\n以下是该买家的长期偏好（来自历史会话，供个性化推荐参考，"
        "like=喜好，dislike=忌口/黑名单）：\n"
        + render_preference_lines(preferences)
        + "\n</buyer-preferences>"
    )


class PreferenceSelector:
    def __init__(
        self,
        embedder: Optional[EmbeddingClient] = None,
        relevance_enabled: bool = False,
    ) -> None:
        self._embedder = embedder
        # 开关关闭（或没有 embedder）时退化为「按时间倒序取 top_k」：
        # 零额外调用、确定性，无凭据的 CI 也能跑
        self._relevance_enabled = relevance_enabled and embedder is not None

    async def select(
        self,
        preferences: Sequence[BuyerPreference],
        query: str,
        top_k: int,
    ) -> list[BuyerPreference]:
        """返回本轮应注入的偏好，保持「dislike 在前、like 在后」的稳定顺序。"""
        if not preferences:
            return []

        dislikes = [p for p in preferences if p.kind == "dislike"]
        likes = [p for p in preferences if p.kind != "dislike"]

        if top_k <= 0:
            # 只保留安全底线，正向偏好全部让位
            return dislikes

        if len(likes) <= top_k:
            selected_likes = likes
        elif self._relevance_enabled:
            selected_likes = await self._rank_by_relevance(likes, query, top_k)
        else:
            selected_likes = self._latest_first(likes, top_k)

        return dislikes + selected_likes

    async def _rank_by_relevance(
        self, likes: list[BuyerPreference], query: str, top_k: int,
    ) -> list[BuyerPreference]:
        """按 statement 与 query 的余弦相似度取 top_k。

        embedding 走 CachedEmbeddingClient 时，重复的 statement 不会重复计费；
        query 每轮一次调用。任何异常都降级为按时间倒序，不能因为选偏好把对话搞挂。
        """
        try:
            vectors = await self._embedder.embed_batch([p.statement for p in likes])
            query_vector = await self._embedder.embed(query)
        except Exception as err:  # noqa: BLE001 —— 选偏好失败不阻断对话
            logger.warning("偏好相关性排序失败，降级按时间倒序：%s", err)
            return self._latest_first(likes, top_k)

        if len(vectors) != len(likes) or not query_vector:
            logger.warning("偏好向量数量与偏好数不一致，降级按时间倒序")
            return self._latest_first(likes, top_k)

        scored = [
            (_cosine(vector, query_vector), index, preference)
            for index, (vector, preference) in enumerate(zip(vectors, likes))
        ]
        # index 作为次序键，保证同分时顺序稳定可测
        scored.sort(key=lambda item: (-item[0], item[1]))
        picked = [preference for _, _, preference in scored[:top_k]]
        # 回到原始相对顺序输出，避免注入块的行序每轮抖动导致重复注入
        return [p for p in likes if p in picked]

    @staticmethod
    def _latest_first(likes: list[BuyerPreference], top_k: int) -> list[BuyerPreference]:
        """按 created_at 倒序取 top_k，再复原相对顺序。"""
        newest = sorted(likes, key=lambda p: p.created_at, reverse=True)[:top_k]
        return [p for p in likes if p in newest]
