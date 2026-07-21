# -*- coding: utf-8 -*-
"""SemanticCache

语义缓存：把「买家问句 → 最终回复」按向量相似度复用，相似提问直接跳过整轮 LLM。
这是四期里对网关限流最直接的缓解手段（命中即零模型调用）。

安全边界（很重要，不能放宽）：
    1. 只缓存检索/咨询类问句。下单、取消这类写操作意图一律不进缓存——
       同样的话第二次说，业务含义可能完全不同（比如再下一单）。
    2. 按 buyer 分桶。买家 A 的回复里可能含其偏好与地址，不能给买家 B 复用。
    2b. 桶 key 还带买家的**偏好指纹**（scope）：回复是在当时偏好下生成的，
        偏好新增或撤回后，旧回复必须自动失效（否则会拿已删偏好当卖点说给用户）。
    3. 会话内已有多轮上下文时不命中。"刚才那款多少钱"依赖上下文，
       跨会话复用会答错。
    4. namespace 带模型名与提示词指纹：改 prompt / 换模型后旧缓存自动作废。

相似度用余弦；阈值默认 0.95，低于阈值视为未命中（宁可多花一次调用，
也不能给出答非所问的回复）。

评测注意：跑回归时应关掉语义缓存（SEMANTIC_CACHE_ENABLED=0），
否则评的是缓存而不是 Agent 行为。
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

from app.domain.catalog.ports.retrieval_ports import EmbeddingClient
from app.infrastructure.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)

_TTL_SECONDS = 24 * 3600
# 单个 buyer 桶内保留的最近条数，避免键无限膨胀
_BUCKET_LIMIT = 30

# 写操作/上下文依赖意图，命中任一则整条不进缓存也不查缓存
_UNSAFE_PATTERNS = (
    r"下单", r"买了", r"购买", r"付款", r"支付",
    r"取消", r"退单", r"退款", r"改地址",
    r"刚才", r"刚刚", r"上面", r"前面", r"那个", r"这个", r"它",
    r"我的订单", r"订单号", r"GBX-",
)
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS))


@dataclass(frozen=True)
class SemanticHit:
    reply: str
    similarity: float
    matched_query: str


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _normalize(query: str) -> str:
    """归一化问句：压空白 + 去尾部标点，让"多少钱?"与"多少钱"共享缓存。"""
    collapsed = re.sub(r"\s+", "", query.strip())
    return collapsed.rstrip("？?。.!！~")


def is_cacheable_query(query: str) -> bool:
    """写操作与上下文依赖类问句不进语义缓存。"""
    return not _UNSAFE_RE.search(query)


class SemanticCache:
    def __init__(
        self,
        cache: RedisCache,
        embedder: EmbeddingClient,
        threshold: float = 0.95,
        enabled: bool = True,
        namespace: str = "",
    ) -> None:
        self._cache = cache
        self._embedder = embedder
        self._threshold = threshold
        self._enabled = enabled and cache.enabled
        # namespace 应包含模型名与提示词指纹：改 prompt 或换模型后旧回复必须自动失效，
        # 否则修了 Agent 行为也会被 24h 内的旧缓存盖掉（实测踩过：改完 prompt 评测回复一字不差）
        self._namespace = namespace

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _bucket_key(self, buyer_id: str, scope: str = "") -> str:
        """桶 key = namespace + buyer_id + scope。

        scope 用来装载“会改变回复内容的买家侧状态”，目前是长期偏好指纹。

        为何必须：回复是在“当时的偏好”下生成的。买家把“不要塑料”撤回后再问一句
        语义相近的话，若桶 key 不含偏好指纹，就会命中撤回前的旧回复——里面还写着
        “符合您不要塑料的偏好”，直接向用户断言了一条它刚删掉的偏好。实测踩过。
        """
        digest = hashlib.sha256(
            f"{self._namespace}\n{buyer_id}\n{scope}".encode(),
        ).hexdigest()[:16]
        return f"semcache:{digest}"

    async def _load_entries(self, buyer_id: str, scope: str = "") -> list:
        """读桶；缓存异常一律当空桶（纵深防御，不依赖底层一定吞异常）。"""
        try:
            entries = await self._cache.get_json(self._bucket_key(buyer_id, scope))
        except Exception as err:  # noqa: BLE001
            logger.warning("语义缓存读异常，按未命中处理：%s", err)
            return []
        return entries if isinstance(entries, list) else []

    async def lookup(
        self, buyer_id: str, query: str, has_history: bool, scope: str = "",
    ) -> Optional[SemanticHit]:
        if not self._enabled or has_history or not is_cacheable_query(query):
            return None
        entries = await self._load_entries(buyer_id, scope)
        if not entries:
            return None

        try:
            vector = await self._embedder.embed(_normalize(query))
        except Exception as err:  # noqa: BLE001 —— 算不出向量就当未命中，不影响主链路
            logger.warning("语义缓存取向量失败，按未命中处理：%s", err)
            return None

        best: Optional[SemanticHit] = None
        for entry in entries:
            similarity = _cosine(vector, entry.get("vector", []))
            if similarity >= self._threshold and (best is None or similarity > best.similarity):
                best = SemanticHit(
                    reply=entry.get("reply", ""),
                    similarity=round(similarity, 4),
                    matched_query=entry.get("query", ""),
                )
        return best if best and best.reply else None

    async def remember(
        self, buyer_id: str, query: str, reply: str, has_history: bool, scope: str = "",
    ) -> None:
        if not self._enabled or has_history or not is_cacheable_query(query):
            return
        if not reply or reply.startswith("[error]"):
            return  # 失败回复绝不能进缓存，否则错误会被反复复用
        try:
            vector = await self._embedder.embed(_normalize(query))
        except Exception as err:  # noqa: BLE001
            logger.warning("语义缓存写向量失败，跳过：%s", err)
            return

        key = self._bucket_key(buyer_id, scope)
        entries = await self._load_entries(buyer_id, scope)
        entries.append({"query": _normalize(query), "reply": reply, "vector": vector})
        try:
            await self._cache.set_json(key, entries[-_BUCKET_LIMIT:], _TTL_SECONDS)
        except Exception as err:  # noqa: BLE001
            logger.warning("语义缓存写异常，跳过：%s", err)
