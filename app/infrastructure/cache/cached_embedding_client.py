# -*- coding: utf-8 -*-
"""CachedEmbeddingClient

给 EmbeddingClient 套一层缓存的装饰器实现（不改原客户端，也不污染 UseCase）。

key 用 sha256(model + text)：embedding 是纯函数，同模型同文本必然同向量，
可以放心长 TTL。批量请求只对未命中的文本回源，命中的直接拼回原位置。
"""
from __future__ import annotations

import hashlib
import logging

from app.domain.catalog.ports.retrieval_ports import EmbeddingClient
from app.infrastructure.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)

_TTL_SECONDS = 7 * 24 * 3600


class CachedEmbeddingClient(EmbeddingClient):
    def __init__(self, inner: EmbeddingClient, cache: RedisCache, model: str) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self._model}\n{text}".encode()).hexdigest()
        return f"emb:{self._model}:{digest}"

    async def _safe_get(self, key: str):
        """缓存读取的纵深防御。

        RedisCache 自己已经吞异常，但装饰器不能依赖这个假设——换实现或
        客户端在其他层报错时，不能让 embedding 主链路跟着挂。
        """
        try:
            return await self._cache.get_json(key)
        except Exception as err:  # noqa: BLE001
            logger.warning("embedding 缓存读异常，按未命中处理：%s", err)
            return None

    async def _safe_set(self, key: str, vector: list[float]) -> None:
        try:
            await self._cache.set_json(key, vector, _TTL_SECONDS)
        except Exception as err:  # noqa: BLE001
            logger.warning("embedding 缓存写异常，跳过：%s", err)

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = []
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = await self._safe_get(self._key(text))
            if isinstance(cached, list) and cached:
                self.hits += 1
                results.append(cached)
            else:
                self.misses += 1
                results.append(None)
                pending.append((index, text))

        if pending:
            fresh = await self._inner.embed_batch([text for _, text in pending])
            for (index, text), vector in zip(pending, fresh):
                results[index] = vector
                await self._safe_set(self._key(text), vector)

        # 到这里不应再有 None；若上游少返回则直接暴露，不静默补零向量
        missing = [index for index, vector in enumerate(results) if vector is None]
        if missing:
            raise RuntimeError(f"embedding 结果缺失，下标={missing}")
        return results  # type: ignore[return-value]
