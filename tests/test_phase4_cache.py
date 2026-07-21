# -*- coding: utf-8 -*-
"""四期模块二：Redis 缓存（embedding 缓存 / 语义缓存 / 幂等键 / 旁路降级）

用内存替身模拟 Redis，重点验证两件容易出事的事：
    1. Redis 故障必须旁路，不能把缓存变成新的单点；
    2. 写操作意图（下单/取消）与上下文依赖问句绝不能进语义缓存。
"""
from __future__ import annotations

import pytest

from app.infrastructure.cache.cached_embedding_client import CachedEmbeddingClient
from app.infrastructure.cache.redis_cache import RedisCache
from app.infrastructure.cache.semantic_cache import SemanticCache, is_cacheable_query

pytestmark = pytest.mark.asyncio


class InMemoryCache(RedisCache):
    """RedisCache 的内存替身：只覆盖存取，行为语义保持一致。"""

    def __init__(self) -> None:
        super().__init__("")
        self._store: dict = {}
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def ping(self) -> bool:
        return self._enabled

    async def get_json(self, key):
        return self._store.get(key)

    async def set_json(self, key, value, ttl_seconds):
        self._store[key] = value

    async def delete(self, key):
        self._store.pop(key, None)

    async def set_if_absent(self, key, value, ttl_seconds):
        if key in self._store:
            return False
        self._store[key] = value
        return True


class BrokenCache(RedisCache):
    """所有操作都抛异常的 Redis：验证调用方是否真的旁路。"""

    def __init__(self) -> None:
        super().__init__("")
        self._client = object()  # 假装已连接，让 enabled 为 True

    async def get_json(self, key):
        raise RuntimeError("connection refused")

    async def set_json(self, key, value, ttl_seconds):
        raise RuntimeError("connection refused")


class CountingEmbedder:
    """按文本首字符造可控向量，同时统计上游调用次数。"""

    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        # 同文本必得同向量；不同文本方向差异明显，便于测相似度
        seed = sum(ord(ch) for ch in text) % 97
        return [1.0, seed / 97.0, 0.1]


class TestCachedEmbeddingClient:
    async def test_second_call_hits_cache(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "text-embedding-v4")
        first = await client.embed("露营灯")
        second = await client.embed("露营灯")
        assert first == second
        assert inner.calls == 1, "同文本第二次不应再打上游"
        assert (client.hits, client.misses) == (1, 1)

    async def test_batch_only_fetches_missing_texts(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "text-embedding-v4")
        await client.embed_batch(["A", "B"])
        inner.texts.clear()
        result = await client.embed_batch(["A", "B", "C"])

        assert inner.texts == ["C"], "只应回源未命中的文本"
        assert len(result) == 3
        # 命中项要拼回原位置，顺序不能乱
        assert result[0] == CountingEmbedder._vector("A")
        assert result[2] == CountingEmbedder._vector("C")

    async def test_broken_redis_falls_through_to_upstream(self):
        """Redis 故障时必须照常返回结果，只是没有缓存收益。"""
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, BrokenCache(), "text-embedding-v4")
        assert await client.embed("露营灯") == CountingEmbedder._vector("露营灯")
        assert await client.embed("露营灯") == CountingEmbedder._vector("露营灯")
        assert inner.calls == 2  # 没有缓存，但没有失败

    async def test_empty_batch_short_circuits(self):
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "m")
        assert await client.embed_batch([]) == []
        assert inner.calls == 0


class TestSemanticCacheSafety:
    @pytest.mark.parametrize(
        "query",
        [
            "帮我下单这款露营灯",
            "取消我的订单",
            "刚才那款多少钱",
            "订单号 GBX-000001 查一下",
            "这个能寄美国吗",
        ],
    )
    async def test_write_and_context_dependent_queries_not_cacheable(self, query):
        assert is_cacheable_query(query) is False

    @pytest.mark.parametrize(
        "query",
        ["300 元以内的露营灯推荐", "降噪耳机怎么挑", "美国免税额度是多少"],
    )
    async def test_read_only_queries_are_cacheable(self, query):
        assert is_cacheable_query(query) is True

    async def test_order_intent_never_written_to_cache(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember("b1", "帮我下单这款露营灯", "已为你创建订单 GBX-000001", has_history=False)
        assert await sem.lookup("b1", "帮我下单这款露营灯", has_history=False) is None


class TestSemanticCacheHit:
    async def test_same_query_hits(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.95)
        await sem.remember("b1", "300 元以内的露营灯推荐", "推荐 LumenGo 露营灯，89 元", has_history=False)

        hit = await sem.lookup("b1", "300 元以内的露营灯推荐", has_history=False)
        assert hit is not None
        assert hit.reply == "推荐 LumenGo 露营灯，89 元"
        assert hit.similarity >= 0.95

    async def test_punctuation_variant_hits(self):
        """归一化后"多少钱?"与"多少钱"应共享缓存。"""
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.95)
        await sem.remember("b1", "美国免税额度是多少", "800 美元", has_history=False)
        assert await sem.lookup("b1", "美国免税额度是多少？", has_history=False) is not None

    async def test_buyers_are_isolated(self):
        """回复里可能含买家偏好与地址，不能跨买家复用。"""
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember("b1", "露营灯推荐", "按你不要塑料的偏好推荐 X", has_history=False)
        assert await sem.lookup("b2", "露营灯推荐", has_history=False) is None

    async def test_no_hit_when_session_has_history(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember("b1", "露营灯推荐", "推荐 X", has_history=False)
        assert await sem.lookup("b1", "露营灯推荐", has_history=True) is None

    async def test_error_reply_not_cached(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember("b1", "露营灯推荐", "[error] 上游超时", has_history=False)
        assert await sem.lookup("b1", "露营灯推荐", has_history=False) is None

    async def test_disabled_when_redis_absent(self):
        sem = SemanticCache(RedisCache(""), CountingEmbedder())
        assert sem.enabled is False
        assert await sem.lookup("b1", "露营灯推荐", has_history=False) is None


class TestPreferenceScopedCache:
    """偏好变化必须使旧回复失效。

    真实踩过的坑：买家把“不要塑料”撤回后又问了一句语义相近的话，
    缓存桶 key 不含偏好指纹，于是命中了撤回前的回复——里面还写着
    “符合您不要塑料的偏好”，直接向用户断言了一条它刚删掉的偏好。
    """

    async def test_scope_change_invalidates(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember(
            "b1", "旅行三件套推荐", "按你不要塑料的偏好推荐 X",
            has_history=False, scope="pref-v1",
        )

        # 偏好撤回后指纹变了，旧回复不得命中
        assert await sem.lookup(
            "b1", "旅行三件套推荐", has_history=False, scope="pref-v2",
        ) is None

    async def test_same_scope_still_hits(self):
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember(
            "b1", "旅行三件套推荐", "推荐 X", has_history=False, scope="pref-v1",
        )
        hit = await sem.lookup(
            "b1", "旅行三件套推荐", has_history=False, scope="pref-v1",
        )
        assert hit is not None and hit.reply == "推荐 X"

    async def test_empty_scope_isolated_from_nonempty(self):
        """从无偏好到有偏好（首次记住）同样要失效。"""
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.9)
        await sem.remember("b1", "旅行三件套推荐", "无偏好时的推荐", has_history=False)
        assert await sem.lookup(
            "b1", "旅行三件套推荐", has_history=False, scope="pref-v1",
        ) is None


class TestIdempotencyKey:
    async def test_first_wins_second_rejected(self):
        cache = InMemoryCache()
        assert await cache.set_if_absent("idem:s1:h1", "1", 600) is True
        assert await cache.set_if_absent("idem:s1:h1", "1", 600) is False

    async def test_absent_redis_lets_request_through(self):
        """没有 Redis 时应放行（退化为无幂等保护），不能拒绝服务。"""
        assert await RedisCache("").set_if_absent("idem:s1:h1", "1", 600) is True
