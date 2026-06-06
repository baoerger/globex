# -*- coding: utf-8 -*-
"""OpenAIEmbeddingClient 回归单测。

起因是一个真实事故：内部 embedding 网关在单批超过 10 条时返回
**HTTP 200 + 空 body**，`raise_for_status()` 放行，最终在 `.json()` 处抛
`JSONDecodeError`，报错完全指不到真因；而 `bootstrap_product_index` 会吞掉异常
降级关键词召回，于是线上表现为「向量检索静默失效」。

商品种子库原本恰好 10 个 SPU（正好卡在上限内），所以问题一直没暴露，
直到为召回评测把商品库扩到 60 个才被发现。这几条用例把它钉死。
"""
from types import SimpleNamespace

import httpx
import pytest

from app.infrastructure.embedding import openai_embedding_client as mod
from app.infrastructure.embedding.openai_embedding_client import OpenAIEmbeddingClient


def _settings():
    return SimpleNamespace(
        embedding_base_url="http://fake-gateway/v1",
        embedding_api_key="k",
        embedding_model="text-embedding-v4",
    )


def _install(monkeypatch, handler):
    """把 httpx.AsyncClient 换成走 MockTransport 的版本。"""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(mod.httpx, "AsyncClient", factory)


class TestBatchChunking:
    async def test_splits_into_chunks_within_gateway_limit(self, monkeypatch):
        monkeypatch.setattr(mod, "_MAX_BATCH", 10)
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            payload = _json.loads(request.content)
            seen.append(len(payload["input"]))
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": i, "embedding": [float(len(text))]}
                        for i, text in enumerate(payload["input"])
                    ],
                },
            )

        _install(monkeypatch, handler)
        texts = [f"t{i}" for i in range(60)]
        vectors = await OpenAIEmbeddingClient(_settings()).embed_batch(texts)

        assert len(vectors) == 60, "分片后必须补齐所有向量"
        assert seen == [10] * 6, "每批不得超过网关上限"

    async def test_preserves_global_order_across_chunks(self, monkeypatch):
        """跨分片的顺序必须与入参一致，否则商品和向量会错配。"""
        monkeypatch.setattr(mod, "_MAX_BATCH", 3)

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            payload = _json.loads(request.content)
            # 故意乱序返回，考验按 index 回位
            data = [
                {"index": i, "embedding": [float(text)]}
                for i, text in enumerate(payload["input"])
            ]
            return httpx.Response(200, json={"data": list(reversed(data))})

        _install(monkeypatch, handler)
        texts = [str(i) for i in range(7)]
        vectors = await OpenAIEmbeddingClient(_settings()).embed_batch(texts)

        assert [v[0] for v in vectors] == [float(i) for i in range(7)]

    async def test_no_request_for_empty_input(self, monkeypatch):
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"data": []})

        _install(monkeypatch, handler)
        assert await OpenAIEmbeddingClient(_settings()).embed_batch([]) == []
        assert not called


class TestEmptyBodyDiagnostics:
    async def test_empty_body_raises_actionable_error(self, monkeypatch):
        """HTTP 200 + 空 body 必须报出「批量上限」，而不是 JSONDecodeError。"""
        monkeypatch.setattr(mod, "_MAX_BATCH", 10)
        _install(monkeypatch, lambda request: httpx.Response(200, content=b""))

        with pytest.raises(RuntimeError) as excinfo:
            await OpenAIEmbeddingClient(_settings()).embed_batch(["a", "b"])

        message = str(excinfo.value)
        assert "空 body" in message
        assert "EMBEDDING_MAX_BATCH" in message

    async def test_missing_data_field_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "_MAX_BATCH", 10)
        _install(monkeypatch, lambda request: httpx.Response(200, json={"error": "boom"}))

        with pytest.raises(RuntimeError, match="embedding 响应异常"):
            await OpenAIEmbeddingClient(_settings()).embed_batch(["a"])

    async def test_http_error_still_propagates(self, monkeypatch):
        monkeypatch.setattr(mod, "_MAX_BATCH", 10)
        _install(monkeypatch, lambda request: httpx.Response(500, content=b"boom"))

        with pytest.raises(httpx.HTTPStatusError):
            await OpenAIEmbeddingClient(_settings()).embed_batch(["a"])
