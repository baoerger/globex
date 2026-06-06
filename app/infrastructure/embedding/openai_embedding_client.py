# -*- coding: utf-8 -*-
"""OpenAIEmbeddingClient

OpenAI 兼容 /v1/embeddings 客户端（httpx 直连，不引入 openai SDK 的 embedding 封装，
便于对接任意兼容网关）。模型默认 text-embedding-v4。
"""
from __future__ import annotations

import os

import httpx

from app.domain.catalog.ports.retrieval_ports import EmbeddingClient
from app.infrastructure.settings import Settings

# 单次请求最多带多少条文本。
#
# 实测坑：内部 OpenAI 兼容网关在 input 超过 10 条时，会返回
# **HTTP 200 + content-type: application/json + 空 body**——
# `raise_for_status()` 因为状态码是 200 而放行，最终在 `response.json()` 处
# 抛出 `JSONDecodeError: Expecting value`，报错信息完全指不到真因。
#
# 更隐蔽的是：商品种子库原本恰好 10 个 SPU，正好卡在上限内，所以这个问题一直没暴露；
# 直到商品库扩到 60 个做召回评测，建库才开始整批失败——而 `bootstrap_product_index`
# 会吞掉异常降级到关键词召回，于是表现为「向量检索静默失效」而不是报错。
_MAX_BATCH = int(os.getenv("EMBEDDING_MAX_BATCH", "10"))


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings, timeout_seconds: float = 15.0) -> None:
        self._base_url = settings.embedding_base_url.rstrip("/")
        self._api_key = settings.embedding_api_key
        self._model = settings.embedding_model
        self._timeout = timeout_seconds

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # 分片串行请求：批量上限是网关侧约束，超限不会报错只会返回空 body
            for start in range(0, len(texts), _MAX_BATCH):
                chunk = texts[start : start + _MAX_BATCH]
                vectors.extend(await self._embed_chunk(client, chunk))
        return vectors

    async def _embed_chunk(
        self, client: httpx.AsyncClient, chunk: list[str],
    ) -> list[list[float]]:
        response = await client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": chunk},
        )
        response.raise_for_status()
        if not response.content:
            # 明确指向批量上限，不要让调用方对着 JSONDecodeError 猜
            raise RuntimeError(
                f"embedding 网关返回空 body（HTTP {response.status_code}，本批 {len(chunk)} 条）："
                f"通常是单次批量超过网关上限，可调小 EMBEDDING_MAX_BATCH（当前 {_MAX_BATCH}）",
            )
        body = response.json()
        if "data" not in body:
            raise RuntimeError(f"embedding 响应异常：{str(body)[:200]}")
        # 按 index 回位，避免网关乱序
        ordered = sorted(body["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]
