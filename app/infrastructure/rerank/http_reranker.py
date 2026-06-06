# -*- coding: utf-8 -*-
"""HttpReranker

HTTP 精排客户端（对接 Qwen3-Reranker 等 /rerank 协议服务）。
RERANKER_BASE_URL 未配置时组装根不会实例化本类；调用失败抛异常，
由 CatalogSearchUseCase 降级为按向量分排序并标注 rerank_applied=false。
"""
from __future__ import annotations

import httpx

from app.domain.catalog.ports.retrieval_ports import Reranker
from app.infrastructure.settings import Settings


class HttpReranker(Reranker):
    def __init__(self, settings: Settings, timeout_seconds: float = 3.0) -> None:
        self._base_url = settings.reranker_base_url.rstrip("/")
        self._model = settings.reranker_model
        self._timeout = timeout_seconds

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/rerank",
                json={"model": self._model, "query": query, "documents": documents},
            )
            response.raise_for_status()
            body = response.json()
        # 兼容 {results:[{index, relevance_score}]} 协议（Jina/TEI/vLLM rerank 通用形态）
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise RuntimeError(f"rerank 响应异常：{str(body)[:200]}")
        scores = [0.0] * len(documents)
        for item in results:
            scores[item["index"]] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores
