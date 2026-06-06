# -*- coding: utf-8 -*-
"""QdrantProductIndex

商品向量索引的 Qdrant 实现（COSINE）。两种形态同一套代码：
    - QDRANT_URL 已配置 → 连 Qdrant 服务端（Docker / 远程）
    - 未配置          → qdrant-client 本地嵌入模式（落盘 DATA_DIR/qdrant，零外部依赖）

point id 用 product_id 的确定性 UUID5，payload 存 product_id，upsert 幂等。
"""
from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.domain.catalog.ports.retrieval_ports import ProductVectorIndex, VectorHit
from app.domain.catalog.product import Product
from app.infrastructure.settings import Settings


def _point_id(product_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"globex/product/{product_id}"))


class QdrantProductIndex(ProductVectorIndex):
    def __init__(self, settings: Settings) -> None:
        if settings.qdrant_url:
            self._client = AsyncQdrantClient(url=settings.qdrant_url)
        else:
            local_path = settings.data_dir / "qdrant"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client = AsyncQdrantClient(path=str(local_path))
        self._collection = settings.qdrant_collection

    async def ensure_ready(self, vector_dim: int) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

    async def upsert_products(self, products: list[Product], embeddings: list[list[float]]) -> None:
        if len(products) != len(embeddings):
            raise ValueError("products 与 embeddings 数量不一致")
        if not products:
            return
        points = [
            PointStruct(
                id=_point_id(product.product_id),
                vector=embedding,
                payload={"product_id": product.product_id},
            )
            for product, embedding in zip(products, embeddings)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, embedding: list[float], top_n: int) -> list[VectorHit]:
        result = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=top_n,
            with_payload=True,
        )
        return [
            VectorHit(product_id=point.payload["product_id"], score=point.score)
            for point in result.points
            if point.payload and "product_id" in point.payload
        ]

    async def close(self) -> None:
        await self._client.close()
