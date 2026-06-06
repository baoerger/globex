# -*- coding: utf-8 -*-
"""index_bootstrap

启动时对种子商品建向量库：searchable_text 批量 embed → ensure_ready → upsert（幂等）。
embedding 服务异常时仅告警不阻塞启动，检索链路自动降级关键词召回。
"""
from __future__ import annotations

import logging

from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient, ProductVectorIndex

logger = logging.getLogger(__name__)


async def bootstrap_product_index(
    product_repo: ProductRepository,
    embedder: EmbeddingClient,
    vector_index: ProductVectorIndex,
) -> bool:
    """建库成功返回 True；失败告警返回 False（检索走关键词降级）。"""
    try:
        products = await product_repo.list_all()
        embeddings = await embedder.embed_batch([p.searchable_text() for p in products])
        if not embeddings:
            logger.warning("商品库为空，跳过向量建库")
            return False
        await vector_index.ensure_ready(vector_dim=len(embeddings[0]))
        await vector_index.upsert_products(products, embeddings)
        logger.info("向量索引就绪：%d 个商品（dim=%d）", len(products), len(embeddings[0]))
        return True
    except Exception as err:  # noqa: BLE001 —— 建库失败不阻塞启动
        logger.warning("向量建库失败，检索将降级关键词召回：%s", err)
        return False
