# -*- coding: utf-8 -*-
"""CatalogSearchUseCase

商品检索核心 UseCase，对齐参考实现五步流程：
    1. EmbeddingClient 把 normalized_query 向量化
    2. ProductVectorIndex.search(top_n) 拿候选 product_id（Qdrant，COSINE）
    3. ProductRepository.find_by_ids 还原 Product 聚合
    4. Reranker 精排取 top_k；失败/未配置降级按向量分排序（rerank_applied=false）
    5. 组装商品卡 JSON；命中 ship_to 时内联到手价（小计+运费+关税，统一目标币种）

降级链（recall_strategy 如实标注）：
    embedding_rerank → embedding_only → keyword_2gram（embedding 服务异常时兜底）

计价收敛设计：到手价在检索链路内联计算（TariffSchedule 规则内核），
不给 Agent 单独暴露比价/运费工具，减少不必要的工具调用轮次。

过滤可观测：被 ship_to / price_max_major 硬约束挡掉的候选以 filtered_out 摘要回传，
让模型能区分"库里没有这个商品"与"有但不满足约束"，不致于给出误导性结论。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    Reranker,
)
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.shipping.tariff_schedule import TariffSchedule

logger = logging.getLogger(__name__)

# 一阶段召回候选数（> top_k，给精排留空间）
_RECALL_TOP_N = 8

# 被硬约束挡掉的候选回传条数上限（只回摘要，避免上下文膨胀）
_FILTERED_OUT_LIMIT = 3


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        return card


def tokenize(text: str) -> set[str]:
    """极简分词：空格切词 + 中文连续段落的 2-gram（关键词降级召回用）。"""
    terms: set[str] = set()
    for chunk in text.lower().split():
        terms.add(chunk)
        # 对含 CJK 的 chunk 补 2-gram，缓解中文无空格问题
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) >= 2:
            terms.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


class CatalogSearchUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        embedder: Optional[EmbeddingClient] = None,
        vector_index: Optional[ProductVectorIndex] = None,
        reranker: Optional[Reranker] = None,
        tariff_schedule: Optional[TariffSchedule] = None,
    ) -> None:
        self._product_repo = product_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._tariff = tariff_schedule or TariffSchedule(rates=ExchangeRateTable())

    async def execute(self, spec: ProductSearchSpec) -> dict:
        scored: list[tuple[float, Product]] = []
        recall_strategy = "keyword_2gram"
        rerank_applied = False

        if self._embedder is not None and self._vector_index is not None:
            try:
                scored = await self._vector_recall(spec)
                recall_strategy = "embedding_only"
            except Exception as err:  # noqa: BLE001 —— 召回基建异常必须降级而非失败
                logger.warning("向量召回不可用，降级关键词召回：%s", err)
                scored = []

        if recall_strategy == "embedding_only" and scored:
            # 二阶段精排；失败降级按向量分排序
            try:
                scored = await self._rerank(spec, scored)
                recall_strategy = "embedding_rerank"
                rerank_applied = True
            except Exception as err:  # noqa: BLE001
                logger.warning("rerank 不可用，按向量分排序：%s", err)
        elif not scored:
            scored = await self._keyword_recall(spec)
            recall_strategy = "keyword_2gram"

        # ship_to / 价格硬约束过滤 + top_k 截断（硬约束走结构化过滤，不交给模型）
        filtered: list[tuple[float, Product]] = []
        filtered_out: list[dict] = []
        for score, product in scored:
            reason = self._reject_reason(product, spec)
            if reason is None:
                filtered.append((score, product))
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(product, spec, reason))

        hits = [self._to_card(score, product, spec) for score, product in filtered[: spec.top_k]]
        result = {
            "hits": [card.to_dict() for card in hits],
            "total_candidates": len(filtered),
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
        }
        if filtered_out:
            # 如实告知"召回到了但被硬约束挡掉"，否则模型分不清"库里没有"与"被过滤"，
            # 会把超预算商品答成"没有这个商品"
            result["filtered_out"] = filtered_out
        return result

    def _reject_reason(self, product: Product, spec: ProductSearchSpec) -> Optional[str]:
        """返回硬约束拒绝原因，None 表示通过。"""
        if spec.ship_to and spec.ship_to not in product.ships_to:
            return "ship_to_unavailable"
        if not self._within_price_cap(product, spec):
            return "over_price_cap"
        return None

    def _to_rejected(self, product: Product, spec: ProductSearchSpec, reason: str) -> dict:
        primary_in_target = self._tariff.rates.convert(product.primary_sku().price, spec.target_currency)
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "price_major": round(primary_in_target.to_major_units(), 2),
            "currency": spec.target_currency,
            "reason": reason,
        }

    def _within_price_cap(self, product: Product, spec: ProductSearchSpec) -> bool:
        if spec.price_max_major is None:
            return True
        primary_in_target = self._tariff.rates.convert(product.primary_sku().price, spec.target_currency)
        return primary_in_target.to_major_units() <= spec.price_max_major

    # ---- 一阶段：向量召回 ----

    async def _vector_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        embedding = await self._embedder.embed(spec.normalized_query)
        vector_hits = await self._vector_index.search(embedding, top_n=_RECALL_TOP_N)
        products = await self._product_repo.find_by_ids([hit.product_id for hit in vector_hits])
        by_id = {product.product_id: product for product in products}
        return [
            (hit.score, by_id[hit.product_id])
            for hit in vector_hits
            if hit.product_id in by_id
        ]

    # ---- 二阶段：精排 ----

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [product.searchable_text() for _, product in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        reranked = [
            (rerank_scores[i], product)
            for i, (_, product) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    # ---- 兜底：关键词召回 ----

    async def _keyword_recall(self, spec: ProductSearchSpec) -> list[tuple[float, Product]]:
        query_terms = tokenize(spec.normalized_query)
        candidates: list[tuple[float, Product]] = []
        for product in await self._product_repo.list_all():
            score = self._keyword_score(query_terms, product, spec)
            if score > 0:
                candidates.append((score, product))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates

    @staticmethod
    def _keyword_score(query_terms: set[str], product: Product, spec: ProductSearchSpec) -> float:
        doc_terms = tokenize(product.searchable_text())
        matched = query_terms & doc_terms
        if not matched:
            return 0.0
        score = float(len(matched))
        # 品类槽位命中加权，让"槽位过滤"优于全文命中
        if spec.category and spec.category in product.category:
            score += 3.0
        return score

    # ---- 商品卡组装（含到手价内联）----

    def _to_card(self, score: float, product: Product, spec: ProductSearchSpec) -> ProductCard:
        primary = product.primary_sku()
        landed_price: Optional[dict] = None
        if spec.ship_to:
            try:
                quote = self._tariff.quote(
                    subtotal=primary.price,
                    category=product.category,
                    ship_to=spec.ship_to,
                    quantity=1,
                    target_currency=spec.target_currency,
                )
                landed_price = quote.to_dict()
            except ValueError as err:
                # 目的国不在规则表内：如实标注，不编造数字
                landed_price = {"unavailable_reason": str(err)}
        return ProductCard(
            product_id=product.product_id,
            title=product.title,
            brand=product.brand,
            category=product.category,
            origin_country=product.origin_country,
            price_major=primary.price.to_major_units(),
            currency=primary.price.currency,
            highlights=[f"{h.label}：{h.detail}" if h.detail else h.label for h in product.highlights],
            skus=[
                {
                    "sku_id": sku.sku_id,
                    "spec": sku.spec,
                    "price_major": sku.price.to_major_units(),
                    "currency": sku.price.currency,
                    "stock": sku.stock,
                }
                for sku in product.skus
            ],
            score=score,
            landed_price=landed_price,
        )
