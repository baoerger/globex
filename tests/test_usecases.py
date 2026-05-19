# -*- coding: utf-8 -*-
"""usecase 层单测：商品召回 + 订单闭环（不依赖 LLM）。"""
import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.order.address import Address
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
    InMemoryProductRepository,
)


@pytest.fixture()
def product_repo() -> InMemoryProductRepository:
    return InMemoryProductRepository()


@pytest.fixture()
def order_repo() -> InMemoryOrderRepository:
    return InMemoryOrderRepository()


def _address() -> Address:
    return Address(
        recipient_name="张三",
        country="CN",
        state="浙江",
        city="杭州",
        address_line="西湖区某路 1 号",
        postal_code="310000",
        phone="13800000000",
    )


class TestCatalogSearch:
    async def test_recall_travel_set(self, product_repo):
        usecase = CatalogSearchUseCase(product_repo)
        result = await usecase.execute(ProductSearchSpec(normalized_query="旅行三件套 抗造 轻便 无塑料"))
        assert result["hits"], "旅行三件套应能召回"
        assert result["hits"][0]["product_id"] == "P1001", "语义最相关的 SPU 应排第一"

    async def test_ship_to_filter(self, product_repo):
        usecase = CatalogSearchUseCase(product_repo)
        result = await usecase.execute(ProductSearchSpec(normalized_query="旅行茶具", ship_to="US"))
        # P1006 只发 CN/JP，指定 ship_to=US 后不应出现
        assert all(hit["product_id"] != "P1006" for hit in result["hits"])

    async def test_top_k_limit(self, product_repo):
        usecase = CatalogSearchUseCase(product_repo)
        result = await usecase.execute(ProductSearchSpec(normalized_query="旅行", top_k=2))
        assert len(result["hits"]) <= 2

    async def test_no_hit_returns_empty(self, product_repo):
        usecase = CatalogSearchUseCase(product_repo)
        result = await usecase.execute(ProductSearchSpec(normalized_query="quantum flux capacitor"))
        assert result["hits"] == []


class TestOrderLifecycle:
    async def test_place_query_cancel_roundtrip(self, product_repo, order_repo):
        place = PlaceOrderUseCase(product_repo, order_repo)
        query = QueryOrderUseCase(order_repo)
        cancel = CancelOrderUseCase(product_repo, order_repo)

        snapshot = await place.execute(
            buyer_id="buyer-1",
            items=[OrderItemInput(product_id="P1001", sku_id="P1001-S1", quantity=2)],
            shipping_address=_address(),
        )
        assert snapshot["status"] == "CONFIRMED"
        assert snapshot["total_amount_major"] == 378.0
        assert snapshot["currency"] == "CNY"

        # 下单扣库存
        product = await product_repo.find_by_id("P1001")
        assert product.find_sku("P1001-S1").stock == 48

        queried = await query.execute(snapshot["order_id"])
        assert queried["order_id"] == snapshot["order_id"]

        cancelled = await cancel.execute(snapshot["order_id"], "买家改主意了")
        assert cancelled["status"] == "CANCELLED"
        # 取消回补库存
        assert product.find_sku("P1001-S1").stock == 50

    async def test_insufficient_stock_rolls_back(self, product_repo, order_repo):
        place = PlaceOrderUseCase(product_repo, order_repo)
        product = await product_repo.find_by_id("P1006")
        original_stock = product.find_sku("P1006-S1").stock

        with pytest.raises(ValueError, match="库存不足"):
            await place.execute(
                buyer_id="buyer-1",
                items=[
                    OrderItemInput(product_id="P1006", sku_id="P1006-S1", quantity=5),
                    OrderItemInput(product_id="P1006", sku_id="P1006-S1", quantity=99),
                ],
                shipping_address=_address(),
            )
        # 首行已扣的库存必须回滚
        assert product.find_sku("P1006-S1").stock == original_stock

    async def test_query_unknown_order(self, order_repo):
        query = QueryOrderUseCase(order_repo)
        with pytest.raises(ValueError, match="订单不存在"):
            await query.execute("GBX-999999")
