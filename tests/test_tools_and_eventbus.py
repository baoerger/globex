# -*- coding: utf-8 -*-
"""工具层与事件总线单测：工具直调（绕过 LLM）+ EventBus 订阅。"""
import asyncio
import json

import pytest

from app.application.tools.order_tools import build_create_order_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.application.usecases.order_usecases import PlaceOrderUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
    InMemoryProductRepository,
)

ADDRESS = {
    "recipient_name": "张三",
    "country": "CN",
    "state": "浙江",
    "city": "杭州",
    "address_line": "西湖区某路 1 号",
    "postal_code": "310000",
    "phone": "13800000000",
}


class TestTradeEventBus:
    async def test_publish_routes_to_subscriber(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        other = bus.subscribe("s2")
        bus.publish("s1", "final.result", {"text": "done"})

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "final.result"
        assert other.empty(), "事件不能串台到其他会话"

    def test_reject_unknown_event_type(self):
        bus = TradeEventBus()
        with pytest.raises(ValueError, match="未知事件类型"):
            bus.publish("s1", "not.a.type", {})


class TestToolsDirectInvoke:
    async def test_product_search_tool(self):
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套 抗造")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.content[0].text)
        assert payload["hits"][0]["product_id"] == "P1001"
        # tool.invoke + tool.result 两条事件
        assert queue.qsize() == 2

    async def test_product_search_tool_accepts_numeric_string(self):
        """回归：模型（如 qwen3-max）会把数字参数传成字符串，工具必须接住并强转，
        而不是在 schema 校验层被拒收（实测 price_max_major="300" 曾导致检索全程失败）。"""
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套 抗造", price_max_major="300", top_k="3")
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(response.content[0].text)
        # 字符串被强转为数字后进检索链路，预算硬约束生效：候选主价均不超过 300
        assert payload["hits"], "传字符串价格上限不应导致检索为空"
        for hit in payload["hits"]:
            assert hit["price_major"] <= 300
        # tool.invoke + tool.result 两条事件
        assert queue.qsize() == 2

    async def test_product_search_tool_rejects_bad_numeric_string(self):
        """非法数字字符串应返回 [error] 而不是抛异常。"""
        bus = TradeEventBus()
        bus.subscribe("s1")
        tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository()), bus)

        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            response = await tool(normalized_query="旅行三件套", price_max_major="不是数字")
        finally:
            ShoppingContext.reset(token)

        assert response.content[0].text.startswith("[error] price_max_major 非法")

    async def test_create_order_tool_and_error_path(self):
        bus = TradeEventBus()
        product_repo = InMemoryProductRepository()
        tool = build_create_order_tool(PlaceOrderUseCase(product_repo, InMemoryOrderRepository()), bus)

        # 买家身份由 ShoppingContext 注入，而非模型入参
        token = ShoppingContext.set(
            ShoppingContextSnapshot(shopping_session_id="s1", buyer_id="b1", locale="zh-CN", currency="CNY"),
        )
        try:
            ok = await tool(
                items=[{"product_id": "P1001", "sku_id": "P1001-S1", "quantity": 1}],
                shipping_address=ADDRESS,
            )
            snapshot = json.loads(ok.content[0].text)
            assert snapshot["status"] == "CONFIRMED"
            assert snapshot["order_id"].startswith("GBX-")
            assert snapshot["buyer_id"] == "b1"

            bad = await tool(
                items=[{"product_id": "P9999", "sku_id": "X", "quantity": 1}],
                shipping_address=ADDRESS,
            )
            assert bad.content[0].text.startswith("[error]")
        finally:
            ShoppingContext.reset(token)
