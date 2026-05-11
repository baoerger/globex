# -*- coding: utf-8 -*-
"""OrderLine 实体

订单行：下单时对 Sku 价格做快照（unit_price），避免商品后续调价影响已下单据。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.catalog.money import Money


@dataclass(frozen=True)
class OrderLine:
    product_id: str
    sku_id: str
    title: str
    unit_price: Money
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"OrderLine.quantity 必须为正整数：{self.sku_id}")

    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)
