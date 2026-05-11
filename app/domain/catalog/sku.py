# -*- coding: utf-8 -*-
"""Sku 实体

Product（SPU）下的最小可售卖单元，携带价格与库存。
TradeAgent 创建订单时以 Sku 粒度结算。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.catalog.money import Money


@dataclass
class Sku:
    sku_id: str
    spec: str  # 规格描述，如 "黑色 / 20寸"
    price: Money
    stock: int

    def __post_init__(self) -> None:
        if not self.sku_id:
            raise ValueError("Sku.sku_id required")
        if self.stock < 0:
            raise ValueError(f"Sku.stock 必须非负：{self.sku_id}")

    def has_stock(self, quantity: int) -> bool:
        return self.stock >= quantity

    def deduct_stock(self, quantity: int) -> None:
        if not self.has_stock(quantity):
            raise ValueError(f"Sku 库存不足：{self.sku_id}，剩余 {self.stock}，需要 {quantity}")
        self.stock -= quantity

    def restore_stock(self, quantity: int) -> None:
        if quantity < 0:
            raise ValueError("restore_stock.quantity 必须非负")
        self.stock += quantity
