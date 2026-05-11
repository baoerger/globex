# -*- coding: utf-8 -*-
"""Order 聚合根

状态机（移除支付/物流，仅作下单意向单据）：
    DRAFT → CONFIRMED → CANCELLED

不变量：
    - 至少一条订单行，且全部订单行币种一致；
    - 只有 CONFIRMED 态可以取消，取消必须带 reason。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from app.domain.catalog.money import Money
from app.domain.order.address import Address
from app.domain.order.order_line import OrderLine


class OrderStatus(str, Enum):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Order:
    order_id: str
    buyer_id: str
    shipping_address: Address
    lines: list[OrderLine]
    status: OrderStatus = OrderStatus.DRAFT
    created_at: datetime = field(default_factory=_now)
    confirmed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancel_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("Order.order_id required")
        if not self.buyer_id:
            raise ValueError("Order.buyer_id required")
        if not self.lines:
            raise ValueError(f"Order 至少要有一条订单行：{self.order_id}")
        currencies = {line.unit_price.currency for line in self.lines}
        if len(currencies) > 1:
            raise ValueError(f"Order 订单行币种不一致：{currencies}")

    @staticmethod
    def place(order_id: str, buyer_id: str, shipping_address: Address, lines: list[OrderLine]) -> "Order":
        """下单即确认：MVP 中 MainAgent 已经完成用户确认，创建后直接进入 CONFIRMED。"""
        order = Order(order_id=order_id, buyer_id=buyer_id, shipping_address=shipping_address, lines=lines)
        order.confirm()
        return order

    def confirm(self) -> None:
        if self.status is not OrderStatus.DRAFT:
            raise ValueError(f"仅 DRAFT 态可确认，当前={self.status.value}：{self.order_id}")
        self.status = OrderStatus.CONFIRMED
        self.confirmed_at = _now()

    def cancel(self, reason: str) -> None:
        if self.status is not OrderStatus.CONFIRMED:
            raise ValueError(f"仅 CONFIRMED 态可取消，当前={self.status.value}：{self.order_id}")
        if not reason or not reason.strip():
            raise ValueError("Order.cancel 必须提供 reason")
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = _now()
        self.cancel_reason = reason

    def total_amount(self) -> Money:
        total = self.lines[0].subtotal()
        for line in self.lines[1:]:
            total = total.add(line.subtotal())
        return total

    def snapshot(self) -> dict:
        """给工具层回传的结构化快照，金额只出现在这里，Agent 不允许自行计算。"""
        return {
            "order_id": self.order_id,
            "buyer_id": self.buyer_id,
            "status": self.status.value,
            "total_amount_major": self.total_amount().to_major_units(),
            "currency": self.total_amount().currency,
            "shipping_address": self.shipping_address.one_line(),
            "lines": [
                {
                    "product_id": line.product_id,
                    "sku_id": line.sku_id,
                    "title": line.title,
                    "unit_price_major": line.unit_price.to_major_units(),
                    "quantity": line.quantity,
                }
                for line in self.lines
            ],
            "created_at": self.created_at.isoformat(),
            "cancel_reason": self.cancel_reason,
        }
