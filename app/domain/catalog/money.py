# -*- coding: utf-8 -*-
"""Money 值对象

跨境业务中价格是核心不变量，统一以最小货币单位（分）存储，避免浮点误差。
不变量：amount_in_minor_units 必须为非负整数；currency 必须是受支持的 ISO-4217 三位代码。
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CNY", "HKD", "AUD", "CAD", "SGD")


@dataclass(frozen=True)
class Money:
    amount_in_minor_units: int
    currency: str

    @staticmethod
    def of(amount: int, currency: str) -> "Money":
        if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
            raise ValueError(f"Money.amount_in_minor_units 必须是非负整数，实际={amount}")
        if currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Money.currency 不受支持：{currency}")
        return Money(amount, currency)

    @staticmethod
    def from_major_units(major: float, currency: str) -> "Money":
        # 简化处理：所有支持币种均按 100 倍换算（JPY 实际为 1，demo 暂统一）
        return Money.of(round(major * 100), currency)

    def add(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount_in_minor_units + other.amount_in_minor_units, self.currency)

    def multiply(self, quantity: int) -> "Money":
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0:
            raise ValueError("Money.multiply.quantity 必须是非负整数")
        return Money(self.amount_in_minor_units * quantity, self.currency)

    def to_major_units(self) -> float:
        return self.amount_in_minor_units / 100

    def __str__(self) -> str:
        return f"{self.to_major_units():.2f} {self.currency}"

    def _assert_same_currency(self, other: "Money") -> None:
        if other.currency != self.currency:
            raise ValueError(f"Money 币种不一致：{self.currency} vs {other.currency}")
