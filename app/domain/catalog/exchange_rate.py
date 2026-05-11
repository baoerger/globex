# -*- coding: utf-8 -*-
"""ExchangeRateTable 值对象

静态汇率表（各币种 → CNY 中间价，最小单位口径一致均为"分"级换算后再取整）。
MVP 用硬编码快照；生产替换为汇率服务时只需换实现，Money.convert_to 口径不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.catalog.money import SUPPORTED_CURRENCIES, Money

# 静态快照：1 单位外币 = N CNY
_RATES_TO_CNY: dict[str, float] = {
    "CNY": 1.0,
    "USD": 7.10,
    "EUR": 7.80,
    "GBP": 9.10,
    "JPY": 0.048,
    "HKD": 0.91,
    "AUD": 4.70,
    "CAD": 5.20,
    "SGD": 5.30,
}


@dataclass(frozen=True)
class ExchangeRateTable:
    rates_to_cny: dict[str, float] = field(default_factory=lambda: dict(_RATES_TO_CNY))

    def rate(self, from_currency: str, to_currency: str) -> float:
        for currency in (from_currency, to_currency):
            if currency not in self.rates_to_cny:
                raise ValueError(f"汇率表不支持币种：{currency}")
        return self.rates_to_cny[from_currency] / self.rates_to_cny[to_currency]

    def convert(self, money: Money, to_currency: str) -> Money:
        if to_currency == money.currency:
            return money
        if to_currency not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Money.currency 不受支持：{to_currency}")
        converted_minor = round(money.amount_in_minor_units * self.rate(money.currency, to_currency))
        return Money.of(converted_minor, to_currency)
