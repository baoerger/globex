# -*- coding: utf-8 -*-
"""TariffSchedule / ShippingQuote

跨境到手价的规则内核：国家×品类关税费率 + 基础运费 + 免税额度。
纯规则纯函数，可单测；生产替换为关税服务时只需换实现，ShippingQuote 结构不变。

到手价口径（目标币种统一折算后计算）：
    landed = 商品小计 + 运费 + 关税
    关税   = 商品小计超出免税额度部分 × 费率（不足免税额度则为 0）
"""
from __future__ import annotations

from dataclasses import dataclass

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money

# 目的国 → 品类 → 关税费率（未列品类走 "*" 兜底）
_TARIFF_RATES: dict[str, dict[str, float]] = {
    "CN": {"数码配件": 0.13, "旅行装备": 0.09, "户外运动": 0.09, "家居生活": 0.09, "*": 0.09},
    "US": {"数码配件": 0.0, "旅行装备": 0.075, "户外运动": 0.075, "家居生活": 0.05, "*": 0.06},
    "EU": {"*": 0.12},
    "JP": {"*": 0.08},
    "SG": {"*": 0.07},
}

# 目的国免税额度（CNY 分）
_DE_MINIMIS_CNY_MINOR: dict[str, int] = {
    "CN": 5_000_00,   # 5000 元（个人物品行邮口径，简化）
    "US": 800 * 710,  # 800 USD 折 CNY 分
    "EU": 150 * 780,
    "JP": 10_000 * 5,  # 简化口径
    "SG": 400 * 530,
}

# 目的国基础运费（CNY 分，单件；多件按 60% 递增，简化的续重逻辑）
_BASE_FREIGHT_CNY_MINOR: dict[str, int] = {
    "CN": 25_00,
    "US": 65_00,
    "EU": 75_00,
    "JP": 45_00,
    "SG": 40_00,
}


@dataclass(frozen=True)
class ShippingQuote:
    ship_to: str
    subtotal: Money
    freight: Money
    tariff: Money
    tariff_rate: float
    de_minimis_applied: bool  # 是否命中免税额度（关税为 0）

    def landed_total(self) -> Money:
        return self.subtotal.add(self.freight).add(self.tariff)

    def to_dict(self) -> dict:
        return {
            "ship_to": self.ship_to,
            "subtotal_major": self.subtotal.to_major_units(),
            "freight_major": self.freight.to_major_units(),
            "tariff_major": self.tariff.to_major_units(),
            "tariff_rate": self.tariff_rate,
            "de_minimis_applied": self.de_minimis_applied,
            "landed_total_major": self.landed_total().to_major_units(),
            "currency": self.subtotal.currency,
        }


@dataclass(frozen=True)
class TariffSchedule:
    rates: ExchangeRateTable

    def supported_destinations(self) -> list[str]:
        return sorted(_TARIFF_RATES.keys())

    def quote(self, subtotal: Money, category: str, ship_to: str, quantity: int, target_currency: str) -> ShippingQuote:
        """按目的国规则计算到手价三要素，全部折算为 target_currency。"""
        if ship_to not in _TARIFF_RATES:
            raise ValueError(f"暂不支持的目的国：{ship_to}（支持 {self.supported_destinations()}）")
        if quantity <= 0:
            raise ValueError("quantity 必须为正整数")

        subtotal_target = self.rates.convert(subtotal, target_currency)

        # 运费：首件全价 + 续件 60%
        base_freight_cny = Money.of(_BASE_FREIGHT_CNY_MINOR[ship_to], "CNY")
        freight_minor_cny = round(base_freight_cny.amount_in_minor_units * (1 + 0.6 * (quantity - 1)))
        freight_target = self.rates.convert(Money.of(freight_minor_cny, "CNY"), target_currency)

        # 关税：小计（CNY 口径）超出免税额度部分 × 费率
        rate_table = _TARIFF_RATES[ship_to]
        tariff_rate = rate_table.get(category, rate_table["*"])
        subtotal_cny = self.rates.convert(subtotal, "CNY")
        de_minimis_minor = _DE_MINIMIS_CNY_MINOR[ship_to]
        taxable_minor_cny = max(0, subtotal_cny.amount_in_minor_units - de_minimis_minor)
        de_minimis_applied = taxable_minor_cny == 0
        tariff_cny = Money.of(round(taxable_minor_cny * tariff_rate), "CNY")
        tariff_target = self.rates.convert(tariff_cny, target_currency)

        return ShippingQuote(
            ship_to=ship_to,
            subtotal=subtotal_target,
            freight=freight_target,
            tariff=tariff_target,
            tariff_rate=tariff_rate,
            de_minimis_applied=de_minimis_applied,
        )
