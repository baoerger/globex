# -*- coding: utf-8 -*-
"""二期计价规则单测：汇率换算 + 关税运费到手价。"""
import pytest

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.shipping.tariff_schedule import TariffSchedule


@pytest.fixture()
def rates() -> ExchangeRateTable:
    return ExchangeRateTable()


@pytest.fixture()
def schedule(rates) -> TariffSchedule:
    return TariffSchedule(rates=rates)


class TestExchangeRate:
    def test_same_currency_is_identity(self, rates):
        money = Money.from_major_units(100, "CNY")
        assert rates.convert(money, "CNY") is money

    def test_usd_to_cny(self, rates):
        converted = rates.convert(Money.from_major_units(100, "USD"), "CNY")
        assert converted.currency == "CNY"
        assert converted.to_major_units() == pytest.approx(710.0, abs=0.01)

    def test_cny_to_usd_roundtrip_close(self, rates):
        original = Money.from_major_units(710, "CNY")
        roundtrip = rates.convert(rates.convert(original, "USD"), "CNY")
        assert roundtrip.to_major_units() == pytest.approx(710.0, abs=0.05)

    def test_reject_unknown_currency(self, rates):
        with pytest.raises(ValueError, match="不支持"):
            rates.rate("USD", "KRW")


class TestTariffSchedule:
    def test_de_minimis_zero_tariff(self, schedule):
        # 189 CNY 寄 CN，远低于免税额度 → 关税 0
        quote = schedule.quote(
            subtotal=Money.from_major_units(189, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        assert quote.de_minimis_applied is True
        assert quote.tariff.to_major_units() == 0.0
        assert quote.freight.to_major_units() == 25.0
        assert quote.landed_total().to_major_units() == pytest.approx(214.0)

    def test_tariff_above_de_minimis(self, schedule):
        # 6000 CNY 寄 CN 旅行装备：应税 1000 × 9% = 90
        quote = schedule.quote(
            subtotal=Money.from_major_units(6000, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        assert quote.de_minimis_applied is False
        assert quote.tariff.to_major_units() == pytest.approx(90.0)

    def test_us_electronics_zero_rate(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(2000, "USD"),
            category="数码配件", ship_to="US", quantity=1, target_currency="USD",
        )
        assert quote.tariff_rate == 0.0
        assert quote.tariff.to_major_units() == 0.0

    def test_multi_quantity_freight_increment(self, schedule):
        single = schedule.quote(
            subtotal=Money.from_major_units(100, "CNY"),
            category="旅行装备", ship_to="CN", quantity=1, target_currency="CNY",
        )
        triple = schedule.quote(
            subtotal=Money.from_major_units(300, "CNY"),
            category="旅行装备", ship_to="CN", quantity=3, target_currency="CNY",
        )
        # 首件全价 + 续件 60%：3 件 = 1 + 0.6*2 = 2.2 倍
        assert triple.freight.to_major_units() == pytest.approx(single.freight.to_major_units() * 2.2)

    def test_target_currency_conversion(self, schedule):
        quote = schedule.quote(
            subtotal=Money.from_major_units(710, "CNY"),
            category="旅行装备", ship_to="US", quantity=1, target_currency="USD",
        )
        assert quote.subtotal.currency == "USD"
        assert quote.subtotal.to_major_units() == pytest.approx(100.0, abs=0.05)

    def test_unsupported_destination(self, schedule):
        with pytest.raises(ValueError, match="暂不支持的目的国"):
            schedule.quote(
                subtotal=Money.from_major_units(100, "CNY"),
                category="旅行装备", ship_to="BR", quantity=1, target_currency="CNY",
            )
