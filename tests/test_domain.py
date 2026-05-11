# -*- coding: utf-8 -*-
"""domain 层单测：Money 值对象与 Order 状态机。"""
import pytest

from app.domain.catalog.money import Money
from app.domain.order.address import Address
from app.domain.order.order import Order, OrderStatus
from app.domain.order.order_line import OrderLine


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


def _line(currency: str = "CNY", major: float = 189.0, quantity: int = 2) -> OrderLine:
    return OrderLine(
        product_id="P1001",
        sku_id="P1001-S1",
        title="Nomadica 旅行三件套（军绿色）",
        unit_price=Money.from_major_units(major, currency),
        quantity=quantity,
    )


class TestMoney:
    def test_minor_units_storage(self):
        money = Money.from_major_units(189.0, "CNY")
        assert money.amount_in_minor_units == 18900
        assert money.to_major_units() == 189.0

    def test_add_and_multiply(self):
        total = Money.from_major_units(10.5, "USD").add(Money.from_major_units(0.5, "USD"))
        assert total.amount_in_minor_units == 1100
        assert total.multiply(3).amount_in_minor_units == 3300

    def test_reject_negative_amount(self):
        with pytest.raises(ValueError):
            Money.of(-1, "CNY")

    def test_reject_currency_mismatch(self):
        with pytest.raises(ValueError, match="币种不一致"):
            Money.of(100, "CNY").add(Money.of(100, "USD"))

    def test_reject_unsupported_currency(self):
        with pytest.raises(ValueError):
            Money.of(100, "XXX")


class TestOrderStateMachine:
    def test_place_enters_confirmed(self):
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        assert order.status is OrderStatus.CONFIRMED
        assert order.confirmed_at is not None

    def test_total_amount(self):
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line(quantity=2)])
        assert order.total_amount().to_major_units() == 378.0

    def test_cancel_confirmed_order(self):
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        order.cancel("买家改主意了")
        assert order.status is OrderStatus.CANCELLED
        assert order.cancel_reason == "买家改主意了"

    def test_cancel_requires_reason(self):
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        with pytest.raises(ValueError, match="reason"):
            order.cancel("  ")

    def test_cancel_twice_rejected(self):
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        order.cancel("买家改主意了")
        with pytest.raises(ValueError, match="仅 CONFIRMED"):
            order.cancel("再取消一次")

    def test_reject_mixed_currency_lines(self):
        with pytest.raises(ValueError, match="币种不一致"):
            Order.place("GBX-000001", "buyer-1", _address(), [_line("CNY"), _line("USD")])

    def test_reject_empty_lines(self):
        with pytest.raises(ValueError, match="订单行"):
            Order.place("GBX-000001", "buyer-1", _address(), [])
