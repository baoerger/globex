# -*- coding: utf-8 -*-
"""OrderRepository 端口

Domain 不关心实现，Infrastructure 提供基于内存（后续可换 PG）的具体仓储。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.order.order import Order


class OrderRepository(ABC):
    @abstractmethod
    async def save(self, order: Order) -> None:
        ...

    @abstractmethod
    async def find_by_id(self, order_id: str) -> Optional[Order]:
        ...

    @abstractmethod
    async def next_order_id(self) -> str:
        ...
