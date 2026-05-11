# -*- coding: utf-8 -*-
"""ProductSearchSpec 值对象

SearchAgent 把买家自然语言 query 改写为标准化检索规格：
normalized_query 用于召回，槽位（category / price_band / ship_to / locale）用于过滤。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProductSearchSpec:
    normalized_query: str
    category: Optional[str] = None
    ship_to: Optional[str] = None
    locale: str = "zh-CN"
    top_k: int = 5
    # 到手价目标币种：命中 ship_to 时商品卡内联 landed_price（小计+运费+关税）
    target_currency: str = "CNY"
    # 价格硬约束（目标币种主单位）：硬约束由检索链路结构化过滤，不交给 embedding/reranker
    price_max_major: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.normalized_query or not self.normalized_query.strip():
            raise ValueError("ProductSearchSpec.normalized_query required")
        if self.top_k <= 0:
            raise ValueError("ProductSearchSpec.top_k 必须为正整数")
