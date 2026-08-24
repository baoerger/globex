# -*- coding: utf-8 -*-
"""标注集自检 —— 见教程 13-2 §2.4。

标注集是评测的基准。基准本身错了，后面所有指标都不可信，而且错法很隐蔽：
把一个「会被硬约束挡掉」的商品标成 relevant，不会报错，只会让 Recall 上限
悄悄低于 1，然后你对着一个永远达不到满分的指标反复调检索。

所以标注集在拿去评测前必须先过这一关。跑测脚本不强制依赖它，但改完标注就该跑：

    uv run python scripts/eval/validate_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.infrastructure.persistence.in_memory_repositories import (  # noqa: E402
    InMemoryProductRepository,
)

_PRODUCT_DATASET = Path("eval/product_recall.jsonl")
_CATEGORY_DATASET = Path("eval/category_recall.jsonl")
_KNOWLEDGE_DIR = Path("knowledge")


def _load(path: Path) -> list[tuple[int, dict]]:
    cases = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw.strip():
            cases.append((lineno, json.loads(raw)))
    return cases


def _check_common(lineno: int, case: dict) -> list[str]:
    problems = []
    relevant = case.get("relevant") or []
    if not relevant:
        problems.append(f"L{lineno} [{case.get('query')}] relevant 为空")
    if len(set(relevant)) != len(relevant):
        problems.append(f"L{lineno} [{case.get('query')}] relevant 有重复")
    return problems


async def validate_products() -> list[str]:
    products = {p.product_id: p for p in await InMemoryProductRepository().list_all()}
    problems: list[str] = []

    for lineno, case in _load(_PRODUCT_DATASET):
        query = case["query"]
        problems.extend(_check_common(lineno, case))
        if case.get("kind") not in (None, "lexical", "semantic"):
            problems.append(f"L{lineno} [{query}] kind 取值非法：{case['kind']}")

        for pid in case.get("relevant", []):
            product = products.get(pid)
            if product is None:
                problems.append(f"L{lineno} [{query}] {pid} 不在商品库")
                continue

            # 价格约束：只在币种与约束口径一致时判定，避免在评测里重算汇率
            limit = case.get("price_max_major")
            if limit is not None:
                cheapest = min(sku.price.to_major_units() for sku in product.skus)
                currency = product.skus[0].price.currency
                if currency == "CNY" and cheapest > limit:
                    problems.append(
                        f"L{lineno} [{query}] {pid} 最低价 {cheapest}{currency} > 上限 {limit}"
                        f"（标注自相矛盾：会被硬约束挡掉的商品不该标为 relevant）",
                    )

            ship_to = case.get("ship_to")
            if ship_to and ship_to not in product.ships_to:
                problems.append(
                    f"L{lineno} [{query}] {pid} ships_to={product.ships_to} 不含 {ship_to}"
                    f"（标注自相矛盾）",
                )
    return problems


def validate_categories() -> list[str]:
    docs = {p.name for p in _KNOWLEDGE_DIR.glob("*.md")}
    problems: list[str] = []
    for lineno, case in _load(_CATEGORY_DATASET):
        problems.extend(_check_common(lineno, case))
        for name in case.get("relevant", []):
            if name not in docs:
                problems.append(
                    f"L{lineno} [{case['query']}] 知识文档不存在：{name}"
                    f"（标注单位应为 knowledge/*.md 的文件名）",
                )
    return problems


async def main() -> None:
    product_problems = await validate_products()
    category_problems = validate_categories()

    print(f"{_PRODUCT_DATASET}：{len(_load(_PRODUCT_DATASET))} 条")
    print(f"{_CATEGORY_DATASET}：{len(_load(_CATEGORY_DATASET))} 条")

    problems = product_problems + category_problems
    if problems:
        print(f"\n发现 {len(problems)} 处问题：")
        for message in problems:
            print("  -", message)
        sys.exit(1)
    print("\n标注集自检通过")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
