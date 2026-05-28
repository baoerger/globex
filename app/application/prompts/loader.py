# -*- coding: utf-8 -*-
"""PromptLoader

读取并缓存 app/application/prompts/globex.yml，全项目提示词只从这里取。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent / "globex.yml"


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
