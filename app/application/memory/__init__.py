# -*- coding: utf-8 -*-
"""memory —— 长期记忆的应用层能力（选取策略）。

`PreferenceStore` 端口只管持久化（append / list_by_buyer / delete）。
「按相关性挑要注入哪几条偏好」不放在端口上：那会迫使每个持久化实现
（JSON 文件 / SQLite / 未来的 PG）都依赖 embedding 模型，把模型依赖倒灌进
持久化层，破坏分层。相关性排序是应用层策略，故独立成 `PreferenceSelector`。
"""
