# -*- coding: utf-8 -*-
"""eval —— 离线召回评测脚手架（13-1 / 13-2 章）

评测是离线工具，不属于运行时应用，因此放在 `scripts/eval/` 而非 `app/`：

    metrics.py               Recall@K / MRR / NDCG@K 纯函数 + 聚合 + 发版门禁
    validate_datasets.py     标注集自检（拿去评测前先跑这个）
    run_product_recall.py    商品检索（product_search）召回评测
    run_category_recall.py   品类知识库（CategoryInsight）召回评测

两个跑测脚本共用 metrics，直连 UseCase / KnowledgeBase，不过 HTTP、不过 Agent——
召回评测的定位是模块级「日常体检」，必须快且便宜，才可能常驻 CI。
端到端质量由 `scripts/eval_regression.py` 的 Rubric 评测负责，两者互补。
"""
