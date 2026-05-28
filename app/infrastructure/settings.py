# -*- coding: utf-8 -*-
"""settings

从 .env / 环境变量读取全部配置，Infrastructure 之外不允许直接触碰 os.environ。

二期增量：embedding / Qdrant / Reranker / Tavily / OTLP / 数据目录。
三期增量：品类知识库 collection、Context 工程（压缩阈值/结果截断/Token 预算）、工具超时与熔断、CORS。
四期增量：模型回退与网关配额闸门（并发上限/请求间隔/重试次数）。
可选能力全部按"空值即关闭/降级"设计，保证零外部依赖也能启动。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（globex-agent/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    port: int
    log_level: str
    # ---- 检索升级（模块一）----
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dim: int  # 知识库建库需显式维度（text-embedding-v4 实测 1024）
    qdrant_url: str  # 空 = qdrant-client 本地嵌入模式（DATA_DIR/qdrant）
    qdrant_collection: str
    reranker_base_url: str  # 空 = 降级为按向量分排序
    reranker_model: str
    tavily_api_key: str  # 空 = 不注册 web_search_tool
    # ---- 可观测（模块四）----
    otlp_endpoint: str  # 空 = 不启用 TracingMiddleware
    # ---- 数据目录（模块三）----
    data_dir: Path
    # ---- 三期：品类知识库 ----
    category_kb_collection: str
    # ---- 三期：Context 工程 ----
    context_size: int  # 模型上下文窗口，压缩阈值按此比例计算
    tool_result_limit: int  # 单个工具结果字符上限（商品卡 JSON 较大，需比默认收紧）
    reply_token_budget: int  # 0 = 不启用 Token 预算护栏
    # ---- 三期：工具韧性 ----
    tool_failure_threshold: int  # 连续失败达阈值后熔断
    tool_circuit_reset_seconds: float  # 熔断后多久转半开探测
    # ---- 三期：前端 ----
    cors_origins: list[str]
    # ---- 四期：模型回退与网关配额闸门 ----
    # 这组给默认值：前面几期每次扩字段都会打断测试里手工构造的 Settings，
    # 新增可选配置一律带默认值，避免同样的修改成本反复发生。
    llm_fallback_model: str = ""  # 空 = 不回退，重试用尽直接报错
    llm_max_concurrency: int = 2  # 同时在飞的模型请求上限
    llm_min_interval_seconds: float = 1.0  # 相邻请求起跑最小间隔，治速率爬升过快
    llm_max_retries: int = 2  # 瞬时故障重试次数（指数退避）
    # ---- 四期：存储 ----
    # 默认 SQLite（零外部依赖，落在 DATA_DIR/globex.db）。
    # 换服务型数据库需自行装异步驱动（aiomysql / asyncpg）并改此 URL，本仓未验证。
    # 特殊值 "file" = 退回三期的 JSON 文件存储（无数据库）
    database_url: str = ""
    # ---- 四期：Redis 缓存 ----
    redis_url: str = ""  # 空 = 全部缓存能力关闭（零外部依赖）
    semantic_cache_enabled: bool = True  # Redis 可用时是否开启语义缓存
    semantic_cache_threshold: float = 0.95  # 余弦相似度阈值，调低会提高答非所问风险
    # ---- 四期：队列削峰 ----
    queue_enabled: bool = True  # 需同时配上 REDIS_URL 才生效；否则意图在 API 进程内直跑
    queue_wait_seconds: float = 300.0  # 同步接口等待队列结果的上限
    worker_concurrency: int = 2  # 单个 worker 同时处理的任务数
    # ---- 五期：运行时护栏（Harness / 安全 / 预算）----
    # 默认值取向：零成本的纯本地护栏默认开，
    # 会额外调模型或改变模型选择的一律默认关，必须是显式开启的选择。
    harness_enabled: bool = True  # 单步断言 + 循环检测 + L3 内容过滤
    loop_repeat_threshold: int = 3  # 同一工具连续调用达此数即注入收敛提示
    output_guard_enabled: bool = True  # L4 输出审核（纯正则）
    drift_detect_enabled: bool = False  # 需额外轻量 LLM 调用，默认关
    token_budget_total: int = 0  # 0 = 不启用请求级 Token 预算与四档降级
    breaker_shared: bool = False  # 熔断状态跨实例共享（需 REDIS_URL）
    # 长期记忆：偏好注入策略
    preference_relevance_enabled: bool = False  # 向量相关性筛选，每轮多一次 embedding，默认关
    preference_top_k: int = 5  # like 注入上限；dislike（黑名单）不受此限
    preference_subagent_inject: bool = True  # 给检索子 Agent 注入偏好（纯本地拼装，零成本）
    queue_priority_enabled: bool = True  # 双队列优先级（无 Redis 时自动无效）
    queue_large_request_turns: int = 30  # 对话轮数 >= 此值走大请求队列


def load_settings() -> Settings:
    llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_api_key = os.getenv("LLM_API_KEY", "")
    if not llm_api_key:
        raise RuntimeError(
            "未配置 LLM_API_KEY，无法启动。请通过环境变量注入（推荐）："
            "export LLM_API_KEY=<你的密钥>；或在项目根目录创建本地 .env 文件"
            "（参考 .env.example，该文件已被 gitignore，不会入库）。"
        )
    data_dir = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)  # SQLite 默认落在此目录，建库前必须存在
    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=os.getenv("LLM_MODEL", "qwen3-max"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        # embedding 默认复用 LLM 网关（OpenAI 兼容 /v1/embeddings）
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", llm_base_url),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", llm_api_key),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        qdrant_url=os.getenv("QDRANT_URL", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "globex_products"),
        reranker_base_url=os.getenv("RERANKER_BASE_URL", ""),
        reranker_model=os.getenv("RERANKER_MODEL", ""),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        data_dir=data_dir,
        category_kb_collection=os.getenv("CATEGORY_KB_COLLECTION", "globex_category_kb"),
        context_size=int(os.getenv("CONTEXT_SIZE", "128000")),
        tool_result_limit=int(os.getenv("TOOL_RESULT_LIMIT", "20000")),
        reply_token_budget=int(os.getenv("REPLY_TOKEN_BUDGET", "0")),
        tool_failure_threshold=int(os.getenv("TOOL_FAILURE_THRESHOLD", "3")),
        tool_circuit_reset_seconds=float(os.getenv("TOOL_CIRCUIT_RESET_SECONDS", "60")),
        cors_origins=[
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        ],
        # 实测 qwen3.7-plus 配额池极紧（单发一条也可能 429），默认配上备用模型保底，
        # 重试用尽后自动回退并发 model.fallback 事件，不静默降级
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", "qwen-plus"),
        # 默认 2 而不是 1：三期真并行 fork 实测 1.84x 加速，设 1 会把并行收益完全抹掉
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "2")),
        llm_min_interval_seconds=float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "1.0")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        # 兼容早期变量名 MYSQL_URL；两者都没配时默认本地 SQLite
        database_url=(
            os.getenv("DATABASE_URL")
            or os.getenv("MYSQL_URL")
            or f"sqlite+aiosqlite:///{data_dir / 'globex.db'}"
        ),
        redis_url=os.getenv("REDIS_URL", ""),
        semantic_cache_enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "1") not in ("0", "false", "False"),
        semantic_cache_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95")),
        queue_enabled=os.getenv("QUEUE_ENABLED", "1") not in ("0", "false", "False"),
        queue_wait_seconds=float(os.getenv("QUEUE_WAIT_SECONDS", "300")),
        worker_concurrency=int(os.getenv("WORKER_CONCURRENCY", "2")),
        harness_enabled=os.getenv("HARNESS_ENABLED", "1") not in ("0", "false", "False"),
        loop_repeat_threshold=int(os.getenv("LOOP_REPEAT_THRESHOLD", "3")),
        output_guard_enabled=os.getenv("OUTPUT_GUARD_ENABLED", "1") not in ("0", "false", "False"),
        drift_detect_enabled=os.getenv("DRIFT_DETECT_ENABLED", "0") not in ("0", "false", "False"),
        token_budget_total=int(os.getenv("TOKEN_BUDGET_TOTAL", "0")),
        breaker_shared=os.getenv("BREAKER_SHARED", "0") not in ("0", "false", "False"),
        preference_relevance_enabled=os.getenv("PREFERENCE_RELEVANCE_ENABLED", "0")
        not in ("0", "false", "False"),
        preference_top_k=int(os.getenv("PREFERENCE_TOP_K", "5")),
        preference_subagent_inject=os.getenv("PREFERENCE_SUBAGENT_INJECT", "1")
        not in ("0", "false", "False"),
        queue_priority_enabled=os.getenv("QUEUE_PRIORITY_ENABLED", "1") not in ("0", "false", "False"),
        queue_large_request_turns=int(os.getenv("QUEUE_LARGE_REQUEST_TURNS", "30")),
    )
