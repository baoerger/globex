# Globex - 跨境电商 Agent（AgentScope 2.0）

基于 AgentScope 2.0 的跨境电商超级搜索框 Agent 系统，DDD 洋葱架构落地：

- **MainAgent**（CommerceConcierge）：超级框总调度，**持有全部业务工具可直接单干**；
  内置 Task 计划四件套管理任务清单；满足"可并行 / 上下文隔离 / 链深"任一条件时经 `task_dispatch` 派发子 Agent；
  发现稳定偏好时经 `remember_preference_tool` 写入长期记忆
- **SearchAgent**（CatalogSearchAgent）：商品检索专家，query 改写 → **embedding+rerank 二阶段召回**（Qdrant），
    失败逐级降级（embedding_only → keyword_2gram）；可选 web_search 兜底跨境政策/关税问答
- **TradeAgent**（OrderTradeAgent）：下单交易专家（订单创建 / 查询 / 取消，买家身份由 ShoppingContext 注入）

分期设计脉络、关键取舍与踩坑记录见 [docs/设计演进记录.md](docs/设计演进记录.md)。

## 技术栈

- Python 3.11 + uv
- AgentScope 2.x（Agent + ContextConfig 上下文压缩 + Toolkit/FunctionTool + 内置 Task 计划工具
  + reply_stream 类型化事件流 + TracingMiddleware / ReplyBudgetControlMiddleware / 自定义工具中间件）
- 检索：OpenAI 兼容 embedding（text-embedding-v4）+ Qdrant（服务端/本地嵌入双形态）+ HTTP Reranker（可降级）
- 知识库：AgentScope `rag.KnowledgeBase`（品类洞察 Markdown → 切片 → Qdrant）
- FastAPI + Uvicorn + WebSocket；React 18 + Vite + TS 前端；Docker Compose（app + worker + qdrant + redis + frontend）
- 持久化：SQLite（SQLAlchemy 2.0 async）存对话流水/事件轨迹/会话状态/订单/偏好；商品目录仍为内存仓储 + 种子数据
- 缓存与削峰：Redis（可选）——语义缓存 + embedding 缓存 + 幂等键 + Stream 任务队列 + 跨进程事件背板

## 架构

```text
app/
├── domain/            # 领域层：Product/Sku/Money、Order 状态机、汇率表、关税运费规则、偏好、会话/队列/仓储端口
├── application/
│   ├── usecases/      # CatalogSearch（二阶段召回+到手价内联）、PlaceOrder/QueryOrder/CancelOrder
│   ├── tools/         # product_search、订单三工具、web_search、remember_preference、task_dispatch
│   ├── agents/        # MainAgent / SearchAgent / TradeAgent 工厂 + Orchestrator + SessionRegistry
│   └── prompts/       # globex.yml：主 / 子 Agent 系统提示词
├── infrastructure/    # llm/embedding/qdrant/reranker/tracing、rag 知识库、缓存、队列、韧性与闸门、仓储
├── presentation/      # FastAPI 路由、WebSocket ConnectionManager、DTO
├── composition.py     # 装配容器（API 与 worker 共用一份接线）
└── worker.py          # 意图消费进程入口
knowledge/             # 品类洞察知识文档（Markdown，服务启动时幂等入库）
frontend/              # React + Vite 前端：对话流 + 商品卡 + 事件时间线
eval/                  # 评测用例集 cases.yaml + 回归报告
docs/                  # 设计演进记录（分期取舍与踩坑档案）
docker/                # docker-compose.yaml（app + worker + qdrant + redis + frontend）
```

关键设计：

- **网关配额治理**：`GatewayThrottle` 同时限并发（`LLM_MAX_CONCURRENCY`，默认 2）与请求起点间隔
  （`LLM_MIN_INTERVAL_SECONDS`）；流式请求的名额持有到流耗尽才释放；瞬时故障指数退避重试，
  用尽后回退 `LLM_FALLBACK_MODEL` 并发 `model.fallback` 事件（不静默降级）
- **语义缓存**：相似问句（余弦 ≥ `SEMANTIC_CACHE_THRESHOLD`，默认 0.95）直接复用历史回复，
  命中即零模型调用并发 `cache.hit` 事件；**写操作意图（下单/取消）与上下文依赖问句不入缓存**，
  按 buyer 分桶避免跨买家复用
- **存储可替换**：`SessionStore` / `ConversationStore` / `OrderRepository` / `PreferenceStore` 四个端口，
  SQLite（默认）/ JSON 文件两套实现共存，换存储只改 `app/composition.py`
- **异步削峰**：`TaskQueue` 端口 + Redis Stream 实现（消费者组 / ack / pending 重投 / 死信），
  独立 worker 进程消费；`POST /commerce/intents` **同步语义不变**（内部入队+等结果），
  另提供 `/commerce/intents/async` + `/commerce/tasks/{id}`；队列是 at-least-once，靠幂等键防重复下单
- **跨进程事件**：worker 与 API 是两个进程，事件总线接 Redis Pub/Sub 背板后前端仍能收到流式事件；
  广播带 `origin` 标识以跳过自己发的消息（否则事件会回环投递两次）
- **主 Agent 单干优先**：MainAgent 与子 Agent 持有同一批业务工具（`build_tools()` 复用），
  只在"可并行 / 上下文隔离 / 调用链深"时派发
- **二阶段召回**：embed → Qdrant 向量召回 topN → rerank 精排 topK；降级链
  embedding_rerank → embedding_only → keyword_2gram，`recall_strategy` 如实标注；
  价格等硬约束走工具参数结构化过滤（price_max_major），不交给模型
- **过滤可观测**：被 ship_to / 价格上限挡掉的候选以 `filtered_out`（含 reason）回传，
  让模型能区分"库里没有"与"有但不满足约束"，避免把超预算商品答成"没有这个商品"
- **品类洞察 RAG**：`category_insight_tool` 查 `rag.KnowledgeBase`（选购口径、价格区间、避坑点、
  跨境通则），先给判断标准再给商品清单
- **上下文工程**：ContextConfig 定制压缩（trigger_ratio 0.75 / reserve_ratio 0.15 + 工具结果截断），
  摘要落 AgentState.summary 并推送 `context.compressed` 事件；配合 Token 预算中间件收口单轮开销
- **工具韧性**：ToolResilienceMiddleware 分级超时 + 按工具熔断（closed→open→half_open），
  触发时返回 [error] 让模型如实告知，不编造数字
- **真并行**：同一轮内多个 `task_dispatch` 由 2.0 并发批执行（`is_concurrency_safe`），
  `scripts/verify_parallel.py` 用事件时间戳比对并行/串行墙钟耗时
- **到手价内联**：传 ship_to 时商品卡自动内联 landed_price（小计+运费+关税，汇率统一折算），
  比价/运费不单独暴露工具，减少不必要的工具调用轮次
- **长期记忆**：写路径 remember_preference_tool → JSON 文件 Store；读路径 orchestrator
  在偏好变化时注入 `<buyer-preferences>` hint，跨会话、跨重启生效
- **会话持久化**：AgentState 每轮落盘 DATA_DIR/sessions/，服务重启后恢复多轮对话
- **SubAgent as Tool**：2.0 库级无 subagent 原语（官方 Agent Team 在 agentscope.app 平台层），
  用 FunctionTool 包装 `task_dispatch(subagent_type, demands)` 实现同等语义
- **事件流**：reply_stream → token.delta / plan.update；工具自身发布 tool.invoke/tool.result；
  TradeEventBus 按会话路由 WebSocket
- **可观测**：全部 Agent 挂 TracingMiddleware，OTEL_EXPORTER_OTLP_ENDPOINT 配置后导出 OTLP Trace

## 启动

```bash
uv sync
# 敏感配置通过环境变量注入（推荐），不落盘、不入库
export LLM_BASE_URL=<OpenAI 兼容网关地址>
export LLM_API_KEY=<密钥>
export LLM_MODEL=qwen-plus   # 可选，缺省 qwen3-max（限流时自动回退 LLM_FALLBACK_MODEL，缺省 qwen-plus）
uv run uvicorn app.presentation.server:app --port 8000

# 启用队列削峰时（需 REDIS_URL）另起消费进程：
uv run python -m app.worker
```

> 本地开发也可 `cp .env.example .env` 填值兜底（已被 gitignore，勿提交真实密钥）；
> 同名环境变量优先于 .env。

## API 概览

- `POST /commerce/intents` 提交买家自然语言意图（同步返回最终回复）
- `WS   /commerce/events` 订阅会话事件流（连上后先发 `{"shopping_session_id": "..."}`）
- `GET  /commerce/orders/{order_id}` 查询订单
- `POST /commerce/orders/{order_id}/cancel` 取消订单
- `GET  /health` 健康检查

## 验证

```bash
uv run pytest                          # 137 个单测：domain / 召回降级与过滤回传 / 计价规则 / 记忆持久化 / 压缩策略 / 韧性中间件
uv run python scripts/smoke_e2e.py    # 端到端冒烟：WS 订阅 + 提交意图，实时打印事件流
uv run python scripts/verify_parallel.py   # 并行验证：同轮多派 vs 串行的墙钟耗时与事件重叠数对比
uv run python scripts/eval_regression.py   # 评测回归：13 条 case，LLM judge 按 P0/P1/P2 Rubric 打分出报告
```

评测 case 支持 `prior_context` 字段：把跨会话已成立的事实（如上一 case 写入的长期偏好）告知 judge，
否则 judge 只看本会话记录，会把"正确应用历史偏好"误判为"无据添加"。

## Docker 部署

```bash
export LLM_BASE_URL=<网关地址> LLM_API_KEY=<密钥>   # 敏感配置走环境变量，compose 透传
docker compose -f docker/docker-compose.yaml up -d --build   # app + qdrant + frontend
# 前端 http://localhost:5173  后端 http://localhost:8000
```

本地开发不依赖 Docker：QDRANT_URL 置空时自动用 qdrant-client 本地嵌入模式（单进程文件锁，
多实例/生产请用 compose 的 Qdrant 服务端）。
