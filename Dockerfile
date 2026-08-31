# syntax=docker/dockerfile:1
# ---- 构建阶段：uv 安装依赖到独立虚拟环境 ----
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 先装依赖（利用层缓存），再拷源码
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app

# ---- 运行阶段：slim 运行时，仅带 venv 与源码 ----
FROM python:3.11-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv
COPY app ./app
# 品类洞察知识库文档（启动时灌入 RAG 向量库）
COPY knowledge ./knowledge

# 运行时数据目录（会话/偏好；向量库走 QDRANT_URL 指向 qdrant 服务）
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 8000
CMD ["uvicorn", "app.presentation.server:app", "--host", "0.0.0.0", "--port", "8000"]
