# -*- coding: utf-8 -*-
"""三期单测：品类知识库 RAG / Context 策略 / 工具韧性（超时+熔断）。

全部不依赖真实 LLM 与外部服务：embedding 用确定性桩，向量库用 Qdrant 本地嵌入模式。
"""
import asyncio
import json

import pytest
from agentscope.credential import OpenAICredential
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse
from agentscope.message import TextBlock, ToolResultState
from agentscope.rag import KnowledgeBase, QdrantStore
from agentscope.tool import FunctionTool, ToolChunk

from app.application.agents.context_policy import build_context_config
from app.infrastructure.transient import is_transient_error as _is_transient
from app.application.tools.category_insight_tool import build_category_insight_tool
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.rag.category_knowledge import bootstrap_category_knowledge
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)

_TERMS = ("露营灯", "登山杖", "免税额度", "塑料", "茶具", "耳机", "行李箱", "运费")


class TermAxisEmbedding(EmbeddingModelBase):
    """确定性 embedding 桩：按词表命中构造向量，可反映关键词重合度。"""

    def __init__(self) -> None:
        super().__init__(
            credential=OpenAICredential(api_key="sk-test"),
            model="term-axis",
            dimensions=len(_TERMS),
            parameters=None,
            context_size=8192,
            batch_size=16,
            max_retries=0,
            retry_delay=0.0,
        )

    async def __call__(self, inputs, **kwargs) -> EmbeddingResponse:
        texts = inputs if isinstance(inputs, list) else [inputs]
        embeddings = [[1.0 if term in str(t) else 0.0 for term in _TERMS] for t in texts]
        return EmbeddingResponse(embeddings=embeddings)


@pytest.fixture()
async def knowledge_base(tmp_path):
    kb = KnowledgeBase(
        name="category_insight_test",
        description="测试用品类知识库",
        embedding_model=TermAxisEmbedding(),
        vector_store=QdrantStore(path=str(tmp_path / "kb")),
        collection="test_category_kb",
    )
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "outdoor.md").write_text(
        "# 户外\n露营灯看防水等级与续航，登山杖优先钛合金。", encoding="utf-8",
    )
    (docs / "guide.md").write_text(
        "# 通则\n美国免税额度约 800 美元，运费按首件全价加续件折价计。", encoding="utf-8",
    )
    inserted = await bootstrap_category_knowledge(kb, knowledge_dir=docs)
    assert inserted == 2
    yield kb


class TestCategoryKnowledge:
    async def test_bootstrap_is_idempotent(self, knowledge_base, tmp_path):
        # 第二次灌同一目录不应重复插入
        again = await bootstrap_category_knowledge(knowledge_base, knowledge_dir=tmp_path / "knowledge")
        assert again == 0
        assert len(await knowledge_base.list_documents()) == 2

    async def test_insight_tool_returns_relevant_chunk(self, knowledge_base):
        bus = TradeEventBus()
        queue = bus.subscribe("anonymous")
        tool = build_category_insight_tool(knowledge_base, bus)

        response = await tool(question="美国免税额度是多少", top_k=2)
        payload = json.loads(response.content[0].text)
        assert payload["insights"], "应有知识命中"
        assert "免税额度" in payload["insights"][0]["content"]
        assert payload["insights"][0]["source"].endswith(".md")
        assert queue.qsize() == 2  # tool.invoke + tool.result

    async def test_insight_tool_degrades_when_kb_broken(self):
        class BrokenKnowledgeBase:
            async def search(self, *args, **kwargs):
                raise RuntimeError("向量库连接失败")

        bus = TradeEventBus()
        tool = build_category_insight_tool(BrokenKnowledgeBase(), bus)
        response = await tool(question="露营灯怎么挑")
        assert response.state == ToolResultState.ERROR
        assert "品类知识库不可用" in response.content[0].text


class TestContextPolicy:
    def test_thresholds_and_limit(self):
        config = build_context_config(context_size=128000, tool_result_limit=20000)
        assert config.trigger_ratio == 0.75
        assert config.reserve_ratio == 0.15
        assert config.tool_result_limit == 20000

    def test_compression_prompt_pins_critical_facts(self):
        config = build_context_config(context_size=128000, tool_result_limit=20000)
        for keyword in ("product_id", "sku_id", "订单号", "偏好"):
            assert keyword in config.compression_prompt
        # 模板占位符必须与 2.0 summary_schema 字段一致，否则压缩时抛 KeyError
        for field in (
            "task_overview",
            "current_state",
            "important_discoveries",
            "next_steps",
            "context_to_preserve",
        ):
            assert "{" + field + "}" in config.summary_template

    def test_summary_template_renders_with_schema_fields(self):
        """用 schema 字段实际渲染一次，防止占位符写错（二期冒烟曾暂错 KeyError）。"""
        config = build_context_config(context_size=128000, tool_result_limit=20000)
        rendered = config.summary_template.format(
            task_overview="买露营灯",
            current_state="已推荐 P1008",
            important_discoveries="P1008-S1 89 CNY",
            next_steps="等待确认下单",
            context_to_preserve="不要塑料",
        )
        assert "P1008-S1 89 CNY" in rendered


class TestTransientRetryPolicy:
    """上游瞬时故障识别：网关把限流错误写在 SSE 流中间，2.0 模型层重试盖不到，
    靠 orchestrator 这一层按错误特征兜底（三期冒烟实际遇到过）。"""

    def test_gateway_concurrency_error_is_transient(self):
        assert _is_transient(RuntimeError("Too many concurrent requests."))

    def test_rate_limit_variants_are_transient(self):
        for message in (
            "Request rate increased too quickly.",
            "429 Too Many Requests",
            "Service Unavailable",
            "Read timeout",
        ):
            assert _is_transient(RuntimeError(message)), message

    def test_business_errors_are_not_transient(self):
        for message in (
            "商品不存在：P9999",
            "仅 CONFIRMED 态可取消",
            "Invalid API key",
        ):
            assert not _is_transient(RuntimeError(message)), message


def _ok_tool_factory(name: str, delay: float = 0.0, fail: bool = False):
    async def tool_func() -> ToolChunk:
        """测试用工具。"""
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return ToolChunk(
                content=[TextBlock(type="text", text="[error] 下游报错")],
                state=ToolResultState.ERROR,
            )
        return ToolChunk(
            content=[TextBlock(type="text", text="ok")],
            state=ToolResultState.SUCCESS,
        )

    tool_func.__name__ = name
    return tool_func


async def _call(tool: FunctionTool) -> ToolChunk:
    """调工具并抽出最后一个 chunk（ToolBase.__call__ 是协程，
    带中间件时返回异步生成器）。"""
    result = await tool()
    if hasattr(result, "__aiter__"):
        chunks = [chunk async for chunk in result]
        return chunks[-1]
    return result


class TestToolResilience:
    async def test_timeout_returns_error_chunk(self):
        registry = CircuitBreakerRegistry(failure_threshold=3, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry, timeouts={"slow_tool": 0.05})
        tool = FunctionTool(_ok_tool_factory("slow_tool", delay=0.5), middlewares=[middleware])

        result = await _call(tool)
        assert result.state == ToolResultState.ERROR
        assert "超过" in result.content[0].text
        assert registry.status("slow_tool") == "closed"  # 一次失败还没到阈值

    async def test_circuit_opens_after_threshold(self):
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)
        tool = FunctionTool(_ok_tool_factory("flaky_tool", fail=True), middlewares=[middleware])

        assert (await _call(tool)).state == ToolResultState.ERROR
        assert (await _call(tool)).state == ToolResultState.ERROR
        assert registry.status("flaky_tool") == "open"

        # 熔断后短路，返回降级提示而不是再次执行
        short_circuited = await _call(tool)
        assert "已熔断" in short_circuited.content[0].text

    async def test_half_open_probe_recovers(self):
        registry = CircuitBreakerRegistry(failure_threshold=1, reset_seconds=0)
        middleware = ToolResilienceMiddleware(registry)
        failing = FunctionTool(_ok_tool_factory("recover_tool", fail=True), middlewares=[middleware])
        assert (await _call(failing)).state == ToolResultState.ERROR
        assert registry.status("recover_tool") == "open"

        # reset_seconds=0 → 立即可转半开；探测成功后闭合
        healthy = FunctionTool(_ok_tool_factory("recover_tool"), middlewares=[middleware])
        assert (await _call(healthy)).state == ToolResultState.SUCCESS
        assert registry.status("recover_tool") == "closed"

    async def test_success_resets_failure_counter(self):
        registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)
        failing = FunctionTool(_ok_tool_factory("mixed_tool", fail=True), middlewares=[middleware])
        healthy = FunctionTool(_ok_tool_factory("mixed_tool"), middlewares=[middleware])

        await _call(failing)  # 1 次失败
        await _call(healthy)  # 成功清零
        await _call(failing)  # 再 1 次失败，仍未达阈值
        assert registry.status("mixed_tool") == "closed"
