# -*- coding: utf-8 -*-
"""二期检索链路单测：二阶段召回 / 降级链 / 价格硬约束 / 到手价内联。

embedding 用确定性桩实现（关键词特征轴 + 余弦），向量索引用 Qdrant 本地嵌入模式，
全程不依赖外部服务与 LLM。
"""
import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient, Reranker
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.settings import Settings
from app.infrastructure.vector.index_bootstrap import bootstrap_product_index
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex

# 特征轴词表：文本命中即该维置 1，余弦相似度即可反映关键词重合度
_FEATURE_TERMS = ("露营灯", "登山杖", "毛巾", "睡袋", "行李箱", "耳机", "充电器", "三件套", "背包", "茶具")


class AxisEmbeddingClient(EmbeddingClient):
    """确定性桩：按特征词命中构造向量。"""

    async def embed(self, text: str) -> list[float]:
        return [1.0 if term in text else 0.0 for term in _FEATURE_TERMS]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class BrokenEmbeddingClient(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding 服务不可用")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding 服务不可用")


class ReverseReranker(Reranker):
    """确定性桩：把候选顺序整体反转（分数与原顺序相反）。"""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(i) for i in range(len(documents))]


def _settings(tmp_path) -> Settings:
    base = Settings(
        llm_base_url="", llm_api_key="", llm_model="", port=8000, log_level="info",
        embedding_base_url="", embedding_api_key="", embedding_model="", embedding_dim=8,
        qdrant_url="", qdrant_collection="test_products",
        reranker_base_url="", reranker_model="", tavily_api_key="",
        otlp_endpoint="", data_dir=tmp_path,
        category_kb_collection="test_category_kb",
        context_size=128000, tool_result_limit=20000, reply_token_budget=0,
        tool_failure_threshold=3, tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
    )
    return base


@pytest.fixture()
async def indexed(tmp_path):
    """已建库的（repo, embedder, index）三元组。"""
    repo = InMemoryProductRepository()
    embedder = AxisEmbeddingClient()
    index = QdrantProductIndex(_settings(tmp_path))
    ok = await bootstrap_product_index(repo, embedder, index)
    assert ok, "本地嵌入模式建库应成功"
    yield repo, embedder, index
    await index.close()


class TestTwoStageRecall:
    async def test_embedding_recall_ranks_camping_light_first(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "embedding_only"
        assert result["rerank_applied"] is False
        assert result["hits"][0]["product_id"] == "P1008", "露营灯应排第一"

    async def test_rerank_applied_changes_order(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(
            repo, embedder=embedder, vector_index=index, reranker=ReverseReranker(),
        )
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert result["recall_strategy"] == "embedding_rerank"
        assert result["rerank_applied"] is True
        # 反转桩生效：露营灯不再是第一位
        assert result["hits"][0]["product_id"] != "P1008"

    async def test_degrade_to_keyword_when_embedding_broken(self, indexed):
        repo, _, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=BrokenEmbeddingClient(), vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯 抗造"))
        assert result["recall_strategy"] == "keyword_2gram"
        assert result["hits"], "关键词降级仍应有召回"
        assert result["hits"][0]["product_id"] == "P1008"

    async def test_price_cap_hard_filter(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 登机", price_max_major=500.0),
        )
        # P1002 行李箱 899 CNY 超预算，必须被结构化过滤
        assert all(hit["product_id"] != "P1002" for hit in result["hits"])

    async def test_over_price_cap_candidate_reported_in_filtered_out(self, indexed):
        """超预算候选必须如实回传，否则模型会把"有但超预算"答成"没有这个商品"
        （三期评测 long-context-memory 曾暴露此缺陷）。"""
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="行李箱 登机", price_max_major=500.0),
        )
        rejected = {item["product_id"]: item for item in result["filtered_out"]}
        assert "P1002" in rejected, "被价格上限挡掉的候选必须在 filtered_out 里可见"
        assert rejected["P1002"]["reason"] == "over_price_cap"
        # 价格按目标币种给出，模型才能直接告知买家超了多少
        assert rejected["P1002"]["currency"] == "CNY"
        assert rejected["P1002"]["price_major"] > 500.0

    async def test_unshippable_candidate_reported_in_filtered_out(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="露营灯 抗造", ship_to="BR"),
        )
        reasons = {item["reason"] for item in result["filtered_out"]}
        assert reasons == {"ship_to_unavailable"}, "不可达目的国应标注为 ship_to_unavailable"

    async def test_no_filtered_out_key_without_hard_constraints(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert "filtered_out" not in result, "无硬约束时不应污染工具返回"

    async def test_landed_price_inlined_with_ship_to(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="露营灯", ship_to="US", target_currency="USD"),
        )
        top = result["hits"][0]
        assert "landed_price" in top
        landed = top["landed_price"]
        assert landed["currency"] == "USD"
        assert landed["landed_total_major"] == pytest.approx(
            landed["subtotal_major"] + landed["freight_major"] + landed["tariff_major"],
            abs=0.02,
        )

    async def test_no_landed_price_without_ship_to(self, indexed):
        repo, embedder, index = indexed
        usecase = CatalogSearchUseCase(repo, embedder=embedder, vector_index=index)
        result = await usecase.execute(ProductSearchSpec(normalized_query="露营灯"))
        assert "landed_price" not in result["hits"][0]

    async def test_tool_event_carries_hits_for_frontend(self, indexed):
        """tool.result 事件必须带 hits，否则前端商品卡渲染不出来（三期浏览器验证曾暴露此缺陷）。"""
        from app.application.tools.product_search_tool import build_product_search_tool
        from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
        from app.infrastructure.eventbus import TradeEventBus

        repo, embedder, index = indexed
        bus = TradeEventBus()
        queue = bus.subscribe("s-cards")
        tool = build_product_search_tool(
            CatalogSearchUseCase(repo, embedder=embedder, vector_index=index),
            bus,
        )
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s-cards", buyer_id="b", locale="zh-CN", currency="CNY",
            ),
        )
        try:
            await tool(normalized_query="露营灯", ship_to="US", target_currency="USD")
        finally:
            ShoppingContext.reset(token)

        queue.get_nowait()  # tool.invoke
        result_event = queue.get_nowait()
        assert result_event.type == "tool.result"
        hits = result_event.payload["hits"]
        assert hits and hits[0]["product_id"] == "P1008"
        assert hits[0]["landed_price"]["currency"] == "USD"
