# -*- coding: utf-8 -*-
"""remember_preference_tool

长期记忆写路径：MainAgent 在对话中发现买家的稳定偏好（材质忌口、风格取向、
预算习惯等）时调用，跨会话持久化；读路径由 orchestrator 每轮注入 system-hint。

买家身份从 ShoppingContext 取真实值，不信任模型入参。

注意：本模块不能用 `from __future__ import annotations`（AgentScope schema 生成依赖运行时注解）。
"""
from typing import Literal

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.domain.buyer.preference import BuyerPreference, PreferenceStore
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_remember_preference_tool(store: PreferenceStore, bus: TradeEventBus):
    async def remember_preference_tool(
        kind: Literal["like", "dislike"],
        statement: str,
    ) -> ToolChunk:
        """记住买家的一条长期偏好（跨会话生效）。仅在买家表达出稳定偏好时调用，
        一次性的临时要求（如"这次要军绿色"）不要记。

        Args:
            kind (`str`):
                "like"（正向偏好，如"喜欢小众设计"）或 "dislike"（忌口/黑名单，如"不要塑料材质"）。
            statement (`str`):
                一句话偏好陈述，10 字以内最佳，如"不要塑料材质"。
        """
        snapshot = ShoppingContext.current()
        buyer_id = snapshot.buyer_id if snapshot else "anonymous"
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "remember_preference_tool", "args": {"kind": kind, "statement": statement}},
        )
        try:
            await store.append(BuyerPreference(buyer_id=buyer_id, kind=kind, statement=statement))
        except ValueError as err:
            bus.publish(session_id, "tool.result", {"tool": "remember_preference_tool", "error": str(err)})
            return ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {err}")],
                state=ToolResultState.ERROR,
            )
        bus.publish(session_id, "tool.result", {"tool": "remember_preference_tool", "saved": statement})
        return ToolChunk(
            content=[TextBlock(type="text", text=f"已记住买家偏好：[{kind}] {statement}")],
            state=ToolResultState.SUCCESS,
        )

    return remember_preference_tool
