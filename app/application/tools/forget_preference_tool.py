# -*- coding: utf-8 -*-
"""forget_preference_tool

长期记忆撤回路径：买家明确表示某条历史偏好不再适用（"以后不用避开塑料了"）时，
MainAgent 调它把该偏好从 Store 删掉，下一轮起不再注入。

与 remember_preference_tool 对称的两条纪律：
    1. buyer_id 从 ShoppingContext 取真实值，不信任模型入参；
    2. 只做精确 statement 匹配。删偏好不可逆，模糊匹配会误删
       （"不要塑料" 和 "不要塑料包装" 很像），未命中就把现存偏好回给模型让它用原文重试。

注意：本模块不能用 `from __future__ import annotations`（AgentScope schema 生成依赖运行时注解）。
"""
from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.domain.buyer.preference import PreferenceStore
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_forget_preference_tool(store: PreferenceStore, bus: TradeEventBus):
    async def forget_preference_tool(statement: str) -> ToolChunk:
        """删除买家的一条长期偏好（撤回后不再影响后续推荐）。

        仅在买家明确表示某条历史偏好不再适用时调用，例如"以后不用避开塑料了"。
        本轮的一次性例外（如"这次可以接受塑料"）不要调用，那属于临时要求。

        Args:
            statement (`str`):
                要删除的偏好原文，必须与 <buyer-preferences> 里那一行的文字**完全一致**，
                如"不要塑料材质"。写错不会误删，工具会把现存偏好列出来供你重试。
        """
        snapshot = ShoppingContext.current()
        buyer_id = snapshot.buyer_id if snapshot else "anonymous"
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "forget_preference_tool", "args": {"statement": statement}},
        )

        try:
            deleted = await store.delete(buyer_id, statement)
        except Exception as err:  # noqa: BLE001 —— 记忆写失败如实回报，不假装成功
            bus.publish(session_id, "tool.result", {"tool": "forget_preference_tool", "error": str(err)})
            return ToolChunk(
                content=[TextBlock(type="text", text=f"[error] 撤回偏好失败：{err}")],
                state=ToolResultState.ERROR,
            )

        if deleted:
            bus.publish(session_id, "tool.result", {"tool": "forget_preference_tool", "deleted": statement})
            return ToolChunk(
                content=[TextBlock(type="text", text=f"已撤回买家偏好：{statement}")],
                state=ToolResultState.SUCCESS,
            )

        # 未命中不算错误：把现存偏好回给模型，让它用原文重试，而不是让它以为删成功了
        remaining = await store.list_by_buyer(buyer_id)
        listing = (
            "\n".join(f"- [{p.kind}] {p.statement}" for p in remaining)
            if remaining
            else "（该买家当前没有任何长期偏好）"
        )
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "forget_preference_tool", "not_found": statement, "remaining": len(remaining)},
        )
        return ToolChunk(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"未找到偏好「{statement}」，没有删除任何内容。"
                        f"现存偏好如下，如需撤回请用其中一行的原文重试：\n{listing}"
                    ),
                ),
            ],
            state=ToolResultState.SUCCESS,
        )

    return forget_preference_tool
