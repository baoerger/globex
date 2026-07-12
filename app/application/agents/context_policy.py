# -*- coding: utf-8 -*-
"""context_policy

Context 工程策略：把 2.0 内置的上下文压缩配置成跨境购物场景的口径
（即教程 Cache Breakpoint 章节要解决的问题——长对话不爆 token 且关键事实不丢）。

压缩触发：上下文占用达 context_size * trigger_ratio 时，Agent 自动把早期消息
压缩成摘要写入 AgentState.summary，保留末段 reserve_ratio 的原始消息。

关键取舍：摘要提示词显式列出"必须逐字保留"的事实清单（偏好、product_id/sku_id、
订单号与金额、待确认动作），避免压缩后 Agent 忘记已确认的商品或订单。

注意：summary_template 的占位符必须与 2.0 内置 summary_schema 的五个字段一致
（task_overview / current_state / important_discoveries / next_steps / context_to_preserve），
否则压缩时渲染摘要会抛 KeyError。
"""
from __future__ import annotations

from agentscope.agent import ContextConfig

_COMPRESSION_PROMPT = """<system-hint>当前对话上下文即将超出窗口，请把此前的工作压缩成一份中文摘要，
供你后续继续为这位买家服务。当前时间：{current_time}。

必须逐字保留的事实（丢失会导致后续回答出错）：
1. 买家的偏好与硬约束：材质忌口、风格取向、预算上限、收货国家/地址；
2. 已经推荐或买家已认可的商品：完整 product_id、sku_id、标题、价格与币种；
3. 订单相关：订单号、状态、总金额与币种、取消原因；
4. 当前待确认的动作：是否有等待买家确认的确认卡、下一步该做什么。

可以压缩或丢弃的内容：工具返回的完整候选列表（只留最终推荐项）、寒暄、重复表述、
已被更新覆盖的中间结论。

摘要中的所有数字必须来自此前的工具返回，不得重新估算。</system-hint>"""

_SUMMARY_TEMPLATE = """<system-info>以下是你此前为该买家服务的工作摘要，视作事实基准继续服务。
# 买家诉求与约束
{task_overview}

# 当前进展（已推荐商品 / 已创建订单）
{current_state}

# 关键事实（product_id / sku_id / 价格 / 订单号）
{important_discoveries}

# 下一步与待确认事项
{next_steps}

# 买家偏好与必须保留的上下文
{context_to_preserve}</system-info>"""


def build_context_config(context_size: int, tool_result_limit: int) -> ContextConfig:
    """构造 Globex 的上下文压缩策略。

    Args:
        context_size (`int`):
            模型上下文窗口大小（与 create_chat_model 保持一致）。
        tool_result_limit (`int`):
            单个工具结果的字符上限，超出会被截断，防止商品卡 JSON 挤爆上下文。
    """
    del context_size  # 窗口由 model 侧提供，这里仅保留参数以标明配套关系
    return ContextConfig(
        trigger_ratio=0.75,
        reserve_ratio=0.15,
        compression_prompt=_COMPRESSION_PROMPT,
        summary_template=_SUMMARY_TEMPLATE,
        tool_result_limit=tool_result_limit,
    )
