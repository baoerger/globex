# -*- coding: utf-8 -*-
"""关系库表结构定义（SQLAlchemy 2.0 声明式）

四期落库范围：对话记录 + 会话状态 + 订单 + 买家偏好。
商品目录不落库（会牵动向量建库与评测事实表构造，留后续）。

建表策略：启动时用 `create_all` 幂等建表。生产环境应换成 Alembic 迁移
（本项目为教学工程，避免引入迁移目录的额外复杂度）。

自增主键用 `BigInteger().with_variant(Integer, "sqlite")`：SQLite 的 AUTOINCREMENT
只能用于 INTEGER PRIMARY KEY，不做 variant 则单测无法用内存库跑真实 SQL。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 自增主键类型：SQLite 降为 INTEGER（其 AUTOINCREMENT 只认 INTEGER PRIMARY KEY），
# 其他驱动用 BIGINT
_AutoPk = BigInteger().with_variant(Integer, "sqlite")
# 金额类（最小货币单位）同理，保证两边都能建表
_BigInt = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


class ConversationSessionRow(Base):
    __tablename__ = "conversation_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    buyer_id: Mapped[str] = mapped_column(String(64), index=True)
    locale: Mapped[str] = mapped_column(String(16), default="zh-CN")
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(),
    )


class ConversationMessageRow(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(_AutoPk, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    buyer_id: Mapped[str] = mapped_column(String(64), default="")
    role: Mapped[str] = mapped_column(String(16))  # buyer / agent
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(64), default="")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_msg_session_turn", "session_id", "turn_index"),)


class ConversationEventRow(Base):
    __tablename__ = "conversation_events"

    id: Mapped[int] = mapped_column(_AutoPk, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)
    occurred_at: Mapped[str] = mapped_column(String(40))


class AgentSessionStateRow(Base):
    """AgentState 全量快照。单会话一行，每轮覆盖写。"""

    __tablename__ = "agent_session_states"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(),
    )


class OrderRow(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    buyer_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(8))
    # 金额存最小货币单位（分），避免浮点误差——与 domain 的 Money 口径一致
    total_amount_minor: Mapped[int] = mapped_column(_BigInt)
    shipping_address_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)


class OrderLineRow(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(_AutoPk, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(32), ForeignKey("orders.order_id"), index=True)
    product_id: Mapped[str] = mapped_column(String(32))
    sku_id: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(255))
    unit_price_minor: Mapped[int] = mapped_column(_BigInt)
    currency: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)


class BuyerPreferenceRow(Base):
    __tablename__ = "buyer_preferences"

    id: Mapped[int] = mapped_column(_AutoPk, primary_key=True, autoincrement=True)
    buyer_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # like / dislike
    statement: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[str] = mapped_column(String(40))

    # 幂等去重从应用层下沉到数据库约束
    __table_args__ = (
        UniqueConstraint("buyer_id", "kind", "statement", name="uq_pref_buyer_kind_statement"),
    )
