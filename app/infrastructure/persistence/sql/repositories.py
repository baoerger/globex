# -*- coding: utf-8 -*-
"""关系库持久化实现（SQLAlchemy 2.0 async）

当前只验证与交付 sqlite+aiosqlite（零外部依赖，开箱即用）。
要换 MySQL / PostgreSQL：装上对应异步驱动（aiomysql / asyncpg）并把 DATABASE_URL
改成该驱动即可，仓储代码不需要改；但本仓未验证过那些驱动的特有行为。

实现四个领域端口：SessionStore / ConversationStore / OrderRepository / PreferenceStore。
domain 与 application 不感知本模块的存在，替换存储只改组装根。

并发安全要点：
    - 订单保存用 merge 覆盖写（订单号唯一，状态机由 domain 保证合法迁移）
    - 偏好去重靠唯一约束，重复插入吞掉 IntegrityError（比先查后插更可靠）
    - turn_index 按会话取当前最大值 +1，同会话并发写有极小概率撞号，
      撞号只影响展示顺序不影响数据完整性，故不加分布式锁

SQLite 的边界（重要）：单写者模型。模块三的 worker 是独立进程，与 API 进程并发写
同一个 db 文件时可能碰到 "database is locked"；WAL 模式能缓解，高并发仍应换服务型数据库。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.domain.buyer.preference import BuyerPreference, PreferenceStore
from app.domain.catalog.money import Money
from app.domain.order.address import Address
from app.domain.order.order import Order, OrderStatus
from app.domain.order.order_line import OrderLine
from app.domain.order.ports.order_repository import OrderRepository
from app.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from app.domain.session.ports.session_store import SessionStore
from app.infrastructure.persistence.sql.tables import (
    AgentSessionStateRow,
    Base,
    BuyerPreferenceRow,
    ConversationEventRow,
    ConversationMessageRow,
    ConversationSessionRow,
    OrderLineRow,
    OrderRow,
)

logger = logging.getLogger(__name__)


def create_engine(database_url: str) -> AsyncEngine:
    """创建异步引擎。连接池参数必须按驱动分开给。

    SQLite：不能传 pool_size / max_overflow（对其默认池无意义），pool_recycle 也无处可用
    （本地文件连接不会被服务端回收）。开 WAL 让读写不互斥，缓解 worker 与 API
    双进程并发写时的 "database is locked"。
    服务型数据库：必需 pool_pre_ping，否则空闲连接被服务端回收后首次查询必报断连。
    """
    if database_url.startswith("sqlite"):
        engine = create_async_engine(database_url, echo=False)

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_wal(dbapi_conn, _record):  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")  # 锁竞争时等待而不是立即报错
            cursor.close()

        return engine
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


async def bootstrap_schema(engine: AsyncEngine) -> None:
    """幂等建表。生产环境应改用 Alembic 迁移。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表结构已就绪（%s）", engine.url.get_backend_name())


class SqlSessionStore(SessionStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def save(self, session_id: str, state_json: str) -> None:
        async with self._session_factory() as db:
            await db.merge(AgentSessionStateRow(session_id=session_id, state_json=state_json))
            await db.commit()

    async def load(self, session_id: str) -> Optional[str]:
        async with self._session_factory() as db:
            row = await db.get(AgentSessionStateRow, session_id)
            return row.state_json if row else None


class SqlConversationStore(ConversationStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def touch_session(self, session_id: str, buyer_id: str, locale: str, currency: str) -> None:
        async with self._session_factory() as db:
            existing = await db.get(ConversationSessionRow, session_id)
            if existing is None:
                db.add(
                    ConversationSessionRow(
                        session_id=session_id, buyer_id=buyer_id, locale=locale, currency=currency,
                    ),
                )
            else:
                existing.last_active_at = datetime.now(timezone.utc)
            await db.commit()

    async def append_turn(self, turn: ConversationTurn) -> None:
        async with self._session_factory() as db:
            max_index = await db.scalar(
                select(func.max(ConversationMessageRow.turn_index)).where(
                    ConversationMessageRow.session_id == turn.session_id,
                ),
            )
            db.add(
                ConversationMessageRow(
                    session_id=turn.session_id,
                    turn_index=(max_index or 0) + 1,
                    buyer_id=turn.buyer_id,
                    role=turn.role,
                    content=turn.content,
                    model=turn.model,
                    latency_ms=turn.latency_ms,
                ),
            )
            await db.commit()

    async def append_events(self, events: list[ConversationEventRecord]) -> None:
        if not events:
            return
        async with self._session_factory() as db:
            db.add_all(
                [
                    ConversationEventRow(
                        session_id=event.session_id,
                        type=event.type,
                        payload=event.payload,
                        occurred_at=event.occurred_at,
                    )
                    for event in events
                ],
            )
            await db.commit()

    async def list_turns(self, session_id: str, limit: int = 50) -> list[ConversationTurn]:
        async with self._session_factory() as db:
            rows = (
                await db.scalars(
                    select(ConversationMessageRow)
                    .where(ConversationMessageRow.session_id == session_id)
                    .order_by(ConversationMessageRow.turn_index)
                    .limit(limit),
                )
            ).all()
        return [
            ConversationTurn(
                session_id=row.session_id,
                buyer_id=row.buyer_id,
                role=row.role,
                content=row.content,
                model=row.model,
                latency_ms=row.latency_ms,
                created_at=row.created_at.isoformat() if row.created_at else "",
            )
            for row in rows
        ]

    async def find_session(self, session_id: str) -> Optional[dict]:
        async with self._session_factory() as db:
            row = await db.get(ConversationSessionRow, session_id)
            if row is None:
                return None
            return {
                "session_id": row.session_id,
                "buyer_id": row.buyer_id,
                "locale": row.locale,
                "currency": row.currency,
                "last_active_at": row.last_active_at.isoformat() if row.last_active_at else "",
            }


class SqlOrderRepository(OrderRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def save(self, order: Order) -> None:
        total = order.total_amount()
        async with self._session_factory() as db:
            await db.merge(
                OrderRow(
                    order_id=order.order_id,
                    buyer_id=order.buyer_id,
                    status=order.status.value,
                    currency=total.currency,
                    total_amount_minor=total.amount_in_minor_units,
                    shipping_address_json=_address_to_dict(order.shipping_address),
                    created_at=order.created_at,
                    confirmed_at=order.confirmed_at,
                    cancelled_at=order.cancelled_at,
                    cancel_reason=order.cancel_reason,
                ),
            )
            # 订单行整体重写：行数固定且量小，比逐行 diff 更简单可靠
            await db.execute(delete(OrderLineRow).where(OrderLineRow.order_id == order.order_id))
            db.add_all(
                [
                    OrderLineRow(
                        order_id=order.order_id,
                        product_id=line.product_id,
                        sku_id=line.sku_id,
                        title=line.title,
                        unit_price_minor=line.unit_price.amount_in_minor_units,
                        currency=line.unit_price.currency,
                        quantity=line.quantity,
                    )
                    for line in order.lines
                ],
            )
            await db.commit()

    async def find_by_id(self, order_id: str) -> Optional[Order]:
        async with self._session_factory() as db:
            row = await db.get(OrderRow, order_id)
            if row is None:
                return None
            line_rows = (
                await db.scalars(select(OrderLineRow).where(OrderLineRow.order_id == order_id))
            ).all()
        return _row_to_order(row, line_rows)

    async def next_order_id(self) -> str:
        """按已有订单数递增。生产应改用独立序列或雪花 ID，避免并发撞号。"""
        async with self._session_factory() as db:
            count = await db.scalar(select(func.count()).select_from(OrderRow))
        return f"GBX-{(count or 0) + 1:06d}"


class SqlPreferenceStore(PreferenceStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def append(self, preference: BuyerPreference) -> None:
        async with self._session_factory() as db:
            db.add(
                BuyerPreferenceRow(
                    buyer_id=preference.buyer_id,
                    kind=preference.kind,
                    statement=preference.statement,
                    created_at=preference.created_at,
                ),
            )
            try:
                await db.commit()
            except IntegrityError:
                # 唯一约束命中 = 该偏好已存在，幂等语义下静默跳过
                await db.rollback()

    async def list_by_buyer(self, buyer_id: str) -> list[BuyerPreference]:
        async with self._session_factory() as db:
            rows = (
                await db.scalars(
                    select(BuyerPreferenceRow)
                    .where(BuyerPreferenceRow.buyer_id == buyer_id)
                    .order_by(BuyerPreferenceRow.id),
                )
            ).all()
        return [
            BuyerPreference(
                buyer_id=row.buyer_id,
                kind=row.kind,
                statement=row.statement,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def delete(self, buyer_id: str, statement: str) -> bool:
        """精确匹配 statement 删除；返回是否真的删到了行。"""
        async with self._session_factory() as db:
            result = await db.execute(
                delete(BuyerPreferenceRow).where(
                    BuyerPreferenceRow.buyer_id == buyer_id,
                    BuyerPreferenceRow.statement == statement,
                ),
            )
            await db.commit()
        return bool(result.rowcount)


# ---- 领域对象 <-> 行记录转换 ----


def _address_to_dict(address: Address) -> dict:
    return {
        "recipient_name": address.recipient_name,
        "country": address.country,
        "state": address.state,
        "city": address.city,
        "address_line": address.address_line,
        "postal_code": address.postal_code,
        "phone": address.phone,
    }


def _row_to_order(row: OrderRow, line_rows: list[OrderLineRow]) -> Order:
    order = Order(
        order_id=row.order_id,
        buyer_id=row.buyer_id,
        shipping_address=Address(**row.shipping_address_json),
        lines=[
            OrderLine(
                product_id=line.product_id,
                sku_id=line.sku_id,
                title=line.title,
                unit_price=Money(amount_in_minor_units=line.unit_price_minor, currency=line.currency),
                quantity=line.quantity,
            )
            for line in line_rows
        ],
        status=OrderStatus(row.status),
        created_at=row.created_at,
        confirmed_at=row.confirmed_at,
        cancelled_at=row.cancelled_at,
        cancel_reason=row.cancel_reason,
    )
    return order
