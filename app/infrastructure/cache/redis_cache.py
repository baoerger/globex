# -*- coding: utf-8 -*-
"""RedisCache

Redis 客户端薄封装。全部读写都吞异常并降级为「未命中 / 未写入」——
缓存的作用是省钱省时，绝不能因为 Redis 挂了让主链路一起挂。

REDIS_URL 未配置时构造 disabled 实例，所有操作直接短路，
调用方不需要写 if enabled 分支。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_url: str = "") -> None:
        self._client: Any = None
        self._url = redis_url
        if not redis_url:
            return
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(redis_url, decode_responses=True)
        except Exception as err:  # noqa: BLE001 —— 连不上也要能启动
            logger.warning("Redis 初始化失败，缓存能力关闭：%s", err)
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> Any:
        """原始客户端。仅供需要 Stream / Pub-Sub 等高级命令的模块使用（队列与事件背板）；
        普通缓存读写请走本类方法，那些方法已统一做了异常旁路。
        """
        return self._client

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except Exception:  # noqa: BLE001
            return False

    async def get_json(self, key: str) -> Optional[Any]:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
        except Exception as err:  # noqa: BLE001
            logger.warning("Redis 读失败，按未命中处理：%s（%s）", key, err)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            # 脏值当未命中，顺手删掉
            await self.delete(key)
            return None

    async def get_raw(self, key: str) -> Optional[str]:
        """取裸字符串值。

        幂等键存的是裸 task_id，不能走 get_json（那里 json.loads 失败会把键当脏值删掉，
        直接破坏幂等语义）。
        """
        if self._client is None:
            return None
        try:
            return await self._client.get(key)
        except Exception as err:  # noqa: BLE001
            logger.warning("Redis 读失败，按未命中处理：%s（%s）", key, err)
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if self._client is None:
            return
        try:
            await self._client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)
        except Exception as err:  # noqa: BLE001
            logger.warning("Redis 写失败，跳过缓存：%s（%s）", key, err)

    async def delete(self, key: str) -> None:
        if self._client is None:
            return
        try:
            await self._client.delete(key)
        except Exception:  # noqa: BLE001
            pass

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        """SET NX：抢到返回 True。用于幂等键与分布式锁。

        Redis 不可用时返回 True（视为抢到），否则整条链路会因为拿不到幂等键而卡死——
        宁可退化成"没有幂等保护"，也不能因缓存故障拒绝服务。
        """
        if self._client is None:
            return True
        try:
            return bool(await self._client.set(key, value, ex=ttl_seconds, nx=True))
        except Exception as err:  # noqa: BLE001
            logger.warning("Redis SETNX 失败，放行本次请求：%s（%s）", key, err)
            return True

    async def zadd_and_count(self, key: str, member: str, now: float, window_seconds: float) -> int:
        """滑动窗口计数：清理过期成员后加入本次，返回窗口内当前数量。

        用于跨进程限流（模块三多 worker 时替代进程内信号量）。
        """
        if self._client is None:
            return 0
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, 0, now - window_seconds)
                pipe.zadd(key, {member: now})
                pipe.zcard(key)
                pipe.expire(key, int(window_seconds) + 1)
                results = await pipe.execute()
            return int(results[2])
        except Exception as err:  # noqa: BLE001
            logger.warning("Redis 滑窗计数失败，按不限流处理：%s（%s）", key, err)
            return 0

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.aclose()
        except Exception:  # noqa: BLE001
            pass
