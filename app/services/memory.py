import json
from typing import Literal, TypedDict, cast

from redis.asyncio import Redis

from app.core.config import settings


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatMemory:
    """Stores each session's messages as a Redis list, with a size cap and expiry."""

    def __init__(self, ttl_seconds: int = 3600, max_messages: int = 20) -> None:
        self._redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._max = max_messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}"

    async def add_message(
        self, session_id: str, role: Literal["user", "assistant"], content: str
    ) -> None:
        key = self._key(session_id)
        await self._redis.rpush(key, json.dumps({"role": role, "content": content}))
        await self._redis.ltrim(key, -self._max, -1)
        await self._redis.expire(key, self._ttl)

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        raw = await self._redis.lrange(self._key(session_id), 0, -1)
        return [cast(ChatMessage, json.loads(item)) for item in raw]

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))