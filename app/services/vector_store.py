import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings


class VectorStore:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim, distance=Distance.COSINE
                ),
            )

    async def add_chunks(
        self, document_id: str, chunks: list[str], vectors: list[list[float]]
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"document_id": document_id, "chunk_index": i, "text": chunk},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(self, vector: list[float], limit: int = 4) -> list[dict[str, Any]]:
        result = await self._client.query_points(
            collection_name=self._collection, query=vector, limit=limit
        )
        return [
            {"score": p.score, **(p.payload or {})} for p in result.points
        ]