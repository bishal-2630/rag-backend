from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ingestion import router as ingestion_router, vector_store
from app.core.database import Base, engine
from app.models import document  # noqa: F401  (registers the table)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await vector_store.ensure_collection()
    yield


app = FastAPI(
    title="RAG Backend",
    description="Document ingestion and conversational RAG APIs",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingestion_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}