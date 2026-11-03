from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.ingestion import router as ingestion_router
from app.core.database import Base, engine
from app.core.deps import vector_store
from app.models import booking, document  # noqa: F401  (registers the tables)
from app.services.llm import LLMUnavailableError


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await vector_store.ensure_collection()
    yield


app = FastAPI(
    title="PalmMind RAG Backend",
    description="Document ingestion and conversational RAG APIs",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingestion_router)
app.include_router(chat_router)


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(_: Request, exc: LLMUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}