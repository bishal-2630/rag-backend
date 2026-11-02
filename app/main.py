from fastapi import FastAPI

app = FastAPI(
    title="RAG Backend",
    description="Document ingestion and conversational RAG APIs",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}