import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.models.document import Document
from app.schemas.document import DocumentOut
from app.services.chunking import ChunkingStrategy, get_chunker
from app.services.embeddings import EmbeddingService
from app.services.extraction import UnsupportedFileTypeError, extract_text
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["ingestion"])

embedder = EmbeddingService()
vector_store = VectorStore()


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_session)],
    strategy: Annotated[ChunkingStrategy, Form()] = ChunkingStrategy.SENTENCE,
) -> Document:
    filename = file.filename or "unknown"
    content = await file.read()

    try:
        text = extract_text(filename, content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found")

    chunks = get_chunker(strategy).chunk(text)
    vectors = embedder.embed_documents(chunks)

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        filename=filename,
        file_type=filename.rsplit(".", 1)[-1].lower(),
        chunking_strategy=strategy.value,
        chunk_count=len(chunks),
    )

    await vector_store.add_chunks(doc_id, chunks, vectors)
    session.add(doc)
    await session.commit()
    return doc