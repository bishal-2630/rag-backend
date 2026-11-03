from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.deps import booking_service, embedder, llm, memory, vector_store
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.booking import BookingSlots
from app.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])

rag_service = RAGService(llm, embedder, vector_store, memory)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ChatResponse:
    state = await booking_service.get_state(request.session_id)
    if state is None and await booking_service.detect_intent(request.message):
        state = BookingSlots()

    if state is not None:
        answer = await booking_service.handle_turn(
            db, request.session_id, request.message, state
        )
        await memory.add_message(request.session_id, "user", request.message)
        await memory.add_message(request.session_id, "assistant", answer)
        return ChatResponse(session_id=request.session_id, answer=answer, sources=[])

    result = await rag_service.answer(request.session_id, request.message)
    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        sources=[SourceChunk(**chunk) for chunk in result.sources],
    )


@router.delete("/{session_id}", status_code=204)
async def clear_session(session_id: str) -> None:
    await memory.clear(session_id)