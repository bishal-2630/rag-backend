from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]