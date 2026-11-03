import asyncio
from dataclasses import dataclass
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.memory import ChatMemory, ChatMessage
from app.services.vector_store import VectorStore

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. If the context does not contain the answer, say: "
    "\"I don't have that information in the uploaded documents.\" "
    "Do not make up facts. Keep answers concise."
)

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a single standalone question, using the "
    "conversation history to resolve references like 'it', 'that' or 'what about X'. "
    "If it is already standalone, return it unchanged. "
    "Output only the question, nothing else."
)


@dataclass(frozen=True)
class RAGResult:
    answer: str
    sources: list[dict[str, Any]]


def format_history(history: list[ChatMessage]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


class RAGService:
    """Custom retrieval-augmented generation: no LangChain chains involved."""

    def __init__(
        self,
        llm: LLMService,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        memory: ChatMemory,
        top_k: int = 4,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._vector_store = vector_store
        self._memory = memory
        self._top_k = top_k

    async def _rewrite_question(self, history: list[ChatMessage], question: str) -> str:
        if not history:
            return question
        prompt = (
            f"Conversation history:\n{format_history(history)}\n\n"
            f"Latest message: {question}\n\nStandalone question:"
        )
        rewritten = await self._llm.generate(prompt, system=REWRITE_SYSTEM_PROMPT)
        return rewritten.strip() or question

    @staticmethod
    def _build_prompt(
        history: list[ChatMessage], chunks: list[dict[str, Any]], question: str
    ) -> str:
        context = "\n\n".join(
            f"[{i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks)
        )
        return (
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n{format_history(history) or '(none)'}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    async def answer(self, session_id: str, question: str) -> RAGResult:
        history = await self._memory.get_history(session_id)
        standalone = await self._rewrite_question(history, question)

        vector = await asyncio.to_thread(self._embedder.embed_query, standalone)
        chunks = await self._vector_store.search(vector, limit=self._top_k)

        prompt = self._build_prompt(history, chunks, question)
        answer = await self._llm.generate(prompt, system=ANSWER_SYSTEM_PROMPT)

        await self._memory.add_message(session_id, "user", question)
        await self._memory.add_message(session_id, "assistant", answer)
        return RAGResult(answer=answer, sources=chunks)