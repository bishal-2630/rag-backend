from app.services.embeddings import EmbeddingService
from app.services.llm import GeminiLLM, LLMService
from app.services.memory import ChatMemory
from app.services.vector_store import VectorStore
from app.services.booking import BookingService

embedder = EmbeddingService()
vector_store = VectorStore()
memory = ChatMemory()
llm: LLMService = GeminiLLM()
booking_service = BookingService(llm)