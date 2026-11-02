from fastembed import TextEmbedding

from app.core.config import settings


class EmbeddingService:
    """Turns text into vectors using a local model (no API key)."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]