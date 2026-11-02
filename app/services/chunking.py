import re
from enum import Enum
from typing import Protocol


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


class FixedSizeChunker:
    """Slices text into fixed character windows with overlap."""

    def __init__(self, size: int = 800, overlap: int = 100) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        step = self.size - self.overlap
        chunks = [text[i : i + self.size] for i in range(0, len(text), step)]
        return [c.strip() for c in chunks if c.strip()]


class SentenceChunker:
    """Groups whole sentences until a max size is reached."""

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > self.max_chars:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current.strip())
        return chunks


def get_chunker(strategy: ChunkingStrategy) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedSizeChunker()
    return SentenceChunker()