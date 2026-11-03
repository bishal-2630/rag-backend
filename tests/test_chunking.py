import pytest

from app.services.chunking import FixedSizeChunker, SentenceChunker


def test_fixed_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(size=100, overlap=100)


def test_fixed_chunker_splits_with_overlap() -> None:
    chunks = FixedSizeChunker(size=50, overlap=10).chunk("a" * 120)
    assert len(chunks) == 3
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_sentence_chunker_keeps_sentences_whole() -> None:
    chunks = SentenceChunker(max_chars=10).chunk("One. Two. Three.")
    assert chunks == ["One. Two.", "Three."]


def test_sentence_chunker_handles_empty_text() -> None:
    assert SentenceChunker().chunk("") == []