from app.services.chunking import ChunkingStrategy, get_chunker

text = "Cats sleep a lot. Dogs bark loudly! Do birds fly? Yes, they do. " * 30
for strategy in ChunkingStrategy:
    chunks = get_chunker(strategy).chunk(text)
    print(strategy.value, len(chunks), "chunks; first:", chunks[0][:60])

print("FIXED ends with:   ", repr(get_chunker(ChunkingStrategy.FIXED).chunk(text)[0][-25:]))
print("SENTENCE ends with:", repr(get_chunker(ChunkingStrategy.SENTENCE).chunk(text)[0][-25:]))