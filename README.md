# PalmMind RAG Backend

FastAPI backend with two APIs: **document ingestion** (PDF/TXT → chunking → embeddings → Qdrant, metadata in PostgreSQL) and a **conversational RAG chat** with Redis memory and LLM-driven interview booking.

## Tech stack

| Concern                       | Choice                                                  |
| ----------------------------- | ------------------------------------------------------- |
| API                           | FastAPI (async)                                         |
| Vector DB                     | Qdrant                                                  |
| Metadata and bookings         | PostgreSQL (SQLAlchemy async)                           |
| Chat memory and booking state | Redis                                                   |
| Embeddings                    | fastembed, `BAAI/bge-small-en-v1.5` (local, no API key) |
| LLM                           | Google Gemini (behind an `LLMService` protocol)         |

No FAISS, no Chroma, no `RetrievalQAChain`, no UI.

## Setup

1. Start the services:
   ```bash
   docker compose up -d
   ```
   PostgreSQL is exposed on host port **5433** to avoid clashing with a local install.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Configure the environment:
   ```bash
   cp .env.example .env             # Windows: copy .env.example .env
   ```
   Set `GEMINI_API_KEY` (free key from Google AI Studio). Model names are retired often, so if the default model (`gemini-3.5-flash`) stops working, set `GEMINI_MODEL` to a current one.
4. Run:
   ```bash
   uvicorn app.main:app --reload
   ```
5. Open http://127.0.0.1:8000/docs for the interactive API.

The first start downloads the embedding model (~70 MB).

## API

### `POST /documents/upload`

Multipart form: `file` (.pdf or .txt) and `strategy` (`fixed` or `sentence`).
Returns the document id, filename, strategy, chunk count and timestamp. Try it with the included `sample.txt`.

### `POST /chat`

```json
{
  "session_id": "demo",
  "message": "How many days of annual leave do employees get?"
}
```

Returns the answer plus the source chunks used. Follow-ups such as "What about the AcmeLift 80?" work because history is kept in Redis.

### Interview booking

Say "I want to book an interview" in `/chat`. The bot collects name, email, date and time over several messages, validates them, and stores the booking in PostgreSQL (`bookings` table). Say "cancel" to stop.

### `DELETE /chat/{session_id}`

Clears a session's chat history.

## Tests

```bash
python -m pytest -q
python -m mypy app
```

Unit tests cover chunking and booking validation. The LLM-dependent flows were tested manually; a fake `LLMService` would be the next step for automated tests.

## Known limitations

- Tables are created at startup; a real deployment would use Alembic migrations.
- Chunk size and top-k are fixed in code rather than configurable per request.
- During a booking, unrelated questions are not answered until the booking finishes or the user says "cancel".
- No authentication or rate limiting.
- Scanned PDFs (images without text) are not supported, since there is no OCR.
