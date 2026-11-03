import asyncio
from typing import Protocol

from google import genai
from google.genai import errors, types

from app.core.config import settings


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM provider keeps failing after retries."""


class LLMService(Protocol):
    """Anything that can turn a prompt into text. Swap providers by
    writing another class with this method."""

    async def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> str: ...


class GeminiLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self._model = model or settings.gemini_model
        self._max_retries = max_retries

    async def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else None,
        )
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
                return response.text or ""
            except errors.APIError as exc:
                retryable = exc.code in (429, 500, 503)
                if not retryable or attempt == self._max_retries:
                    raise LLMUnavailableError(
                        f"LLM request failed with status {exc.code}"
                    ) from exc
                await asyncio.sleep(2**attempt)
        raise LLMUnavailableError("LLM request failed")