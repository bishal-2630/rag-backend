import asyncio

from google import genai

from app.core.config import settings


async def try_model(client: genai.Client, model: str) -> None:
    try:
        response = await client.aio.models.generate_content(
            model=model, contents="Say 'working' and nothing else."
        )
        print(f"{model} -> {response.text}")
    except Exception as exc:
        print(f"{model} -> FAILED: {str(exc)[:200]}")


async def main() -> None:
    client = genai.Client(api_key=settings.gemini_api_key)
    print("configured model:", settings.gemini_model)
    for model in [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ]:
        await try_model(client, model)


asyncio.run(main())