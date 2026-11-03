import asyncio

from app.services.llm import GeminiLLM
from app.services.memory import ChatMemory


async def main() -> None:
    memory = ChatMemory()
    await memory.add_message("test", "user", "Hello")
    await memory.add_message("test", "assistant", "Hi there")
    print("History:", await memory.get_history("test"))
    await memory.clear("test")

    llm = GeminiLLM()
    print("LLM:", await llm.generate("Say 'working' and nothing else."))


asyncio.run(main())