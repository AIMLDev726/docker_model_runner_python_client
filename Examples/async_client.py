import asyncio
from docker_model_runner import AsyncClient  # Use AsyncClient instead of Client

async def main():
    client = AsyncClient()  # Automatically uses http://localhost:12434/engines/llama.cpp/v1

    response = await client.chat.completions.create(
        model="ai/gemma3n:2B-Q4_K_M",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(response)

# Run the async function
asyncio.run(main())

