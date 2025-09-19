import asyncio
from docker_model_runner import AsyncClient

async def test_async_stream():
    async with AsyncClient() as client:
        
        async for item in client.chat.completions.stream(
            model="ai/gemma3n:2B-Q4_K_M",
            messages=[{"role": "user", "content": "hello how are u"}]
        ):
            if isinstance(item, dict) and 'choices' in item and item['choices']:
                # This is a streaming chunk
                choice = item['choices'][0]
                if 'delta' in choice and choice['delta'] and 'content' in choice['delta']:
                    content = choice['delta']['content']
                    if content:
                        print(content, end='', flush=True)
                elif 'message' in choice and choice['message'] and 'content' in choice['message']:
                    content = choice['message']['content']
                    if content:
                        print(content, end='', flush=True)
            else:
                # This is the full response
                if isinstance(item, dict) and 'choices' in item and item['choices']:
                    print(f"\n\nFull response: {item['choices'][0]['message']['content']}")

# Run the async test
asyncio.run(test_async_stream())