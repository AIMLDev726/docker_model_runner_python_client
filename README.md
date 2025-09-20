# Docker Model Runner Client

A Python client for interacting with Docker Model Runner, providing a standard API interface for chat completions, embeddings, and more.

(**Note** :- In Code We Have Used gemma3 model please use tool callling supported model else it might possible you face issue or errors for more information please check dockerhub AI Models.)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/AIMLDev726/docker_model_runner_python_client)

## Installation

Install via pip:

```bash
pip install docker-model-runner
```

## Quick Start

### Synchronous Client

```python
from docker_model_runner import Client

client = Client(base_url="http://localhost:12434/engines/v1")  # API key optional

# Chat completion
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello, world!"}]
)

print(response['choices'][0]['message']['content'])
```

### Asynchronous Client

```python
import asyncio
from docker_model_runner import AsyncClient

async def main():
    async with AsyncClient(base_url="http://localhost:12434/engines/v1") as client:  # API key optional
        response = await client.chat.completions.create(
            model="your-model",
            messages=[{"role": "user", "content": "Hello, world!"}]
        )
        print(response['choices'][0]['message']['content'])

asyncio.run(main())
```

### Tool Calls

```python
from docker_model_runner import Client

client = Client()  # Uses default base_url, API key optional

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
message = response['choices'][0]['message']
if message.get("tool_calls"):
    for tool_call in message["tool_calls"]:
        print(f"Tool: {tool_call['function']['name']}")
```

## Features

- Synchronous and asynchronous clients
- Chat completions with streaming support
- Embeddings
- Tool calls with local handling
- Compatible with standard API format

## API Reference

### Client

- `Client(base_url, api_key)`: Initialize sync client (api_key optional)
- `client.chat.completions.create(model, messages, **kwargs)`: Create chat completion
- `client.chat.completions.stream(model, messages, **kwargs)`: Stream chat completion
- `client.embeddings.create(model, input, **kwargs)`: Create embeddings
- `client.models.list()`: List available models

### AsyncClient

- `AsyncClient(base_url, api_key)`: Initialize async client (api_key optional)
- Similar methods as Client, but async

### Tool Choice

- `"auto"`: Let model decide (default)
- `"none"`: Don't use tools
- `"always"`: Force tool usage

## License

MIT License - see LICENSE file for details
