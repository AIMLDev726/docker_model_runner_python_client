import asyncio
import json
from docker_model_runner import AsyncClient

async def main():
    async with AsyncClient(api_key="nan") as client:
        # Execute Python code using MCP tool
        response = await client.chat.completions.create(
            model="ai/gemma3",
            messages=[{"role": "user", "content": "Calculate: x = 10, y = 75 print(x * y)"}],
            tools=[{
                "type": "mcp",
                "server_label": "mcp-code-interpreter",
                "server_description": "Python code execution server",
                "command": "docker",
                "args": ["run", "-i", "--rm", "mcp/mcp-code-interpreter"]
            }],
            tool_choice="always"
        )

        message = response["choices"][0]["message"]

        # Print reasoning_content only if it exists
        reasoning = message.get("reasoning_content")
        if reasoning:
            print("Reasoning:", reasoning)

        # Always print main content
        print("Content:", message.get("content"))

asyncio.run(main())
