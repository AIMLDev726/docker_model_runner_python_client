import asyncio
from docker_model_runner import AsyncClient

async def main():
    async with AsyncClient(api_key="nan") as client:
        response = await client.chat.completions.create(
            model="ai/gemma3",
            messages=[{"role": "user", "content": "Search for latest news of AI and provide detailed summary for each."}],
            tools=[{
                "type": "mcp",
                "server_label": "duckduckgo",
                "server_description": "DuckDuckGo search and content fetching MCP server.",
                "command": "docker",
                "args": ["run", "-i", "--rm", "mcp/duckduckgo"]
            }],
            tool_choice="always",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ai_news_summary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "news_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "url": {"type": "string"}
                                    },
                                    "required": ["title", "summary", "url"]
                                }
                            },
                            "overall_trends": {"type": "string"}
                        },
                        "required": ["news_items", "overall_trends"]
                    },
                    "strict": True
                }
            }
        )
        print(response['choices'][0]['message']['content'])

asyncio.run(main())