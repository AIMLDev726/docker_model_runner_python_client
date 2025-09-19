from docker_model_runner import Client

client = Client(api_key="nan")  # Automatically uses http://localhost:12434/engines/llama.cpp/v1


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                },
                "required": ["location"]
            }
        }
    }
]


response = client.chat.completions.create(
    model="ai/gemma3n:2B-Q4_K_M",
    messages=[
        {"role": "user", "content": "What is the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="always",
    n_ctx_slot = 8192
)

# Check if model returned a tool call
message = response['choices'][0]['message']
if message.get("tool_calls"):
    for tool_call in message["tool_calls"]:
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]
        print(f"Tool to call: {tool_name}")
        print(f"Arguments: {tool_args}")