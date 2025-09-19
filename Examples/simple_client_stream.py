from docker_model_runner import Client


client = Client()  # Automatically uses http://localhost:12434/engines/llama.cpp/v1

# Create a streaming response
stream = client.chat.completions.stream(
    model="ai/gemma3n:2B-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

# Iterate over streaming chunks
for chunk in stream:
    # Handle dictionary-based streaming chunks
    if 'choices' in chunk and chunk['choices']:
        choice = chunk['choices'][0]
        if 'delta' in choice and choice['delta'] and 'content' in choice['delta']:
            content = choice['delta']['content']
            if content:
                print(content, end="", flush=True)
        elif 'message' in choice and choice['message'] and 'content' in choice['message']:
            content = choice['message']['content']
            if content:
                print(content, end="", flush=True)

print("\n--- Stream finished ---")
