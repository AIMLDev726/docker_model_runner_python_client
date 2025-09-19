from docker_model_runner import Client

client = Client()  # Automatically uses http://localhost:12434/engines/llama.cpp/v1

response = client.chat.completions.create(
    model="ai/gemma3n:2B-Q4_K_M",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=8192
)

print(response)