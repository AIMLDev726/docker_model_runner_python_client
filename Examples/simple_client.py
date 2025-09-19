from docker_model_runner import Client

client = Client()  # Automatically uses http://localhost:12434/engines/llama.cpp/v1

response = client.chat.completions.create(
    model="ai/smollm3",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response)