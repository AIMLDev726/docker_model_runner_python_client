from docker_model_runner import Client

client = Client(api_key="nan")  # Automatically uses http://localhost:12434/engines/v1

response = client.embeddings.create(
    input="Hello how are you?",
    model="ai/embeddinggemma"
)

print(response['data'][0]['embedding'])