from docker_model_runner import Client

def test_sync_stream():
    client = Client()
    
    for item in client.chat.completions.stream(
        model="ai/smollm3",
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

# Run the sync test
test_sync_stream()