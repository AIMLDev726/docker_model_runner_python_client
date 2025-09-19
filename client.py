import json
import requests
from typing import Optional, Dict, Any, Iterator, List, Literal
from typing_extensions import TypedDict

class Message(TypedDict):
    role: str  # e.g., "user", "assistant", "system"
    content: str
    # Optional fields like tool_calls can be added if needed

class Stream:
    def __init__(self, iterator: Iterator[Dict[str, Any]]):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator

class Client:
    def __init__(self, base_url: str = "http://localhost:12434/engines/llama.cpp/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Automatically configure UTF-8 encoding for proper character support
        self._configure_utf8()

    def _configure_utf8(self):
        """Automatically configure UTF-8 encoding for proper character support"""
        import sys
        import locale
        
        # Configure stdout for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        
        # Configure stderr for UTF-8
        if hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        
        # Set locale for Windows
        if sys.platform == "win32":
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                except locale.Error:
                    # If locale setting fails, continue anyway
                    pass

    @property
    def chat(self):
        return Chat(self)

    @property
    def completions(self):
        return Completions(self)

    @property
    def embeddings(self):
        return Embeddings(self)

    @property
    def models(self):
        return Models(self)

class Chat:
    def __init__(self, client: Client):
        self.client = client

    @property
    def completions(self):
        return ChatCompletions(self.client)

class ChatCompletions:
    def __init__(self, client: Client):
        self.client = client

    def create(self, model: str, messages: List[Message], tool_choice: Optional[Literal["auto", "none", "always"]] = None, **kwargs) -> Dict[str, Any]:
        url = f"{self.client.base_url}/chat/completions"
        data = {"model": model, "messages": messages, **kwargs}
        
        # Handle tool_choice locally
        if tool_choice == "none":
            data.pop("tools", None)
        elif tool_choice == "always":
            if "tools" in data:
                tool_names = [tool["function"]["name"] for tool in data["tools"]]
                tool_names_str = ", ".join(tool_names)
                # Modify the last user message
                for msg in reversed(data["messages"]):
                    if msg["role"] == "user":
                        msg["content"] += f" Use tool {tool_names_str} to respond strictly use tool."
                        break
        elif tool_choice == "auto":
            # Send tools and let model decide (default behavior)
            pass
        # Remove tool_choice from data as server doesn't support it
        data.pop("tool_choice", None)
        
        if kwargs.get("stream", False):
            return Stream(self._stream_response(url, data))
        response = self.client.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def stream(self, model: str, messages: List[Message], **kwargs) -> Iterator[Dict[str, Any]]:
        """Stream method that yields chunks and then the full response"""
        url = f"{self.client.base_url}/chat/completions"
        data = {"model": model, "messages": messages, "stream": True, **kwargs}
        
        # First yield all streaming chunks
        for chunk in self._stream_response(url, data):
            yield chunk
        
        # Then yield the full response (non-streaming)
        data_no_stream = {**data}
        data_no_stream.pop('stream', None)  # Remove stream parameter if present
        response = self.client.session.post(url, json=data_no_stream)
        response.raise_for_status()
        full_response = response.json()
        yield full_response

    def _stream_response(self, url: str, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        with self.client.session.post(url, json=data, stream=True) as response:
            response.raise_for_status()
            # Ensure proper encoding
            response.encoding = 'utf-8'
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024):
                # Explicitly decode as UTF-8
                decoded_chunk = chunk.decode('utf-8', errors='replace')
                buffer += decoded_chunk
                lines = buffer.split('\n')
                buffer = lines.pop()
                for line in lines:
                    line = line.strip()
                    if line:
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                return
                            try:
                                chunk_data = json.loads(data_str)
                                yield chunk_data
                            except json.JSONDecodeError:
                                continue
                        else:
                            try:
                                chunk_data = json.loads(line)
                                yield chunk_data
                            except json.JSONDecodeError:
                                continue

class Completions:
    def __init__(self, client: Client):
        self.client = client

    def create(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.client.base_url}/completions"
        data = {"model": model, "prompt": prompt, **kwargs}
        response = self.client.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

class Embeddings:
    def __init__(self, client: Client):
        self.client = client

    def create(self, model: str, input: List[str], **kwargs) -> Dict[str, Any]:
        url = f"{self.client.base_url}/embeddings"
        data = {"model": model, "input": input, **kwargs}
        response = self.client.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

class Models:
    def __init__(self, client: Client):
        self.client = client

    def list(self) -> Dict[str, Any]:
        url = f"{self.client.base_url}/models"
        response = self.client.session.get(url)
        response.raise_for_status()
        return response.json()

    def retrieve(self, model: str) -> Dict[str, Any]:
        url = f"{self.client.base_url}/models/{model}"
        response = self.client.session.get(url)
        response.raise_for_status()
        return response.json()

    # DMR specific methods
    def create(self, model: str, **kwargs) -> Dict[str, Any]:
        base = self.client.base_url.replace("/engines/llama.cpp/v1", "")
        url = f"{base}/models/create"
        data = {"model": model, **kwargs}
        response = self.client.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def delete(self, model: str) -> Dict[str, Any]:
        base = self.client.base_url.replace("/engines/llama.cpp/v1", "")
        url = f"{base}/models/{model}"
        response = self.client.session.delete(url)
        response.raise_for_status()
        return response.json()