import json
import aiohttp
from typing import Optional, Dict, Any, AsyncIterator, List, Literal
from typing_extensions import TypedDict

class Message(TypedDict):
    role: str  # e.g., "user", "assistant", "system"
    content: str
    # Optional fields like tool_calls can be added if needed

class AsyncStream:
    def __init__(self, aiterator: AsyncIterator[Dict[str, Any]]):
        self.aiterator = aiterator

    def __aiter__(self):
        return self.aiterator

class AsyncClient:
    def __init__(self, base_url: str = "http://localhost:12434/engines/llama.cpp/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
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

    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        if self.session and not self.session.closed:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(self.session.close())
            except:
                pass

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    @property
    def chat(self):
        return AsyncChat(self)

    @property
    def completions(self):
        return AsyncCompletions(self)

    @property
    def embeddings(self):
        return AsyncEmbeddings(self)

    @property
    def models(self):
        return AsyncModels(self)

class AsyncChat:
    def __init__(self, client: AsyncClient):
        self.client = client

    @property
    def completions(self):
        return AsyncChatCompletions(self.client)

class AsyncChatCompletions:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def create(self, model: str, messages: List[Message], tool_choice: Optional[Literal["auto", "none", "always"]] = None, **kwargs):
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
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
            return AsyncStream(self._stream_response(url, data))
        async with self.client.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

    async def stream(self, model: str, messages: List[Message], **kwargs):
        """Async stream method that yields chunks and then the full response"""
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/chat/completions"
        data = {"model": model, "messages": messages, "stream": True,  **kwargs}
        
        # First yield all streaming chunks
        async for chunk in self._stream_response(url, data):
            yield chunk
        
        # Then yield the full response (non-streaming)
        data_no_stream = {**data}
        data_no_stream.pop('stream', None)  # Remove stream parameter if present
        async with self.client.session.post(url, json=data_no_stream) as response:
            response.raise_for_status()
            full_response = await response.json()
            yield full_response

    async def _stream_response(self, url: str, data: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        async with self.client.session.post(url, json=data) as response:
            response.raise_for_status()
            # Ensure proper encoding
            response.encoding = 'utf-8'
            buffer = ""
            async for chunk in response.content.iter_chunked(1024):
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

class AsyncCompletions:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def create(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/completions"
        data = {"model": model, "prompt": prompt, **kwargs}
        async with self.client.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

class AsyncEmbeddings:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def create(self, model: str, input: List[str], **kwargs) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/embeddings"
        data = {"model": model, "input": input, **kwargs}
        async with self.client.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

class AsyncModels:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def list(self) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/models"
        async with self.client.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def retrieve(self, model: str) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/models/{model}"
        async with self.client.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def create(self, model: str, **kwargs) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        base = self.client.base_url.replace("/engines/llama.cpp/v1", "")
        url = f"{base}/models/create"
        data = {"model": model, **kwargs}
        async with self.client.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

    async def delete(self, model: str) -> Dict[str, Any]:
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        base = self.client.base_url.replace("/engines/llama.cpp/v1", "")
        url = f"{base}/models/{model}"
        async with self.client.session.delete(url) as response:
            response.raise_for_status()
            return await response.json()