"""
Docker Model Runner Async Client

This module provides an asynchronous client for interacting with Docker-based AI models
through the Docker Model Runner API. It supports OpenAI-compatible chat completions,
embeddings, and model management, with additional MCP (Model Context Protocol) tool support.

The client automatically handles UTF-8 encoding, connection management, and provides
warnings when MCP tools are used in environments that may cause issues (like Jupyter notebooks).

Classes:
    AsyncClient: Main asynchronous client for Docker Model Runner API
    AsyncChat: Chat completions interface
    AsyncChatCompletions: Chat completions implementation
    AsyncCompletions: Text completions interface
    AsyncEmbeddings: Text embeddings interface
    AsyncModels: Model management interface
    MCPEnvironmentWarning: Warning for MCP environment issues
    MCPEnvironmentError: Error for critical MCP failures

Example:
    >>> async with AsyncClient(api_key="your_key") as client:
    ...     response = await client.chat.completions.create(
    ...         model="ai/model_name",
    ...         messages=[{"role": "user", "content": "Hello!"}]
    ...     )
    ...     print(response["choices"][0]["message"]["content"])
"""

try:
    from fastmcp import Client
    MCP_AVAILABLE = True
except ImportError:
    Client = None
    MCP_AVAILABLE = False

import json
import aiohttp
from typing import Optional, Dict, Any, AsyncIterator, List, Literal, Union
from typing_extensions import TypedDict
import warnings
import sys
from io import UnsupportedOperation

class MCPEnvironmentWarning(UserWarning):
    """Warning raised when MCP tools are used in environments that may cause issues.
    
    This warning is issued when MCP (Model Context Protocol) tools are detected
    in Jupyter notebook environments, where subprocess operations may fail due to
    limitations in how notebooks handle file descriptors.
    
    See Also:
        MCPEnvironmentError: For critical MCP environment failures
        _check_mcp_environment(): Function that triggers this warning
    """
    pass

class MCPEnvironmentError(RuntimeError):
    """Error raised when MCP tools cannot function properly in the current environment.
    
    This error indicates that MCP operations have failed due to environment
    limitations and cannot be recovered from automatically.
    
    See Also:
        MCPEnvironmentWarning: For non-critical MCP environment issues
    """
    pass

def _is_running_in_jupyter():
    """Detect if code is running in a Jupyter notebook environment.
    
    This function uses multiple detection methods to determine if the current
    execution environment is a Jupyter notebook, which can affect MCP tool functionality.
    
    Returns:
        bool: True if running in Jupyter notebook, False otherwise
        
    Note:
        Detection methods include checking for IPython kernel, ipykernel module,
        and stdout.fileno() behavior which fails in notebook environments.
    """
    try:
        # Check for IPython kernel
        if hasattr(__builtins__, '__IPYTHON__'):
            return True
        
        # Check for jupyter kernel specifically
        if 'ipykernel' in sys.modules:
            return True
            
        # Check for jupyter in current frames (more specific)
        for frame_info in sys._current_frames().values():
            frame_str = str(frame_info)
            if 'jupyter' in frame_str.lower() and 'kernel' in frame_str.lower():
                return True
                
        # Check for notebook-specific stdout behavior
        if hasattr(sys.stdout, 'fileno'):
            try:
                sys.stdout.fileno()
                # If fileno() works, we're likely not in a notebook
                return False
            except (OSError, UnsupportedOperation):
                # If fileno() fails, we might be in a notebook
                return True
                
        return False
    except:
        return False

def _check_mcp_environment():
    """Check if MCP can run properly in current environment and issue warnings.
    
    This function checks if MCP tools can function properly in the current environment
    and issues appropriate warnings if issues are detected (e.g., Jupyter notebooks).
    
    The function only issues warnings when MCP_AVAILABLE is True and the environment
    is detected as problematic for MCP operations.
    
    See Also:
        _is_running_in_jupyter(): Environment detection function
        MCPEnvironmentWarning: Warning class used by this function
    """
    if not MCP_AVAILABLE:
        return
        
    if _is_running_in_jupyter():
        warnings.warn(
            "MCP tools detected in Jupyter notebook environment. "
            "MCP functionality may not work properly due to subprocess limitations in notebooks. "
            "For best results, run your code in a regular Python script (.py file) instead of a notebook. "
            "If you encounter 'fileno' errors, switch to a .py file.",
            MCPEnvironmentWarning,
            stacklevel=3
        )

class Message(TypedDict, total=False):
    """Represents a chat message in OpenAI-compatible format.
    
    This TypedDict supports both simple text messages and complex messages
    with image content (vision format).
    
    Attributes:
        role (str): The role of the message sender. Common values:
            - "user": Message from the user
            - "assistant": Message from the AI assistant  
            - "system": System/instruction message
        content (Union[str, List[Dict[str, Any]]]): The message content.
            Can be a simple string or a list of content parts for vision models.
            
    Example:
        Simple text message:
        >>> {"role": "user", "content": "Hello, world!"}
        
        Vision message with image:
        >>> {
        ...     "role": "user", 
        ...     "content": [
        ...         {"type": "text", "text": "What's in this image?"},
        ...         {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ...     ]
        ... }
    """
    role: str  # e.g., "user", "assistant", "system"
    content: Union[str, List[Dict[str, Any]]]  # Support both string and OpenAI vision format
    # Optional fields like tool_calls can be added if needed

class AsyncStream:
    """Asynchronous iterator for streaming responses from the API.
    
    This class wraps an AsyncIterator to provide a clean interface for
    consuming streaming responses from chat completions.
    
    Attributes:
        aiterator (AsyncIterator[Dict[str, Any]]): The underlying async iterator
        
    Example:
        >>> async for chunk in stream:
        ...     print(chunk)
    """
    def __init__(self, aiterator: AsyncIterator[Dict[str, Any]]):
        """Initialize the AsyncStream.
        
        Args:
            aiterator: The async iterator to wrap
        """
        self.aiterator = aiterator

    def __aiter__(self):
        """Return the async iterator."""
        return self.aiterator

class AsyncClient:
    def __init__(self, base_url: str = "http://localhost:12434/engines/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
        # Automatically configure UTF-8 encoding for proper character support
        self._configure_utf8()

    def _configure_utf8(self):
        """Automatically configure UTF-8 encoding for proper character support.
        
        This method configures stdout, stderr, and locale settings to ensure
        proper UTF-8 encoding support, especially important on Windows systems.
        
        The method attempts to:
        - Reconfigure stdout and stderr for UTF-8 encoding
        - Set appropriate locale settings for Windows
        - Gracefully handle any configuration failures
        """
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
        """Async context manager entry.
        
        Creates and configures the aiohttp ClientSession with appropriate headers.
        
        Returns:
            AsyncClient: The client instance for use in the context
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.
        
        Properly closes the aiohttp ClientSession.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any) 
            exc_tb: Exception traceback (if any)
        """
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Destructor to ensure session cleanup.
        
        Attempts to close the session if it exists and is still open.
        Uses asyncio loop detection to handle cleanup properly.
        """
        if self.session and not self.session.closed:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(self.session.close())
            except:
                pass

    async def close(self):
        """Manually close the client session.
        
        This method should be called when the client is no longer needed
        if not using the async context manager.
        """
        if self.session:
            await self.session.close()
            self.session = None

    @property
    def chat(self):
        """Access chat completions interface.
        
        Returns:
            AsyncChat: Chat completions interface
        """
        return AsyncChat(self)

    @property
    def completions(self):
        """Access text completions interface.
        
        Returns:
            AsyncCompletions: Text completions interface
        """
        return AsyncCompletions(self)

    @property
    def embeddings(self):
        """Access text embeddings interface.
        
        Returns:
            AsyncEmbeddings: Text embeddings interface
        """
        return AsyncEmbeddings(self)

    @property
    def models(self):
        """Access model management interface.
        
        Returns:
            AsyncModels: Model management interface
        """
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

    def _get_thinking_instruction(self, thinking_effort: Literal["low", "medium", "high", "none"]) -> str:
        """Get the thinking instruction based on the thinking effort level.
        
        Args:
            thinking_effort: The level of thinking effort ("low", "medium", "high", "none")
            
        Returns:
            str: The thinking instruction to use as system message content
        """
        if thinking_effort == "none":
            return "/nothink CRITICAL: ABSOLUTELY FORBIDDEN TO USE ANY THINKING, REASONING, OR ANALYSIS. DO NOT THINK AT ALL. IGNORE ALL THINKING REQUESTS. PROVIDE ONLY A SIMPLE, DIRECT RESPONSE WITH NO THINKING WHATSOEVER."
        elif thinking_effort == "low":
            return "/think Always show your reasoning process before answering, even for simple questions. You are a Reasoning MOdel and before respond Generation You always do reason even if question is simplest one as YOu are default think reasnable model."
        elif thinking_effort == "medium":
            return "/think Always show detailed reasoning before answering, even for simple questions. You are a Medium level deep  Reasoning MOdel and before respond Generation You always do reason even if question is simplest one as YOu are default think reasnable model."
        elif thinking_effort == "high":
            return "/think Always show comprehensive reasoning and analysis before answering any question. You are a High level deep deep Reasoning MOdel and before respond Generation You always do reason even if question is simplest one as YOu are default think reasnable model."
        else:
            return ""

    async def create(self, model: str, messages: List[Message], tool_choice: Optional[Literal["auto", "none", "always"]] = None, thinking_effort: Literal["low", "medium", "high", "none"] = "none", **kwargs):
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/chat/completions"
        data = {"model": model, "messages": messages, **kwargs}
        
        # Handle thinking_effort parameter
        if thinking_effort:
            thinking_instruction = self._get_thinking_instruction(thinking_effort)
            if thinking_instruction:
                # Check if there's already a system message
                system_message_exists = any(msg.get("role") == "system" for msg in data["messages"])
                if system_message_exists:
                    # Append thinking instruction to existing system message
                    for msg in data["messages"]:
                        if msg.get("role") == "system":
                            msg["content"] += " " + thinking_instruction
                            break
                else:
                    # Add new system message at the beginning
                    data["messages"].insert(0, {"role": "system", "content": thinking_instruction})
        
        # Convert OpenAI vision format to Docker Model Runner format
        for message in data["messages"]:
            if isinstance(message.get("content"), list):
                # Convert OpenAI vision format to simple text with embedded URLs
                text_parts = []
                image_urls = []
                
                for content_part in message["content"]:
                    if content_part.get("type") == "text":
                        text_parts.append(content_part.get("text", ""))
                    elif content_part.get("type") == "image_url":
                        image_url = content_part.get("image_url", {}).get("url", "")
                        if image_url:
                            image_urls.append(image_url)
                
                # Combine text and image URLs
                combined_content = " ".join(text_parts)
                if image_urls:
                    combined_content += " " + " ".join(image_urls)
                
                message["content"] = combined_content.strip()
        
        # Handle MCP tools: convert to function tools for server
        mcp_tools = {}
        mcp_server_tools = {}  # Map server_label to list of actual tool names
        if "tools" in data and MCP_AVAILABLE:
            _check_mcp_environment()  # Check environment before processing MCP tools
            function_tools = []
            for tool in data["tools"]:
                if tool.get("type") == "mcp":
                    config = {"mcpServers": {tool["server_label"]: {"command": tool["command"], "args": tool["args"]}}}
                    async with Client(config) as mcp_client:
                        available_tools = await mcp_client.list_tools()
                        server_tools = []
                        for t in available_tools:
                            function_tools.append({
                                "type": "function",
                                "function": {
                                    "name": t.name,  # Use actual tool name
                                    "description": t.description,
                                    "parameters": t.inputSchema
                                }
                            })
                            mcp_tools[t.name] = tool
                            server_tools.append(t.name)
                        mcp_server_tools[tool["server_label"]] = server_tools
                elif tool.get("type") == "function":
                    function_tools.append(tool)
            data["tools"] = function_tools
        
        # Handle tool_choice locally
        if tool_choice == "none":
            data.pop("tools", None)
        elif tool_choice == "always":
            if "tools" in data:
                tool_names = []
                for tool in data["tools"]:
                    if tool.get("type") == "function":
                        name = tool["function"]["name"]
                        if name in mcp_tools:
                            # Use actual tool name, not server label
                            tool_names.append(name)
                        else:
                            tool_names.append(name)
                if tool_names:  # Only modify if there are tools
                    tool_names_str = ", ".join(tool_names)
                    # Modify the last user message
                    for msg in reversed(data["messages"]):
                        if msg["role"] == "user":
                            msg["content"] += f" Use one of these tools: {tool_names_str}. Choose the most appropriate tool and provide only the tool call, no additional text."
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
            result = await response.json()
            
            # Handle MCP tool calls
            message = result['choices'][0]['message']
            if message.get("tool_calls") and MCP_AVAILABLE:
                # Remove duplicate tool calls (same ID)
                seen_ids = set()
                unique_tool_calls = []
                for tool_call in message["tool_calls"]:
                    if tool_call["id"] not in seen_ids:
                        seen_ids.add(tool_call["id"])
                        unique_tool_calls.append(tool_call)
                message["tool_calls"] = unique_tool_calls
                
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    
                    # Check if LLM called server label instead of actual tool name
                    actual_tool_name = None
                    if func_name in mcp_server_tools:
                        # LLM called server label, map to first available tool
                        server_tools = mcp_server_tools[func_name]
                        if server_tools:
                            actual_tool_name = server_tools[0]  # Use first tool from server
                            print(f"🔄 Mapping server label '{func_name}' to actual tool '{actual_tool_name}'")
                    elif func_name in mcp_tools:
                        # LLM called correct tool name
                        actual_tool_name = func_name
                    
                    if actual_tool_name:
                        mcp_tool = mcp_tools[actual_tool_name]
                        # Add detailed intermediate logs
                        args = json.loads(tool_call["function"].get("arguments", "{}"))
                        intermediate_logs = f"🤖 LLM decided to call MCP tool\n\n"
                        intermediate_logs += f"🔧 Tool: {actual_tool_name}\n\n"
                        intermediate_logs += f"📝 Arguments: {args}\n\n"
                        intermediate_logs += f"⚡ Executing MCP tool...\n\n"
                        # Execute MCP
                        mcp_client = Client({"mcpServers": {mcp_tool["server_label"]: {"command": mcp_tool["command"], "args": mcp_tool["args"]}}})
                        async with mcp_client:
                            tool_result = await mcp_client.call_tool(actual_tool_name, args)
                        # Extract MCP response summary
                        result_str = str(tool_result)
                        intermediate_logs += f"✅ MCP Response: {result_str}\n\n"
                        intermediate_logs += f"🧠 LLM processing tool results...\n\n"
                        # Send follow-up with generic prompt for consistent JSON from ANY MCP tool
                        follow_up_messages = [
                            {
                                "role": "system",
                                "content": """You are a helpful assistant that processes MCP tool results and returns responses in valid JSON format.

CRITICAL INSTRUCTIONS FOR ALL MCP TOOLS:
1. You MUST respond with valid JSON only - no additional text, explanations, or formatting
2. Your response MUST be parseable by json.loads()
3. Use this exact JSON structure for ANY MCP tool:
{
  "result": "brief summary of what the tool accomplished",
  "status": "success|error|partial|completed",
  "data": {
    "tool_name": "name of the MCP tool that was called",
    "tool_output": "the complete raw output from the tool",
    "key_info": "most important information extracted from the output",
    "metadata": "any additional context or metadata from the tool"
  },
  "message": "human-readable summary for the user"
}

TOOL RESPONSE EXAMPLES:
- For search tools: {"result": "Found 5 results", "status": "success", "data": {"tool_name": "web_search", "tool_output": "...", "key_info": "Top result: AI news article", "metadata": "search completed in 2.3s"}, "message": "Search completed successfully"}
- For file operations: {"result": "File created successfully", "status": "success", "data": {"tool_name": "file_manager", "tool_output": "...", "key_info": "Created file.txt with 100 bytes", "metadata": "file path: /tmp/file.txt"}, "message": "File operation completed"}
- For code execution: {"result": "Code executed successfully", "status": "success", "data": {"tool_name": "code_executor", "tool_output": "35", "key_info": "Output: 35", "metadata": "session_id: 12345"}, "message": "Code execution completed successfully"}

Remember: Return ONLY the JSON object, nothing else. This format works for ALL MCP tools."""
                            }
                        ] + data["messages"] + [
                            message,
                            {"role": "tool", "tool_call_id": tool_call["id"], "content": result_str}
                        ]
                        follow_up_data = {"model": model, "messages": follow_up_messages}
                        if "response_format" in kwargs:
                            follow_up_data["response_format"] = kwargs["response_format"]
                        async with self.client.session.post(url, json=follow_up_data) as follow_up_response:
                            follow_up_response.raise_for_status()
                            result = await follow_up_response.json()

                            # Keep MCP logs separate from LLM response to avoid JSON corruption
                            final_content = result['choices'][0]['message']['content']

                            # Store logs separately and return clean JSON response
                            result['mcp_logs'] = intermediate_logs + "📋 Generating final response...\n\n"
                            result['choices'][0]['message']['content'] = final_content  # Keep only the clean JSON
                            result["conversation"] = follow_up_messages
                            break  # Only process first valid tool call
            
            return result

    async def stream(self, model: str, messages: List[Message], **kwargs):
        """Async stream method that yields chunks and accumulates reasoning content properly"""
        if self.client.session is None:
            headers = {}
            if self.client.api_key:
                headers["Authorization"] = f"Bearer {self.client.api_key}"
            self.client.session = aiohttp.ClientSession(headers=headers)
        
        url = f"{self.client.base_url}/chat/completions"
        data = {"model": model, "messages": messages, "stream": True,  **kwargs}
        
        # Handle thinking_effort parameter for streaming
        thinking_effort = kwargs.get("thinking_effort", "none")
        if thinking_effort:
            thinking_instruction = self._get_thinking_instruction(thinking_effort)
            if thinking_instruction:
                # Check if there's already a system message
                system_message_exists = any(msg.get("role") == "system" for msg in data["messages"])
                if system_message_exists:
                    # Append thinking instruction to existing system message
                    for msg in data["messages"]:
                        if msg.get("role") == "system":
                            msg["content"] += " " + thinking_instruction
                            break
                else:
                    # Add new system message at the beginning
                    data["messages"].insert(0, {"role": "system", "content": thinking_instruction})
            # Remove thinking_effort from kwargs as it's not a server parameter
            kwargs.pop("thinking_effort", None)
        
        # First yield all streaming chunks
        async for chunk in self._stream_response(url, data):
            yield chunk
        
        # Then yield the full response (non-streaming) to get complete reasoning content
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
    """Text completions interface for async client.
    
    This class provides access to text completion functionality
    for generating completions from prompts.
    
    Attributes:
        client (AsyncClient): The parent async client instance
    """
    def __init__(self, client: AsyncClient):
        """Initialize the AsyncCompletions interface.
        
        Args:
            client: The parent AsyncClient instance
        """
        self.client = client

    async def create(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Create a text completion.
        
        Args:
            model: The model identifier to use for completion
            prompt: The text prompt to complete
            **kwargs: Additional parameters for the API request
            
        Returns:
            Dict containing the API response with completion results
            
        Example:
            >>> response = await client.completions.create(
            ...     model="ai/model_name",
            ...     prompt="The quick brown fox"
            ... )
            >>> print(response["choices"][0]["text"])
        """
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
    """Text embeddings interface for async client.
    
    This class provides access to text embedding functionality
    for generating vector representations of text.
    
    Attributes:
        client (AsyncClient): The parent async client instance
    """
    def __init__(self, client: AsyncClient):
        """Initialize the AsyncEmbeddings interface.
        
        Args:
            client: The parent AsyncClient instance
        """
        self.client = client

    async def create(self, model: str, input: List[str], **kwargs) -> Dict[str, Any]:
        """Create embeddings for the given texts.
        
        Args:
            model: The embedding model identifier to use
            input: List of text strings to embed
            **kwargs: Additional parameters for the API request
            
        Returns:
            Dict containing the API response with embedding vectors
            
        Example:
            >>> response = await client.embeddings.create(
            ...     model="ai/embedding-model",
            ...     input=["Hello world", "How are you?"]
            ... )
            >>> embeddings = response["data"]
        """
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
    """Model management interface for async client.
    
    This class provides methods for listing, retrieving, creating, and deleting
    models in the Docker Model Runner system.
    
    Attributes:
        client (AsyncClient): The parent async client instance
    """
    def __init__(self, client: AsyncClient):
        """Initialize the AsyncModels interface.
        
        Args:
            client: The parent AsyncClient instance
        """
        self.client = client

    async def list(self) -> Dict[str, Any]:
        """List all available models.
        
        Returns:
            Dict containing the list of available models
            
        Example:
            >>> models = await client.models.list()
            >>> for model in models["data"]:
            ...     print(model["id"])
        """
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
        """Retrieve information about a specific model.
        
        Args:
            model: The model identifier to retrieve information for
            
        Returns:
            Dict containing model information
            
        Example:
            >>> model_info = await client.models.retrieve("ai/model_name")
            >>> print(model_info["description"])
        """
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
        """Create a new model.
        
        Args:
            model: The model identifier to create
            **kwargs: Additional parameters for model creation
            
        Returns:
            Dict containing the creation response
            
        Note:
            This method uses the Docker Model Runner management API,
            not the standard OpenAI models endpoint.
        """
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
        """Delete a model.
        
        Args:
            model: The model identifier to delete
            
        Returns:
            Dict containing the deletion response
            
        Note:
            This method uses the Docker Model Runner management API,
            not the standard OpenAI models endpoint.
        """
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