"""
Main client for the Local LLM library.

This module provides the LocalLLM class which implements an OpenAI-like API
for interacting with local Ollama models.
"""

import json
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Union
import requests

from .exceptions import (
    APIError,
    LocalLLMError,
    ModelNotFoundError,
    OllamaConnectionError,
    TimeoutError,
)
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    Completion,
    CompletionChunk,
    CompletionRequest,
    Embedding,
    EmbeddingRequest,
    Model,
    ModelList,
    Usage,
)


class ChatCompletions:
    """Handles chat completion requests."""
    
    def __init__(self, client: 'LocalLLM'):
        self.client = client
    
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: Optional[bool] = False,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Create a chat completion."""
        
        # Convert messages to ChatMessage objects
        chat_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                chat_messages.append(ChatMessage(**msg))
            else:
                chat_messages.append(msg)
        
        # Prepare request data
        request_data = {
            "model": model,
            "messages": [msg.dict() for msg in chat_messages],
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        if stream:
            return self._create_stream(request_data)
        else:
            return self._create_completion(request_data)
    
    def _create_completion(self, request_data: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        response = self.client._make_request("POST", "/api/chat", request_data)
        
        # Convert Ollama response to OpenAI format
        return ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request_data["model"],
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.get("message", {}).get("content", ""),
                    "tool_calls": response.get("message", {}).get("tool_calls"),
                },
                "finish_reason": response.get("done", False) and "stop" or None,
            }],
            usage=Usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            ) if "prompt_eval_count" in response else None,
        )
    
    def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Create a streaming chat completion."""
        response = self.client._make_request(
            "POST", "/api/chat", request_data, stream=True
        )
        
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Create chunk
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request_data["model"],
                        choices=[{
                            "index": 0,
                            "delta": {
                                "content": data.get("message", {}).get("content", ""),
                                "tool_calls": data.get("message", {}).get("tool_calls"),
                            },
                            "finish_reason": data.get("done", False) and "stop" or None,
                        }],
                    )
                    
                    yield chunk
                    
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue


class Completions:
    """Handles text completion requests."""
    
    def __init__(self, client: 'LocalLLM'):
        self.client = client
    
    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        n: Optional[int] = 1,
        stream: Optional[bool] = False,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = 0.0,
        frequency_penalty: Optional[float] = 0.0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Completion, Generator[CompletionChunk, None, None]]:
        """Create a text completion."""
        
        # Prepare request data
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user,
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        if stream:
            return self._create_stream(request_data)
        else:
            return self._create_completion(request_data)
    
    def _create_completion(self, request_data: Dict[str, Any]) -> Completion:
        """Create a non-streaming text completion."""
        response = self.client._make_request("POST", "/api/generate", request_data)
        
        return Completion(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request_data["model"],
            choices=[{
                "index": 0,
                "text": response.get("response", ""),
                "finish_reason": response.get("done", False) and "stop" or None,
            }],
            usage=Usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            ) if "prompt_eval_count" in response else None,
        )
    
    def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> Generator[CompletionChunk, None, None]:
        """Create a streaming text completion."""
        response = self.client._make_request(
            "POST", "/api/generate", request_data, stream=True
        )
        
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    chunk = CompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request_data["model"],
                        choices=[{
                            "index": 0,
                            "text": data.get("response", ""),
                            "finish_reason": data.get("done", False) and "stop" or None,
                        }],
                    )
                    
                    yield chunk
                    
                    if data.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue


class Embeddings:
    """Handles embedding requests."""
    
    def __init__(self, client: 'LocalLLM'):
        self.client = client
    
    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Embedding:
        """Create embeddings."""
        
        # Prepare request data
        request_data = {
            "model": model,
            "prompt": input if isinstance(input, str) else input[0],  # Ollama supports single input
            "user": user,
        }
        
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}
        
        response = self.client._make_request("POST", "/api/embeddings", request_data)
        
        # Convert Ollama response to OpenAI format
        embeddings = []
        if isinstance(input, str):
            embeddings.append({
                "object": "embedding",
                "embedding": response.get("embedding", []),
                "index": 0,
            })
        else:
            # For multiple inputs, we need to make separate requests
            for i, text in enumerate(input):
                single_response = self.client._make_request(
                    "POST", "/api/embeddings", {"model": model, "prompt": text}
                )
                embeddings.append({
                    "object": "embedding",
                    "embedding": single_response.get("embedding", []),
                    "index": i,
                })
        
        return Embedding(
            object="list",
            data=embeddings,
            model=model,
            usage=Usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=0,
                total_tokens=response.get("prompt_eval_count", 0),
            ) if "prompt_eval_count" in response else None,
        )


class Models:
    """Handles model management."""
    
    def __init__(self, client: 'LocalLLM'):
        self.client = client
    
    def list(self) -> ModelList:
        """List available models."""
        response = self.client._make_request("GET", "/api/tags")
        
        models = []
        for model_data in response.get("models", []):
            models.append(Model(
                id=model_data.get("name", ""),
                created=int(time.time()),
                owned_by="ollama",
                size=model_data.get("size"),
                digest=model_data.get("digest"),
            ))
        
        return ModelList(object="list", data=models)
    
    def pull(self, model: str) -> Dict[str, Any]:
        """Pull a model from Ollama."""
        return self.client._make_request("POST", "/api/pull", {"name": model})
    
    def delete(self, model: str) -> Dict[str, Any]:
        """Delete a model."""
        return self.client._make_request("DELETE", f"/api/delete", {"name": model})


class LocalLLM:
    """
    Main client for interacting with local Ollama models.
    
    This class provides an OpenAI-like API for local LLMs, making it easy
    for developers to switch between cloud and local models.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the LocalLLM client.
        
        Args:
            base_url: The base URL for the Ollama API
            default_model: Default model to use if none specified
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize API components
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test the connection to Ollama."""
        try:
            self._make_request("GET", "/api/tags")
        except Exception as e:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running and accessible. Error: {e}"
            )
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], requests.Response]:
        """Make a request to the Ollama API."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                elif method == "POST":
                    if stream:
                        response = requests.post(
                            url, 
                            json=data, 
                            headers=headers, 
                            timeout=self.timeout,
                            stream=True
                        )
                    else:
                        response = requests.post(
                            url, 
                            json=data, 
                            headers=headers, 
                            timeout=self.timeout
                        )
                elif method == "DELETE":
                    response = requests.delete(
                        url, 
                        json=data, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if stream:
                    return response
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    raise ModelNotFoundError(f"Model not found: {data.get('model', 'unknown')}")
                else:
                    raise APIError(
                        f"API request failed with status {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response=response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                    )
                    
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Request timed out after {self.timeout} seconds")
                continue
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    raise OllamaConnectionError(
                        f"Unable to connect to Ollama at {self.base_url}"
                    )
                continue
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LocalLLMError(f"Request failed: {e}")
                continue
        
        raise LocalLLMError("Request failed after all retry attempts") 