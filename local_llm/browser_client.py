"""
Browser client for the Local LLM library using WebLLM.

This module provides a browser-compatible version of the LocalLLM client
that uses WebLLM for in-browser inference without requiring Ollama.
"""

import asyncio
from typing import Any, Dict, Generator, List, Optional, Union
import json

from .exceptions import LocalLLMError, ModelNotFoundError
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    Completion,
    CompletionChunk,
    Embedding,
    Model,
    ModelList,
    Usage,
)


class WebLLMClient:
    """
    Browser client using WebLLM for in-browser LLM inference.
    
    This client provides the same OpenAI-like API as the main LocalLLM client
    but runs entirely in the browser using WebGPU acceleration.
    """
    
    def __init__(
        self,
        model: str = "Llama-3.1-8B-Instruct-q4f32_1-MLC",
        init_progress_callback: Optional[callable] = None,
        app_config: Optional[Dict[str, Any]] = None,
        chat_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the WebLLM client.
        
        Args:
            model: Model to use (from WebLLM's supported models)
            init_progress_callback: Callback for initialization progress
            app_config: Custom app configuration
            chat_opts: Custom chat options
        """
        self.model = model
        self.init_progress_callback = init_progress_callback
        self.app_config = app_config or {}
        self.chat_opts = chat_opts or {}
        self.engine = None
        self.initialized = False
        
        # Initialize API components
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)
    
    async def initialize(self):
        """Initialize the WebLLM engine."""
        try:
            # Import WebLLM (this will only work in browser environments)
            try:
                from mlc_ai.web_llm import CreateMLCEngine
            except ImportError:
                # Fallback for different import paths
                try:
                    from web_llm import CreateMLCEngine
                except ImportError:
                    raise LocalLLMError(
                        "WebLLM not available. Please install @mlc-ai/web-llm "
                        "or ensure you're running in a browser environment."
                    )
            
            # Create the engine
            self.engine = await CreateMLCEngine(
                self.model,
                {
                    "initProgressCallback": self.init_progress_callback,
                    "appConfig": self.app_config,
                },
                self.chat_opts,
            )
            
            self.initialized = True
            
        except Exception as e:
            raise LocalLLMError(f"Failed to initialize WebLLM: {e}")
    
    def _ensure_initialized(self):
        """Ensure the engine is initialized."""
        if not self.initialized or not self.engine:
            raise LocalLLMError(
                "WebLLM engine not initialized. Call await client.initialize() first."
            )


class ChatCompletions:
    """Handles chat completion requests for WebLLM."""
    
    def __init__(self, client: 'WebLLMClient'):
        self.client = client
    
    async def create(
        self,
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
        """Create a chat completion using WebLLM."""
        
        self.client._ensure_initialized()
        
        # Prepare request data
        request_data = {
            "messages": messages,
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
            return await self._create_completion(request_data)
    
    async def _create_completion(self, request_data: Dict[str, Any]) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        try:
            response = await self.client.engine.chat.completions.create(request_data)
            
            # Convert WebLLM response to our format
            return ChatCompletion(
                id=response.get("id", "webllm-chat"),
                created=response.get("created", 0),
                model=self.client.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                    },
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                }],
                usage=Usage(
                    prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=response.get("usage", {}).get("total_tokens", 0),
                ) if "usage" in response else None,
            )
            
        except Exception as e:
            raise LocalLLMError(f"WebLLM chat completion failed: {e}")
    
    def _create_stream(
        self, request_data: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Create a streaming chat completion."""
        async def _stream_generator():
            try:
                chunks = await self.client.engine.chat.completions.create(request_data)
                
                for chunk in chunks:
                    yield ChatCompletionChunk(
                        id=chunk.get("id", "webllm-stream"),
                        created=chunk.get("created", 0),
                        model=self.client.model,
                        choices=[{
                            "index": 0,
                            "delta": {
                                "content": chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""),
                            },
                            "finish_reason": chunk.get("choices", [{}])[0].get("finish_reason"),
                        }],
                    )
                    
            except Exception as e:
                raise LocalLLMError(f"WebLLM streaming failed: {e}")
        
        # Convert async generator to sync generator
        loop = asyncio.get_event_loop()
        async_gen = _stream_generator()
        
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break


class Completions:
    """Handles text completion requests for WebLLM."""
    
    def __init__(self, client: 'WebLLMClient'):
        self.client = client
    
    async def create(
        self,
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
        """Create a text completion using WebLLM."""
        
        self.client._ensure_initialized()
        
        # Convert to chat format for WebLLM
        messages = [{"role": "user", "content": prompt}]
        
        # Use chat completions as fallback
        chat_response = await self.client.chat.create(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
        )
        
        # Convert chat response to completion format
        if stream:
            # Handle streaming
            for chunk in chat_response:
                yield CompletionChunk(
                    id=chunk.id,
                    created=chunk.created,
                    model=chunk.model,
                    choices=[{
                        "index": 0,
                        "text": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason,
                    }],
                )
        else:
            # Handle non-streaming
            return Completion(
                id=chat_response.id,
                created=chat_response.created,
                model=chat_response.model,
                choices=[{
                    "index": 0,
                    "text": chat_response.choices[0].message.content,
                    "finish_reason": chat_response.choices[0].finish_reason,
                }],
                usage=chat_response.usage,
            )


class Embeddings:
    """Handles embedding requests for WebLLM."""
    
    def __init__(self, client: 'WebLLMClient'):
        self.client = client
    
    async def create(
        self,
        input: Union[str, List[str]],
        user: Optional[str] = None,
        **kwargs: Any,
    ) -> Embedding:
        """Create embeddings using WebLLM."""
        
        # Note: WebLLM doesn't natively support embeddings
        # This is a placeholder implementation
        raise LocalLLMError(
            "Embeddings are not currently supported in WebLLM. "
            "Consider using a different model or API for embeddings."
        )


class Models:
    """Handles model management for WebLLM."""
    
    def __init__(self, client: 'WebLLMClient'):
        self.client = client
    
    async def list(self) -> ModelList:
        """List available WebLLM models."""
        try:
            # Import WebLLM to get model list
            try:
                from mlc_ai.web_llm import prebuiltAppConfig
            except ImportError:
                try:
                    from web_llm import prebuiltAppConfig
                except ImportError:
                    raise LocalLLMError("WebLLM not available")
            
            models = []
            for model_config in prebuiltAppConfig.model_list:
                models.append(Model(
                    id=model_config.get("model_id", "unknown"),
                    created=0,  # WebLLM doesn't provide creation time
                    owned_by="webllm",
                    size=None,  # WebLLM doesn't provide size info
                    digest=None,
                ))
            
            return ModelList(object="list", data=models)
            
        except Exception as e:
            raise LocalLLMError(f"Failed to list WebLLM models: {e}")
    
    async def reload(self, model: str) -> Dict[str, Any]:
        """Reload a model in WebLLM."""
        try:
            self.client._ensure_initialized()
            await self.client.engine.reload(model)
            self.client.model = model
            return {"status": "success", "model": model}
        except Exception as e:
            raise LocalLLMError(f"Failed to reload model: {e}")


# Factory function for easy creation
async def create_web_llm_client(
    model: str = "Llama-3.1-8B-Instruct-q4f32_1-MLC",
    init_progress_callback: Optional[callable] = None,
    app_config: Optional[Dict[str, Any]] = None,
    chat_opts: Optional[Dict[str, Any]] = None,
) -> WebLLMClient:
    """
    Create and initialize a WebLLM client.
    
    Args:
        model: Model to use
        init_progress_callback: Progress callback
        app_config: App configuration
        chat_opts: Chat options
    
    Returns:
        Initialized WebLLMClient instance
    """
    client = WebLLMClient(
        model=model,
        init_progress_callback=init_progress_callback,
        app_config=app_config,
        chat_opts=chat_opts,
    )
    
    await client.initialize()
    return client 