"""
iOS client for the Local LLM library using Apple Foundation Models Framework.

This module provides an iOS-compatible version of the LocalLLM client
that uses Apple's on-device foundation models without requiring Ollama.
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


class IOSLLMClient:
    """
    iOS client using Apple Foundation Models Framework for on-device inference.
    
    This client provides the same OpenAI-like API as the main LocalLLM client
    but runs entirely on iOS devices using Apple's optimized foundation models.
    """
    
    def __init__(
        self,
        instructions: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 512,
        use_case: Optional[str] = None,
    ):
        """
        Initialize the iOS LLM client.
        
        Args:
            instructions: System instructions for the model
            temperature: Generation temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            use_case: Specific use case for the model
        """
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_case = use_case
        self.model = None
        self.session = None
        self.initialized = False
        
        # Initialize API components
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)
    
    def initialize(self):
        """Initialize the iOS Foundation Models engine."""
        try:
            # Import iOS-specific modules
            try:
                from Foundation import SystemLanguageModel, LanguageModelSession
                from Foundation import GenerationOptions, Prompt
            except ImportError:
                raise LocalLLMError(
                    "Apple Foundation Models Framework not available. "
                    "Please ensure you're running on iOS with the correct dependencies."
                )
            
            # Get the system language model
            self.model = SystemLanguageModel.default
            
            # Check availability
            if self.model.availability != SystemLanguageModel.Availability.available:
                raise LocalLLMError(
                    f"Foundation model not available: {self.model.availability}"
                )
            
            # Create session with instructions if provided
            if self.instructions:
                self.session = LanguageModelSession(instructions=self.instructions)
            else:
                self.session = LanguageModelSession()
            
            self.initialized = True
            
        except Exception as e:
            raise LocalLLMError(f"Failed to initialize iOS LLM: {e}")
    
    def _ensure_initialized(self):
        """Ensure the engine is initialized."""
        if not self.initialized or not self.model or not self.session:
            raise LocalLLMError(
                "iOS LLM engine not initialized. Call client.initialize() first."
            )


class ChatCompletions:
    """Handles chat completion requests for iOS LLM."""
    
    def __init__(self, client: 'IOSLLMClient'):
        self.client = client
    
    async def create(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Create a chat completion using iOS LLM."""
        
        self.client._ensure_initialized()
        
        # Use client defaults if not specified
        temp = temperature if temperature is not None else self.client.temperature
        tokens = max_tokens if max_tokens is not None else self.client.max_tokens
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        if stream:
            return self._create_stream(prompt, temp, tokens)
        else:
            return await self._create_completion(prompt, temp, tokens)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # System messages are handled by instructions in iOS
                continue
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                # iOS doesn't support assistant messages in the same way
                # We'll include them as context
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    async def _create_completion(self, prompt: str, temperature: float, max_tokens: int) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        try:
            # Create generation options
            options = GenerationOptions()
            options.temperature = temperature
            
            # Create prompt object
            prompt_obj = Prompt(prompt)
            
            # Generate response
            result = await self.client.session.respond(
                to=prompt_obj,
                options=options
            )
            
            # Convert to our format
            return ChatCompletion(
                id="ios-llm-chat",
                created=0,  # iOS API doesn't provide timestamp
                model="ios-foundation-model",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(result),
                    },
                    "finish_reason": "stop",
                }],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),  # Rough estimate
                    completion_tokens=len(str(result).split()),
                    total_tokens=len(prompt.split()) + len(str(result).split()),
                ),
            )
            
        except Exception as e:
            raise LocalLLMError(f"iOS LLM chat completion failed: {e}")
    
    def _create_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Create a streaming chat completion."""
        async def _stream_generator():
            try:
                # iOS Foundation Models don't support streaming directly
                # We'll simulate it by generating the full response and yielding chunks
                
                options = GenerationOptions()
                options.temperature = temperature
                prompt_obj = Prompt(prompt)
                
                result = await self.client.session.respond(
                    to=prompt_obj,
                    options=options
                )
                
                result_str = str(result)
                words = result_str.split()
                chunk_size = max(1, len(words) // 10)  # 10 chunks
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    yield ChatCompletionChunk(
                        id="ios-llm-stream",
                        created=0,
                        model="ios-foundation-model",
                        choices=[{
                            "index": 0,
                            "delta": {
                                "content": chunk_text + (" " if i + chunk_size < len(words) else ""),
                            },
                            "finish_reason": None if i + chunk_size < len(words) else "stop",
                        }],
                    )
                    
            except Exception as e:
                raise LocalLLMError(f"iOS LLM streaming failed: {e}")
        
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
    """Handles text completion requests for iOS LLM."""
    
    def __init__(self, client: 'IOSLLMClient'):
        self.client = client
    
    async def create(
        self,
        prompt: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Completion, Generator[CompletionChunk, None, None]]:
        """Create a text completion using iOS LLM."""
        
        self.client._ensure_initialized()
        
        # Convert list to string if needed
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        
        # Use client defaults if not specified
        temp = temperature if temperature is not None else self.client.temperature
        tokens = max_tokens if max_tokens is not None else self.client.max_tokens
        
        if stream:
            return self._create_stream(prompt, temp, tokens)
        else:
            return await self._create_completion(prompt, temp, tokens)
    
    async def _create_completion(self, prompt: str, temperature: float, max_tokens: int) -> Completion:
        """Create a non-streaming text completion."""
        try:
            options = GenerationOptions()
            options.temperature = temperature
            prompt_obj = Prompt(prompt)
            
            result = await self.client.session.respond(
                to=prompt_obj,
                options=options
            )
            
            return Completion(
                id="ios-llm-completion",
                created=0,
                model="ios-foundation-model",
                choices=[{
                    "index": 0,
                    "text": str(result),
                    "finish_reason": "stop",
                }],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(str(result).split()),
                    total_tokens=len(prompt.split()) + len(str(result).split()),
                ),
            )
            
        except Exception as e:
            raise LocalLLMError(f"iOS LLM completion failed: {e}")
    
    def _create_stream(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Generator[CompletionChunk, None, None]:
        """Create a streaming text completion."""
        async def _stream_generator():
            try:
                options = GenerationOptions()
                options.temperature = temperature
                prompt_obj = Prompt(prompt)
                
                result = await self.client.session.respond(
                    to=prompt_obj,
                    options=options
                )
                
                result_str = str(result)
                words = result_str.split()
                chunk_size = max(1, len(words) // 10)
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    yield CompletionChunk(
                        id="ios-llm-stream",
                        created=0,
                        model="ios-foundation-model",
                        choices=[{
                            "index": 0,
                            "text": chunk_text + (" " if i + chunk_size < len(words) else ""),
                            "finish_reason": None if i + chunk_size < len(words) else "stop",
                        }],
                    )
                    
            except Exception as e:
                raise LocalLLMError(f"iOS LLM streaming failed: {e}")
        
        # Convert async generator to sync generator
        loop = asyncio.get_event_loop()
        async_gen = _stream_generator()
        
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break


class Embeddings:
    """Handles embedding requests for iOS LLM."""
    
    def __init__(self, client: 'IOSLLMClient'):
        self.client = client
    
    async def create(
        self,
        input: Union[str, List[str]],
        **kwargs: Any,
    ) -> Embedding:
        """Create embeddings using iOS LLM."""
        
        # Note: iOS Foundation Models Framework doesn't natively support embeddings
        # This is a placeholder implementation
        raise LocalLLMError(
            "Embeddings are not currently supported in iOS Foundation Models Framework. "
            "Consider using a different model or API for embeddings."
        )


class Models:
    """Handles model management for iOS LLM."""
    
    def __init__(self, client: 'IOSLLMClient'):
        self.client = client
    
    def list(self) -> ModelList:
        """List available iOS LLM models."""
        # iOS Foundation Models Framework provides a single system model
        models = [
            Model(
                id="system-foundation-model",
                created=0,
                owned_by="apple",
                size=None,
                digest=None,
            ),
        ]
        
        return ModelList(object="list", data=models)
    
    def reload(self, model_id: str) -> Dict[str, Any]:
        """Reload a model in iOS LLM."""
        try:
            self.client._ensure_initialized()
            # iOS Foundation Models Framework doesn't support model reloading
            return {"status": "not_supported", "message": "Model reloading not supported in iOS Foundation Models Framework"}
        except Exception as e:
            raise LocalLLMError(f"Failed to reload model: {e}")


# Factory function for easy creation
def create_ios_llm_client(
    instructions: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 512,
    use_case: Optional[str] = None,
) -> IOSLLMClient:
    """
    Create and initialize an iOS LLM client.
    
    Args:
        instructions: System instructions for the model
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        use_case: Specific use case for the model
    
    Returns:
        Initialized IOSLLMClient instance
    """
    client = IOSLLMClient(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        use_case=use_case,
    )
    
    client.initialize()
    return client 