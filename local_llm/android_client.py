"""
Android client for the Local LLM library using Google AI Edge LLM Inference API.

This module provides an Android-compatible version of the LocalLLM client
that uses Google's on-device LLM inference without requiring Ollama.
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


class AndroidLLMClient:
    """
    Android client using Google AI Edge LLM Inference API for on-device inference.
    
    This client provides the same OpenAI-like API as the main LocalLLM client
    but runs entirely on Android devices using Google's optimized inference engine.
    """
    
    def __init__(
        self,
        model_path: str = "/data/local/tmp/llm/model_version.task",
        max_tokens: int = 512,
        top_k: int = 40,
        temperature: float = 0.8,
        random_seed: int = 0,
        lora_path: Optional[str] = None,
        max_num_images: int = 0,
        enable_vision: bool = False,
    ):
        """
        Initialize the Android LLM client.
        
        Args:
            model_path: Path to the model file on device
            max_tokens: Maximum number of tokens to generate
            top_k: Number of top tokens to consider during generation
            temperature: Randomness in generation (0.0-2.0)
            random_seed: Random seed for reproducible generation
            lora_path: Path to LoRA model (optional)
            max_num_images: Maximum number of images for multimodal
            enable_vision: Enable vision modality support
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.random_seed = random_seed
        self.lora_path = lora_path
        self.max_num_images = max_num_images
        self.enable_vision = enable_vision
        self.llm_inference = None
        self.session = None
        self.initialized = False
        
        # Initialize API components
        self.chat = ChatCompletions(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)
    
    def initialize(self, context=None):
        """Initialize the Android LLM Inference engine."""
        try:
            # Import Android-specific modules
            try:
                from com.google.mediapipe.tasks.genai import LlmInference, LlmInferenceOptions
                from com.google.mediapipe.tasks.genai import LlmInferenceSession, LlmInferenceSessionOptions
                from com.google.mediapipe.framework import GraphOptions
            except ImportError:
                raise LocalLLMError(
                    "Android LLM Inference API not available. "
                    "Please ensure you're running on Android with the correct dependencies."
                )
            
            # Build options
            options_builder = LlmInferenceOptions.builder()
            options_builder.setModelPath(self.model_path)
            options_builder.setMaxTokens(self.max_tokens)
            options_builder.setTopK(self.top_k)
            options_builder.setTemperature(self.temperature)
            options_builder.setRandomSeed(self.random_seed)
            
            if self.lora_path:
                options_builder.setLoraPath(self.lora_path)
            
            if self.max_num_images > 0:
                options_builder.setMaxNumImages(self.max_num_images)
            
            options = options_builder.build()
            
            # Create LLM Inference instance
            if context is None:
                # Try to get context from Android environment
                try:
                    from android import mActivity
                    context = mActivity
                except ImportError:
                    raise LocalLLMError("Android context required for initialization")
            
            self.llm_inference = LlmInference.createFromOptions(context, options)
            
            # Create session options if vision is enabled
            if self.enable_vision:
                session_options_builder = LlmInferenceSessionOptions.builder()
                session_options_builder.setTopK(self.top_k)
                session_options_builder.setTemperature(self.temperature)
                session_options_builder.setGraphOptions(
                    GraphOptions.builder().setEnableVisionModality(True).build()
                )
                session_options = session_options_builder.build()
                self.session = LlmInferenceSession.createFromOptions(self.llm_inference, session_options)
            
            self.initialized = True
            
        except Exception as e:
            raise LocalLLMError(f"Failed to initialize Android LLM: {e}")
    
    def _ensure_initialized(self):
        """Ensure the engine is initialized."""
        if not self.initialized or not self.llm_inference:
            raise LocalLLMError(
                "Android LLM engine not initialized. Call client.initialize() first."
            )


class ChatCompletions:
    """Handles chat completion requests for Android LLM."""
    
    def __init__(self, client: 'AndroidLLMClient'):
        self.client = client
    
    def create(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """Create a chat completion using Android LLM."""
        
        self.client._ensure_initialized()
        
        # Use client defaults if not specified
        temp = temperature if temperature is not None else self.client.temperature
        tokens = max_tokens if max_tokens is not None else self.client.max_tokens
        k = top_k if top_k is not None else self.client.top_k
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        if stream:
            return self._create_stream(prompt, temp, tokens, k)
        else:
            return self._create_completion(prompt, temp, tokens, k)
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def _create_completion(self, prompt: str, temperature: float, max_tokens: int, top_k: int) -> ChatCompletion:
        """Create a non-streaming chat completion."""
        try:
            # Use session if available (for vision), otherwise direct inference
            if self.client.session:
                # Clear previous session state
                self.client.session.clear()
                
                # Add prompt to session
                self.client.session.addQueryChunk(prompt)
                
                # Generate response
                result = self.client.session.generateResponse()
            else:
                # Direct inference
                result = self.client.llm_inference.generateResponse(prompt)
            
            # Convert to our format
            return ChatCompletion(
                id="android-llm-chat",
                created=0,  # Android API doesn't provide timestamp
                model="android-llm",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result,
                    },
                    "finish_reason": "stop",
                }],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),  # Rough estimate
                    completion_tokens=len(result.split()),
                    total_tokens=len(prompt.split()) + len(result.split()),
                ),
            )
            
        except Exception as e:
            raise LocalLLMError(f"Android LLM chat completion failed: {e}")
    
    def _create_stream(
        self, prompt: str, temperature: float, max_tokens: int, top_k: int
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Create a streaming chat completion."""
        try:
            # Android API doesn't support streaming directly, so we simulate it
            # by generating the full response and yielding it in chunks
            
            if self.client.session:
                self.client.session.clear()
                self.client.session.addQueryChunk(prompt)
                result = self.client.session.generateResponse()
            else:
                result = self.client.llm_inference.generateResponse(prompt)
            
            # Split result into chunks and yield them
            words = result.split()
            chunk_size = max(1, len(words) // 10)  # 10 chunks
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                yield ChatCompletionChunk(
                    id="android-llm-stream",
                    created=0,
                    model="android-llm",
                    choices=[{
                        "index": 0,
                        "delta": {
                            "content": chunk_text + (" " if i + chunk_size < len(words) else ""),
                        },
                        "finish_reason": None if i + chunk_size < len(words) else "stop",
                    }],
                )
                
        except Exception as e:
            raise LocalLLMError(f"Android LLM streaming failed: {e}")


class Completions:
    """Handles text completion requests for Android LLM."""
    
    def __init__(self, client: 'AndroidLLMClient'):
        self.client = client
    
    def create(
        self,
        prompt: Union[str, List[str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Completion, Generator[CompletionChunk, None, None]]:
        """Create a text completion using Android LLM."""
        
        self.client._ensure_initialized()
        
        # Convert list to string if needed
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        
        # Use client defaults if not specified
        temp = temperature if temperature is not None else self.client.temperature
        tokens = max_tokens if max_tokens is not None else self.client.max_tokens
        k = top_k if top_k is not None else self.client.top_k
        
        if stream:
            return self._create_stream(prompt, temp, tokens, k)
        else:
            return self._create_completion(prompt, temp, tokens, k)
    
    def _create_completion(self, prompt: str, temperature: float, max_tokens: int, top_k: int) -> Completion:
        """Create a non-streaming text completion."""
        try:
            if self.client.session:
                self.client.session.clear()
                self.client.session.addQueryChunk(prompt)
                result = self.client.session.generateResponse()
            else:
                result = self.client.llm_inference.generateResponse(prompt)
            
            return Completion(
                id="android-llm-completion",
                created=0,
                model="android-llm",
                choices=[{
                    "index": 0,
                    "text": result,
                    "finish_reason": "stop",
                }],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(result.split()),
                    total_tokens=len(prompt.split()) + len(result.split()),
                ),
            )
            
        except Exception as e:
            raise LocalLLMError(f"Android LLM completion failed: {e}")
    
    def _create_stream(
        self, prompt: str, temperature: float, max_tokens: int, top_k: int
    ) -> Generator[CompletionChunk, None, None]:
        """Create a streaming text completion."""
        try:
            if self.client.session:
                self.client.session.clear()
                self.client.session.addQueryChunk(prompt)
                result = self.client.session.generateResponse()
            else:
                result = self.client.llm_inference.generateResponse(prompt)
            
            # Split result into chunks
            words = result.split()
            chunk_size = max(1, len(words) // 10)
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                yield CompletionChunk(
                    id="android-llm-stream",
                    created=0,
                    model="android-llm",
                    choices=[{
                        "index": 0,
                        "text": chunk_text + (" " if i + chunk_size < len(words) else ""),
                        "finish_reason": None if i + chunk_size < len(words) else "stop",
                    }],
                )
                
        except Exception as e:
            raise LocalLLMError(f"Android LLM streaming failed: {e}")


class Embeddings:
    """Handles embedding requests for Android LLM."""
    
    def __init__(self, client: 'AndroidLLMClient'):
        self.client = client
    
    def create(
        self,
        input: Union[str, List[str]],
        **kwargs: Any,
    ) -> Embedding:
        """Create embeddings using Android LLM."""
        
        # Note: Android LLM Inference API doesn't natively support embeddings
        # This is a placeholder implementation
        raise LocalLLMError(
            "Embeddings are not currently supported in Android LLM Inference API. "
            "Consider using a different model or API for embeddings."
        )


class Models:
    """Handles model management for Android LLM."""
    
    def __init__(self, client: 'AndroidLLMClient'):
        self.client = client
    
    def list(self) -> ModelList:
        """List available Android LLM models."""
        # Android models are typically stored in specific locations
        # This is a simplified implementation
        models = [
            Model(
                id="gemma-3-1b",
                created=0,
                owned_by="google",
                size=None,
                digest=None,
            ),
            Model(
                id="gemma-3n-e2b",
                created=0,
                owned_by="google",
                size=None,
                digest=None,
            ),
            Model(
                id="gemma-3n-e4b",
                created=0,
                owned_by="google",
                size=None,
                digest=None,
            ),
        ]
        
        return ModelList(object="list", data=models)
    
    def reload(self, model_path: str) -> Dict[str, Any]:
        """Reload a model in Android LLM."""
        try:
            self.client._ensure_initialized()
            # Android API doesn't support dynamic model reloading
            # This would require reinitializing the client
            return {"status": "not_supported", "message": "Model reloading not supported in Android API"}
        except Exception as e:
            raise LocalLLMError(f"Failed to reload model: {e}")


# Factory function for easy creation
def create_android_llm_client(
    model_path: str = "/data/local/tmp/llm/model_version.task",
    max_tokens: int = 512,
    top_k: int = 40,
    temperature: float = 0.8,
    random_seed: int = 0,
    lora_path: Optional[str] = None,
    max_num_images: int = 0,
    enable_vision: bool = False,
    context=None,
) -> AndroidLLMClient:
    """
    Create and initialize an Android LLM client.
    
    Args:
        model_path: Path to the model file
        max_tokens: Maximum tokens to generate
        top_k: Top-k sampling parameter
        temperature: Generation temperature
        random_seed: Random seed
        lora_path: Path to LoRA model
        max_num_images: Maximum images for multimodal
        enable_vision: Enable vision support
        context: Android context
    
    Returns:
        Initialized AndroidLLMClient instance
    """
    client = AndroidLLMClient(
        model_path=model_path,
        max_tokens=max_tokens,
        top_k=top_k,
        temperature=temperature,
        random_seed=random_seed,
        lora_path=lora_path,
        max_num_images=max_num_images,
        enable_vision=enable_vision,
    )
    
    client.initialize(context)
    return client 