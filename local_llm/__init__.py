"""
Local LLM - OpenAI-like API for local Large Language Models using Ollama.

This library provides a familiar interface for developers to use local LLMs
with the same API patterns as OpenAI's Python client.
"""

from .client import LocalLLM
from .browser_client import WebLLMClient, create_web_llm_client
from .android_client import AndroidLLMClient, create_android_llm_client
from .ios_client import IOSLLMClient, create_ios_llm_client
from .exceptions import LocalLLMError, ModelNotFoundError, OllamaConnectionError
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
from .utils import (
    check_ollama_installation,
    create_model_selector,
    download_model_with_progress,
    estimate_model_performance,
    get_model_info,
    get_ollama_install_instructions,
    get_recommended_models,
    get_setup_status,
    get_system_info,
    install_ollama_automatically,
    open_ollama_download_page,
    recommend_model_for_use_case,
    setup_local_ai_with_guidance,
)

__version__ = "0.1.0"
__author__ = "Local LLM Team"
__email__ = "dev@localllm.ai"

__all__ = [
    # Main clients
    "LocalLLM",
    "WebLLMClient",
    "create_web_llm_client",
    "AndroidLLMClient",
    "create_android_llm_client",
    "IOSLLMClient",
    "create_ios_llm_client",
    
    # Exceptions
    "LocalLLMError",
    "ModelNotFoundError", 
    "OllamaConnectionError",
    
    # Types
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatMessage",
    "Completion",
    "CompletionChunk",
    "Embedding",
    "Model",
    "ModelList",
    "Usage",
    
    # Utility functions
    "check_ollama_installation",
    "create_model_selector",
    "download_model_with_progress",
    "estimate_model_performance",
    "get_model_info",
    "get_ollama_install_instructions",
    "get_recommended_models",
    "get_setup_status",
    "get_system_info",
    "install_ollama_automatically",
    "open_ollama_download_page",
    "recommend_model_for_use_case",
    "setup_local_ai_with_guidance",
] 