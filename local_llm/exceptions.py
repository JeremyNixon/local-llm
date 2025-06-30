"""
Custom exceptions for the Local LLM library.
"""


class LocalLLMError(Exception):
    """Base exception for all Local LLM errors."""
    pass


class OllamaConnectionError(LocalLLMError):
    """Raised when unable to connect to Ollama service."""
    pass


class ModelNotFoundError(LocalLLMError):
    """Raised when a requested model is not found."""
    pass


class InvalidParameterError(LocalLLMError):
    """Raised when invalid parameters are provided."""
    pass


class RateLimitError(LocalLLMError):
    """Raised when rate limits are exceeded."""
    pass


class AuthenticationError(LocalLLMError):
    """Raised when authentication fails."""
    pass


class APIError(LocalLLMError):
    """Raised when the Ollama API returns an error."""
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class TimeoutError(LocalLLMError):
    """Raised when a request times out."""
    pass 