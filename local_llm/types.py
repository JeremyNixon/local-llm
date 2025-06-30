"""
Type definitions for the Local LLM library.

These types are designed to be compatible with OpenAI's API structure
while working with local Ollama models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="The name of the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made by the model")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call")


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion."""
    index: int = Field(..., description="The index of the choice")
    message: ChatMessage = Field(..., description="The message content")
    finish_reason: Optional[str] = Field(None, description="The reason the completion finished")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class ChatCompletion(BaseModel):
    """A chat completion response."""
    id: str = Field(..., description="The completion ID")
    object: str = Field("chat.completion", description="The object type")
    created: int = Field(..., description="The creation timestamp")
    model: str = Field(..., description="The model used")
    choices: List[ChatCompletionChoice] = Field(..., description="The completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class ChatCompletionChunk(BaseModel):
    """A streaming chat completion chunk."""
    id: str = Field(..., description="The completion ID")
    object: str = Field("chat.completion.chunk", description="The object type")
    created: int = Field(..., description="The creation timestamp")
    model: str = Field(..., description="The model used")
    choices: List[Dict[str, Any]] = Field(..., description="The completion choices")


class CompletionChoice(BaseModel):
    """A choice in a text completion."""
    index: int = Field(..., description="The index of the choice")
    text: str = Field(..., description="The generated text")
    finish_reason: Optional[str] = Field(None, description="The reason the completion finished")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities")


class Completion(BaseModel):
    """A text completion response."""
    id: str = Field(..., description="The completion ID")
    object: str = Field("text_completion", description="The object type")
    created: int = Field(..., description="The creation timestamp")
    model: str = Field(..., description="The model used")
    choices: List[CompletionChoice] = Field(..., description="The completion choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class CompletionChunk(BaseModel):
    """A streaming text completion chunk."""
    id: str = Field(..., description="The completion ID")
    object: str = Field("text_completion.chunk", description="The object type")
    created: int = Field(..., description="The creation timestamp")
    model: str = Field(..., description="The model used")
    choices: List[Dict[str, Any]] = Field(..., description="The completion choices")


class EmbeddingData(BaseModel):
    """Embedding data."""
    object: str = Field("embedding", description="The object type")
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., description="The index of the embedding")


class Embedding(BaseModel):
    """An embedding response."""
    object: str = Field("list", description="The object type")
    data: List[EmbeddingData] = Field(..., description="The embedding data")
    model: str = Field(..., description="The model used")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class Model(BaseModel):
    """A model."""
    id: str = Field(..., description="The model ID")
    object: str = Field("model", description="The object type")
    created: int = Field(..., description="The creation timestamp")
    owned_by: str = Field("ollama", description="The owner of the model")
    size: Optional[int] = Field(None, description="The model size in bytes")
    digest: Optional[str] = Field(None, description="The model digest")


class ModelList(BaseModel):
    """A list of models."""
    object: str = Field("list", description="The object type")
    data: List[Model] = Field(..., description="The list of models")


class Tool(BaseModel):
    """A tool definition for function calling."""
    type: str = Field(..., description="The tool type")
    function: Dict[str, Any] = Field(..., description="The function definition")


class ChatCompletionRequest(BaseModel):
    """Request parameters for chat completion."""
    model: str = Field(..., description="The model to use")
    messages: List[ChatMessage] = Field(..., description="The messages to complete")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=128, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    tools: Optional[List[Tool]] = Field(None, description="Tools to use")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice")


class CompletionRequest(BaseModel):
    """Request parameters for text completion."""
    model: str = Field(..., description="The model to use")
    prompt: Union[str, List[str]] = Field(..., description="The prompt to complete")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(1, ge=1, le=128, description="Number of completions to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")


class EmbeddingRequest(BaseModel):
    """Request parameters for embedding."""
    model: str = Field(..., description="The model to use")
    input: Union[str, List[str]] = Field(..., description="The input to embed")
    user: Optional[str] = Field(None, description="User identifier") 