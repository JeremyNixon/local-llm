"""
Basic tests for the Local LLM library.
"""

import pytest
from unittest.mock import Mock, patch
import json

from local_llm import LocalLLM, LocalLLMError, OllamaConnectionError
from local_llm.types import ChatCompletion, ChatMessage


class TestLocalLLM:
    """Test the LocalLLM client."""
    
    def test_initialization(self):
        """Test LocalLLM initialization."""
        with patch('local_llm.client.requests.get') as mock_get:
            # Mock successful connection test
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}
            
            llm = LocalLLM()
            assert llm.base_url == "http://localhost:11434"
            assert llm.timeout == 30
            assert llm.max_retries == 3
    
    def test_connection_error(self):
        """Test connection error handling."""
        with patch('local_llm.client.requests.get') as mock_get:
            # Mock connection failure
            mock_get.side_effect = Exception("Connection failed")
            
            with pytest.raises(OllamaConnectionError):
                LocalLLM()
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        with patch('local_llm.client.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}
            
            llm = LocalLLM(
                base_url="http://custom:11434",
                timeout=60,
                max_retries=5
            )
            
            assert llm.base_url == "http://custom:11434"
            assert llm.timeout == 60
            assert llm.max_retries == 5


class TestChatCompletions:
    """Test chat completion functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LocalLLM instance."""
        with patch('local_llm.client.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}
            
            llm = LocalLLM()
            
            # Mock the _make_request method
            with patch.object(llm, '_make_request') as mock_request:
                yield llm, mock_request
    
    def test_chat_completion(self, mock_llm):
        """Test basic chat completion."""
        llm, mock_request = mock_llm
        
        # Mock successful response
        mock_request.return_value = {
            "message": {"content": "Hello! How can I help you?"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        
        response = llm.chat.completions.create(
            model="llama2",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert isinstance(response, ChatCompletion)
        assert response.model == "llama2"
        assert response.choices[0].message.content == "Hello! How can I help you?"
        assert response.usage.total_tokens == 15
    
    def test_chat_completion_with_parameters(self, mock_llm):
        """Test chat completion with various parameters."""
        llm, mock_request = mock_llm
        
        mock_request.return_value = {
            "message": {"content": "Test response"},
            "done": True
        }
        
        response = llm.chat.completions.create(
            model="llama2",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9
        )
        
        # Verify the request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        request_data = call_args[0][2]  # The data parameter
        
        assert request_data["model"] == "llama2"
        assert request_data["temperature"] == 0.5
        assert request_data["max_tokens"] == 100
        assert request_data["top_p"] == 0.9


class TestModels:
    """Test model management functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LocalLLM instance."""
        with patch('local_llm.client.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}
            
            llm = LocalLLM()
            
            with patch.object(llm, '_make_request') as mock_request:
                yield llm, mock_request
    
    def test_list_models(self, mock_llm):
        """Test listing models."""
        llm, mock_request = mock_llm
        
        mock_request.return_value = {
            "models": [
                {"name": "llama2", "size": 4000000000, "digest": "abc123"},
                {"name": "codellama", "size": 4000000000, "digest": "def456"}
            ]
        }
        
        models = llm.models.list()
        
        assert len(models.data) == 2
        assert models.data[0].id == "llama2"
        assert models.data[1].id == "codellama"
    
    def test_pull_model(self, mock_llm):
        """Test pulling a model."""
        llm, mock_request = mock_llm
        
        mock_request.return_value = {"status": "success"}
        
        result = llm.models.pull("llama2")
        
        assert result["status"] == "success"
        mock_request.assert_called_once_with("POST", "/api/pull", {"name": "llama2"})


class TestExceptions:
    """Test exception handling."""
    
    def test_local_llm_error(self):
        """Test LocalLLMError."""
        error = LocalLLMError("Test error")
        assert str(error) == "Test error"
    
    def test_ollama_connection_error(self):
        """Test OllamaConnectionError."""
        error = OllamaConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, LocalLLMError)


if __name__ == "__main__":
    pytest.main([__file__]) 