#!/usr/bin/env python3
"""
Unified Client Example

This example demonstrates how to use both the Ollama-based LocalLLM client
and the WebLLM browser client with automatic fallback based on the environment.
"""

import asyncio
import sys
from typing import Optional, Dict, Any

# Try to import both clients
try:
    from local_llm import LocalLLM, WebLLMClient, create_web_llm_client
    from local_llm.exceptions import OllamaConnectionError, LocalLLMError
except ImportError:
    print("❌ Local LLM library not found. Please install it first.")
    sys.exit(1)


class UnifiedLLMClient:
    """
    Unified client that automatically chooses between Ollama and WebLLM
    based on the environment and availability.
    """
    
    def __init__(self):
        self.ollama_client = None
        self.webllm_client = None
        self.active_client = None
        self.client_type = None
    
    async def initialize(self, preferred_client: str = "auto") -> bool:
        """
        Initialize the appropriate client.
        
        Args:
            preferred_client: "ollama", "webllm", or "auto"
        
        Returns:
            True if initialization was successful
        """
        if preferred_client == "ollama" or preferred_client == "auto":
            if await self._try_ollama():
                return True
        
        if preferred_client == "webllm" or (preferred_client == "auto" and not self.active_client):
            if await self._try_webllm():
                return True
        
        return False
    
    async def _try_ollama(self) -> bool:
        """Try to initialize Ollama client."""
        try:
            print("🔧 Trying Ollama client...")
            self.ollama_client = LocalLLM()
            # Test connection
            self.ollama_client._test_connection()
            
            self.active_client = self.ollama_client
            self.client_type = "ollama"
            print("✅ Ollama client initialized successfully!")
            return True
            
        except OllamaConnectionError:
            print("❌ Ollama not available (not installed or not running)")
            return False
        except Exception as e:
            print(f"❌ Ollama initialization failed: {e}")
            return False
    
    async def _try_webllm(self) -> bool:
        """Try to initialize WebLLM client."""
        try:
            print("🌐 Trying WebLLM client...")
            self.webllm_client = await create_web_llm_client(
                model="Llama-3.1-8B-Instruct-q4f32_1-MLC"
            )
            
            self.active_client = self.webllm_client
            self.client_type = "webllm"
            print("✅ WebLLM client initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ WebLLM initialization failed: {e}")
            return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the active client."""
        if not self.active_client:
            return {"status": "not_initialized"}
        
        info = {
            "type": self.client_type,
            "status": "active",
        }
        
        if self.client_type == "ollama":
            info["base_url"] = self.ollama_client.base_url
            info["timeout"] = self.ollama_client.timeout
        elif self.client_type == "webllm":
            info["model"] = self.webllm_client.model
        
        return info
    
    async def chat_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False,
        **kwargs
    ):
        """Create a chat completion using the active client."""
        if not self.active_client:
            raise LocalLLMError("No client initialized. Call initialize() first.")
        
        # Prepare request parameters
        request_params = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        # Add model if specified (only for Ollama)
        if model and self.client_type == "ollama":
            request_params["model"] = model
        
        # Make the request
        if self.client_type == "ollama":
            return self.active_client.chat.completions.create(**request_params)
        else:  # webllm
            return await self.active_client.chat.completions.create(**request_params)
    
    async def text_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False,
        **kwargs
    ):
        """Create a text completion using the active client."""
        if not self.active_client:
            raise LocalLLMError("No client initialized. Call initialize() first.")
        
        # Prepare request parameters
        request_params = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        # Add model if specified (only for Ollama)
        if model and self.client_type == "ollama":
            request_params["model"] = model
        
        # Make the request
        if self.client_type == "ollama":
            return self.active_client.completions.create(**request_params)
        else:  # webllm
            return await self.active_client.completions.create(**request_params)
    
    async def list_models(self):
        """List available models."""
        if not self.active_client:
            raise LocalLLMError("No client initialized. Call initialize() first.")
        
        if self.client_type == "ollama":
            return self.active_client.models.list()
        else:  # webllm
            return await self.active_client.models.list()


async def main():
    """Main example function."""
    print("🚀 Unified LLM Client Example")
    print("=" * 50)
    
    # Create unified client
    client = UnifiedLLMClient()
    
    # Initialize with automatic client selection
    print("\n🔧 Initializing client...")
    success = await client.initialize(preferred_client="auto")
    
    if not success:
        print("❌ Failed to initialize any client.")
        print("\nTroubleshooting:")
        print("1. For Ollama: Install Ollama from https://ollama.ai/ and start it")
        print("2. For WebLLM: Ensure you're in a browser environment with WebGPU support")
        return
    
    # Show client information
    info = client.get_client_info()
    print(f"\n✅ Using {info['type'].upper()} client")
    print(f"Status: {info['status']}")
    
    # List available models
    print("\n📋 Available Models:")
    try:
        models = await client.list_models()
        for model in models.data[:5]:  # Show first 5 models
            print(f"  • {model.id}")
        if len(models.data) > 5:
            print(f"  ... and {len(models.data) - 5} more")
    except Exception as e:
        print(f"  Error listing models: {e}")
    
    # Interactive chat loop
    print("\n💬 Interactive Chat (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("AI: ", end="", flush=True)
            
            # Generate response
            response = await client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract and print response
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                print(content)
            else:
                print("Error: No response generated")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def demonstrate_client_selection():
    """Demonstrate different client selection strategies."""
    print("\n🔍 Client Selection Strategies")
    print("=" * 50)
    
    strategies = [
        ("Auto (prefer Ollama)", "auto"),
        ("Ollama only", "ollama"),
        ("WebLLM only", "webllm"),
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n📋 Strategy: {strategy_name}")
        print("-" * 30)
        
        async def test_strategy():
            client = UnifiedLLMClient()
            success = await client.initialize(preferred_client=strategy)
            
            if success:
                info = client.get_client_info()
                print(f"✅ Success: Using {info['type']} client")
            else:
                print("❌ Failed to initialize any client")
        
        # Run the test
        asyncio.run(test_strategy())


if __name__ == "__main__":
    # Check if we're in a browser environment
    try:
        import browser
        print("🌐 Browser environment detected")
        # In browser, we might want to prefer WebLLM
        asyncio.run(main())
    except ImportError:
        print("💻 Desktop environment detected")
        # In desktop, we might want to prefer Ollama
        asyncio.run(main())
    
    # Demonstrate client selection strategies
    demonstrate_client_selection() 