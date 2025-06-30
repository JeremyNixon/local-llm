#!/usr/bin/env python3
"""
Cross-Platform Local LLM Example

This example demonstrates how to use the Local LLM library across all platforms:
- Desktop/Server: Ollama
- Browser: WebLLM
- Android: Google AI Edge LLM Inference API
- iOS: Apple Foundation Models Framework

The example automatically detects the platform and initializes the appropriate client.
"""

import asyncio
import sys
import platform
from typing import Optional, Dict, Any

# Try to import all clients
try:
    from local_llm import (
        LocalLLM, 
        WebLLMClient, 
        create_web_llm_client,
        AndroidLLMClient,
        create_android_llm_client,
        IOSLLMClient,
        create_ios_llm_client,
        OllamaConnectionError,
        LocalLLMError
    )
except ImportError:
    print("❌ Local LLM library not found. Please install it first.")
    sys.exit(1)


class CrossPlatformLLMClient:
    """
    Cross-platform client that automatically detects the platform and
    initializes the appropriate LLM client.
    """
    
    def __init__(self):
        self.platform = self._detect_platform()
        self.client = None
        self.client_type = None
        self.initialized = False
    
    def _detect_platform(self) -> str:
        """Detect the current platform."""
        system = platform.system().lower()
        
        if system == "darwin":
            # Check if it's iOS (mobile) or macOS (desktop)
            try:
                import Foundation
                return "ios"
            except ImportError:
                return "desktop"  # macOS
        elif system == "linux":
            # Check if it's Android
            try:
                import android
                return "android"
            except ImportError:
                return "desktop"  # Linux
        elif system == "windows":
            return "desktop"
        else:
            return "unknown"
    
    async def initialize(self, preferred_platform: Optional[str] = None) -> bool:
        """
        Initialize the appropriate client for the detected platform.
        
        Args:
            preferred_platform: Override platform detection ("ollama", "webllm", "android", "ios")
        
        Returns:
            True if initialization was successful
        """
        platforms_to_try = []
        
        if preferred_platform:
            platforms_to_try = [preferred_platform]
        else:
            # Auto-detect based on platform
            if self.platform == "desktop":
                platforms_to_try = ["ollama", "webllm"]
            elif self.platform == "android":
                platforms_to_try = ["android", "webllm"]
            elif self.platform == "ios":
                platforms_to_try = ["ios", "webllm"]
            else:
                platforms_to_try = ["ollama", "webllm", "android", "ios"]
        
        for platform_type in platforms_to_try:
            if await self._try_platform(platform_type):
                return True
        
        return False
    
    async def _try_platform(self, platform_type: str) -> bool:
        """Try to initialize a specific platform client."""
        try:
            print(f"🔧 Trying {platform_type.upper()} client...")
            
            if platform_type == "ollama":
                return await self._try_ollama()
            elif platform_type == "webllm":
                return await self._try_webllm()
            elif platform_type == "android":
                return await self._try_android()
            elif platform_type == "ios":
                return await self._try_ios()
            else:
                print(f"❌ Unknown platform: {platform_type}")
                return False
                
        except Exception as e:
            print(f"❌ {platform_type.upper()} initialization failed: {e}")
            return False
    
    async def _try_ollama(self) -> bool:
        """Try to initialize Ollama client."""
        try:
            self.client = LocalLLM()
            # Test connection
            self.client._test_connection()
            
            self.client_type = "ollama"
            self.initialized = True
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
            self.client = await create_web_llm_client(
                model="Llama-3.1-8B-Instruct-q4f32_1-MLC"
            )
            
            self.client_type = "webllm"
            self.initialized = True
            print("✅ WebLLM client initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ WebLLM initialization failed: {e}")
            return False
    
    async def _try_android(self) -> bool:
        """Try to initialize Android client."""
        try:
            # For Android, we need a context
            context = None
            try:
                from android import mActivity
                context = mActivity
            except ImportError:
                print("❌ Android context not available")
                return False
            
            self.client = create_android_llm_client(
                model_path="/data/local/tmp/llm/model_version.task",
                context=context
            )
            
            self.client_type = "android"
            self.initialized = True
            print("✅ Android client initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Android initialization failed: {e}")
            return False
    
    async def _try_ios(self) -> bool:
        """Try to initialize iOS client."""
        try:
            self.client = create_ios_llm_client(
                instructions="You are a helpful AI assistant."
            )
            
            self.client_type = "ios"
            self.initialized = True
            print("✅ iOS client initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ iOS initialization failed: {e}")
            return False
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get information about the active client."""
        if not self.initialized or not self.client:
            return {"status": "not_initialized"}
        
        info = {
            "platform": self.platform,
            "client_type": self.client_type,
            "status": "active",
        }
        
        if self.client_type == "ollama":
            info["base_url"] = self.client.base_url
            info["timeout"] = self.client.timeout
        elif self.client_type == "webllm":
            info["model"] = self.client.model
        elif self.client_type == "android":
            info["model_path"] = self.client.model_path
        elif self.client_type == "ios":
            info["instructions"] = self.client.instructions
        
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
        if not self.initialized or not self.client:
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
        
        # Make the request based on client type
        if self.client_type == "ollama":
            return self.client.chat.completions.create(**request_params)
        elif self.client_type == "webllm":
            return await self.client.chat.completions.create(**request_params)
        elif self.client_type == "android":
            return self.client.chat.completions.create(**request_params)
        elif self.client_type == "ios":
            return await self.client.chat.completions.create(**request_params)
        else:
            raise LocalLLMError(f"Unknown client type: {self.client_type}")
    
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
        if not self.initialized or not self.client:
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
        
        # Make the request based on client type
        if self.client_type == "ollama":
            return self.client.completions.create(**request_params)
        elif self.client_type == "webllm":
            return await self.client.completions.create(**request_params)
        elif self.client_type == "android":
            return self.client.completions.create(**request_params)
        elif self.client_type == "ios":
            return await self.client.completions.create(**request_params)
        else:
            raise LocalLLMError(f"Unknown client type: {self.client_type}")
    
    async def list_models(self):
        """List available models."""
        if not self.initialized or not self.client:
            raise LocalLLMError("No client initialized. Call initialize() first.")
        
        if self.client_type == "ollama":
            return self.client.models.list()
        elif self.client_type == "webllm":
            return await self.client.models.list()
        elif self.client_type == "android":
            return self.client.models.list()
        elif self.client_type == "ios":
            return self.client.models.list()
        else:
            raise LocalLLMError(f"Unknown client type: {self.client_type}")


async def main():
    """Main example function."""
    print("🚀 Cross-Platform Local LLM Example")
    print("=" * 60)
    
    # Create cross-platform client
    client = CrossPlatformLLMClient()
    
    # Show platform detection
    print(f"\n🔍 Detected Platform: {client.platform.upper()}")
    
    # Initialize with automatic platform selection
    print("\n🔧 Initializing client...")
    success = await client.initialize()
    
    if not success:
        print("❌ Failed to initialize any client.")
        print("\nTroubleshooting:")
        print("1. For Ollama: Install Ollama from https://ollama.ai/ and start it")
        print("2. For WebLLM: Ensure you're in a browser environment with WebGPU support")
        print("3. For Android: Ensure you have the Google AI Edge dependencies")
        print("4. For iOS: Ensure you have the Foundation Models Framework")
        return
    
    # Show client information
    info = client.get_client_info()
    print(f"\n✅ Using {info['client_type'].upper()} client on {info['platform'].upper()}")
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
    print("-" * 60)
    
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


def demonstrate_platform_capabilities():
    """Demonstrate different platform capabilities."""
    print("\n🔍 Platform Capabilities")
    print("=" * 60)
    
    platforms = [
        ("Desktop/Server (Ollama)", "ollama", "Full OpenAI API compatibility, embeddings, model management"),
        ("Browser (WebLLM)", "webllm", "WebGPU acceleration, no installation required, privacy-first"),
        ("Android (Google AI Edge)", "android", "On-device inference, multimodal support, LoRA tuning"),
        ("iOS (Foundation Models)", "ios", "Apple Intelligence integration, system-level optimization"),
    ]
    
    for platform_name, platform_type, capabilities in platforms:
        print(f"\n📱 {platform_name}")
        print(f"   Type: {platform_type}")
        print(f"   Capabilities: {capabilities}")


if __name__ == "__main__":
    # Show platform capabilities
    demonstrate_platform_capabilities()
    
    # Run the main example
    asyncio.run(main()) 