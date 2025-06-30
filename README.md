# Private AI

A Python library that provides an OpenAI-like API for local Large Language Models across all platforms. **For developers** to include in their applications, enabling their **users** to run powerful AI models locally on their own machines.

## Overview

This library is designed for **application developers** who want to give their users the ability to run local LLMs across all platforms. Instead of sending data to cloud APIs, users can run models directly on their devices for privacy, speed, and cost savings.

### Supported Platforms

- **Desktop/Server**: Ollama (macOS, Linux, Windows)
- **Browser**: WebLLM (Chrome, Firefox, Safari with WebGPU)
- **Android**: Google AI Edge LLM Inference API
- **iOS**: Apple Foundation Models Framework

### Use Cases

- **Desktop Applications**: Add AI capabilities to desktop apps without cloud dependencies
- **Web Applications**: Provide local AI options alongside cloud services
- **Mobile Apps**: Enable offline AI processing on mobile devices
- **Development Tools**: Code completion, documentation generation, etc.
- **Content Creation**: Local text generation, summarization, translation
- **Data Analysis**: Local processing of sensitive data

## Features

- 🚀 **OpenAI-like API**: Familiar interface for developers already using OpenAI
- 🏠 **User's Local Execution**: Models run on the end user's machine for privacy
- 📦 **Automatic Setup**: Handles installation and model management for users
- 🔄 **Streaming Support**: Real-time streaming responses for better UX
- 🛠️ **Flexible Configuration**: Customize model parameters, system prompts, and more
- 📊 **Token Counting**: Built-in token counting for cost estimation
- 🎯 **Multiple Use Cases**: Chat, completion, embedding, and function calling support
- 🔧 **Developer Tools**: Easy integration, error handling, and user feedback
- 🌐 **Cross-Platform**: Works on desktop, browser, Android, and iOS
- 🤖 **Platform Optimization**: Each platform uses the best available inference engine

## Installation

### For Developers

```bash
pip install private-ai
```

For development and streaming support:
```bash
pip install private-ai[dev,streaming]
```

For browser support:
```bash
pip install private-ai[browser]
```

### For End Users

The library automatically handles setup for each platform:

1. **Desktop**: Guides users through Ollama installation
2. **Browser**: Works immediately with WebGPU acceleration
3. **Android**: Uses Google AI Edge LLM Inference API
4. **iOS**: Uses Apple Foundation Models Framework

## Quick Start for Developers

### Basic Integration (Cross-Platform)

```python
from private_ai import CrossPlatformLLMClient

# Automatically detects platform and initializes appropriate client
client = CrossPlatformLLMClient()
await client.initialize()

# Works the same on all platforms
response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
```

### Platform-Specific Usage

#### Desktop/Server (Ollama)

```python
from private_ai import LocalLLM

# Initialize the client (handles Ollama setup automatically)
llm = LocalLLM()

# Simple chat completion
response = llm.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

#### Browser (WebLLM)

```python
import asyncio
from private_ai import create_web_llm_client

async def browser_chat():
    # Initialize WebLLM client
    client = await create_web_llm_client(
        model="Llama-3.1-8B-Instruct-q4f32_1-MLC"
    )
    
    # Chat completion (works entirely in browser)
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": "Hello from the browser!"}
        ]
    )
    
    print(response.choices[0].message.content)

# Run in browser environment
asyncio.run(browser_chat())
```

#### Android (Google AI Edge)

```python
from private_ai import create_android_llm_client

# Initialize Android client
client = create_android_llm_client(
    model_path="/data/local/tmp/llm/model_version.task",
    context=android_context,  # Android context required
    enable_vision=True,  # Enable multimodal support
    max_num_images=1
)

# Chat completion with vision support
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Describe this image"},
        {"role": "user", "content": image_data}  # MPImage object
    ]
)

print(response.choices[0].message.content)
```

#### iOS (Apple Foundation Models)

```python
from private_ai import create_ios_llm_client

# Initialize iOS client
client = create_ios_llm_client(
    instructions="You are a helpful AI assistant.",
    temperature=0.8,
    max_tokens=512
)

# Chat completion
response = await client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Hello from iOS!"}
    ]
)

print(response.choices[0].message.content)
```

### Unified Client (Automatic Fallback)

For applications that should work across all platforms:

```python
import asyncio
from private_ai import CrossPlatformLLMClient

async def unified_example():
    client = CrossPlatformLLMClient()
    
    # Automatically chooses the best available backend
    success = await client.initialize(preferred_platform="auto")
    
    if success:
        # Works the same regardless of platform
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
    else:
        print("No local AI backend available")

asyncio.run(unified_example())
```

### User-Friendly Setup

```python
from private_ai import CrossPlatformLLMClient, OllamaConnectionError

async def setup_local_ai():
    """Guide users through local AI setup on any platform."""
    client = CrossPlatformLLMClient()
    
    try:
        success = await client.initialize()
        if success:
            info = client.get_client_info()
            print(f"✅ Private AI ready using {info['client_type']} on {info['platform']}!")
            return client
        else:
            print("❌ No local AI backend available")
            return None
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None

# In your application
client = await setup_local_ai()
if client:
    # Use local AI
    response = await client.chat_completion(...)
else:
    # Fall back to cloud or show setup instructions
    pass
```

### Streaming Responses

```python
from private_ai import CrossPlatformLLMClient

client = CrossPlatformLLMClient()
await client.initialize()

# Streaming chat completion (works on all platforms)
stream = await client.chat_completion(
    messages=[
        {"role": "user", "content": "Tell me a story about a robot."}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Text Completion

```python
from private_ai import CrossPlatformLLMClient

client = CrossPlatformLLMClient()
await client.initialize()

# Text completion
response = await client.text_completion(
    prompt="The future of artificial intelligence is",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Model Management for Users

```python
from private_ai import CrossPlatformLLMClient

client = CrossPlatformLLMClient()
await client.initialize()

# List available models (platform-specific)
models = await client.list_models()
print("Available models:", [model.id for model in models.data])

# Download a model (Ollama only)
if client.client_type == "ollama":
    client.client.models.pull("codellama:7b")
    print("✅ CodeLlama is ready to use!")
```

## Advanced Usage

### Custom Configuration

#### Desktop/Server (Ollama)

```python
from private_ai import LocalLLM

# Configure with custom settings
llm = LocalLLM(
    base_url="http://localhost:11434",  # Custom Ollama URL
    default_model="llama2:13b",         # Default model for your app
    timeout=30,                         # Request timeout
    max_retries=3                       # Retry attempts
)
```

#### Browser (WebLLM)

```python
from private_ai import create_web_llm_client

client = await create_web_llm_client(
    model="Llama-3.1-8B-Instruct-q4f32_1-MLC",
    init_progress_callback=lambda p: print(f"Loading: {p.progress}%"),
    app_config={"custom_config": "value"},
    chat_opts={"temperature": 0.7}
)
```

#### Android (Google AI Edge)

```python
from private_ai import create_android_llm_client

client = create_android_llm_client(
    model_path="/data/local/tmp/llm/gemma-3.task",
    max_tokens=1000,
    top_k=64,
    temperature=0.8,
    random_seed=42,
    lora_path="/data/local/tmp/llm/lora.task",  # LoRA support
    enable_vision=True,
    max_num_images=1
)
```

#### iOS (Apple Foundation Models)

```python
from private_ai import create_ios_llm_client

client = create_ios_llm_client(
    instructions="You are a helpful AI assistant specialized in coding.",
    temperature=0.8,
    max_tokens=512,
    use_case="code_generation"
)
```

### Function Calling

```python
from private_ai import CrossPlatformLLMClient

client = CrossPlatformLLMClient()
await client.initialize()

# Define functions for your application
def get_user_preferences(user_id: str) -> dict:
    """Get user preferences from your app's database."""
    return {"theme": "dark", "language": "en"}

# Create function calling request (Ollama and WebLLM support)
response = await client.chat_completion(
    messages=[
        {"role": "user", "content": "What are my current preferences?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_user_preferences",
                "description": "Get user preferences",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"}
                    },
                    "required": ["user_id"]
                }
            }
        }
    ]
)
```

### Embeddings

```python
from private_ai import CrossPlatformLLMClient

client = CrossPlatformLLMClient()
await client.initialize()

# Create embeddings for local search (Ollama only)
if client.client_type == "ollama":
    response = client.client.embeddings.create(
        model="llama2",
        input=["Hello world", "How are you?"]
    )

    for embedding in response.data:
        print(f"Embedding: {embedding.embedding[:5]}...")  # Show first 5 dimensions
```

## Application Integration Examples

### Desktop Application

```python
import tkinter as tk
from private_ai import CrossPlatformLLMClient

class AIAssistantApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Private AI Assistant")
        
        # Initialize cross-platform client
        self.client = CrossPlatformLLMClient()
        self.setup_ui()
    
    async def setup_ui(self):
        # Initialize client
        success = await self.client.initialize()
        if not success:
            self.show_setup_instructions()
            return
        
        # Create your app's UI here
        self.text_input = tk.Entry(self.root)
        self.text_input.pack()
        
        self.send_button = tk.Button(self.root, text="Ask AI", command=self.ask_ai)
        self.send_button.pack()
        
        self.response_text = tk.Text(self.root)
        self.response_text.pack()
    
    async def ask_ai(self):
        user_input = self.text_input.get()
        response = await self.client.chat_completion(
            messages=[{"role": "user", "content": user_input}]
        )
        self.response_text.insert(tk.END, f"AI: {response.choices[0].message.content}\n")
    
    def show_setup_instructions(self):
        # Show instructions for setting up local AI
        pass

if __name__ == "__main__":
    app = AIAssistantApp()
    app.root.mainloop()
```

### Web Application (Flask)

```python
from flask import Flask, request, jsonify
from private_ai import CrossPlatformLLMClient

app = Flask(__name__)

# Initialize cross-platform client
client = CrossPlatformLLMClient()

@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    message = data.get('message', '')
    
    try:
        # Initialize if not already done
        if not client.initialized:
            await client.initialize()
        
        response = await client.chat_completion(
            messages=[{"role": "user", "content": message}]
        )
        
        return jsonify({
            'response': response.choices[0].message.content,
            'platform': client.get_client_info()['client_type']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Mobile Application (React Native)

```javascript
import { CrossPlatformLLMClient } from 'private-ai';

class MobileAI {
  constructor() {
    this.client = new CrossPlatformLLMClient();
  }

  async initialize() {
    try {
      await this.client.initialize();
      console.log('Private AI ready!');
      return true;
    } catch (error) {
      console.error('Failed to initialize private AI:', error);
      return false;
    }
  }

  async chat(message) {
    try {
      const response = await this.client.chat_completion({
        messages: [{ role: 'user', content: message }]
      });
      return response.choices[0].message.content;
    } catch (error) {
      console.error('Chat error:', error);
      throw error;
    }
  }
}

// Usage
const ai = new MobileAI();
await ai.initialize();
const response = await ai.chat("Hello from mobile!");
```

### Browser Application (HTML/JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Private AI Chat</title>
</head>
<body>
    <div id="chat-container"></div>
    <input type="text" id="message-input" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>

    <script type="module">
        import { CrossPlatformLLMClient } from 'https://esm.run/private-ai';
        
        let client = null;
        
        async function initialize() {
            try {
                client = new CrossPlatformLLMClient();
                await client.initialize();
                console.log('Private AI initialized!');
            } catch (error) {
                console.error('Failed to initialize:', error);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value;
            
            if (client && message) {
                const response = await client.chat_completion({
                    messages: [{ role: 'user', content: message }]
                });
                
                const chatContainer = document.getElementById('chat-container');
                chatContainer.innerHTML += `<p><strong>AI:</strong> ${response.choices[0].message.content}</p>`;
                
                input.value = '';
            }
        }
        
        // Initialize on page load
        initialize();
    </script>
</body>
</html>
```

### Code Generation Tool

```python
from private_ai import CrossPlatformLLMClient

class CodeGenerator:
    def __init__(self):
        self.client = CrossPlatformLLMClient()
    
    async def setup(self):
        """Set up the local AI for code generation."""
        await self.client.initialize()
    
    async def generate_function(self, description: str, language: str = "python") -> str:
        """Generate code based on description."""
        response = await self.client.chat_completion(
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert {language} programmer. Generate only code, no explanations."
                },
                {"role": "user", "content": f"Create a function that: {description}"}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        return response.choices[0].message.content
    
    async def explain_code(self, code: str) -> str:
        """Explain what code does."""
        response = await self.client.chat_completion(
            messages=[
                {"role": "user", "content": f"Explain what this code does:\n\n{code}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content

# Usage
generator = CodeGenerator()
await generator.setup()
function_code = await generator.generate_function("calculates fibonacci numbers")
print(function_code)
```

## Available Models

### Desktop/Server (Ollama)

The library works with any model available in Ollama. Popular models include:

- `llama2` - Meta's Llama 2 (good general purpose)
- `llama2:13b` - Llama 2 13B parameter version (better quality)
- `codellama` - Code-focused Llama variant (excellent for programming)
- `mistral` - Mistral AI's 7B model (fast and efficient)
- `dolphin-llama3` - Dolphin-optimized Llama 3 (helpful assistant)
- `phi` - Microsoft's Phi models (small and fast)
- `gemma` - Google's Gemma models (good balance)

### Browser (WebLLM)

For browser-based usage, WebLLM supports:

- `Llama-3.1-8B-Instruct-q4f32_1-MLC` - Llama 3.1 8B (recommended)
- `Phi-3-mini-4k-instruct-q4f32_1-MLC` - Phi 3 Mini (fast)
- `Mistral-7B-Instruct-v0.3-q4f32_1-MLC` - Mistral 7B (balanced)
- `Qwen2.5-7B-Instruct-q4f32_1-MLC` - Qwen 2.5 7B (multilingual)

### Android (Google AI Edge)

For Android devices, Google AI Edge supports:

- `gemma-3-1b` - Gemma 3 1B model
- `gemma-3n-e2b` - Gemma 3N 2B model (multimodal)
- `gemma-3n-e4b` - Gemma 3N 4B model (multimodal)
- Custom models in `.task` format

### iOS (Apple Foundation Models)

For iOS devices, Apple Foundation Models Framework provides:

- `system-foundation-model` - Apple's system language model

## Platform-Specific Features

### Desktop/Server (Ollama)

- ✅ Full OpenAI API compatibility
- ✅ Embeddings support
- ✅ Model management (download, list, delete)
- ✅ Function calling
- ✅ Streaming responses
- ✅ Custom model support
- ✅ LoRA fine-tuning

### Browser (WebLLM)

- ✅ WebGPU acceleration
- ✅ No installation required
- ✅ Privacy-first (no data leaves browser)
- ✅ Streaming responses
- ✅ Function calling (WIP)
- ✅ JSON mode generation
- ✅ Service worker support

### Android (Google AI Edge)

- ✅ On-device inference
- ✅ Multimodal support (text + images)
- ✅ LoRA fine-tuning
- ✅ GPU acceleration
- ✅ Optimized for mobile devices
- ✅ Custom model support

### iOS (Apple Foundation Models)

- ✅ Apple Intelligence integration
- ✅ System-level optimization
- ✅ Privacy-focused
- ✅ Battery-optimized
- ✅ Seamless iOS integration
- ✅ Guided generation support

## User Experience Considerations

### First-Time Setup

```python
from private_ai import CrossPlatformLLMClient

async def ensure_local_ai_ready():
    """Check if local AI is available and guide user through setup."""
    client = CrossPlatformLLMClient()
    
    try:
        success = await client.initialize()
        if success:
            info = client.get_client_info()
            print(f"✅ Private AI ready using {info['client_type']}!")
            return client
    except Exception as e:
        print(f"❌ Private AI setup failed: {e}")
    
    # Show platform-specific setup instructions
    platform = client.platform
    if platform == "desktop":
        print("🔧 To use private AI, install Ollama from https://ollama.ai/")
    elif platform == "android":
        print("🔧 To use private AI, ensure Google AI Edge is available")
    elif platform == "ios":
        print("🔧 To use private AI, enable Apple Intelligence in Settings")
    else:
        print("🔧 To use private AI, ensure WebGPU is supported in your browser")
    
    return None

# In your application
client = await ensure_local_ai_ready()
if client:
    # Private AI is ready
    pass
else:
    # Show alternative options or disable features
    pass
```

### Model Download Progress

```python
from private_ai import CrossPlatformLLMClient

async def download_model_with_progress(model_name: str):
    """Download a model with progress feedback."""
    client = CrossPlatformLLMClient()
    await client.initialize()
    
    if client.client_type == "ollama":
        print(f"📥 Downloading {model_name}...")
        print("This may take several minutes depending on your internet connection.")
        
        try:
            result = client.client.models.pull(model_name)
            print(f"✅ {model_name} downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            return False
    else:
        print(f"📋 Model {model_name} is available on this platform")
        return True

# Usage
if await download_model_with_progress("codellama:7b"):
    # Model is ready to use
    pass
```

## Error Handling

The library provides comprehensive error handling for different scenarios:

```python
from private_ai import CrossPlatformLLMClient, LocalLLMError, OllamaConnectionError

client = CrossPlatformLLMClient()

try:
    await client.initialize()
    
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}]
    )
    
except OllamaConnectionError:
    print("Ollama is not running. Please start Ollama and try again.")
except LocalLLMError as e:
    print(f"Private AI error: {e}")
    # Handle gracefully or fall back to cloud
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Performance Considerations

### Platform-Specific Optimization

```python
from private_ai import CrossPlatformLLMClient

async def get_optimized_client():
    """Get a client optimized for the current platform."""
    client = CrossPlatformLLMClient()
    
    # Platform-specific optimizations
    if client.platform == "desktop":
        # Use larger models on desktop
        await client.initialize(preferred_platform="ollama")
    elif client.platform == "android":
        # Use mobile-optimized models
        await client.initialize(preferred_platform="android")
    elif client.platform == "ios":
        # Use Apple's optimized models
        await client.initialize(preferred_platform="ios")
    else:
        # Use WebLLM for browser
        await client.initialize(preferred_platform="webllm")
    
    return client

# Usage
client = await get_optimized_client()
```

### Caching and Optimization

```python
from private_ai import CrossPlatformLLMClient

class OptimizedPrivateAI:
    def __init__(self):
        self.client = None
        self.response_cache = {}
    
    async def setup(self):
        """Set up the client."""
        self.client = CrossPlatformLLMClient()
        await self.client.initialize()
    
    async def get_cached_response(self, prompt: str) -> str:
        """Get cached response or generate new one."""
        cache_key = f"{self.client.client_type}:{hash(prompt)}"
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        response = await self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        self.response_cache[cache_key] = result
        return result
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/jeremynixon/private-ai/issues)
- Documentation: [Full API documentation](https://private-ai.readthedocs.io)
- Discord: [Join our community](https://discord.gg/private-ai)

## Roadmap

- [ ] Multi-modal support (images, audio)
- [ ] Model fine-tuning capabilities
- [ ] Advanced caching and optimization
- [ ] Web UI for model management
- [ ] Integration with other local LLM backends
- [ ] Batch processing capabilities
- [ ] Automatic model optimization for user's hardware
- [ ] Cloud fallback when local models are unavailable 