# Local LLM

A Python library that provides an OpenAI-like API for local Large Language Models using Ollama. **For developers** to include in their applications, enabling their **users** to run powerful AI models locally on their own machines.

## Overview

This library is designed for **application developers** who want to give their users the ability to run local LLMs. Instead of sending data to cloud APIs, users can run models directly on their devices for privacy, speed, and cost savings.

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
- 📦 **Automatic Setup**: Handles Ollama installation and model management for users
- 🔄 **Streaming Support**: Real-time streaming responses for better UX
- 🛠️ **Flexible Configuration**: Customize model parameters, system prompts, and more
- 📊 **Token Counting**: Built-in token counting for cost estimation
- 🎯 **Multiple Use Cases**: Chat, completion, embedding, and function calling support
- 🔧 **Developer Tools**: Easy integration, error handling, and user feedback

## Installation

### For Developers

```bash
pip install local-llm
```

For development and streaming support:
```bash
pip install local-llm[dev,streaming]
```

### For End Users

The library automatically handles Ollama installation and setup. Users just need to:

1. Install your application that includes this library
2. The library will guide them through Ollama installation if needed
3. Models are downloaded automatically when first used

## Quick Start for Developers

### Basic Integration

```python
from local_llm import LocalLLM

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

### User-Friendly Setup

```python
from local_llm import LocalLLM, OllamaConnectionError

def setup_local_ai():
    """Guide users through local AI setup."""
    try:
        llm = LocalLLM()
        print("✅ Local AI is ready!")
        return llm
    except OllamaConnectionError:
        print("🔧 Setting up local AI for the first time...")
        print("Please install Ollama from https://ollama.ai/")
        print("After installation, restart this application.")
        return None

# In your application
llm = setup_local_ai()
if llm:
    # Use local AI
    response = llm.chat.completions.create(...)
else:
    # Fall back to cloud or show setup instructions
    pass
```

### Streaming Responses

```python
from local_llm import LocalLLM

llm = LocalLLM()

# Streaming chat completion
stream = llm.chat.completions.create(
    model="llama2",
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
from local_llm import LocalLLM

llm = LocalLLM()

# Text completion
response = llm.completions.create(
    model="llama2",
    prompt="The future of artificial intelligence is",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

### Model Management for Users

```python
from local_llm import LocalLLM

llm = LocalLLM()

# List available models
models = llm.models.list()
print("Available models:", [model.id for model in models.data])

# Download a model for the user
print("Downloading CodeLlama for code generation...")
llm.models.pull("codellama:7b")
print("✅ CodeLlama is ready to use!")
```

## Advanced Usage

### Custom Configuration

```python
from local_llm import LocalLLM

# Configure with custom settings
llm = LocalLLM(
    base_url="http://localhost:11434",  # Custom Ollama URL
    default_model="llama2:13b",         # Default model for your app
    timeout=30,                         # Request timeout
    max_retries=3                       # Retry attempts
)
```

### Function Calling

```python
from local_llm import LocalLLM

llm = LocalLLM()

# Define functions for your application
def get_user_preferences(user_id: str) -> dict:
    """Get user preferences from your app's database."""
    return {"theme": "dark", "language": "en"}

# Create function calling request
response = llm.chat.completions.create(
    model="llama2",
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
from local_llm import LocalLLM

llm = LocalLLM()

# Create embeddings for local search
response = llm.embeddings.create(
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
from local_llm import LocalLLM, OllamaConnectionError

class AIAssistantApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Local AI Assistant")
        
        # Try to initialize local AI
        try:
            self.llm = LocalLLM()
            self.setup_ui()
        except OllamaConnectionError:
            self.show_setup_instructions()
    
    def setup_ui(self):
        # Create your app's UI here
        self.text_input = tk.Entry(self.root)
        self.text_input.pack()
        
        self.send_button = tk.Button(self.root, text="Ask AI", command=self.ask_ai)
        self.send_button.pack()
        
        self.response_text = tk.Text(self.root)
        self.response_text.pack()
    
    def ask_ai(self):
        user_input = self.text_input.get()
        response = self.llm.chat.completions.create(
            model="llama2",
            messages=[{"role": "user", "content": user_input}]
        )
        self.response_text.insert(tk.END, f"AI: {response.choices[0].message.content}\n")
    
    def show_setup_instructions(self):
        # Show instructions for installing Ollama
        pass

if __name__ == "__main__":
    app = AIAssistantApp()
    app.root.mainloop()
```

### Web Application (Flask)

```python
from flask import Flask, request, jsonify
from local_llm import LocalLLM, OllamaConnectionError

app = Flask(__name__)

# Initialize local AI
try:
    llm = LocalLLM()
    local_ai_available = True
except OllamaConnectionError:
    local_ai_available = False

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    if not local_ai_available:
        return jsonify({
            'error': 'Local AI not available. Please install Ollama.',
            'setup_url': 'https://ollama.ai/'
        }), 400
    
    try:
        response = llm.chat.completions.create(
            model="llama2",
            messages=[{"role": "user", "content": message}]
        )
        
        return jsonify({
            'response': response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Code Generation Tool

```python
from local_llm import LocalLLM

class CodeGenerator:
    def __init__(self):
        self.llm = LocalLLM()
    
    def generate_function(self, description: str, language: str = "python") -> str:
        """Generate code based on description."""
        response = self.llm.chat.completions.create(
            model="codellama",
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
    
    def explain_code(self, code: str) -> str:
        """Explain what code does."""
        response = self.llm.chat.completions.create(
            model="codellama",
            messages=[
                {"role": "user", "content": f"Explain what this code does:\n\n{code}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content

# Usage
generator = CodeGenerator()
function_code = generator.generate_function("calculates fibonacci numbers")
print(function_code)
```

## Available Models

The library works with any model available in Ollama. Popular models include:

- `llama2` - Meta's Llama 2 (good general purpose)
- `llama2:13b` - Llama 2 13B parameter version (better quality)
- `codellama` - Code-focused Llama variant (excellent for programming)
- `mistral` - Mistral AI's 7B model (fast and efficient)
- `dolphin-llama3` - Dolphin-optimized Llama 3 (helpful assistant)
- `phi` - Microsoft's Phi models (small and fast)
- `gemma` - Google's Gemma models (good balance)

## User Experience Considerations

### First-Time Setup

```python
from local_llm import LocalLLM, OllamaConnectionError
import subprocess
import sys

def ensure_ollama_installed():
    """Check if Ollama is installed and guide user through installation."""
    try:
        llm = LocalLLM()
        return llm
    except OllamaConnectionError:
        print("🔧 Local AI Setup Required")
        print("To use local AI features, you need to install Ollama.")
        print("1. Visit https://ollama.ai/")
        print("2. Download and install Ollama for your system")
        print("3. Restart this application")
        
        # Optionally, open the download page
        if input("Open Ollama download page? (y/n): ").lower() == 'y':
            import webbrowser
            webbrowser.open("https://ollama.ai/")
        
        return None

# In your application
llm = ensure_ollama_installed()
if llm:
    # Local AI is ready
    pass
else:
    # Show alternative options or disable features
    pass
```

### Model Download Progress

```python
from local_llm import LocalLLM

def download_model_with_progress(model_name: str):
    """Download a model with progress feedback."""
    llm = LocalLLM()
    
    print(f"📥 Downloading {model_name}...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        result = llm.models.pull(model_name)
        print(f"✅ {model_name} downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

# Usage
if download_model_with_progress("codellama:7b"):
    # Model is ready to use
    pass
```

## Error Handling

The library provides comprehensive error handling for different scenarios:

```python
from local_llm import LocalLLM, LocalLLMError, ModelNotFoundError, OllamaConnectionError

llm = LocalLLM()

try:
    response = llm.chat.completions.create(
        model="llama2",
        messages=[{"role": "user", "content": "Hello"}]
    )
except OllamaConnectionError:
    print("Ollama is not running. Please start Ollama and try again.")
except ModelNotFoundError:
    print("Model not found. Downloading...")
    llm.models.pull("llama2")
    # Retry the request
except LocalLLMError as e:
    print(f"Local AI error: {e}")
    # Handle gracefully or fall back to cloud
```

## Performance Considerations

### Model Selection

```python
# For fast responses (smaller models)
fast_model = "phi"  # ~1GB, very fast

# For better quality (larger models)
quality_model = "llama2:13b"  # ~7GB, better responses

# For code generation
code_model = "codellama"  # Optimized for programming
```

### Caching and Optimization

```python
from local_llm import LocalLLM

class OptimizedLocalAI:
    def __init__(self):
        self.llm = LocalLLM()
        self.response_cache = {}
    
    def get_cached_response(self, prompt: str, model: str) -> str:
        """Get cached response or generate new one."""
        cache_key = f"{model}:{hash(prompt)}"
        
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        response = self.llm.chat.completions.create(
            model=model,
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

- GitHub Issues: [Report bugs or request features](https://github.com/jeremynixon/local-llm/issues)
- Documentation: [Full API documentation](https://local-llm.readthedocs.io)
- Discord: [Join our community](https://discord.gg/local-llm)

## Roadmap

- [ ] Multi-modal support (images, audio)
- [ ] Model fine-tuning capabilities
- [ ] Advanced caching and optimization
- [ ] Web UI for model management
- [ ] Integration with other local LLM backends
- [ ] Batch processing capabilities
- [ ] Automatic model optimization for user's hardware
- [ ] Cloud fallback when local models are unavailable 