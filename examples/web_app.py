#!/usr/bin/env python3
"""
Web Application Example

This example shows how to integrate local LLMs into a Flask web application,
providing AI capabilities to users through a web interface.
"""

from flask import Flask, render_template, request, jsonify, session
from local_llm import (
    LocalLLM, 
    setup_local_ai_with_guidance, 
    get_setup_status,
    download_model_with_progress,
    get_recommended_models
)
import json
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global LLM instance
llm = None
setup_complete = False


def initialize_local_ai():
    """Initialize local AI for the web application."""
    global llm, setup_complete
    
    try:
        # Check setup status
        status = get_setup_status()
        
        if status['ollama_running']:
            llm = LocalLLM()
            setup_complete = True
            print("✅ Local AI initialized successfully!")
            return True
        else:
            print("❌ Local AI not available")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing Local AI: {e}")
        return False


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get local AI status."""
    global llm, setup_complete
    
    status = get_setup_status()
    
    return jsonify({
        'local_ai_available': setup_complete,
        'ollama_installed': status['ollama_installed'],
        'ollama_running': status['ollama_running'],
        'available_models': status['available_models'],
        'system_requirements': status['system_requirements'],
        'recommendations': get_recommended_models()
    })


@app.route('/api/setup', methods=['POST'])
def api_setup():
    """Handle local AI setup."""
    global llm, setup_complete
    
    try:
        # Try to initialize local AI
        if initialize_local_ai():
            return jsonify({
                'success': True,
                'message': 'Local AI setup successful!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Local AI setup failed. Please install Ollama first.',
                'setup_url': 'https://ollama.ai/'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Setup error: {str(e)}'
        })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handle chat requests."""
    global llm, setup_complete
    
    if not setup_complete or not llm:
        return jsonify({
            'success': False,
            'message': 'Local AI not available. Please set up Ollama first.',
            'setup_url': 'https://ollama.ai/'
        }), 400
    
    try:
        data = request.json
        message = data.get('message', '')
        model = data.get('model', 'llama2')
        
        if not message:
            return jsonify({
                'success': False,
                'message': 'No message provided'
            }), 400
        
        # Generate response
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return jsonify({
            'success': True,
            'response': response.choices[0].message.content,
            'model': model
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating response: {str(e)}'
        }), 500


@app.route('/api/stream-chat', methods=['POST'])
def api_stream_chat():
    """Handle streaming chat requests."""
    global llm, setup_complete
    
    if not setup_complete or not llm:
        return jsonify({
            'success': False,
            'message': 'Local AI not available. Please set up Ollama first.'
        }), 400
    
    try:
        data = request.json
        message = data.get('message', '')
        model = data.get('model', 'llama2')
        
        if not message:
            return jsonify({
                'success': False,
                'message': 'No message provided'
            }), 400
        
        # Generate streaming response
        def generate():
            try:
                stream = llm.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return app.response_class(
            generate(),
            mimetype='text/plain'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting stream: {str(e)}'
        }), 500


@app.route('/api/download-model', methods=['POST'])
def api_download_model():
    """Download a model."""
    global llm, setup_complete
    
    if not setup_complete or not llm:
        return jsonify({
            'success': False,
            'message': 'Local AI not available. Please set up Ollama first.'
        }), 400
    
    try:
        data = request.json
        model_name = data.get('model', '')
        
        if not model_name:
            return jsonify({
                'success': False,
                'message': 'No model name provided'
            }), 400
        
        # Download model
        success = download_model_with_progress(model_name, show_progress=False)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_name} downloaded successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to download model {model_name}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error downloading model: {str(e)}'
        }), 500


@app.route('/api/generate-code', methods=['POST'])
def api_generate_code():
    """Generate code based on description."""
    global llm, setup_complete
    
    if not setup_complete or not llm:
        return jsonify({
            'success': False,
            'message': 'Local AI not available. Please set up Ollama first.'
        }), 400
    
    try:
        data = request.json
        description = data.get('description', '')
        language = data.get('language', 'python')
        model = data.get('model', 'codellama')
        
        if not description:
            return jsonify({
                'success': False,
                'message': 'No description provided'
            }), 400
        
        # Generate code
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert {language} programmer. Generate only code, no explanations."
                },
                {"role": "user", "content": f"Create code that: {description}"}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        return jsonify({
            'success': True,
            'code': response.choices[0].message.content,
            'language': language,
            'model': model
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating code: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Initialize local AI on startup
    print("🚀 Starting Local AI Web Application...")
    
    if initialize_local_ai():
        print("✅ Local AI is ready!")
    else:
        print("⚠️  Local AI not available. Users will need to set it up.")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local AI Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .status.success { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .status.warning { background-color: #fff3cd; color: #856404; }
        
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message { background-color: #e3f2fd; }
        .ai-message { background-color: #f1f8e9; }
        
        input, textarea, select, button {
            padding: 10px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        button {
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #6c757d; cursor: not-allowed; }
        
        .code-output {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Local AI Web Application</h1>
        
        <div id="status" class="status warning">
            Checking Local AI status...
        </div>
        
        <div id="setup-section" style="display: none;">
            <h2>🔧 Setup Local AI</h2>
            <p>To use local AI features, you need to install Ollama.</p>
            <button onclick="openSetupPage()">Open Ollama Download Page</button>
            <button onclick="checkSetup()">Check Setup</button>
        </div>
        
        <div id="chat-section" style="display: none;">
            <h2>💬 Chat with Local AI</h2>
            <div class="chat-container">
                <div id="messages"></div>
                <div>
                    <select id="model-select">
                        <option value="llama2">Llama 2</option>
                        <option value="codellama">CodeLlama</option>
                        <option value="phi">Phi</option>
                    </select>
                    <textarea id="message-input" placeholder="Type your message..." rows="3" style="width: 100%;"></textarea>
                    <button onclick="sendMessage()">Send</button>
                    <button onclick="sendStreamMessage()">Send (Stream)</button>
                </div>
            </div>
        </div>
        
        <div id="code-section" style="display: none;">
            <h2>💻 Code Generation</h2>
            <div>
                <select id="code-language">
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                    <option value="java">Java</option>
                    <option value="cpp">C++</option>
                </select>
                <textarea id="code-description" placeholder="Describe the code you want to generate..." rows="3" style="width: 100%;"></textarea>
                <button onclick="generateCode()">Generate Code</button>
            </div>
            <div id="code-output" class="code-output" style="display: none;"></div>
        </div>
    </div>

    <script>
        // Check status on page load
        window.onload = function() {
            checkStatus();
        };
        
        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    const setupSection = document.getElementById('setup-section');
                    const chatSection = document.getElementById('chat-section');
                    
                    if (data.local_ai_available) {
                        statusDiv.className = 'status success';
                        statusDiv.textContent = '✅ Local AI is ready!';
                        setupSection.style.display = 'none';
                        chatSection.style.display = 'block';
                        document.getElementById('code-section').style.display = 'block';
                    } else {
                        statusDiv.className = 'status error';
                        statusDiv.textContent = '❌ Local AI not available. Please set up Ollama.';
                        setupSection.style.display = 'block';
                        chatSection.style.display = 'none';
                        document.getElementById('code-section').style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').textContent = '❌ Error checking status';
                });
        }
        
        function openSetupPage() {
            window.open('https://ollama.ai/', '_blank');
        }
        
        function checkSetup() {
            fetch('/api/setup', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        checkStatus();
                    } else {
                        alert('Setup failed: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error during setup:', error);
                    alert('Setup error: ' + error);
                });
        }
        
        function sendMessage() {
            const message = document.getElementById('message-input').value;
            const model = document.getElementById('model-select').value;
            
            if (!message.trim()) return;
            
            // Add user message
            addMessage('user', message);
            document.getElementById('message-input').value = '';
            
            // Send to API
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, model })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('ai', data.response);
                } else {
                    addMessage('ai', 'Error: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error sending message:', error);
                addMessage('ai', 'Error: Could not send message');
            });
        }
        
        function sendStreamMessage() {
            const message = document.getElementById('message-input').value;
            const model = document.getElementById('model-select').value;
            
            if (!message.trim()) return;
            
            // Add user message
            addMessage('user', message);
            document.getElementById('message-input').value = '';
            
            // Add AI message placeholder
            const aiMessageId = addMessage('ai', '');
            
            // Send to streaming API
            fetch('/api/stream-chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, model })
            })
            .then(response => {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                function readStream() {
                    reader.read().then(({ done, value }) => {
                        if (done) return;
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        
                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.content) {
                                        updateMessage(aiMessageId, data.content);
                                    }
                                } catch (e) {
                                    // Ignore parsing errors
                                }
                            }
                        });
                        
                        readStream();
                    });
                }
                
                readStream();
            })
            .catch(error => {
                console.error('Error streaming message:', error);
                updateMessage(aiMessageId, 'Error: Could not stream message');
            });
        }
        
        function generateCode() {
            const description = document.getElementById('code-description').value;
            const language = document.getElementById('code-language').value;
            
            if (!description.trim()) return;
            
            const outputDiv = document.getElementById('code-output');
            outputDiv.style.display = 'block';
            outputDiv.textContent = 'Generating code...';
            
            fetch('/api/generate-code', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description, language })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    outputDiv.textContent = data.code;
                } else {
                    outputDiv.textContent = 'Error: ' + data.message;
                }
            })
            .catch(error => {
                console.error('Error generating code:', error);
                outputDiv.textContent = 'Error: Could not generate code';
            });
        }
        
        function addMessage(type, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return messageDiv;
        }
        
        function updateMessage(messageDiv, content) {
            messageDiv.textContent = content;
            document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
        }
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("🌐 Starting web server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 