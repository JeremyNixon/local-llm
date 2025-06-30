#!/usr/bin/env python3
"""
Code Generator Example

This example shows how to create a code generation tool
that uses local LLMs for programming assistance.
"""

import os
import sys
from local_llm import (
    LocalLLM, 
    setup_local_ai_with_guidance, 
    download_model_with_progress,
    recommend_model_for_use_case
)


class CodeGenerator:
    """Code generation tool using local LLMs."""
    
    def __init__(self):
        self.llm = None
        self.model = "codellama"  # Default code model
    
    def setup(self):
        """Set up the local AI for code generation."""
        print("🔧 Setting up Code Generator...")
        
        # Try to set up local AI
        self.llm = setup_local_ai_with_guidance()
        
        if not self.llm:
            print("❌ Local AI setup failed.")
            return False
        
        # Check if we have a code model
        try:
            models = self.llm.models.list()
            available_models = [model.id for model in models.data]
            
            if self.model not in available_models:
                print(f"📥 Code model '{self.model}' not found. Downloading...")
                if download_model_with_progress(self.model):
                    print(f"✅ {self.model} is ready for code generation!")
                else:
                    print(f"❌ Failed to download {self.model}. Using available models.")
                    # Fall back to any available model
                    if available_models:
                        self.model = available_models[0]
                        print(f"Using model: {self.model}")
                    else:
                        return False
            else:
                print(f"✅ {self.model} is ready for code generation!")
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return False
        
        return True
    
    def generate_function(self, description: str, language: str = "python") -> str:
        """Generate a function based on description."""
        if not self.llm:
            return "❌ Local AI not available."
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an expert {language} programmer. Generate only code, no explanations. Include proper docstrings and type hints."
                    },
                    {"role": "user", "content": f"Create a function that: {description}"}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating code: {e}"
    
    def explain_code(self, code: str) -> str:
        """Explain what code does."""
        if not self.llm:
            return "❌ Local AI not available."
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a programming expert. Explain code clearly and concisely."
                    },
                    {"role": "user", "content": f"Explain what this code does:\n\n{code}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error explaining code: {e}"
    
    def debug_code(self, code: str, error_message: str = "") -> str:
        """Debug code and suggest fixes."""
        if not self.llm:
            return "❌ Local AI not available."
        
        try:
            prompt = f"Debug this code:\n\n{code}"
            if error_message:
                prompt += f"\n\nError message: {error_message}"
            
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a debugging expert. Identify issues and suggest fixes."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error debugging code: {e}"
    
    def generate_tests(self, code: str, language: str = "python") -> str:
        """Generate unit tests for code."""
        if not self.llm:
            return "❌ Local AI not available."
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are a testing expert. Generate comprehensive unit tests for {language} code."
                    },
                    {"role": "user", "content": f"Generate unit tests for this code:\n\n{code}"}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating tests: {e}"


def main():
    """Main code generator application."""
    print("💻 Local AI Code Generator")
    print("=" * 50)
    
    generator = CodeGenerator()
    
    if not generator.setup():
        print("❌ Failed to set up code generator.")
        return
    
    print("\n🎯 Available Commands:")
    print("  1. generate <description> - Generate a function")
    print("  2. explain <code> - Explain what code does")
    print("  3. debug <code> [error] - Debug code and suggest fixes")
    print("  4. test <code> - Generate unit tests")
    print("  5. quit - Exit")
    print("-" * 50)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not command:
                continue
            
            parts = command.split(' ', 1)
            if len(parts) < 2:
                print("❌ Please provide a command and input.")
                continue
            
            cmd, input_text = parts
            
            if cmd == "generate":
                print("\n🔧 Generating function...")
                result = generator.generate_function(input_text)
                print(f"\n{result}")
                
            elif cmd == "explain":
                print("\n📖 Explaining code...")
                result = generator.explain_code(input_text)
                print(f"\n{result}")
                
            elif cmd == "debug":
                print("\n🐛 Debugging code...")
                result = generator.debug_code(input_text)
                print(f"\n{result}")
                
            elif cmd == "test":
                print("\n🧪 Generating tests...")
                result = generator.generate_tests(input_text)
                print(f"\n{result}")
                
            else:
                print("❌ Unknown command. Use: generate, explain, debug, test, or quit.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main() 