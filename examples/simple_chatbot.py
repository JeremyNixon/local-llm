#!/usr/bin/env python3
"""
Simple Chatbot Example

This example shows how to create a basic chatbot application
that uses local LLMs for privacy and speed.
"""

import sys
from local_llm import LocalLLM, setup_local_ai_with_guidance, get_recommended_models


def main():
    """Main chatbot application."""
    print("🤖 Local AI Chatbot")
    print("=" * 50)
    
    # Try to set up local AI with user guidance
    llm = setup_local_ai_with_guidance()
    
    if not llm:
        print("\n❌ Local AI is not available.")
        print("You can still use this chatbot with cloud AI by modifying the code.")
        return
    
    # Show available models
    print("\n📋 Available Models:")
    recommendations = get_recommended_models()
    for key, info in recommendations.items():
        print(f"  • {info['name']} - {info['description']}")
    
    # Simple chat loop
    print("\n💬 Start chatting! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response using local AI
            print("AI: ", end="", flush=True)
            
            response = llm.chat.completions.create(
                model="llama2",  # You can change this to any available model
                messages=[
                    {"role": "system", "content": "You are a helpful and friendly AI assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            print(response.choices[0].message.content)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main() 