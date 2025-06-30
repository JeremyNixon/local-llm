"""
Command-line interface for the Local LLM library.

This module provides a CLI tool for interacting with local Ollama models
directly from the terminal.
"""

import argparse
import json
import sys
from typing import List, Optional

from .client import LocalLLM
from .exceptions import LocalLLMError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local LLM - OpenAI-like API for local Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chat with a model
  local-llm chat --model llama2 "Hello, how are you?"

  # Generate text completion
  local-llm complete --model llama2 "The future of AI is"

  # List available models
  local-llm models list

  # Pull a new model
  local-llm models pull codellama:7b

  # Create embeddings
  local-llm embed --model llama2 "Hello world"
        """
    )
    
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a model")
    chat_parser.add_argument("--model", required=True, help="Model to use")
    chat_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    chat_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    chat_parser.add_argument("--stream", action="store_true", help="Stream the response")
    chat_parser.add_argument("message", help="Message to send")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Generate text completion")
    complete_parser.add_argument("--model", required=True, help="Model to use")
    complete_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    complete_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    complete_parser.add_argument("--stream", action="store_true", help="Stream the response")
    complete_parser.add_argument("prompt", help="Prompt to complete")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Create embeddings")
    embed_parser.add_argument("--model", required=True, help="Model to use")
    embed_parser.add_argument("text", help="Text to embed")
    
    # Models command
    models_parser = subparsers.add_parser("models", help="Manage models")
    models_subparsers = models_parser.add_subparsers(dest="models_command", help="Model operations")
    
    models_list_parser = models_subparsers.add_parser("list", help="List available models")
    models_pull_parser = models_subparsers.add_parser("pull", help="Pull a model")
    models_pull_parser.add_argument("model", help="Model to pull")
    models_delete_parser = models_subparsers.add_parser("delete", help="Delete a model")
    models_delete_parser.add_argument("model", help="Model to delete")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize client
        client = LocalLLM(
            base_url=args.base_url,
            timeout=args.timeout
        )
        
        if args.command == "chat":
            handle_chat(client, args)
        elif args.command == "complete":
            handle_complete(client, args)
        elif args.command == "embed":
            handle_embed(client, args)
        elif args.command == "models":
            handle_models(client, args)
            
    except LocalLLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_chat(client: LocalLLM, args):
    """Handle chat command."""
    messages = [{"role": "user", "content": args.message}]
    
    if args.stream:
        print("Assistant: ", end="", flush=True)
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print(f"Assistant: {response.choices[0].message.content}")


def handle_complete(client: LocalLLM, args):
    """Handle complete command."""
    if args.stream:
        print("Completion: ", end="", flush=True)
        stream = client.completions.create(
            model=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].text:
                print(chunk.choices[0].text, end="", flush=True)
        print()
    else:
        response = client.completions.create(
            model=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print(f"Completion: {response.choices[0].text}")


def handle_embed(client: LocalLLM, args):
    """Handle embed command."""
    response = client.embeddings.create(
        model=args.model,
        input=args.text
    )
    
    # Print embedding as JSON
    embedding_data = {
        "model": response.model,
        "embedding": response.data[0].embedding[:10],  # Show first 10 dimensions
        "embedding_length": len(response.data[0].embedding),
        "usage": response.usage.dict() if response.usage else None
    }
    
    print(json.dumps(embedding_data, indent=2))


def handle_models(client: LocalLLM, args):
    """Handle models command."""
    if args.models_command == "list":
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            size_mb = model.size / (1024 * 1024) if model.size else 0
            print(f"  {model.id} ({size_mb:.1f} MB)")
    
    elif args.models_command == "pull":
        print(f"Pulling model: {args.model}")
        result = client.models.pull(args.model)
        print(f"Model pulled successfully: {result}")
    
    elif args.models_command == "delete":
        print(f"Deleting model: {args.model}")
        result = client.models.delete(args.model)
        print(f"Model deleted successfully: {result}")


if __name__ == "__main__":
    main() 