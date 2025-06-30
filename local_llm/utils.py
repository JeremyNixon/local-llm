"""
Utility functions for the Local LLM library.

This module provides helper functions for developers to easily integrate
local LLM capabilities into their applications.
"""

import os
import platform
import subprocess
import sys
import webbrowser
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .client import LocalLLM
from .exceptions import OllamaConnectionError, LocalLLMError


def get_system_info() -> Dict[str, str]:
    """Get system information for debugging and support."""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }


def check_ollama_installation() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is installed and accessible.
    
    Returns:
        Tuple of (is_installed, error_message)
    """
    try:
        # Try to connect to Ollama
        llm = LocalLLM()
        llm._test_connection()
        return True, None
    except OllamaConnectionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def get_ollama_install_instructions() -> Dict[str, str]:
    """Get platform-specific Ollama installation instructions."""
    system = platform.system().lower()
    
    instructions = {
        "macos": {
            "url": "https://ollama.ai/download/mac",
            "command": "brew install ollama",
            "manual": "Download from https://ollama.ai/download/mac",
        },
        "linux": {
            "url": "https://ollama.ai/download/linux",
            "command": "curl -fsSL https://ollama.ai/install.sh | sh",
            "manual": "Download from https://ollama.ai/download/linux",
        },
        "windows": {
            "url": "https://ollama.ai/download/windows",
            "command": "winget install Ollama.Ollama",
            "manual": "Download from https://ollama.ai/download/windows",
        },
    }
    
    return instructions.get(system, {
        "url": "https://ollama.ai/",
        "command": "See https://ollama.ai/ for installation instructions",
        "manual": "Visit https://ollama.ai/ for installation instructions",
    })


def open_ollama_download_page() -> bool:
    """Open the Ollama download page in the user's browser."""
    try:
        instructions = get_ollama_install_instructions()
        webbrowser.open(instructions["url"])
        return True
    except Exception:
        return False


def install_ollama_automatically() -> bool:
    """
    Attempt to install Ollama automatically using platform-specific methods.
    
    Returns:
        True if installation was attempted, False otherwise
    """
    system = platform.system().lower()
    
    try:
        if system == "macos":
            # Try Homebrew first
            result = subprocess.run(
                ["brew", "install", "ollama"], 
                capture_output=True, 
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return True
        
        elif system == "linux":
            # Try the official install script
            result = subprocess.run(
                ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return True
        
        elif system == "windows":
            # Try winget
            result = subprocess.run(
                ["winget", "install", "Ollama.Ollama"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                return True
                
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return False


def setup_local_ai_with_guidance() -> Optional[LocalLLM]:
    """
    Guide users through local AI setup with helpful prompts.
    
    Returns:
        LocalLLM instance if setup is successful, None otherwise
    """
    print("🤖 Setting up Local AI...")
    
    # Check if Ollama is already installed
    is_installed, error = check_ollama_installation()
    
    if is_installed:
        print("✅ Ollama is already installed and running!")
        try:
            llm = LocalLLM()
            print("✅ Local AI is ready to use!")
            return llm
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            return None
    
    # Ollama is not installed, guide user through installation
    print("🔧 Ollama is not installed. Let's set it up!")
    
    system_info = get_system_info()
    instructions = get_ollama_install_instructions()
    
    print(f"\n📋 Installation Instructions for {system_info['platform']}:")
    print(f"1. Visit: {instructions['url']}")
    print(f"2. Download and install Ollama")
    print(f"3. Start Ollama")
    print(f"4. Restart this application")
    
    # Offer to open download page
    try:
        choice = input("\n🌐 Open Ollama download page in browser? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            if open_ollama_download_page():
                print("✅ Download page opened!")
            else:
                print("❌ Could not open browser. Please visit the URL manually.")
    except (EOFError, KeyboardInterrupt):
        print("\nInstallation cancelled.")
        return None
    
    # Offer automatic installation
    try:
        choice = input("\n⚡ Try automatic installation? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("🔄 Attempting automatic installation...")
            if install_ollama_automatically():
                print("✅ Ollama installed successfully!")
                print("🔄 Please restart this application to use Local AI.")
            else:
                print("❌ Automatic installation failed. Please install manually.")
    except (EOFError, KeyboardInterrupt):
        print("\nAutomatic installation cancelled.")
    
    return None


def get_recommended_models() -> Dict[str, Dict[str, Any]]:
    """
    Get recommended models for different use cases.
    
    Returns:
        Dictionary of model recommendations with metadata
    """
    return {
        "fast": {
            "name": "phi",
            "description": "Fast and lightweight model (~1GB)",
            "use_case": "Quick responses, limited resources",
            "size_gb": 1.0,
            "speed": "Very Fast",
            "quality": "Good",
        },
        "balanced": {
            "name": "llama2",
            "description": "Well-balanced model (~4GB)",
            "use_case": "General purpose, good quality",
            "size_gb": 4.0,
            "speed": "Fast",
            "quality": "Very Good",
        },
        "quality": {
            "name": "llama2:13b",
            "description": "High-quality model (~7GB)",
            "use_case": "Best quality responses",
            "size_gb": 7.0,
            "speed": "Medium",
            "quality": "Excellent",
        },
        "code": {
            "name": "codellama",
            "description": "Code-optimized model (~4GB)",
            "use_case": "Programming and code generation",
            "size_gb": 4.0,
            "speed": "Fast",
            "quality": "Excellent for Code",
        },
        "assistant": {
            "name": "dolphin-llama3",
            "description": "Helpful assistant model (~4GB)",
            "use_case": "Conversation and assistance",
            "size_gb": 4.0,
            "speed": "Fast",
            "quality": "Very Good for Chat",
        },
    }


def recommend_model_for_use_case(use_case: str) -> Optional[str]:
    """
    Recommend a model based on use case.
    
    Args:
        use_case: The intended use case (fast, balanced, quality, code, assistant)
    
    Returns:
        Recommended model name or None
    """
    recommendations = get_recommended_models()
    return recommendations.get(use_case, {}).get("name")


def download_model_with_progress(model_name: str, show_progress: bool = True) -> bool:
    """
    Download a model with optional progress feedback.
    
    Args:
        model_name: Name of the model to download
        show_progress: Whether to show progress messages
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        llm = LocalLLM()
        
        if show_progress:
            print(f"📥 Downloading {model_name}...")
            print("This may take several minutes depending on your internet connection.")
        
        result = llm.models.pull(model_name)
        
        if show_progress:
            print(f"✅ {model_name} downloaded successfully!")
        
        return True
    except Exception as e:
        if show_progress:
            print(f"❌ Failed to download {model_name}: {e}")
        return False


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Model information dictionary or None
    """
    try:
        llm = LocalLLM()
        models = llm.models.list()
        
        for model in models.data:
            if model.id == model_name:
                return {
                    "id": model.id,
                    "size_bytes": model.size,
                    "size_mb": model.size / (1024 * 1024) if model.size else 0,
                    "digest": model.digest,
                }
        
        return None
    except Exception:
        return None


def estimate_model_performance(model_name: str) -> Dict[str, str]:
    """
    Estimate model performance based on size and type.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Performance estimates
    """
    # Get model info
    model_info = get_model_info(model_name)
    size_mb = model_info.get("size_mb", 0) if model_info else 0
    
    # Estimate performance based on size
    if size_mb < 2:
        speed = "Very Fast"
        memory = "Low"
        quality = "Good"
    elif size_mb < 5:
        speed = "Fast"
        memory = "Medium"
        quality = "Very Good"
    elif size_mb < 10:
        speed = "Medium"
        memory = "High"
        quality = "Excellent"
    else:
        speed = "Slow"
        memory = "Very High"
        quality = "Excellent"
    
    # Adjust for specific models
    if "code" in model_name.lower():
        quality = "Excellent for Code"
    elif "dolphin" in model_name.lower():
        quality = "Very Good for Chat"
    
    return {
        "speed": speed,
        "memory_usage": memory,
        "quality": quality,
        "estimated_tokens_per_second": "10-50" if speed == "Very Fast" else "5-20" if speed == "Fast" else "2-10",
    }


def create_model_selector() -> str:
    """
    Create an interactive model selector for users.
    
    Returns:
        Selected model name
    """
    print("🤖 Choose a Local AI Model:")
    print()
    
    recommendations = get_recommended_models()
    
    for i, (key, info) in enumerate(recommendations.items(), 1):
        print(f"{i}. {info['name']} - {info['description']}")
        print(f"   Use case: {info['use_case']}")
        print(f"   Size: {info['size_gb']}GB | Speed: {info['speed']} | Quality: {info['quality']}")
        print()
    
    try:
        choice = input("Enter your choice (1-5): ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(recommendations):
            model_key = list(recommendations.keys())[choice_num - 1]
            selected_model = recommendations[model_key]["name"]
            
            print(f"✅ Selected: {selected_model}")
            return selected_model
        else:
            print("❌ Invalid choice. Using default model: llama2")
            return "llama2"
            
    except (ValueError, EOFError, KeyboardInterrupt):
        print("❌ Invalid input. Using default model: llama2")
        return "llama2"


def check_system_requirements() -> Dict[str, bool]:
    """
    Check if the system meets requirements for local AI.
    
    Returns:
        Dictionary of requirement checks
    """
    system_info = get_system_info()
    
    checks = {
        "python_version": sys.version_info >= (3, 8),
        "platform_supported": system_info["platform"].lower() in ["darwin", "linux", "windows"],
        "architecture_supported": system_info["architecture"] in ["x86_64", "amd64", "arm64", "aarch64"],
    }
    
    # Check available memory (rough estimate)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        checks["sufficient_memory"] = memory_gb >= 4  # At least 4GB RAM
        checks["memory_gb"] = memory_gb
    except ImportError:
        checks["sufficient_memory"] = True  # Assume OK if psutil not available
        checks["memory_gb"] = "Unknown"
    
    return checks


def get_setup_status() -> Dict[str, Any]:
    """
    Get comprehensive setup status for local AI.
    
    Returns:
        Dictionary with setup status information
    """
    status = {
        "ollama_installed": False,
        "ollama_running": False,
        "system_requirements": check_system_requirements(),
        "available_models": [],
        "recommendations": get_recommended_models(),
    }
    
    # Check Ollama installation
    is_installed, error = check_ollama_installation()
    status["ollama_installed"] = is_installed
    
    if is_installed:
        try:
            llm = LocalLLM()
            llm._test_connection()
            status["ollama_running"] = True
            
            # Get available models
            models = llm.models.list()
            status["available_models"] = [model.id for model in models.data]
            
        except Exception as e:
            status["ollama_error"] = str(e)
    
    return status 