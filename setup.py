from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="local-llm",
    version="0.1.0",
    author="Local LLM Team",
    author_email="dev@localllm.ai",
    description="OpenAI-like API for local LLMs across all platforms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/jeremynixon/local-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "streaming": [
            "sseclient-py>=1.7.0",
        ],
        "web": [
            "flask>=2.0.0",
        ],
        "browser": [
            "@mlc-ai/web-llm>=0.2.79",
        ],
        "android": [
            "com.google.mediapipe:tasks-genai>=0.10.24",
        ],
        "ios": [
            "Foundation>=1.0.0",
        ],
        "all": [
            "sseclient-py>=1.7.0",
            "flask>=2.0.0",
            "@mlc-ai/web-llm>=0.2.79",
            "com.google.mediapipe:tasks-genai>=0.10.24",
            "Foundation>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "local-llm=local_llm.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "local_llm": ["py.typed"],
    },
) 