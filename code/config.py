"""Configuration management."""

import os
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the pipeline."""
    
    # API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Configuration - can be changed per technique
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    
    # Available models for testing (OpenRouter compatible)
    AVAILABLE_MODELS: Dict[str, str] = {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3-opus": "anthropic/claude-3-opus",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "gemini-pro": "google/gemini-pro-1.5",
    }
    
    # Data paths
    BASE_PATH: Path = Path(__file__).parent
    CLEAN_TEXT_PATH: Path = BASE_PATH / "clean_text"
    GOLD_RELATIONS_PATH: Path = BASE_PATH / "gold_relations"
    OUTPUT_DIR: Path = BASE_PATH / "results"
    
    # RAG Configuration
    RAG_SOURCE_DIR: Path = BASE_PATH / "rag_sources"  # Directory for source files
    RAG_EMBEDDINGS_DIR: Path = BASE_PATH / "rag_embeddings"  # Directory for cached embeddings
    RAG_EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding model
    RAG_TOP_K: int = 5  # Number of retrieved documents
    
    # LLM Configuration
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.0  
    
    # Evaluation Configuration
    MATCHING_STRATEGY: str = "exact"  # "exact" or "fuzzy"
    
    # Logging Configuration
    LOG_LEVEL: str = "DEBUG"  # "DEBUG", "INFO", "WARNING", "ERROR"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Create necessary directories
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.RAG_SOURCE_DIR.mkdir(exist_ok=True)
        cls.RAG_EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_name(cls, model_key: Optional[str] = None) -> str:
        """
        Get model name from key or return default.
        
        Args:
            model_key: Key from AVAILABLE_MODELS or None for default
            
        Returns:
            Full model name for OpenRouter
        """
        if model_key and model_key in cls.AVAILABLE_MODELS:
            return cls.AVAILABLE_MODELS[model_key]
        return cls.DEFAULT_MODEL
