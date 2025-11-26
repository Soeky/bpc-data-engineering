"""Embedding model utilities."""

import hashlib
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import requests
from openai import OpenAI

from config import Config


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API (via OpenRouter or direct)."""
    
    def __init__(self, model: str = None, use_openrouter: bool = False):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name (defaults to config)
            use_openrouter: Whether to use OpenRouter API (if False, uses OpenAI directly)
        """
        self.model = model or Config.RAG_EMBEDDING_MODEL
        self.use_openrouter = use_openrouter
        self.api_key = Config.OPENROUTER_API_KEY
        
        if not use_openrouter:
            # Try OpenAI client directly (works if OPENROUTER_API_KEY is actually an OpenAI key)
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                # Fallback to OpenRouter
                self.use_openrouter = True
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if self.use_openrouter:
            return self._generate_embedding_openrouter(text)
        else:
            return self._generate_embedding_openai(text)
    
    def _generate_embedding_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI client."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            # Fallback to OpenRouter if OpenAI fails
            self.use_openrouter = True
            return self._generate_embedding_openrouter(text)
    
    def _generate_embedding_openrouter(self, text: str) -> List[float]:
        """Generate embedding using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": f"openai/{self.model}",  # OpenRouter format
            "input": text
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.use_openrouter:
            return self._generate_embeddings_batch_openrouter(texts)
        else:
            return self._generate_embeddings_batch_openai(texts)
    
    def _generate_embeddings_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI client."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            # Fallback to OpenRouter if OpenAI fails
            self.use_openrouter = True
            return self._generate_embeddings_batch_openrouter(texts)
    
    def _generate_embeddings_batch_openrouter(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": f"openai/{self.model}",  # OpenRouter format
            "input": texts
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of file contents.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
