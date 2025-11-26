"""I/O (Input/Output) Prompter - Simple zero-shot prompting."""

import json
import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter


class IOPrompter(LLMPrompter):
    """Simple zero-shot prompting for relation extraction."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize IO Prompter.
        
        Args:
            entity_map: Optional global entity map
            use_exact_spans: Whether to encourage exact text span extraction
            model: Model name/key (defaults to config default)
            logger: Optional logger instance
        """
        super().__init__(entity_map, use_exact_spans, logger)
        self.model = Config.get_model_name(model)
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
    
    @property
    def name(self) -> str:
        """Return technique name."""
        return "IO"
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for IO prompting."""
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += """Extract all biomedical relations from the text above.

For each relation, identify:
1. Head entity (exact text span from the document)
2. Tail entity (exact text span from the document)
3. Relation type (e.g., Association, Positive_Correlation, Negative_Correlation)

Return the results as a JSON array with the following format:
[
  {
    "head_mention": "exact text from document",
    "tail_mention": "exact text from document",
    "relation_type": "Association"
  }
]

IMPORTANT: Use EXACT text spans from the document for entity mentions. Do not paraphrase or modify the text.
"""
        return prompt
    
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response using OpenRouter API.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            LLM response string
        """
        self.logger.info(f"[{self.name}] Processing document: {doc_id}")
        self.logger.debug(f"[{self.name}] Document text length: {len(text)} characters")
        self.logger.debug(f"[{self.name}] Using model: {self.model}")
        
        prompt = self._build_prompt(text, doc_id)
        self.logger.debug(f"[{self.name}] Prompt length: {len(prompt)} characters")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }
        
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Sending request to OpenRouter API...")
            
            response = self._make_api_request_with_retry(
                f"{self.base_url}/chat/completions",
                headers=headers,
                payload=payload,
                timeout=180  # Increased timeout to 180 seconds
            )
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"[{self.name}] Received response in {elapsed_time:.2f} seconds")
            self.logger.debug(f"[{self.name}] Raw LLM response length: {len(llm_response)} characters")
            self.logger.debug(f"[{self.name}] Raw LLM response:\n{llm_response}")
            
            return llm_response
        except Exception as e:
            self.logger.error(f"[{self.name}] OpenRouter API error: {e}")
            raise
    
    def get_responses_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get responses for multiple documents (sequential for now).
        
        Args:
            texts: List of document texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of LLM responses
        """
        if doc_ids is None:
            doc_ids = [None] * len(texts)
        
        responses = []
        for text, doc_id in zip(texts, doc_ids):
            responses.append(self.get_response(text, doc_id))
        
        return responses
