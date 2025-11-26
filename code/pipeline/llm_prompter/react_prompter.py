"""ReAct (Reason + Act) Prompter - Reasoning with actions."""

import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter


class ReActPrompter(LLMPrompter):
    """ReAct prompting with reasoning and action steps."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ReAct Prompter.
        
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
        return "ReAct"
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for ReAct prompting."""
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += """Extract all biomedical relations from the text above using a reasoning and action approach.

You can use the following format:
Thought: [Your reasoning about what to do next]
Action: [Action to take, e.g., IDENTIFY_ENTITY, VERIFY_TYPE, EXTRACT_RELATION]
Observation: [Result of the action]

Available actions:
- IDENTIFY_ENTITY: Identify an entity and extract its exact text span from the document
- VERIFY_TYPE: Verify the type of an identified entity
- EXTRACT_RELATION: Extract a relation between two entities

After reasoning through the text, provide your final answer as a JSON array:

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
                timeout=240  # Longer timeout for ReAct (increased to 240 seconds)
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
