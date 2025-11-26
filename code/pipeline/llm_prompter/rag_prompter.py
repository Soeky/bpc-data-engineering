"""RAG (Retrieval-Augmented Generation) Prompter."""

import time
import logging
from typing import List, Optional

from config import Config
from .base import LLMPrompter
from ..retrieval import VectorStore


class RAGPrompter(LLMPrompter):
    """Retrieval-Augmented Generation prompting with external knowledge."""
    
    def __init__(
        self,
        entity_map=None,
        use_exact_spans: bool = True,
        model: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize RAG Prompter.
        
        Args:
            entity_map: Optional global entity map
            use_exact_spans: Whether to encourage exact text span extraction
            model: Model name/key (defaults to config default)
            vector_store: Vector store instance (creates new one if None)
            top_k: Number of retrieved documents (defaults to config)
            logger: Optional logger instance
        """
        super().__init__(entity_map, use_exact_spans, logger)
        self.model = Config.get_model_name(model)
        self.api_key = Config.OPENROUTER_API_KEY
        self.base_url = Config.OPENROUTER_BASE_URL
        self.top_k = top_k or Config.RAG_TOP_K
        
        # Initialize or use provided vector store
        if vector_store is None:
            self.vector_store = VectorStore()
            # Load documents from source directory
            self.vector_store.add_documents_from_files(Config.RAG_SOURCE_DIR)
        else:
            self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        """Return technique name."""
        return "RAG"
    
    def _retrieve_context(self, text: str) -> str:
        """
        Retrieve relevant context from vector store.
        
        Args:
            text: Document text to use as query
            
        Returns:
            Retrieved context as formatted string
        """
        # Use first few sentences or a summary of the text as query
        query = text[:500] if len(text) > 500 else text
        
        self.logger.debug(f"[{self.name}] Retrieving context with top_k={self.top_k}")
        results = self.vector_store.search(query, top_k=self.top_k)
        
        if not results:
            self.logger.debug(f"[{self.name}] No relevant context found")
            return "No relevant context found."
        
        self.logger.debug(f"[{self.name}] Retrieved {len(results)} context documents")
        context_parts = []
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0.0)
            self.logger.debug(f"[{self.name}] Context {i}: similarity={similarity:.3f}")
            context_parts.append(
                f"[Context {i}] (Similarity: {similarity:.3f})\n"
                f"{result.get('text', '')[:500]}...\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """Build the prompt for RAG prompting."""
        # Retrieve relevant context
        context = self._retrieve_context(text)
        
        prompt = self._build_base_prompt(text, doc_id)
        
        prompt += f"""Relevant Context from Knowledge Base:
{context}

---

Now extract all biomedical relations from the text above. The context provided above may help you understand the entities and relations better, but you must extract entity mentions as EXACT text spans from the original document text (not from the context).

For each relation, identify:
1. Head entity (exact text span from the ORIGINAL document)
2. Tail entity (exact text span from the ORIGINAL document)
3. Relation type (e.g., Association, Positive_Correlation, Negative_Correlation)

Return the results as a JSON array:
[
  {{
    "head_mention": "exact text from original document",
    "tail_mention": "exact text from original document",
    "relation_type": "Association"
  }}
]

IMPORTANT: Use EXACT text spans from the original document for entity mentions. The context is only for understanding - do not copy entity mentions from the context.
"""
        return prompt
    
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response using OpenRouter API with RAG.
        
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
                timeout=240  # Longer timeout for RAG (increased to 240 seconds)
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
