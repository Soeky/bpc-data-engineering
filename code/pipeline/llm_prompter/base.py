"""Base class for LLM prompters."""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING
import requests

if TYPE_CHECKING:
    from ..data.entity_map import GlobalEntityMap


class LLMPrompter(ABC):
    """Abstract base class for LLM prompting techniques."""
    
    def __init__(
        self,
        entity_map: Optional["GlobalEntityMap"] = None,
        use_exact_spans: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the prompter.
        
        Args:
            entity_map: Optional global entity map for context
            use_exact_spans: Whether to encourage exact text span extraction
            logger: Optional logger instance
        """
        self.entity_map = entity_map
        self.use_exact_spans = use_exact_spans
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Get LLM response for a single document text.
        
        Args:
            text: Document text (title + body)
            doc_id: Optional document ID for context
            
        Returns:
            LLM response as string
        """
        pass
    
    @abstractmethod
    def get_responses_batch(
        self, texts: List[str], doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get LLM responses for multiple documents (optional optimization).
        
        Args:
            texts: List of document texts
            doc_ids: Optional list of document IDs
            
        Returns:
            List of LLM responses
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this prompting technique."""
        pass
    
    def _build_base_prompt(self, text: str, doc_id: Optional[str] = None) -> str:
        """
        Build base prompt with common instructions.
        
        Args:
            text: Document text
            doc_id: Optional document ID
            
        Returns:
            Base prompt string
        """
        prompt = """Extract biomedical relations from the following text.

"""
        if doc_id:
            prompt += f"Document ID: {doc_id}\n\n"
        
        if self.use_exact_spans:
            prompt += """IMPORTANT: When extracting entities, use the EXACT text spans from the document. 
Do not paraphrase or modify the entity mentions. Copy them exactly as they appear in the text.

"""
        
        prompt += f"Text:\n{text}\n\n"
        
        return prompt
    
    def _get_entity_context(self) -> Optional[str]:
        """
        Get entity context from global entity map for prompting.
        
        Returns:
            Entity context string or None if entity_map is not available
        """
        if not self.entity_map:
            return None
        
        # Get some common entities to provide context
        # This can be customized by subclasses
        return None
    
    def _make_api_request_with_retry(
        self,
        url: str,
        headers: dict,
        payload: dict,
        timeout: int = 180,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> requests.Response:
        """
        Make API request with retry logic and exponential backoff.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            
        Returns:
            Response object
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s, etc.
                    delay = base_delay * (2 ** (attempt - 1))
                    self.logger.warning(
                        f"[{self.name}] Retry attempt {attempt}/{max_retries} "
                        f"after {delay:.1f}s delay..."
                    )
                    time.sleep(delay)
                
                self.logger.debug(
                    f"[{self.name}] API request attempt {attempt + 1}/{max_retries + 1} "
                    f"(timeout={timeout}s)"
                )
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(
                    f"[{self.name}] Request timeout (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                if attempt < max_retries:
                    continue
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                # For non-timeout errors, only retry if it's a 5xx server error
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if 500 <= status_code < 600 and attempt < max_retries:
                        self.logger.warning(
                            f"[{self.name}] Server error {status_code} "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        continue
                    else:
                        # Don't retry for client errors (4xx)
                        self.logger.error(f"[{self.name}] Client error {status_code}: {e}")
                        raise RuntimeError(f"OpenRouter API error: {e}")
                else:
                    # Network errors - retry
                    self.logger.warning(
                        f"[{self.name}] Network error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    if attempt < max_retries:
                        continue
        
        # All retries exhausted
        error_msg = f"OpenRouter API request failed after {max_retries + 1} attempts"
        if last_exception:
            error_msg += f": {last_exception}"
        self.logger.error(f"[{self.name}] {error_msg}")
        raise RuntimeError(error_msg)
