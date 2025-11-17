"""Base class for LLM prompters."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.entity_map import GlobalEntityMap


class LLMPrompter(ABC):
    """Abstract base class for LLM prompting techniques."""
    
    def __init__(
        self,
        entity_map: Optional["GlobalEntityMap"] = None,
        use_exact_spans: bool = True,
    ):
        """
        Initialize the prompter.
        
        Args:
            entity_map: Optional global entity map for context
            use_exact_spans: Whether to encourage exact text span extraction
        """
        self.entity_map = entity_map
        self.use_exact_spans = use_exact_spans
    
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
