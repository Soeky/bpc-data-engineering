"""PubMed retriever (placeholder for future implementation)."""

from typing import List, Dict, Any
from .base import Retriever


class PubMedRetriever(Retriever):
    """PubMed API retriever (placeholder)."""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve PubMed abstracts (placeholder).
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of retrieved documents
        """
        # TODO: Implement PubMed API integration
        return []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Not applicable for PubMed retriever."""
        pass
