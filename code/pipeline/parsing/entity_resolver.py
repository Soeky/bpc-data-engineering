"""Entity resolver for mapping mentions to IDs."""

import re
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

from ..types import ParsedRelation, GlobalEntity
from ..data.entity_map import GlobalEntityMap


class EntityResolver:
    """Resolves entity mentions to global entity IDs."""
    
    def __init__(self, entity_map: Optional[GlobalEntityMap] = None):
        """
        Initialize entity resolver.
        
        Args:
            entity_map: Global entity map for resolution
        """
        self.entity_map = entity_map
    
    def resolve_mention(
        self, 
        mention_text: str, 
        entity_type: Optional[str] = None,
        source_text: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve a mention to an entity ID.
        
        Args:
            mention_text: Text mention to resolve
            entity_type: Optional entity type hint
            source_text: Optional source text for context
            
        Returns:
            Entity ID or None if not found
        """
        if not self.entity_map:
            return None
        
        mention_text = mention_text.strip()
        if not mention_text:
            return None
        
        # Try exact match first
        matches = self.entity_map.find_entity_by_mention(
            mention_text, 
            entity_type=entity_type, 
            fuzzy=False
        )
        
        if matches:
            return matches[0].id
        
        # Try fuzzy match
        matches = self.entity_map.find_entity_by_mention(
            mention_text, 
            entity_type=entity_type, 
            fuzzy=True
        )
        
        if matches:
            # If multiple matches, prefer the one with highest similarity
            best_match = max(
                matches,
                key=lambda e: self._similarity_score(mention_text, e)
            )
            return best_match.id
        
        return None
    
    def _similarity_score(self, mention_text: str, entity: GlobalEntity) -> float:
        """
        Calculate similarity score between mention and entity.
        
        Args:
            mention_text: Mention text
            entity: GlobalEntity to compare
            
        Returns:
            Similarity score (0-1)
        """
        mention_lower = mention_text.lower().strip()
        
        # Check against canonical name
        if entity.canonical_name:
            canon_sim = SequenceMatcher(None, mention_lower, entity.canonical_name.lower()).ratio()
        else:
            canon_sim = 0.0
        
        # Check against common mentions
        common_sims = [
            SequenceMatcher(None, mention_lower, cm.lower()).ratio()
            for cm in entity.common_mentions[:5]  # Check top 5
        ]
        max_common_sim = max(common_sims) if common_sims else 0.0
        
        return max(canon_sim, max_common_sim)
    
    def resolve_relation(
        self, 
        relation: ParsedRelation,
        source_text: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve both head and tail entities in a relation.
        
        Args:
            relation: ParsedRelation to resolve
            source_text: Optional source text for context
            
        Returns:
            Tuple of (head_id, tail_id)
        """
        head_id = self.resolve_mention(
            relation.head_mention,
            source_text=source_text
        )
        
        tail_id = self.resolve_mention(
            relation.tail_mention,
            source_text=source_text
        )
        
        return head_id, tail_id
    
    def resolve_relations(
        self,
        relations: List[ParsedRelation],
        source_text: Optional[str] = None
    ) -> List[ParsedRelation]:
        """
        Resolve all relations in a list.
        
        Args:
            relations: List of ParsedRelation objects
            source_text: Optional source text for context
            
        Returns:
            List of ParsedRelation objects with resolved IDs
        """
        resolved = []
        for relation in relations:
            head_id, tail_id = self.resolve_relation(relation, source_text)
            relation.head_id = head_id
            relation.tail_id = tail_id
            resolved.append(relation)
        
        return resolved
