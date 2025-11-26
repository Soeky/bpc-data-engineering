"""Response parser for LLM outputs."""

import json
import re
import logging
from typing import List, Optional

from ..types import ParsedRelations, ParsedRelation
from .entity_resolver import EntityResolver


class ResponseParser:
    """Parses LLM text responses into structured relations."""
    
    def __init__(self, entity_map=None, logger: Optional[logging.Logger] = None):
        """
        Initialize response parser.
        
        Args:
            entity_map: Optional global entity map for entity resolution
            logger: Optional logger instance
        """
        self.entity_resolver = EntityResolver(entity_map) if entity_map else None
        self.logger = logger or logging.getLogger(__name__)
    
    def parse(
        self, 
        response: str, 
        doc_id: Optional[str] = None,
        source_text: Optional[str] = None
    ) -> ParsedRelations:
        """
        Parse LLM response into structured relations.
        
        Args:
            response: LLM response text
            doc_id: Optional document ID
            source_text: Optional source text for entity resolution
            
        Returns:
            ParsedRelations object
        """
        self.logger.info(f"[Parser] Parsing response for document: {doc_id}")
        parsed = ParsedRelations(doc_id=doc_id)
        
        # Try to extract JSON from response
        json_data = self._extract_json(response)
        
        if json_data:
            try:
                relations_data = json_data
                if isinstance(relations_data, dict) and "relations" in relations_data:
                    relations_data = relations_data["relations"]
                elif not isinstance(relations_data, list):
                    relations_data = [relations_data]
                
                for rel_data in relations_data:
                    if isinstance(rel_data, dict):
                        relation = ParsedRelation(
                            head_mention=rel_data.get("head_mention", "").strip(),
                            tail_mention=rel_data.get("tail_mention", "").strip(),
                            relation_type=rel_data.get("relation_type", "").strip(),
                            confidence=rel_data.get("confidence")
                        )
                        
                        if relation.head_mention and relation.tail_mention and relation.relation_type:
                            parsed.relations.append(relation)
                
                self.logger.info(f"[Parser] Extracted {len(parsed.relations)} relations from JSON")
                
            except Exception as e:
                error_msg = f"Error parsing JSON: {e}"
                parsed.parsing_errors.append(error_msg)
                self.logger.warning(f"[Parser] {error_msg}")
        else:
            # Try text-based parsing as fallback
            error_msg = "No JSON found, attempting text parsing"
            parsed.parsing_errors.append(error_msg)
            self.logger.warning(f"[Parser] {error_msg}")
            text_relations = self._parse_text_format(response)
            parsed.relations.extend(text_relations)
            self.logger.info(f"[Parser] Extracted {len(text_relations)} relations from text format")
        
        # Log parsed relations
        self.logger.debug(f"[Parser] Parsed {len(parsed.relations)} relations:")
        for i, rel in enumerate(parsed.relations, 1):
            self.logger.debug(
                f"[Parser]   Relation {i}: {rel.head_mention} -> {rel.tail_mention} "
                f"({rel.relation_type})"
            )
        
        # Resolve entity IDs if entity resolver is available
        if self.entity_resolver and parsed.relations:
            self.logger.info(f"[Parser] Resolving entity IDs for {len(parsed.relations)} relations...")
            resolved_relations = self.entity_resolver.resolve_relations(
                parsed.relations,
                source_text=source_text
            )
            
            # Track resolution errors
            resolved_count = 0
            for relation in resolved_relations:
                if not relation.head_id:
                    error_msg = f"Could not resolve head entity: {relation.head_mention}"
                    parsed.entity_resolution_errors.append(error_msg)
                    self.logger.warning(f"[Parser] {error_msg}")
                else:
                    resolved_count += 1
                    
                if not relation.tail_id:
                    error_msg = f"Could not resolve tail entity: {relation.tail_mention}"
                    parsed.entity_resolution_errors.append(error_msg)
                    self.logger.warning(f"[Parser] {error_msg}")
                else:
                    resolved_count += 1
            
            self.logger.info(
                f"[Parser] Resolved {resolved_count}/{len(resolved_relations) * 2} entity IDs"
            )
            parsed.relations = resolved_relations
        
        # Log parsing errors if any
        if parsed.parsing_errors:
            self.logger.warning(f"[Parser] Parsing errors: {len(parsed.parsing_errors)}")
            for error in parsed.parsing_errors:
                self.logger.debug(f"[Parser]   Error: {error}")
        
        if parsed.entity_resolution_errors:
            self.logger.warning(f"[Parser] Entity resolution errors: {len(parsed.entity_resolution_errors)}")
            for error in parsed.entity_resolution_errors:
                self.logger.debug(f"[Parser]   Resolution Error: {error}")
        
        return parsed
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from text response.
        
        Args:
            text: Response text
            
        Returns:
            Parsed JSON dict or None
        """
        # Try to find JSON array or object
        json_patterns = [
            r'\[[\s\S]*\]',  # JSON array
            r'\{[\s\S]*\}',  # JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Try parsing the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _parse_text_format(self, text: str) -> List[ParsedRelation]:
        """
        Parse relations from natural language text (fallback).
        
        Args:
            text: Response text
            
        Returns:
            List of ParsedRelation objects
        """
        relations = []
        
        # Look for patterns like "Entity1 -> Entity2: RelationType"
        pattern = r'([^->:]+)\s*->\s*([^->:]+)\s*:\s*([^\n]+)'
        matches = re.findall(pattern, text)
        
        for head, tail, rel_type in matches:
            relation = ParsedRelation(
                head_mention=head.strip(),
                tail_mention=tail.strip(),
                relation_type=rel_type.strip()
            )
            relations.append(relation)
        
        return relations
