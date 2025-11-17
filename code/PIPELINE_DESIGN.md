# LLM Relation Extraction Pipeline Design

## Overview

This pipeline evaluates different LLM prompting techniques for biomedical relation extraction by comparing their outputs against gold-standard relations. The project focuses on comparing four key prompting methods:

1. **I/O (Input/Output)**: Simple zero-shot prompting - direct relation extraction
2. **CoT (Chain of Thought)**: Step-by-step reasoning for more transparent and accurate extraction
3. **RAG (Retrieval-Augmented Generation)**: Enriches LLM with external knowledge sources (e.g., PubMed abstracts) to improve factual accuracy
4. **ReAct (Reason + Act)**: Combines reasoning with actions (search, API calls) for robust relation extraction in complex texts

The evaluation uses multiple metrics including exact matching, omission/hallucination rates, graph edit distance, and semantic similarity (BERTScore) to comprehensively assess each technique's performance.

## Pipeline Architecture

### 1. Data Loading Layer
**Purpose**: Load input documents and gold relations

**Components**:
- `DocumentLoader`: Loads text files from `clean_text/{split}texts/`
- `GoldRelationsLoader`: Loads gold relations from `gold_relations/{split}/`
- `GlobalEntityMap`: Builds and maintains a global entity registry across all documents
- `Dataset`: Combines documents and gold relations, provides iteration interface

**Data Structures**:
```python
Document:
  - doc_id: str
  - text: str (title + body)

Entity:
  - id: str (global identifier, e.g., "D003409", "6528", "9606")
  - type: str (e.g., "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "OrganismTaxon")
  - mentions: List[Mention] (text, passage_index, char_offset, length)

GlobalEntity:
  - id: str (global identifier)
  - type: str
  - all_mentions: List[Mention] (aggregated across all documents)
  - common_mentions: List[str] (most frequent surface forms)
  - document_count: int (number of documents containing this entity)
  - canonical_name: str (most common mention text)

GoldRelations:
  - doc_id: str
  - entities: List[Entity] (entities present in this document)
  - relations: List[Relation]

Relation:
  - id: str
  - head_id: str (entity identifier - references global entity ID)
  - tail_id: str (entity identifier - references global entity ID)
  - type: str (e.g., "Association", "Positive_Correlation", "Negative_Correlation")
  - novel: str ("Novel" | "No")
```

**Global Entity Map**:
The `GlobalEntityMap` aggregates entities from all documents in the dataset to create a unified entity registry. This is crucial because:
- Entity IDs are **global identifiers** (e.g., "D003409" = "Congenital hypothyroidism" appears across multiple documents)
- The same entity may have different surface forms/mentions in different documents
- This map helps with:
  - **Entity resolution**: Matching LLM-extracted entities to canonical IDs
  - **Prompting**: Providing entity context to LLMs
  - **Evaluation**: Understanding what entities should be recognized
  - **Analysis**: Understanding entity distribution and frequency

### 2. LLM Interface Layer (Modular Prompting)
**Purpose**: Abstract interface for different prompting techniques

**Base Class**: `LLMPrompter`
```python
class LLMPrompter(ABC):
    def __init__(self, entity_map: Optional[GlobalEntityMap] = None):
        """Initialize with optional global entity map for context."""
        self.entity_map = entity_map
    
    @abstractmethod
    def get_response(self, text: str, doc_id: Optional[str] = None) -> str:
        """Get LLM response for a single document text.
        
        Args:
            text: Document text (title + body)
            doc_id: Optional document ID for context
        """
        pass
    
    @abstractmethod
    def get_responses_batch(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> List[str]:
        """Get LLM responses for multiple documents (optional optimization)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this prompting technique."""
        pass
```

**Concrete Implementations**:
- `IOPrompter`: Simple Input/Output prompting (zero-shot, direct relation extraction)
- `ChainOfThoughtPrompter` (CoT): Step-by-step reasoning for relation extraction
- `RAGPrompter`: Retrieval-Augmented Generation - combines LLM with external knowledge sources (e.g., PubMed abstracts) to enrich context
- `ReActPrompter`: Reason + Act framework - combines reasoning with actions (search, API calls) for robust relation extraction

**Key Features**:
- Each prompter can use different LLM APIs (OpenAI, Anthropic, local models, etc.)
- Each prompter defines its own prompt template
- Batch processing support for efficiency
- Error handling and retry logic

**Prompting Technique Details**:

- **I/O Prompter**: Simple, direct prompting without additional context or reasoning steps. Fast but may be less accurate for complex relations.
  - **Prompting strategy**: Should instruct LLM to use exact text spans from the document for entity mentions to maximize exact matches

- **CoT Prompter**: Guides the LLM through step-by-step reasoning (e.g., "First identify entities using exact text from the document, then determine their types, then extract relations"). More transparent and often more accurate.
  - **Prompting strategy**: Each reasoning step should emphasize using exact text spans from the source document

- **RAG Prompter**: 
  - Retrieves relevant context from external knowledge sources (e.g., PubMed abstracts)
  - Enriches the prompt with retrieved information
  - Reduces hallucinations and improves factual accuracy
  - Requires retrieval infrastructure (vector database, embedding model)
  - **Prompting strategy**: Should use retrieved context for understanding, but still extract entity mentions as exact text spans from the original document (not from retrieved context)

- **ReAct Prompter**:
  - Alternates between reasoning steps and actions
  - Actions can include: entity type verification, knowledge base queries, API calls
  - More robust for complex texts but slower and more complex
  - Provides transparent reasoning traces
  - **Prompting strategy**: Actions can verify entity information, but entity mentions should still be extracted as exact text spans from the document

### 3. Response Parsing Layer
**Purpose**: Extract structured relations from LLM text responses

**Components**:
- `ResponseParser`: Base parser interface
- `JSONParser`: Parse JSON-formatted responses
- `TextParser`: Parse natural language responses (regex/LLM-based)
- `StructuredParser`: Parse structured formats (XML, YAML, etc.)
- `EntityResolver`: Resolve entity mentions to global entity IDs using GlobalEntityMap

**Output Format**:
```python
ParsedRelations:
  - relations: List[Relation]
  - entities: Optional[List[Entity]] (if LLM also extracts entities)
  - confidence_scores: Optional[List[float]]
  - parsing_errors: List[str]
  - entity_resolution_errors: List[str]
```

**Entity Resolution**:
The `EntityResolver` component maps LLM-extracted entity mentions to global entity IDs:
- **Exact match**: Match surface form to known mentions in GlobalEntityMap
- **Fuzzy match**: Handle variations, abbreviations, synonyms
- **Type matching**: Consider entity type when resolving
- **Context-aware**: Use document context for disambiguation

**Important Note on Entity Mentions**:
In the gold standard data, entity mentions are **exact text spans** copied directly from the document text. For example:
- "Congenital hypothyroidism" (exact match from title)
- "sodium/iodide symporter" (exact match from text)
- "NIS" (exact match, abbreviation used in text)

**This means that LLM prompts should encourage using exact text spans from the document** for entity mentions, as this will maximize exact matches during entity resolution. If the LLM uses paraphrases or different wording, it will require fuzzy matching which may introduce errors. The pipeline should prioritize exact text span extraction to improve evaluation accuracy.

**Key Features**:
- Handles different output formats
- Error recovery for malformed responses
- Validation of extracted relations
- Entity ID resolution using global entity map

### 4. Evaluation Layer
**Purpose**: Compare predicted relations against gold relations

**Components**:
- `RelationMatcher`: Match predicted relations to gold relations
- `MetricsCalculator`: Compute evaluation metrics
- `EvaluationResult`: Store per-document and aggregate results

**Matching Strategy**:
- **Exact match**: `(head_id, tail_id, type)` tuple - requires all three to match
- **Partial match**: Consider `(head_id, tail_id)` only (ignoring relation type)
- **Entity resolution**: Uses GlobalEntityMap to handle:
  - Entity mention → entity ID mapping
  - Synonym/variant handling
  - Type validation

**Key Insight for Evaluation**:
Since gold standard entity mentions are exact text spans from the document, **LLMs that extract exact text spans will achieve higher exact match rates**. The evaluation pipeline should:
1. First attempt exact text span matching (character-level or word-level)
2. Fall back to fuzzy matching only when exact matches fail
3. Track the proportion of exact vs. fuzzy matches as a quality metric

**Metrics**:

**Quantitative Metrics**:
- **Exact Matching (EM)**: Percentage of correct triplets that exactly match gold-standard relations
- **Omission Rate**: Percentage of missing relations (false negatives)
- **Hallucination Rate**: Percentage of invented relations (false positives that don't exist in gold standard)
- **Redundancy**: Percentage of repeated triplets in predictions
- **Precision**: `TP / (TP + FP)`
- **Recall**: `TP / (TP + FN)`
- **F1-Score**: `2 * (Precision * Recall) / (Precision + Recall)`

**Graph-based Metrics**:
- **Graph Edit Distance (GED)**: Number of edits (additions, deletions, modifications) needed to transform predicted graph to match gold-standard graph

**Semantic Metrics**:
- **BERTScore**: Semantic similarity of predicted vs. gold-standard triplets using contextual embeddings

**Additional Analysis**:
- Per-relation-type metrics
- Novel vs. non-novel relation analysis

**Data Structure**:
```python
EvaluationResult:
  - doc_id: str
  - true_positives: List[Relation]
  - false_positives: List[Relation]
  - false_negatives: List[Relation]
  - precision: float
  - recall: float
  - f1_score: float
  - exact_match_rate: float  # Percentage of exact matches
  - omission_rate: float     # Percentage of missing relations
  - hallucination_rate: float # Percentage of invented relations
  - redundancy_rate: float   # Percentage of repeated triplets
  - graph_edit_distance: float  # GED score
  - bertscore: float         # BERTScore semantic similarity
  - per_type_metrics: Dict[str, Metrics]
```

**Evaluation Challenges**:
- **Ambiguity in correct triples**: Model may output semantically correct relations worded differently than ground truth (e.g., "Haiti is part of CARICOM" vs. "Haiti is member of CARICOM")
- **Hallucinations vs. Incomplete Knowledge**: Distinguishing between invented facts (hallucinations) and omitted valid relations due to lack of context or retrieval

### 5. Comparison & Aggregation Layer
**Purpose**: Aggregate results across documents and compare techniques

**Components**:
- `ResultAggregator`: Aggregate metrics across documents
- `TechniqueComparator`: Compare different prompting techniques
- `ReportGenerator`: Generate comparison reports

**Outputs**:
- Per-document metrics for each technique
- Aggregate metrics (macro/micro averages)
- Statistical significance tests
- Visualization (optional): bar charts, confusion matrices

**Data Structure**:
```python
AggregateResults:
  - technique_name: str  # "I/O", "CoT", "RAG", "ReAct"
  - macro_precision: float
  - macro_recall: float
  - macro_f1: float
  - micro_precision: float
  - micro_recall: float
  - micro_f1: float
  - avg_exact_match_rate: float
  - avg_omission_rate: float
  - avg_hallucination_rate: float
  - avg_redundancy_rate: float
  - avg_graph_edit_distance: float
  - avg_bertscore: float
  - per_document_results: List[EvaluationResult]
```

## Implementation Structure

```
code/
├── PIPELINE_DESIGN.md
├── README.md
├── config.py
├── main.py
├── pipeline
│   ├── __init__.py
│   ├── aggregation
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   └── comparator.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── matcher.py
│   │   └── metrics.py
│   ├── llm_prompter
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cot_prompter.py
│   │   ├── io_prompter.py
│   │   ├── rag_prompter.py
│   │   └── react_prompter.py
│   ├── parsing
│   │   ├── __init__.py
│   │   ├── entity_resolver.py
│   │   └── parser.py
│   ├── retrieval
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── embeddings.py
│   │   ├── pubmed_retriever.py
│   │   └── vector_store.py
│   └── types.py
├── pyproject.toml
├── scripts
│   ├── generate_clean_text_output.py
│   ├── generate_gold_graph_output.py
│   └── graph_viewer_server.py
├── utils
│   ├── __init__.py
│   ├── io.py
│   └── logging.py
└── uv.lock
```

### Module Organization

**1. Data Layer (`pipeline/data/`)**
- Handles all data loading and entity management
- `loader.py`: Loads documents and gold relations from files
- `entity_map.py`: Builds and manages global entity registry

**2. LLM Prompting (`pipeline/llm_prompter/`)**
- Base class and all prompting technique implementations
- Each prompter is self-contained with its own prompt templates
- All prompters inherit from `base.LLMPrompter`

**3. Retrieval (`pipeline/retrieval/`)**
- RAG-specific components for knowledge retrieval
- Supports multiple retrieval backends (PubMed, vector stores, etc.)
- Can be extended with new retrieval methods

**4. Parsing (`pipeline/parsing/`)**
- Parses LLM text responses into structured relations
- Resolves entity mentions to global entity IDs
- Handles different output formats (JSON, text, structured)

**5. Evaluation (`pipeline/evaluation/`)**
- Matches predictions to gold standard
- Computes all evaluation metrics
- Separated into matcher, metrics calculator, and main evaluator

**6. Aggregation (`pipeline/aggregation/`)**
- Aggregates results across documents
- Compares different prompting techniques
- Generates comparison reports

## Configuration

Configuration should include:
- Data paths (clean_text, gold_relations directories)
- LLM API keys and endpoints
- Prompt templates
- Evaluation parameters (matching strategy, metrics)
- Output paths for results
- Logging level

## Global Entity Map Details

The `GlobalEntityMap` is a critical component that provides:

1. **Entity Registry**: Maps entity IDs to canonical entity information
   ```python
   entity_map.get_entity("D003409")
   # Returns: GlobalEntity with id, type, all mentions, common names, etc.
   ```

2. **Entity Lookup**: Find entities by surface form or mention
   ```python
   entity_map.find_entity_by_mention("congenital hypothyroidism")
   # Returns: List of matching GlobalEntity objects
   ```

3. **Statistics**: Entity frequency and distribution
   ```python
   entity_map.get_entity_stats("D003409")
   # Returns: document_count, mention_count, common_mentions, etc.
   ```

4. **Export/Import**: Save and load entity map for reuse
   ```python
   entity_map.save("entity_map.json")
   entity_map = GlobalEntityMap.load("entity_map.json")
   ```

**Use Cases**:
- **Prompting**: Include relevant entity context in prompts (e.g., "Common entities to look for: ...")
- **Entity Resolution**: Map LLM-extracted entity mentions to canonical IDs
- **Evaluation**: Validate that predicted entity IDs exist in the global map
- **Analysis**: Understand entity distribution, most common entities, etc.

## Implementation Steps

### Phase 1: Data Loading & Entity Management
- [ ] **DocumentLoader**: Load text files from `clean_text/{split}texts/`, parse title/body, return `Document` objects
- [ ] **GoldRelationsLoader**: Load JSON files from `gold_relations/{split}/`, parse entities and relations, return `GoldRelations` objects
- [ ] **DatasetLoader**: Combine both loaders, match documents to gold relations by `doc_id`
- [ ] **GlobalEntityMap**: Aggregate entities across documents, implement lookup methods (`get_entity`, `find_entity_by_mention`), add save/load functionality

### Phase 2: LLM Prompting Infrastructure
- [x] **Base Prompter**: Already implemented ✓
- [ ] **IOPrompter**: Simple zero-shot prompting with exact span extraction emphasis
- [ ] **ChainOfThoughtPrompter**: Step-by-step reasoning (identify entities → types → relations)
- [ ] **RAGPrompter**: Retrieval-augmented with context retrieval (placeholder for now)
- [ ] **ReActPrompter**: Reasoning + actions pattern with tool support
- [ ] **Update exports**: Export all prompters in `__init__.py`

### Phase 3: Retrieval Infrastructure (Optional - for RAG)
- [ ] **Base Retriever**: Abstract interface with `retrieve(query, top_k)` method
- [ ] **PubMed Retriever**: Basic PubMed API integration or mock implementation
- [ ] **Vector Store & Embeddings**: Optional - embedding storage and similarity search

### Phase 4: Response Parsing & Entity Resolution
- [ ] **EntityResolver**: Resolve entity mentions to IDs (exact match first, then fuzzy match)
- [ ] **ResponseParser**: Parse LLM JSON responses, extract relations, use EntityResolver to resolve entity IDs

### Phase 5: Evaluation Components
- [ ] **RelationMatcher**: Match predicted relations to gold using `(head_id, tail_id, type)` tuples
- [ ] **MetricsCalculator**: Calculate all metrics:
  - Basic: Precision, Recall, F1, Exact Match Rate, Omission Rate, Hallucination Rate
  - Redundancy: Detect duplicate relations
  - Graph Edit Distance (GED): Transform predicted graph to gold graph
  - BERTScore: Semantic similarity between relations
  - Per-type metrics: Precision/Recall/F1 per relation type
- [ ] **Evaluator**: Combine matcher and metrics calculator, evaluate predictions against gold

### Phase 6: Aggregation & Comparison
- [ ] **ResultAggregator**: Aggregate metrics across documents (macro/micro averages, average rates)
- [ ] **TechniqueComparator**: Compare techniques, rank by metrics, generate and save comparison report

### Phase 7: Main Pipeline Integration
- [ ] **Update main.py**: Uncomment all imports and code, test with small subset
- [ ] **Add progress tracking**: Logging, progress bars, time tracking, error handling

### Phase 8: Configuration & Utilities
- [ ] **Configuration**: Data paths, API keys, evaluation parameters, output paths
- [ ] **Logging**: Structured logging for pipeline steps
- [ ] **I/O Utilities**: JSON save/load helpers

### Phase 9: Testing & Validation
- [ ] **Unit Tests**: Test each component individually
- [ ] **Integration Tests**: Test full pipeline with dev split
- [ ] **End-to-End**: Run complete pipeline, verify all techniques and outputs

### Phase 10: Documentation & Cleanup
- [ ] **Code Documentation**: Docstrings, type hints, comments
- [ ] **README**: Usage instructions, configuration, examples
- [ ] **Final Review**: Code cleanup, remove TODOs, prepare for production

