# Retrieval-Augmented Generation (RAG)

**Definition:**  
- Combines LLM + external knowledge source  

**Retriever sources:**  
- PubMed Abstracts → medicine  
- Legal documents → law  
- Wikidata / DBpedia → general  

**Process:**  
1. Input text contains entity (e.g., "CARICOM")  
2. Retriever provides context ("CARICOM is an international organization")  
3. LLM combines text + retrieved knowledge → more precise relations  

**Example:**  
- Without RAG: ("Haiti", "related to", "CARICOM") imprecise  
- With RAG: ("Haiti", "member of", "CARICOM") correct  

**Pros:** fewer hallucinations, higher factual accuracy  
**Cons:** depends on retrieval quality, more infrastructure needed  


# ReAct (Reason + Act)

**Definition:**  
- Combines logical reasoning + external actions (search, tools)  

**Process:**  
1. Reasoning → plan steps ("CARICOM → is it an organization?")  
2. Action → external lookup  
3. Combine → final relation extraction  

**Example:**  
- Input: "Haiti is a member of CARICOM."  
- Step 1: Reason → check entity type  
- Step 2: Action → verify with knowledge source  
- Step 3: Output → ("Haiti", "member of", "CARICOM")  

**Pros:** less error, transparent steps, good for complex texts  
**Cons:** more complex, slower, requires tool support  


# Showcase Plan

**Demo flow:**  
1. Input text:  
   *"Haiti is a member of CARICOM and the United Nations."*  

2. Compare prompting techniques:  
   - I/O → ("Haiti", "related to", "CARICOM")
   - CoT → ("Haiti", "member of", "CARICOM"), ("Haiti", "member of", "UN")
   - RAG → adds context from knowledge source → precise triplets 
   - ReAct → shows reasoning + external check → robust triplets 

3. Optional: visualize as a small knowledge graph  

**Goal:**  
- Show clear differences between prompting techniques  
- Highlight how RAG / ReAct reduce hallucinations  


# Evaluation Metrics

**Quantitative metrics:**  
- Exact Matching → percentage of correct triplets  
- Omission Rate → missing relations  
- Hallucination Rate → invented relations  
- Redundancy → repeated triplets  

**Graph-based metric:**  
- Graph Edit Distance → number of edits to match gold-standard graph  

**Semantic metric:**  
- BERTScore → semantic similarity of predicted vs. gold-standard triplets  


# Conclusion

- Relation Extraction is key for building Knowledge Graphs  
- Prompting techniques have different strengths:  
  - I/O → simple but often inaccurate  
  - CoT → step-by-step reasoning, more transparent  
  - RAG → adds external knowledge, improves factual accuracy  
  - ReAct → combines reasoning + actions, robust but complex  

**Project goal:**  
- Evaluate which technique works best for knowledge-intensive domains  

**Key message:**  
*"The choice of prompting method determines the quality of the knowledge graph."*  
