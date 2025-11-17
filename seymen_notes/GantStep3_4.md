# 1. Dataset Structure and Purpose

Our dataset is split into three BioC JSON files:

Train Set -> used for development
Dev Set -> used for prompt tuning & method selection
Test Set -> used for final evaluation only

Each file contains:
Biomedical text passages (title, abstract)
Entity annotations (surface strings + IDs + types)
Relation annotations (gold triples)

These annotations are not given to the LLM. They are only used for post-processing, graph construction and evaluation. 

---

# 2. Rule for LLM Inputs and Outputs

### LLM Input

The LLM receives only natural text from the documents:

- No entity identifiers
- No BioC metadata
- No ground-truth relations

### LLM Output

The LLM produces surface-form triples, e.g.:

```
(CenpH, regulates, cyclin B1)
```

### After the LLM

We map the predicted surface strings -> BioC identifiers
(using the annotations in the dataset) and evaluate against the gold graph.

---

# 3. Gantt Step Summary with Dataset Usage

## Dataset Selection & Preparation

Dataset use: All splits, but primarily train
Tasks include:

- Parsing BioC JSON
- Normalizing entity types and identifiers
- Preparing prompt-ready plain-text samples
- Constructing gold triples per document
- Splitting into train/dev/test (if not already split)

Output of this stage:

- Cleaned dataset
- Text-only input sequences
- Gold graphs for later evaluation

## Define Evaluation Metrics & Pipeline

Dataset use: Train for debugging, Dev for calibration
Evaluation metrics implemented:

* Exact Match
* Omission Rate
* Hallucination Rate
* Redundancy (BERTScore?)
* Graph Edit Distance (GED)

## Prompt Design & Model Configuration**

Dataset use: Train set
We build:

* I/O prompts
* Chain-of-Thought prompts
* RAG (retrieval-augmented) setups
* ReAct (reasoning + action) prompts

Few-shot examples (if used) must come from the train set only.

LLM still sees only natural text, not entity IDs.

## Development Experiments & Prompt Tuning**

Dataset use: Dev set
We run controlled experiments to choose the best prompting strategy.

Per method, we compare:

* accuracy
* hallucination rate
* omission rate
* GED distance
* robustness to phrasing changes

We iterate only on dev, never on test.

## **Step 7 — Final Evaluation**

Dataset use: Test set only
We run the best-performing prompt for each technique on the test set.

This stage produces the final numbers for the:

* final tables,
* final plots,
* and the results used in the project report.

Again, the LLM only sees:

* the raw text,
* plus any markers we intentionally include for experiments.

## Analysis, Comparison & Reporting**

Dataset use: Test results
We compare prompting techniques and interpret:

* differences in extraction quality
* types of errors
* relation-type sensitivity
* graph structure differences (via GED)

We finalize the documentation and presentation.

# 4. Final Notes

* **Identifiers (NCBI Gene, MeSH, etc.) are *only* for evaluation and graph-building.**
* **The LLM must never see the identifiers or the gold annotations.**
* **Train → build & design**
* **Dev → tune & compare**
* **Test → final results**

This ensures a clean, unbiased, scientifically valid evaluation of all prompting strategies.
