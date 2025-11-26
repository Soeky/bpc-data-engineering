"""Main pipeline orchestration.

This script runs the complete relation extraction pipeline:
1. Loads test data (documents and gold relations)
2. Builds global entity map
3. Runs all prompting techniques on all documents
4. Parses LLM responses
5. Evaluates predictions against gold standard
6. Aggregates and compares results across techniques
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from utils.logging import setup_logger, get_log_file_path

from pipeline.data import DatasetLoader, GlobalEntityMap
from pipeline.llm_prompter import (
    IOPrompter,
    ChainOfThoughtPrompter,
    RAGPrompter,
    ReActPrompter,
)
from pipeline.parsing import ResponseParser
from pipeline.evaluation import Evaluator
from pipeline.aggregation import ResultAggregator, TechniqueComparator

from pipeline.types import Document, GoldRelations, ParsedRelations, EvaluationResult, AggregateResults


def main(
    split: str = "train",
    models: Optional[Dict[str, str]] = None,
    techniques: Optional[List[str]] = None,
    max_documents: Optional[int] = None,
):
    """
    Run the complete relation extraction pipeline.
    
    Args:
        split: Data split to use ("dev", "test", or "train")
        models: Optional dict mapping technique names to model keys
                e.g., {"IO": "gpt-4o-mini", "CoT": "gpt-4o"}
        techniques: Optional list of techniques to run (defaults to all)
                   e.g., ["IO", "CoT", "RAG", "ReAct"]
        max_documents: Optional limit on number of documents to process (for testing)
    """
    # ========== Configuration ==========
    Config.validate()
    
    clean_text_path = Config.CLEAN_TEXT_PATH
    gold_relations_path = Config.GOLD_RELATIONS_PATH
    
    # Create run-specific directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Config.OUTPUT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summaries subdirectory
    summaries_dir = run_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    
    # ========== Setup Logging ==========
    log_file = run_dir / f"pipeline_{split}_{timestamp}.log" if Config.LOG_TO_FILE else None
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    logger = setup_logger(
        name="pipeline",
        log_file=log_file,
        level=log_level,
        console=Config.LOG_TO_CONSOLE
    )
    
    logger.info("=" * 80)
    logger.info("Starting Relation Extraction Pipeline")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Split: {split}")
    logger.info(f"Max documents: {max_documents if max_documents else 'All'}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Default to all techniques if not specified
    if techniques is None:
        techniques = ["IO", "CoT", "RAG", "ReAct"]
    
    # Default models (can be overridden)
    if models is None:
        models = {}
    
    # ========== Step 1: Load Data ==========
    logger.info("=" * 80)
    logger.info("Step 1: Loading data...")
    logger.info("=" * 80)
    loader = DatasetLoader(clean_text_path, gold_relations_path, logger=logger)
    documents, gold_relations = loader.load(split)
    # logger.info(f"Loaded {len(documents)} documents")
    
    # Log gold relations file names
    logger.info("Gold relations files loaded:")
#   for gold in gold_relations:
#       file_name = Path(gold.file_path).name if gold.file_path else "unknown"
#       logger.info(f"  - {file_name} (doc_id: {gold.doc_id}, {len(gold.relations)} relations)")
    
    # Limit documents if max_documents is specified
    if max_documents and max_documents > 0:
        original_count = len(documents)
        documents = documents[:max_documents]
        gold_relations = gold_relations[:max_documents]
        logger.info(f"Limited to {len(documents)} documents (from {original_count})")
    
    # ========== Step 2: Build Global Entity Map ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Building global entity map...")
    logger.info("=" * 80)
    entity_map = GlobalEntityMap()
    entity_map.build_from_gold_relations(gold_relations)
    logger.info(f"Entity map contains {len(entity_map)} entities")
    
    # ========== Step 3: Initialize Components ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Initializing components...")
    logger.info("=" * 80)
    
    # Initialize all prompting techniques with model configuration
    prompters = []
    
    if "IO" in techniques:
        model = models.get("IO")
        prompter = IOPrompter(
            entity_map=entity_map,
            use_exact_spans=True,
            model=model,
            logger=logger
        )
        prompters.append(prompter)
        logger.info(f"  Initialized {prompter.name} prompter with model: {prompter.model}")
    
    if "CoT" in techniques:
        model = models.get("CoT")
        prompter = ChainOfThoughtPrompter(
            entity_map=entity_map,
            use_exact_spans=True,
            model=model,
            logger=logger
        )
        prompters.append(prompter)
        logger.info(f"  Initialized {prompter.name} prompter with model: {prompter.model}")
    
    if "RAG" in techniques:
        model = models.get("RAG")
        prompter = RAGPrompter(
            entity_map=entity_map,
            use_exact_spans=True,
            model=model,
            logger=logger
        )
        prompters.append(prompter)
        logger.info(f"  Initialized {prompter.name} prompter with model: {prompter.model}")
        logger.info(f"    RAG source directory: {Config.RAG_SOURCE_DIR}")
        logger.info(f"    RAG embeddings directory: {Config.RAG_EMBEDDINGS_DIR}")
    
    if "ReAct" in techniques:
        model = models.get("ReAct")
        prompter = ReActPrompter(
            entity_map=entity_map,
            use_exact_spans=True,
            model=model,
            logger=logger
        )
        prompters.append(prompter)
        logger.info(f"  Initialized {prompter.name} prompter with model: {prompter.model}")
    
    # Initialize parser and evaluator
    parser = ResponseParser(entity_map=entity_map, logger=logger)
    evaluator = Evaluator(entity_map=entity_map, logger=logger)
    
    # Initialize aggregator and comparator
    aggregator = ResultAggregator()
    comparator = TechniqueComparator()
    
    logger.info(f"Initialized {len(prompters)} prompting techniques")
    
    # ========== Step 4: Run Pipeline for Each Technique ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Running pipeline for each technique...")
    logger.info("=" * 80)
    
    # Store results: technique_name -> list of evaluation results per document
    all_results: Dict[str, List[EvaluationResult]] = {}
    aggregated_results: Dict[str, AggregateResults] = {}
    
    for prompter in prompters:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing with {prompter.name} prompter")
        logger.info(f"{'=' * 80}")
        logger.info(f"Processing {len(documents)} documents...")
        
        # Store predictions for this technique
        predictions: List[ParsedRelations] = []
        
        # Process each document
        for i, doc in enumerate(documents, 1):
            logger.info(f"\n--- Document {i}/{len(documents)}: {doc.doc_id} ---")
            logger.info(f"Document text preview: {doc.text[:200]}...")
            
            # Get LLM response
            response = prompter.get_response(doc.text, doc_id=doc.doc_id)
            
            # Parse response
            parsed = parser.parse(
                response, 
                doc_id=doc.doc_id, 
                source_text=doc.text
            )
            parsed.doc_id = doc.doc_id  # Ensure doc_id is set
            predictions.append(parsed)
            
            logger.info(
                f"Document {doc.doc_id}: Parsed {len(parsed.relations)} relations, "
                f"{len(parsed.parsing_errors)} parsing errors, "
                f"{len(parsed.entity_resolution_errors)} resolution errors"
            )
        
        logger.info(f"Completed: {len(predictions)} documents processed")
        
        # ========== Step 5: Evaluate Predictions ==========
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating {prompter.name} predictions...")
        logger.info(f"{'=' * 80}")
        eval_results = evaluator.evaluate(predictions, gold_relations)
        
        all_results[prompter.name] = eval_results
        
        # Aggregate results for this technique
        aggregated = aggregator.aggregate(eval_results, prompter.name)
        
        # Print detailed summary for this technique
        logger.info(f"\n{'=' * 80}")
        logger.info(f"{prompter.name} - Aggregated Results")
        logger.info(f"{'=' * 80}")
        logger.info(f"Number of documents: {len(eval_results)}")
        logger.info(f"\nMacro Averages (average of per-document metrics):")
        logger.info(f"  Precision: {aggregated.macro_precision:.3f}")
        logger.info(f"  Recall: {aggregated.macro_recall:.3f}")
        logger.info(f"  F1 Score: {aggregated.macro_f1:.3f}")
        logger.info(f"\nMicro Averages (calculated from aggregated TP/FP/FN):")
        logger.info(f"  Precision: {aggregated.micro_precision:.3f}")
        logger.info(f"  Recall: {aggregated.micro_recall:.3f}")
        logger.info(f"  F1 Score: {aggregated.micro_f1:.3f}")
        logger.info(f"\nAdditional Metrics:")
        logger.info(f"  Exact Match Rate: {aggregated.avg_exact_match_rate:.3f}")
        logger.info(f"  Omission Rate: {aggregated.avg_omission_rate:.3f}")
        logger.info(f"  Hallucination Rate: {aggregated.avg_hallucination_rate:.3f}")
        logger.info(f"  Redundancy Rate: {aggregated.avg_redundancy_rate:.3f}")
        logger.info(f"  Graph Edit Distance: {aggregated.avg_graph_edit_distance:.2f}")
        
        logger.info(f"\nFuzzy/Partial Match Statistics (entities correct, type may differ):")
        logger.info(f"  Total Partial Matches: {aggregated.total_partial_matches}")
        logger.info(f"  Avg Partial Matches per Document: {aggregated.avg_partial_matches:.2f}")
        logger.info(f"\nFuzzy Macro Averages (average of per-document fuzzy metrics):")
        logger.info(f"  Fuzzy Precision: {aggregated.fuzzy_macro_precision:.3f}")
        logger.info(f"  Fuzzy Recall: {aggregated.fuzzy_macro_recall:.3f}")
        logger.info(f"  Fuzzy F1 Score: {aggregated.fuzzy_macro_f1:.3f}")
        logger.info(f"\nFuzzy Micro Averages (calculated from aggregated TP/FP/FN):")
        logger.info(f"  Fuzzy Precision: {aggregated.fuzzy_micro_precision:.3f}")
        logger.info(f"  Fuzzy Recall: {aggregated.fuzzy_micro_recall:.3f}")
        logger.info(f"  Fuzzy F1 Score: {aggregated.fuzzy_micro_f1:.3f}")
        
        # Store aggregated results for later comparison
        aggregated_results[prompter.name] = aggregated
        
        # Save summary file for this technique
        import json
        summary_path = summaries_dir / f"{prompter.name}_summary.json"
        
        # Get the prompt template (use a sample document to build it)
        # Replace actual text with placeholder to show template structure
        sample_text = "Sample document text for prompt template demonstration."
        sample_prompt = prompter._build_prompt(sample_text, "sample_doc_id") if hasattr(prompter, '_build_prompt') else ""
        
        summary = {
            "technique": prompter.name,
            "model": prompter.model,
            "split": split,
            "num_documents": len(eval_results),
            "timestamp": timestamp,
            "prompt_template": sample_prompt,
            "exact_match_metrics": {
                "macro_precision": aggregated.macro_precision,
                "macro_recall": aggregated.macro_recall,
                "macro_f1": aggregated.macro_f1,
                "micro_precision": aggregated.micro_precision,
                "micro_recall": aggregated.micro_recall,
                "micro_f1": aggregated.micro_f1,
                "avg_exact_match_rate": aggregated.avg_exact_match_rate,
                "avg_omission_rate": aggregated.avg_omission_rate,
                "avg_hallucination_rate": aggregated.avg_hallucination_rate,
                "avg_redundancy_rate": aggregated.avg_redundancy_rate,
                "avg_graph_edit_distance": aggregated.avg_graph_edit_distance,
            },
            "fuzzy_match_metrics": {
                "total_partial_matches": aggregated.total_partial_matches,
                "avg_partial_matches": aggregated.avg_partial_matches,
                "fuzzy_macro_precision": aggregated.fuzzy_macro_precision,
                "fuzzy_macro_recall": aggregated.fuzzy_macro_recall,
                "fuzzy_macro_f1": aggregated.fuzzy_macro_f1,
                "fuzzy_micro_precision": aggregated.fuzzy_micro_precision,
                "fuzzy_micro_recall": aggregated.fuzzy_micro_recall,
                "fuzzy_micro_f1": aggregated.fuzzy_micro_f1,
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved {prompter.name} summary to: {summary_path}")
    
    # ========== Step 6: Aggregate Results ==========
    # Note: Aggregation already done per-technique above, this is just for consistency
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Final aggregation check...")
    logger.info("=" * 80)
    logger.info(f"Aggregated results for {len(aggregated_results)} techniques")
    
    # ========== Step 7: Compare Techniques ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Comparing techniques...")
    logger.info("=" * 80)
    
    comparison_report = comparator.compare(aggregated_results)
    comparator.print_comparison_table(aggregated_results)
    
    # ========== Step 8: Save Results ==========
    logger.info("\n" + "=" * 80)
    logger.info("Step 8: Saving results...")
    logger.info("=" * 80)
    
    # Save comparison report
    report_path = run_dir / f"comparison_{split}.json"
    comparator.save_report(aggregated_results, str(report_path))
    logger.info(f"Saved comparison report to: {report_path}")
    
    # Save per-document results for each technique
    import json
    for technique_name, eval_results in all_results.items():
        results_path = run_dir / f"{technique_name}_{split}_results.json"
        # Convert EvaluationResult objects to dicts for JSON serialization
        results_dict = {
            "technique": technique_name,
            "num_documents": len(eval_results),
            "results": [
                {
                    "doc_id": r.doc_id,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1_score": r.f1_score,
                    "exact_match_rate": r.exact_match_rate,
                    "omission_rate": r.omission_rate,
                    "hallucination_rate": r.hallucination_rate,
                    "redundancy_rate": r.redundancy_rate,
                    "graph_edit_distance": r.graph_edit_distance,
                    "num_true_positives": len(r.true_positives),
                    "num_false_negatives": len(r.false_negatives),
                }
                for r in eval_results
            ]
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Saved {technique_name} results to: {results_path}")
    
    # ========== Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed!")
    logger.info("=" * 80)
    logger.info(f"Processed {len(documents)} documents")
    logger.info(f"Evaluated {len(prompters)} prompting techniques")
    logger.info(f"Results saved to: {run_dir}")
    logger.info(f"Summaries saved to: {summaries_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Final comparison table already printed in Step 7


if __name__ == "__main__":
    main(
        split="train",
        max_documents=1,
        techniques=["IO", "CoT"],
        models={"IO": "gpt-4o-mini", "CoT": "gpt-4o-mini"},
    )
