"""Main pipeline orchestration.

This script runs the complete relation extraction pipeline:
1. Loads test data (documents and gold relations)
2. Builds global entity map
3. Runs all prompting techniques on all documents
4. Parses LLM responses
5. Evaluates predictions against gold standard
6. Aggregates and compares results across techniques
"""

from pathlib import Path
from typing import Dict, List

# TODO: Update imports once modules are implemented
# from pipeline.data.loader import DatasetLoader
# from pipeline.data.entity_map import GlobalEntityMap
# from pipeline.llm_prompter import (
#     IOPrompter,
#     ChainOfThoughtPrompter,
#     RAGPrompter,
#     ReActPrompter,
# )
# from pipeline.parsing.parser import ResponseParser
# from pipeline.evaluation.evaluator import Evaluator
# from pipeline.aggregation.aggregator import ResultAggregator
# from pipeline.aggregation.comparator import TechniqueComparator

from pipeline.types import Document, GoldRelations, ParsedRelations, EvaluationResult, AggregateResults


def main():
    """Run the complete relation extraction pipeline."""
    # ========== Configuration ==========
    base_path = Path(__file__).parent
    clean_text_path = base_path / "clean_text"
    gold_relations_path = base_path / "gold_relations"
    split = "test"  # "dev", "test", or "train"
    output_dir = base_path / "results"
    output_dir.mkdir(exist_ok=True)
    
    # ========== Step 1: Load Data ==========
    print("=" * 60)
    print("Step 1: Loading data...")
    print("=" * 60)
    # TODO: Uncomment when implemented
    # loader = DatasetLoader(clean_text_path, gold_relations_path)
    # documents, gold_relations = loader.load(split)
    # For now, placeholder
    documents: List[Document] = []
    gold_relations: List[GoldRelations] = []
    print(f"Loaded {len(documents)} documents")
    
    # ========== Step 2: Build Global Entity Map ==========
    print("\n" + "=" * 60)
    print("Step 2: Building global entity map...")
    print("=" * 60)
    # TODO: Uncomment when implemented
    # entity_map = GlobalEntityMap()
    # entity_map.build_from_gold_relations(gold_relations)
    entity_map = None
    # print(f"Entity map contains {len(entity_map)} entities")
    
    # ========== Step 3: Initialize Components ==========
    print("\n" + "=" * 60)
    print("Step 3: Initializing components...")
    print("=" * 60)
    
    # Initialize all prompting techniques
    # TODO: Uncomment when implemented
    # prompters = [
    #     IOPrompter(entity_map=entity_map, use_exact_spans=True),
    #     ChainOfThoughtPrompter(entity_map=entity_map, use_exact_spans=True),
    #     RAGPrompter(entity_map=entity_map, use_exact_spans=True),
    #     ReActPrompter(entity_map=entity_map, use_exact_spans=True),
    # ]
    prompters = []
    
    # Initialize parser and evaluator
    # TODO: Uncomment when implemented
    # parser = ResponseParser(entity_map=entity_map)
    # evaluator = Evaluator(entity_map=entity_map)
    parser = None
    evaluator = None
    
    # Initialize aggregator and comparator
    # TODO: Uncomment when implemented
    # aggregator = ResultAggregator()
    # comparator = TechniqueComparator()
    aggregator = None
    comparator = None
    
    print(f"Initialized {len(prompters)} prompting techniques")
    
    # ========== Step 4: Run Pipeline for Each Technique ==========
    print("\n" + "=" * 60)
    print("Step 4: Running pipeline for each technique...")
    print("=" * 60)
    
    # Store results: technique_name -> list of evaluation results per document
    all_results: Dict[str, List[EvaluationResult]] = {}
    
    for prompter in prompters:
        print(f"\n--- Processing with {prompter.name} prompter ---")
        print(f"Processing {len(documents)} documents...")
        
        # Store predictions for this technique
        predictions: List[ParsedRelations] = []
        
        # Process each document
        for i, doc in enumerate(documents, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(documents)} documents")
            
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
        
        print(f"  Completed: {len(predictions)} documents processed")
        
        # ========== Step 5: Evaluate Predictions ==========
        print(f"\n  Evaluating {prompter.name} predictions...")
        # TODO: Uncomment when implemented
        # eval_results = evaluator.evaluate(predictions, gold_relations)
        eval_results: List[EvaluationResult] = []
        
        all_results[prompter.name] = eval_results
        
        # Print summary for this technique
        if eval_results:
            avg_precision = sum(r.precision for r in eval_results) / len(eval_results)
            avg_recall = sum(r.recall for r in eval_results) / len(eval_results)
            avg_f1 = sum(r.f1_score for r in eval_results) / len(eval_results)
            print(f"  {prompter.name} Summary:")
            print(f"    Avg Precision: {avg_precision:.3f}")
            print(f"    Avg Recall: {avg_recall:.3f}")
            print(f"    Avg F1: {avg_f1:.3f}")
    
    # ========== Step 6: Aggregate Results ==========
    print("\n" + "=" * 60)
    print("Step 6: Aggregating results...")
    print("=" * 60)
    
    aggregated_results: Dict[str, AggregateResults] = {}
    for technique_name, eval_results in all_results.items():
        # TODO: Uncomment when implemented
        # aggregated = aggregator.aggregate(eval_results, technique_name)
        # aggregated_results[technique_name] = aggregated
        pass
    
    # ========== Step 7: Compare Techniques ==========
    print("\n" + "=" * 60)
    print("Step 7: Comparing techniques...")
    print("=" * 60)
    
    # TODO: Uncomment when implemented
    # comparison_report = comparator.compare(aggregated_results)
    
    # ========== Step 8: Save Results ==========
    print("\n" + "=" * 60)
    print("Step 8: Saving results...")
    print("=" * 60)
    
    # Save comparison report
    report_path = output_dir / f"comparison_{split}.json"
    # TODO: Uncomment when implemented
    # comparator.save_report(aggregated_results, str(report_path))
    print(f"Saved comparison report to: {report_path}")
    
    # Save per-document results for each technique
    for technique_name, eval_results in all_results.items():
        results_path = output_dir / f"{technique_name}_{split}_results.json"
        # TODO: Save per-document results
        print(f"Saved {technique_name} results to: {results_path}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    print(f"Processed {len(documents)} documents")
    print(f"Evaluated {len(prompters)} prompting techniques")
    print(f"Results saved to: {output_dir}")
    
    # Print final comparison table
    print("\nFinal Comparison:")
    print("-" * 60)
    # TODO: Print comparison table when implemented
    print("(Comparison table will be displayed here)")


if __name__ == "__main__":
    main()
