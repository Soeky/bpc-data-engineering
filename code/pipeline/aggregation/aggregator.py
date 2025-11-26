"""Result aggregator for aggregating results across documents."""

from typing import List

from ..types import EvaluationResult, AggregateResults


class ResultAggregator:
    """Aggregates evaluation results across documents."""
    
    def aggregate(
        self,
        eval_results: List[EvaluationResult],
        technique_name: str
    ) -> AggregateResults:
        """
        Aggregate evaluation results across documents.
        
        Args:
            eval_results: List of per-document evaluation results
            technique_name: Name of the prompting technique
            
        Returns:
            AggregateResults object with aggregated metrics
        """
        if not eval_results:
            return AggregateResults(
                technique_name=technique_name,
                per_document_results=eval_results
            )
        
        n = len(eval_results)
        
        # Macro averages (average of per-document metrics)
        macro_precision = sum(r.precision for r in eval_results) / n
        macro_recall = sum(r.recall for r in eval_results) / n
        macro_f1 = sum(r.f1_score for r in eval_results) / n
        
        # Micro averages (calculated from aggregated TP/FP/FN)
        total_tp = sum(len(r.true_positives) for r in eval_results)
        total_fp = sum(len(r.false_positives) for r in eval_results)
        total_fn = sum(len(r.false_negatives) for r in eval_results)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0.0
        )
        
        # Average rates
        avg_exact_match_rate = sum(r.exact_match_rate for r in eval_results) / n
        avg_omission_rate = sum(r.omission_rate for r in eval_results) / n
        avg_hallucination_rate = sum(r.hallucination_rate for r in eval_results) / n
        avg_redundancy_rate = sum(r.redundancy_rate for r in eval_results) / n
        avg_graph_edit_distance = sum(r.graph_edit_distance for r in eval_results) / n
        avg_bertscore = sum(r.bertscore for r in eval_results) / n
        
        # Calculate partial match statistics (fuzzy matches)
        total_partial_matches = sum(len(r.partial_matches) for r in eval_results)
        avg_partial_matches = total_partial_matches / n if n > 0 else 0.0
        
        # Fuzzy macro averages (average of per-document fuzzy metrics)
        fuzzy_macro_precision = sum(r.fuzzy_precision for r in eval_results) / n
        fuzzy_macro_recall = sum(r.fuzzy_recall for r in eval_results) / n
        fuzzy_macro_f1 = sum(r.fuzzy_f1 for r in eval_results) / n
        
        # Fuzzy micro averages (calculated from aggregated TP/FP/FN including partial matches)
        # Fuzzy TP = exact TP + partial matches
        fuzzy_tp = total_tp + total_partial_matches
        # For fuzzy FP, exclude partial matches from false positives
        fuzzy_fp = total_fp - total_partial_matches
        fuzzy_micro_precision = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if (fuzzy_tp + fuzzy_fp) > 0 else 0.0
        fuzzy_micro_recall = fuzzy_tp / (fuzzy_tp + total_fn) if (fuzzy_tp + total_fn) > 0 else 0.0
        fuzzy_micro_f1 = (
            2 * (fuzzy_micro_precision * fuzzy_micro_recall) / (fuzzy_micro_precision + fuzzy_micro_recall)
            if (fuzzy_micro_precision + fuzzy_micro_recall) > 0 else 0.0
        )
        
        return AggregateResults(
            technique_name=technique_name,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            avg_exact_match_rate=avg_exact_match_rate,
            avg_omission_rate=avg_omission_rate,
            avg_hallucination_rate=avg_hallucination_rate,
            avg_redundancy_rate=avg_redundancy_rate,
            avg_graph_edit_distance=avg_graph_edit_distance,
            avg_bertscore=avg_bertscore,
            total_partial_matches=total_partial_matches,
            avg_partial_matches=avg_partial_matches,
            fuzzy_micro_precision=fuzzy_micro_precision,
            fuzzy_micro_recall=fuzzy_micro_recall,
            fuzzy_micro_f1=fuzzy_micro_f1,
            fuzzy_macro_precision=fuzzy_macro_precision,
            fuzzy_macro_recall=fuzzy_macro_recall,
            fuzzy_macro_f1=fuzzy_macro_f1,
            per_document_results=eval_results
        )
