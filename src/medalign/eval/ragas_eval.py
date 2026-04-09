"""RAGAS evaluation: faithfulness, answer relevancy, context precision/recall."""
from __future__ import annotations


def evaluate_rag(samples: list[dict]) -> dict:
    """samples: list of {'question', 'answer', 'contexts': [str], 'ground_truth'}"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    ds = Dataset.from_list(samples)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    return {k: float(v) for k, v in result.items()}
