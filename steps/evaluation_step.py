from typing import Dict, List

from zenml import step

from src.audio_classifier.evaluation import (
    calculate_metrics,
)


@step
def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Calculate classification metrics from
    model predictions.
    """

    print("Calculating evaluation metrics...")

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print("\nEvaluation Results")
    print("==================")

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}"
    )

    return {
        "accuracy": float(
            metrics["accuracy"]
        ),
        "macro_f1": float(
            metrics["macro_f1"]
        ),
        "macro_precision": float(
            metrics["macro_precision"]
        ),
        "macro_recall": float(
            metrics["macro_recall"]
        ),
    }
