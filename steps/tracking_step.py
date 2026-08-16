import os
from typing import Dict

import mlflow
from zenml import step


@step
def track_with_mlflow(
    metrics: Dict[str, float],
) -> str:
    """
    Log ZenML evaluation results to MLflow.
    """

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "audio-classification"
    )

    with mlflow.start_run(
        run_name="zenml-audio-evaluation"
    ):

        mlflow.log_param(
            "pipeline",
            "zenml",
        )

        mlflow.log_param(
            "model",
            "superb/wav2vec2-base-superb-ks",
        )

        mlflow.log_param(
            "samples_per_class",
            10,
        )

        mlflow.log_param(
            "total_evaluation_samples",
            120,
        )

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

        mlflow.log_metric(
            "macro_f1",
            metrics["macro_f1"],
        )

        mlflow.log_metric(
            "macro_precision",
            metrics["macro_precision"],
        )

        mlflow.log_metric(
            "macro_recall",
            metrics["macro_recall"],
        )

    print(
        "Metrics successfully logged to MLflow."
    )

    return "MLflow tracking completed"
