from pathlib import Path
from typing import Dict

import mlflow


def log_evaluation_run(
    metrics: Dict,
    num_samples: int,
    model_name: str,
    sample_rate: int,
):
    """
    Log an evaluation run to MLflow.
    """

    mlflow.set_experiment("audio-classification")

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset", "google/speech_commands")
        mlflow.log_param("task", "12-class-keyword-spotting")
        mlflow.log_param("sample_rate", sample_rate)
        mlflow.log_param("num_samples", num_samples)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric(
            "macro_precision",
            metrics["macro_precision"],
        )
        mlflow.log_metric(
            "macro_recall",
            metrics["macro_recall"],
        )

        report_path = Path("classification_report.txt")

        report_path.write_text(
            metrics["classification_report"],
            encoding="utf-8",
        )

        mlflow.log_artifact(report_path)

        print("MLflow run ID:", run.info.run_id)

        return run.info.run_id
