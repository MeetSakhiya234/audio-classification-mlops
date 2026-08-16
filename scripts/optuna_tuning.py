import gc
import os
from pathlib import Path

import mlflow
import optuna
import pyarrow.parquet as pq

from src.audio_classifier.data import CLASS_NAMES, map_label
from src.audio_classifier.evaluation import calculate_metrics
from src.audio_classifier.model import load_model, predict
from src.audio_classifier.preprocessing import prepare_audio_bytes


TEST_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--google--speech_commands/"
    "snapshots/a751309c0fd613e8a5d30d77900f30e8b42bc2da/"
    "v0.02/test-00000-of-00001.parquet"
)

SAMPLES_PER_CLASS = 5
MODEL_NAME = "superb/wav2vec2-base-superb-ks"


def load_balanced_samples():
    """Load 5 samples from each of the 12 classes."""

    table = pq.read_table(
        TEST_PATH,
        columns=["audio", "label"],
    )

    selected = {
        class_id: []
        for class_id in range(len(CLASS_NAMES))
    }

    labels = table.column("label").to_pylist()

    for index, original_label in enumerate(labels):

        class_id = map_label(original_label)

        if len(selected[class_id]) < SAMPLES_PER_CLASS:
            selected[class_id].append(index)

        if all(
            len(selected[class_id]) >= SAMPLES_PER_CLASS
            for class_id in range(len(CLASS_NAMES))
        ):
            break

    indices = []

    for class_id in range(len(CLASS_NAMES)):
        indices.extend(selected[class_id])

    return table, indices


def collect_predictions(model, table, indices):
    """Run Wav2Vec2 inference once and cache predictions."""

    y_true = []
    y_pred = []
    confidences = []

    print("\nCollecting model predictions once...")

    for count, index in enumerate(indices, start=1):

        audio = table.column("audio")[index].as_py()
        original_label = table.column("label")[index].as_py()

        input_tensor = prepare_audio_bytes(
            audio["bytes"]
        )

        predicted_class, confidence, _ = predict(
            model,
            input_tensor,
        )

        true_class = map_label(original_label)

        y_true.append(true_class)
        y_pred.append(predicted_class)
        confidences.append(confidence)

        del input_tensor
        gc.collect()

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(indices)}"
            )

    return y_true, y_pred, confidences


def objective(
    trial,
    y_true,
    y_pred,
    confidences,
):
    """Optimize confidence threshold using cached predictions."""

    confidence_threshold = trial.suggest_float(
        "confidence_threshold",
        0.50,
        0.99,
    )

    threshold_predictions = []

    for predicted_class, confidence in zip(
        y_pred,
        confidences,
    ):

        if confidence < confidence_threshold:
            threshold_predictions.append(10)
        else:
            threshold_predictions.append(
                predicted_class
            )

    metrics = calculate_metrics(
        y_true,
        threshold_predictions,
    )

    with mlflow.start_run(
        run_name=f"optuna-trial-{trial.number + 1}",
        nested=True,
    ):

        mlflow.log_param(
            "model_name",
            MODEL_NAME,
        )

        mlflow.log_param(
            "dataset",
            "google/speech_commands",
        )

        mlflow.log_param(
            "dataset_split",
            "test",
        )

        mlflow.log_param(
            "samples_per_class",
            SAMPLES_PER_CLASS,
        )

        mlflow.log_param(
            "confidence_threshold",
            confidence_threshold,
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
        f"Trial {trial.number + 1}/3 "
        f"| Threshold: {confidence_threshold:.4f} "
        f"| Macro F1: {metrics['macro_f1']:.4f}"
    )

    return metrics["macro_f1"]


def main():

    if not Path(TEST_PATH).exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    print("Loading balanced tuning samples...")

    table, indices = load_balanced_samples()

    print(
        "Tuning samples:",
        len(indices),
    )

    print(
        "Samples per class:",
        SAMPLES_PER_CLASS,
    )

    print("\nLoading Wav2Vec2 model once...")

    model = load_model()

    y_true, y_pred, confidences = collect_predictions(
        model,
        table,
        indices,
    )

    print("\nModel inference completed.")

    baseline_metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print(
        f"Baseline tuning-set Macro F1: "
        f"{baseline_metrics['macro_f1']:.4f}"
    )

    mlflow.set_experiment(
        "audio-classification"
    )

    print("\nCreating Optuna study...")

    study = optuna.create_study(
        direction="maximize",
        study_name="wav2vec2-confidence-tuning",
    )

    print("Starting 3 Optuna trials...")

    with mlflow.start_run(
        run_name="optuna-parent-run"
    ):

        study.optimize(
            lambda trial: objective(
                trial,
                y_true,
                y_pred,
                confidences,
            ),
            n_trials=3,
        )

        best_trial = study.best_trial

        mlflow.log_metric(
            "best_macro_f1",
            best_trial.value,
        )

        mlflow.log_param(
            "best_confidence_threshold",
            best_trial.params[
                "confidence_threshold"
            ],
        )

        mlflow.log_param(
            "optuna_trials",
            3,
        )

        print("\nOptuna Results")
        print("==========================")

        print(
            f"Best Macro F1: "
            f"{best_trial.value:.4f}"
        )

        print(
            "Best confidence threshold:",
            best_trial.params[
                "confidence_threshold"
            ],
        )

        print(
            "Best trial number:",
            best_trial.number + 1,
        )

    print(
        "\nOptuna tuning completed successfully."
    )


if __name__ == "__main__":
    main()
