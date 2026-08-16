import gc
import os

import mlflow
import optuna
import pyarrow.parquet as pq

from src.audio_classifier.data import (
    CLASS_NAMES,
    map_label,
)
from src.audio_classifier.evaluation import (
    calculate_metrics,
)
from src.audio_classifier.model import (
    load_model,
    predict,
)
from src.audio_classifier.preprocessing import (
    prepare_audio_bytes,
)


MODEL_NAME = "superb/wav2vec2-base-superb-ks"

SAMPLES_PER_CLASS = 5
N_TRIALS = 3

DATASET_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/"
    "datasets--google--speech_commands/"
    "snapshots/"
    "a751309c0fd613e8a5d30d77900f30e8b42bc2da/"
    "v0.02/"
    "test-00000-of-00001.parquet"
)


def load_tuning_samples():
    """Load a balanced tuning subset from the test dataset."""

    print("Loading balanced tuning samples...")

    table = pq.read_table(
        DATASET_PATH,
        columns=[
            "audio",
            "label",
            "is_unknown",
        ],
    )

    df = table.to_pandas()

    selected_rows = []

    # Known classes
    known_classes = [
        label
        for label in sorted(df["label"].unique())
        if not df.loc[
            df["label"] == label,
            "is_unknown",
        ].iloc[0]
    ]

    # Select SAMPLES_PER_CLASS from each known class.
    for label in known_classes:
        class_df = df[
            (df["label"] == label)
            & (~df["is_unknown"])
        ]

        selected_rows.append(
            class_df.head(SAMPLES_PER_CLASS)
        )

    # Unknown samples
    unknown_df = df[
        df["is_unknown"]
    ].head(SAMPLES_PER_CLASS)

    selected_rows.append(unknown_df)

    tuning_df = (
        __import__("pandas")
        .concat(selected_rows)
        .reset_index(drop=True)
    )

    print(
        "Tuning samples:",
        len(tuning_df),
    )

    print(
        "Samples per class:",
        SAMPLES_PER_CLASS,
    )

    return tuning_df


def collect_predictions(tuning_df):
    """
    Run model inference once.

    Optuna trials reuse these predictions instead of
    loading/running Wav2Vec2 repeatedly.
    """

    print("\nLoading Wav2Vec2 model once...")

    model = load_model()

    y_true = []
    y_pred = []
    confidences = []

    print("\nCollecting model predictions once...")

    for count, (_, row) in enumerate(
        tuning_df.iterrows(),
        start=1,
    ):
        audio_bytes = row["audio"]["bytes"]

        input_tensor = prepare_audio_bytes(
            audio_bytes
        )

        predicted_class, confidence, _ = predict(
            model,
            input_tensor,
        )

        true_class = map_label(
            row["label"]
        )

        y_true.append(true_class)
        y_pred.append(predicted_class)
        confidences.append(confidence)

        del input_tensor
        gc.collect()

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(tuning_df)}"
            )

    del model
    gc.collect()

    print("\nModel inference completed.")

    baseline_metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print(
        "Baseline tuning-set Macro F1:",
        f"{baseline_metrics['macro_f1']:.4f}",
    )

    return (
        y_true,
        y_pred,
        confidences,
    )


def apply_confidence_threshold(
    y_pred,
    confidences,
    threshold,
):
    """
    Convert low-confidence predictions into _unknown_.

    The predicted class for low-confidence samples
    becomes the class represented by _unknown_.
    """

    unknown_class = CLASS_NAMES.index(
        "_unknown_"
    )

    threshold_predictions = []

    for predicted_class, confidence in zip(
        y_pred,
        confidences,
    ):
        if confidence < threshold:
            threshold_predictions.append(
                unknown_class
            )
        else:
            threshold_predictions.append(
                predicted_class
            )

    return threshold_predictions


def objective(
    trial,
    y_true,
    y_pred,
    confidences,
):
    """
    Optuna objective.

    Optimize confidence threshold using
    Macro F1 on the tuning set.
    """

    threshold = trial.suggest_float(
        "confidence_threshold",
        0.50,
        0.99,
    )

    threshold_predictions = (
        apply_confidence_threshold(
            y_pred,
            confidences,
            threshold,
        )
    )

    metrics = calculate_metrics(
        y_true,
        threshold_predictions,
    )

    # Make sure no previous MLflow run is active.
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(
        run_name=f"optuna-trial-{trial.number + 1}"
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
            threshold,
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

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

    print(
        f"Trial {trial.number + 1}: "
        f"threshold={threshold:.4f}, "
        f"Macro F1={metrics['macro_f1']:.4f}"
    )

    return metrics["macro_f1"]


def main():
    """Run the Optuna confidence-threshold optimization."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    tuning_df = load_tuning_samples()

    (
        y_true,
        y_pred,
        confidences,
    ) = collect_predictions(
        tuning_df
    )

    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5001",
        )
    )

    mlflow.set_experiment(
        "audio-classification"
    )

    print("\nCreating Optuna study...")

    study = optuna.create_study(
        direction="maximize",
        study_name="wav2vec2-confidence-tuning",
    )

    print(
        f"Starting {N_TRIALS} Optuna trials..."
    )

    study.optimize(
        lambda trial: objective(
            trial,
            y_true,
            y_pred,
            confidences,
        ),
        n_trials=N_TRIALS,
    )

    best_trial = study.best_trial

    print("\nOptuna Results")
    print("==================")
    print(
        "Best Macro F1:",
        f"{best_trial.value:.4f}",
    )
    print(
        "Best confidence threshold:",
        f"{best_trial.params['confidence_threshold']:.4f}",
    )

    # Log the final best result in a separate run.
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(
        run_name="optuna-best-result"
    ):
        mlflow.log_param(
            "model_name",
            MODEL_NAME,
        )

        mlflow.log_param(
            "optimization",
            "Optuna confidence threshold",
        )

        mlflow.log_param(
            "n_trials",
            N_TRIALS,
        )

        mlflow.log_param(
            "best_confidence_threshold",
            best_trial.params[
                "confidence_threshold"
            ],
        )

        mlflow.log_metric(
            "best_macro_f1",
            best_trial.value,
        )

    print(
        "\nOptuna optimization completed successfully."
    )


if __name__ == "__main__":
    main()
