import gc
import os
from collections import defaultdict

import mlflow
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

SAMPLES_PER_CLASS = 10

MODEL_NAME = "superb/wav2vec2-base-superb-ks"

# Best threshold found by the 3-trial Optuna experiment.
BEST_THRESHOLD = 0.5364194313399226


def select_balanced_samples(table):
    """Use the exact same sample-selection logic as evaluate_model.py."""

    selected = defaultdict(list)

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

    return indices


def main():

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    print("Loading evaluation dataset metadata...")

    table = pq.read_table(
        TEST_PATH,
        columns=["audio", "label"],
    )

    indices = select_balanced_samples(table)

    expected_samples = (
        len(CLASS_NAMES) * SAMPLES_PER_CLASS
    )

    print(
        "Selected samples:",
        len(indices),
    )

    print(
        "Expected samples:",
        expected_samples,
    )

    if len(indices) != expected_samples:
        raise RuntimeError(
            "Balanced sample selection did not "
            "produce the expected number of samples."
        )

    print("\nLoading Wav2Vec2 model...")

    model = load_model()

    y_true = []
    baseline_predictions = []
    threshold_predictions = []

    print("\nRunning inference...")

    for count, index in enumerate(
        indices,
        start=1,
    ):

        audio = table.column(
            "audio"
        )[index].as_py()

        original_label = table.column(
            "label"
        )[index].as_py()

        audio_bytes = audio["bytes"]

        input_tensor = prepare_audio_bytes(
            audio_bytes
        )

        (
            predicted_class,
            confidence,
            _,
        ) = predict(
            model,
            input_tensor,
        )

        true_class = map_label(
            original_label
        )

        # Baseline prediction
        baseline_predictions.append(
            predicted_class
        )

        # Apply Optuna threshold
        # Low-confidence predictions become _unknown_.
        if confidence < BEST_THRESHOLD:
            threshold_class = CLASS_NAMES.index(
                "_unknown_"
            )
        else:
            threshold_class = predicted_class

        threshold_predictions.append(
            threshold_class
        )

        y_true.append(true_class)

        del input_tensor
        gc.collect()

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(indices)} "
                f"| True: {CLASS_NAMES[true_class]} "
                f"| Baseline: {CLASS_NAMES[predicted_class]} "
                f"| Threshold: {CLASS_NAMES[threshold_class]} "
                f"| Confidence: {confidence:.3f}"
            )

    del model
    gc.collect()

    print("\nCalculating metrics...")

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_predictions,
    )

    threshold_metrics = calculate_metrics(
        y_true,
        threshold_predictions,
    )

    print("\nBaseline Results")
    print("==================")

    print(
        "Accuracy:",
        f"{baseline_metrics['accuracy']:.4f}",
    )

    print(
        "Macro F1:",
        f"{baseline_metrics['macro_f1']:.4f}",
    )

    print(
        "Macro Precision:",
        f"{baseline_metrics['macro_precision']:.4f}",
    )

    print(
        "Macro Recall:",
        f"{baseline_metrics['macro_recall']:.4f}",
    )

    print("\nOptuna Threshold Results")
    print("==================")

    print(
        "Threshold:",
        f"{BEST_THRESHOLD:.4f}",
    )

    print(
        "Accuracy:",
        f"{threshold_metrics['accuracy']:.4f}",
    )

    print(
        "Macro F1:",
        f"{threshold_metrics['macro_f1']:.4f}",
    )

    print(
        "Macro Precision:",
        f"{threshold_metrics['macro_precision']:.4f}",
    )

    print(
        "Macro Recall:",
        f"{threshold_metrics['macro_recall']:.4f}",
    )

    f1_change = (
        threshold_metrics["macro_f1"]
        - baseline_metrics["macro_f1"]
    )

    print("\nComparison")
    print("==================")

    print(
        "Macro F1 change:",
        f"{f1_change:+.4f}",
    )

    if f1_change > 0:
        print(
            "Decision: Optuna threshold improves "
            "Macro F1 on the evaluation set."
        )

    elif f1_change < 0:
        print(
            "Decision: Baseline performs better "
            "on the evaluation set."
        )

    else:
        print(
            "Decision: Both approaches have "
            "the same Macro F1."
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

    with mlflow.start_run(
        run_name="optuna-threshold-evaluation"
    ):

        mlflow.log_param(
            "model_name",
            MODEL_NAME,
        )

        mlflow.log_param(
            "evaluation_samples",
            len(indices),
        )

        mlflow.log_param(
            "samples_per_class",
            SAMPLES_PER_CLASS,
        )

        mlflow.log_param(
            "confidence_threshold",
            BEST_THRESHOLD,
        )

        mlflow.log_metric(
            "baseline_accuracy",
            baseline_metrics["accuracy"],
        )

        mlflow.log_metric(
            "baseline_macro_f1",
            baseline_metrics["macro_f1"],
        )

        mlflow.log_metric(
            "baseline_macro_precision",
            baseline_metrics["macro_precision"],
        )

        mlflow.log_metric(
            "baseline_macro_recall",
            baseline_metrics["macro_recall"],
        )

        mlflow.log_metric(
            "threshold_accuracy",
            threshold_metrics["accuracy"],
        )

        mlflow.log_metric(
            "threshold_macro_f1",
            threshold_metrics["macro_f1"],
        )

        mlflow.log_metric(
            "threshold_macro_precision",
            threshold_metrics["macro_precision"],
        )

        mlflow.log_metric(
            "threshold_macro_recall",
            threshold_metrics["macro_recall"],
        )

        mlflow.log_metric(
            "macro_f1_change",
            f1_change,
        )

    print(
        "\nThreshold evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
