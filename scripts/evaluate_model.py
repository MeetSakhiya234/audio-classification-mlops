import gc
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
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
SAMPLE_RATE = 16000
MODEL_NAME = "superb/wav2vec2-base-superb-ks"


def select_balanced_samples(table):
    """Select 10 samples from each of the 12 classes."""

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


def create_confusion_matrix_plot(cm, output_path):
    """Create and save confusion matrix."""

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(cm)

    ax.set_title("Speech Command Classification Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))

    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(
                j,
                i,
                cm[i][j],
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():

    if not Path(TEST_PATH).exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    print("Loading test dataset metadata...")

    table = pq.read_table(
        TEST_PATH,
        columns=["audio", "label"],
    )

    indices = select_balanced_samples(table)

    print("Selected samples:", len(indices))
    print(
        "Expected samples:",
        len(CLASS_NAMES) * SAMPLES_PER_CLASS,
    )

    model = load_model()

    y_true = []
    y_pred = []

    print("Running inference...")

    for count, index in enumerate(indices, start=1):

        audio = table.column("audio")[index].as_py()
        original_label = table.column("label")[index].as_py()

        audio_bytes = audio["bytes"]

        input_tensor = prepare_audio_bytes(audio_bytes)

        predicted_class, confidence, _ = predict(
            model,
            input_tensor,
        )

        true_class = map_label(original_label)

        y_true.append(true_class)
        y_pred.append(predicted_class)

        del input_tensor
        gc.collect()

        if count % 20 == 0:
            print(
                f"Processed {count}/{len(indices)} "
                f"| True: {CLASS_NAMES[true_class]} "
                f"| Predicted: {CLASS_NAMES[predicted_class]} "
                f"| Confidence: {confidence:.3f}"
            )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print("\nEvaluation Results")
    print("==================")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}"
    )

    Path("models").mkdir(exist_ok=True)

    report_path = Path(
        "models/classification_report.txt"
    )

    report_path.write_text(
        metrics["classification_report"],
        encoding="utf-8",
    )

    confusion_matrix_path = Path(
        "models/confusion_matrix.png"
    )

    create_confusion_matrix_plot(
        metrics["confusion_matrix"],
        confusion_matrix_path,
    )

    mlflow.set_experiment(
        "audio-classification"
    )

    with mlflow.start_run(
        run_name="baseline-wav2vec2-evaluation"
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
            "sample_rate",
            SAMPLE_RATE,
        )

        mlflow.log_param(
            "samples_per_class",
            SAMPLES_PER_CLASS,
        )

        mlflow.log_param(
            "total_evaluation_samples",
            len(indices),
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

        mlflow.log_artifact(
            str(report_path)
        )

        mlflow.log_artifact(
            str(confusion_matrix_path)
        )

        print(
            "\nMLflow Run ID:",
            mlflow.active_run().info.run_id,
        )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
