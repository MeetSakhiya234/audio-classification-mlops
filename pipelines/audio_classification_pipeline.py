from zenml import pipeline

from steps.data_step import load_dataset
from steps.preprocessing_step import (
    prepare_evaluation_data,
)
from steps.inference_step import (
    run_model_inference,
)
from steps.evaluation_step import (
    evaluate_predictions,
)
from steps.tracking_step import (
    track_with_mlflow,
)


@pipeline
def audio_classification_pipeline():
    """
    Complete ZenML audio classification pipeline.

    Data
      ↓
    Preprocessing
      ↓
    Model Inference
      ↓
    Evaluation
      ↓
    MLflow Tracking
    """

    test_path = load_dataset()

    indices = prepare_evaluation_data(
        test_path
    )

    y_true, y_pred = run_model_inference(
        test_path,
        indices,
    )

    metrics = evaluate_predictions(
        y_true,
        y_pred,
    )

    track_with_mlflow(
        metrics
    )
