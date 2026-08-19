import os
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "audio-classification"

MODEL_FILE = "models/wav2vec2_int8_linear.onnx"
MODEL_NAME = "audio-classification-wav2vec2"

# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)

print("=" * 70)
print("AUDIO CLASSIFICATION - MLflow MODEL REGISTRATION")
print("=" * 70)

print(f"MLflow version : {mlflow.__version__}")
print(f"Tracking URI   : {mlflow.get_tracking_uri()}")
print(f"Model file     : {MODEL_FILE}")
print(f"Registered name: {MODEL_NAME}")

# ---------------------------------------------------------
# Validate model
# ---------------------------------------------------------
print("\n[1/5] Validating ONNX model...")

import onnx

model = onnx.load(MODEL_FILE)
onnx.checker.check_model(model)

print("ONNX MODEL VALID")
print("Inputs :", [x.name for x in model.graph.input])
print("Outputs:", [x.name for x in model.graph.output])

# ---------------------------------------------------------
# Get/create experiment
# ---------------------------------------------------------
print("\n[2/5] Preparing MLflow experiment...")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

print("Experiment ID:", experiment_id)

# ---------------------------------------------------------
# Create MLflow run
# ---------------------------------------------------------
print("\n[3/5] Creating MLflow run...")

with mlflow.start_run(
    experiment_id=experiment_id,
    run_name="production-model-registration-v2"
) as run:

    run_id = run.info.run_id

    # Parameters
    mlflow.log_params({
        "model_name": "superb/wav2vec2-base-superb-ks",
        "model_format": "ONNX",
        "quantization": "INT8",
        "task": "12-class keyword spotting",
        "dataset": "Google Speech Commands",
        "runtime": "ONNX Runtime",
        "input_name": "input_values",
        "output_name": "logits",
        "model_size_mb": round(
            os.path.getsize(MODEL_FILE) / (1024 * 1024), 2
        )
    })

    # Evaluation metrics already obtained from your project
    mlflow.log_metrics({
        "accuracy": 0.975,
        "macro_f1": 0.9752980937191463,
        "macro_precision": 0.9785353535353535,
        "macro_recall": 0.975
    })

    # Tags
    mlflow.set_tags({
        "project": "audio-classification-mlops",
        "model_type": "wav2vec2",
        "format": "ONNX",
        "quantization": "INT8",
        "deployment_status": "production-candidate"
    })

    # -----------------------------------------------------
    # Upload actual ONNX file as an MLflow artifact
    # -----------------------------------------------------
    print("\n[4/5] Uploading ONNX model artifact...")
    print("This may take a little time because the model is ~117 MB.")

    mlflow.log_artifact(
        MODEL_FILE,
        artifact_path="model"
    )

    print("ONNX artifact uploaded successfully.")
    print("Run ID:", run_id)

# ---------------------------------------------------------
# Register model
# ---------------------------------------------------------
print("\n[5/5] Registering model in MLflow Model Registry...")

client = MlflowClient()

# Create registered model if it does not exist
try:
    registered_model = client.get_registered_model(MODEL_NAME)
    print("Registered model already exists.")
except Exception:
    print("Creating registered model...")
    registered_model = client.create_registered_model(
        MODEL_NAME,
        description=(
            "Production candidate INT8 ONNX Wav2Vec2 model "
            "for 12-class audio keyword classification."
        )
    )

# Create model version from the run artifact
source = f"runs:/{run_id}/model"

model_version = client.create_model_version(
    name=MODEL_NAME,
    source=source,
    run_id=run_id,
    description=(
        "INT8 quantized Wav2Vec2 ONNX model. "
        "Accuracy=0.975, Macro-F1=0.9753."
    )
)

print("\n" + "=" * 70)
print("MODEL REGISTRATION SUCCESSFUL")
print("=" * 70)
print("Registered Model :", MODEL_NAME)
print("Version          :", model_version.version)
print("Run ID           :", run_id)
print("Source           :", source)
print("Status           :", model_version.status)
print("=" * 70)
