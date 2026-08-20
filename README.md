🎙️ Audio Classification MLOps

End-to-End Production Machine Learning Pipeline

An end-to-end MLOps system for 12-class audio keyword classification using a Wav2Vec2-based speech classification model.

This project demonstrates the complete machine learning lifecycle from audio data processing and model evaluation to experiment tracking, model registration, API serving, containerization, cloud deployment, monitoring, and CI/CD automation.

The production model uses an INT8-quantized ONNX Wav2Vec2 model served through FastAPI and deployed using Docker and Render. The ML workflow is orchestrated with ZenML, experiments are tracked using MLflow, and the production candidate is managed through the MLflow Model Registry.

📌 Table of Contents

Project Overview

Aim

Objectives

Problem Statement

Key Features

Technology Stack

Dataset

Supported Classes

Model Architecture

ONNX INT8 Optimization

MLOps Architecture

End-to-End Workflow

ZenML Pipeline

MLflow Experiment Tracking

MLflow Model Registry

Model Performance

FastAPI

API Endpoints

Prediction Example

Docker

Render Deployment

Prometheus Monitoring

Grafana Dashboard

GitHub and GitHub Actions

Project Structure

Installation

Running the API Locally

Running ZenML

Running MLflow

Model Registration

Monitoring Setup

Results

Challenges and Solutions

Key Learnings

Future Scope

Conclusion

Author

📖 Project Overview

Audio keyword classification is an important speech processing task used in applications such as:

Voice-controlled systems

Smart assistants

Human-computer interaction

Accessibility applications

Embedded speech interfaces

Voice command recognition

However, developing an accurate machine learning model is only one part of building a production system.

A production audio classification application also requires:

Reproducible ML pipelines

Experiment tracking

Model versioning

Optimized inference

API serving

Containerization

Cloud deployment

Application monitoring

Inference monitoring

CI/CD automation

This project addresses these requirements by integrating a Wav2Vec2-based audio classification model into a complete MLOps workflow.

🎯 Aim

To design and implement an end-to-end MLOps system for 12-class audio keyword classification using Wav2Vec2, with experiment tracking, model registration, optimized inference, API deployment, monitoring, and CI/CD automation.

🎯 Objectives

The major objectives of the project are:

Build an audio keyword classification system.

Use a Wav2Vec2-based speech classification model.

Support 12 audio classification classes.

Convert the model into ONNX format.

Apply INT8 optimization for efficient CPU inference.

Evaluate the model using classification metrics.

Orchestrate the ML workflow using ZenML.

Track experiments using MLflow.

Register the production candidate model using MLflow Model Registry.

Develop a FastAPI inference service.

Containerize the application using Docker.

Deploy the service on Render.

Collect production metrics using Prometheus.

Visualize monitoring metrics using Grafana.

Use GitHub and GitHub Actions for source control and CI/CD.

❗ Problem Statement

Machine learning models developed in notebooks or local environments are difficult to maintain and deploy reliably without proper MLOps practices.

For an audio classification system, the following problems need to be addressed:

How can ML experiments be tracked?

How can model versions be managed?

How can inference be optimized for CPU environments?

How can the model be exposed as an API?

How can the application be deployed consistently?

How can API errors and latency be monitored?

How can the complete workflow be automated?

This project solves these problems by combining machine learning with a complete MLOps architecture.

⭐ Key Features

🎙️ 12-class audio keyword classification

🧠 Wav2Vec2-based speech classification

⚡ ONNX INT8 optimized inference

🔄 ZenML pipeline orchestration

📊 MLflow experiment tracking

🏆 MLflow Model Registry

🌐 FastAPI REST API

🐳 Docker containerization

☁️ Render cloud deployment

📡 Prometheus monitoring

📈 Grafana dashboard

🔁 GitHub Actions CI/CD

📦 Production-oriented model management

🧰 Technology Stack

Category

Technology

Programming Language

Python 3.10

Audio Model

Wav2Vec2

Dataset

Google Speech Commands

Model Format

ONNX

Quantization

INT8

Inference

ONNX Runtime

Pipeline Orchestration

ZenML

Experiment Tracking

MLflow

Model Registry

MLflow Model Registry

API

FastAPI

API Documentation

Swagger / OpenAPI

Containerization

Docker

Cloud Deployment

Render

Monitoring

Prometheus

Visualization

Grafana

Version Control

Git / GitHub

CI/CD

GitHub Actions

🎧 Dataset

Google Speech Commands Dataset

The project uses the Google Speech Commands / Speech Commands dataset for audio keyword classification.

The production system supports 12 classes.

🔤 Supported Classes

yes
no
up
down
left
right
on
off
stop
go
_unknown_
_silence_

The model therefore performs:

12-class speech keyword classification

🧠 Model Architecture

The project uses:

superb/wav2vec2-base-superb-ks

as the underlying Wav2Vec2-based speech classification model.

Wav2Vec2 is designed to learn useful speech representations from raw audio waveforms and is suitable for speech recognition and audio classification tasks.

The production inference model is:

models/wav2vec2_int8_linear.onnx

⚡ ONNX INT8 Optimization

The production model was converted into ONNX format and optimized using INT8 quantization.

Production Model

Model:
wav2vec2_int8_linear.onnx

Format:
ONNX

Quantization:
INT8

Runtime:
ONNX Runtime

Approximate Size:
117 MB

The model was successfully validated using the ONNX checker.

ONNX MODEL VALID
Inputs: ['input_values']
Outputs: ['logits']

The production inference path uses ONNX Runtime instead of requiring PyTorch during API startup.

This reduces unnecessary production dependencies and provides an optimized CPU inference path.

🏗️ MLOps Architecture

The complete project architecture is:

                    ┌──────────────────────────┐
                    │ Google Speech Commands   │
                    │        Dataset           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Audio Preprocessing     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        Wav2Vec2           │
                    │  Audio Classification     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Evaluation          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐       ┌──────────────────┐
          │      ZenML       │       │      MLflow       │
          │  Orchestration   │       │ Experiment Track │
          └────────┬─────────┘       └────────┬─────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                    ┌──────────────────────────┐
                    │   MLflow Model Registry  │
                    │        Version 1          │
                    │          READY            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      ONNX INT8 Model     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        FastAPI            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │         Docker            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │         Render            │
                    │    Production API         │
                    └────────────┬─────────────┘
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
              ┌───────────────┐       ┌───────────────┐
              │  Prometheus   │──────▶│    Grafana    │
              │    Metrics    │       │   Dashboard   │
              └───────────────┘       └───────────────┘


          GitHub + GitHub Actions
          Source Control + CI/CD

🔄 End-to-End Workflow

The complete workflow is:

1. Audio Dataset
       ↓
2. Audio Preprocessing
       ↓
3. Wav2Vec2 Classification
       ↓
4. Model Evaluation
       ↓
5. ZenML Pipeline
       ↓
6. MLflow Experiment Tracking
       ↓
7. MLflow Model Registry
       ↓
8. ONNX INT8 Production Model
       ↓
9. FastAPI
       ↓
10. Docker
       ↓
11. Render
       ↓
12. Prometheus
       ↓
13. Grafana

GitHub and GitHub Actions provide source control and CI/CD automation across the project.

🔄 ZenML Pipeline

The main ZenML pipeline is:

audio_classification_pipeline

The successful pipeline execution contains five steps:

load_dataset
      ↓
prepare_evaluation_data
      ↓
run_model_inference
      ↓
evaluate_predictions
      ↓
track_with_mlflow

Successful Pipeline Run

Total Steps : 5
Completed   : 5
Failed      : 0
Pending     : 0
Running     : 0

ZenML is responsible for orchestrating and tracking the machine learning workflow.

📊 MLflow Experiment Tracking

MLflow is used for experiment tracking.

MLflow Server

http://127.0.0.1:5001

Experiment

audio-classification

MLflow tracks:

Model information

Dataset information

Model configuration

Quantization

Runtime

Evaluation metrics

Model artifacts

Run metadata

🏆 MLflow Model Registry

The final production candidate has been registered using the MLflow Model Registry.

Registered Model

audio-classification-wav2vec2

Current Version

Version: 1

Status

READY

Model Details

Architecture : Wav2Vec2
Format       : ONNX
Quantization : INT8
Task         : 12-class keyword spotting
Runtime      : ONNX Runtime

Source

runs:/67b38fc4b09d453daa4e744e23176e53/model

Model Registry provides version control and lifecycle management for production model artifacts.

📈 Model Performance

The final evaluated model achieved:

Metric

Result

Accuracy

97.50%

Macro Precision

97.85%

Macro Recall

97.50%

Macro F1

97.53%

Evaluation dataset:

Total Evaluation Samples: 120
Samples per Class: 10
Number of Classes: 12

🌐 FastAPI

The production model is served through FastAPI.

Local API

http://localhost:8001

Swagger Documentation

http://localhost:8001/docs

FastAPI provides an HTTP interface for audio classification.

🔌 API Endpoints

Method

Endpoint

Description

GET

/

Service information

GET

/health

Health and model status

POST

/predict

Audio classification

GET

/metrics

Prometheus metrics

🎤 Prediction Example

A prediction can be generated using:

curl -X POST "http://localhost:8001/predict" \
  -H "accept: application/json" \
  -F "file=@sample.wav"

Example response:

{
  "predicted_class": 10,
  "predicted_label": "_unknown_",
  "confidence": 0.98684
}

The API returns classification information including:

Predicted class

Predicted label

Confidence

Inference time

Probability distribution

❤️ Health Check

The health endpoint:

GET /health

returns:

{
  "status": "healthy",
  "model": "superb/wav2vec2-base-superb-ks",
  "model_loaded": true
}

🐳 Docker

The FastAPI application is containerized using Docker.

Docker provides:

Reproducible environments

Dependency isolation

Consistent application packaging

Portable deployment

Cloud deployment support

The Docker image contains the FastAPI application and the required runtime dependencies for ONNX inference.

☁️ Render Deployment

The production API is deployed on Render.

Service

audio-classification-mlops

Production URL

https://audio-classification-mlops.onrender.com

Production Root Response

{
  "service": "Audio Classification MLOps API",
  "status": "running",
  "docs": "/docs",
  "health": "/health",
  "prediction_endpoint": "/predict"
}

Production prediction was successfully tested.

Example:

Predicted Class : 10
Predicted Label : _unknown_
Confidence       : approximately 0.98684

📡 Prometheus Monitoring

The FastAPI application exposes Prometheus-compatible metrics through:

/metrics

Important metrics include:

audio_api_requests_total
audio_predictions_total
audio_prediction_errors_total
audio_inference_latency_seconds

Prometheus monitors:

API request volume

Prediction volume

Prediction errors

Inference latency

Prediction distribution

The production API target was successfully verified as:

UP

📊 Grafana Dashboard

Grafana is connected to Prometheus.

Datasource

http://prometheus:9090

Dashboard

Audio Classification MLOps Monitoring

Dashboard panels include:

Total API Requests

Total Predictions

Prediction Errors

Average Inference Latency

Predictions by Class

API Request Rate

Prediction Success Rate

Latency Over Time

Prediction Errors Over Time

Latest Monitoring Results

Metric

Value

Total API Requests

11

Total Predictions

11

Prediction Errors

0

Average Inference Latency

~145 ms

Prediction Success Rate

100%

_unknown_ Predictions

11

🔁 GitHub and GitHub Actions

GitHub is used for source control.

Repository

https://github.com/MeetSakhiya234/audio-classification-mlops

GitHub stores the source code, configuration, pipeline definitions, API implementation, Docker configuration, monitoring configuration, and project documentation.

GitHub Actions provides the CI/CD automation layer.

The CI/CD workflow helps automate project validation and deployment-related operations.

📁 Project Structure

audio-classification-mlops/
│
├── .github/
│   └── workflows/
│
├── app/
│   └── main.py
│
├── models/
│   └── wav2vec2_int8_linear.onnx
│
├── scripts/
│   └── register_model.py
│
├── src/
│   └── audio_classifier/
│       ├── labels.py
│       ├── model.py
│       └── preprocessing.py
│
├── steps/
│   ├── evaluation_step.py
│   └── tracking_step.py
│
├── pipelines/
│   └── audio_classification_pipeline.py
│
├── prometheus/
│
├── grafana/
│
├── Dockerfile
├── requirements.txt
├── README.md
├── Gantt_Audio_Classification_MLOps.png
└── WBS_Audio_Classification_MLOps.png

⚙️ Installation

1. Clone Repository

git clone https://github.com/MeetSakhiya234/audio-classification-mlops.git
cd audio-classification-mlops

2. Create Virtual Environment

python3.10 -m venv .venv

3. Activate Environment

Ubuntu / WSL

source .venv/bin/activate

Windows

.venv\Scripts\activate

4. Install Dependencies

pip install -r requirements.txt

🚀 Running the API Locally

Start FastAPI:

uvicorn app.main:app --host 0.0.0.0 --port 8001

Open:

http://localhost:8001

Swagger:

http://localhost:8001/docs

Health:

http://localhost:8001/health

Metrics:

http://localhost:8001/metrics

🧪 Testing Prediction

Use an audio file such as:

sample.wav

Run:

curl -X POST "http://localhost:8001/predict" \
  -H "accept: application/json" \
  -F "file=@sample.wav"

🧠 Running ZenML

Check ZenML:

zenml status

List pipelines:

zenml pipeline list

Main pipeline:

audio_classification_pipeline

The pipeline performs:

Dataset Loading
      ↓
Evaluation Data Preparation
      ↓
Model Inference
      ↓
Evaluation
      ↓
MLflow Tracking

📊 Running MLflow

Start the MLflow server:

mlflow server \
  --host 127.0.0.1 \
  --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts

Open:

http://127.0.0.1:5001

🏷️ Registering the Model

The project provides:

scripts/register_model.py

Run:

python scripts/register_model.py

The script performs:

ONNX validation

MLflow experiment setup

Run creation

Parameter logging

Metric logging

Model artifact upload

Model Registry creation

Model version creation

Model version verification

Current registered model:

audio-classification-wav2vec2

Current version:

1

Current status:

READY

📡 Prometheus Setup

The FastAPI service exposes:

/metrics

Prometheus can scrape this endpoint and collect:

audio_api_requests_total
audio_predictions_total
audio_prediction_errors_total
audio_inference_latency_seconds

📊 Grafana Setup

Start Grafana and connect Prometheus as the data source.

Prometheus datasource:

http://prometheus:9090

Grafana:

http://localhost:3000

Dashboard:

Audio Classification MLOps Monitoring

🔎 Project Verification

The complete system can be verified using the following sequence:

ONNX Model Validation
        ↓
ZenML Pipeline
        ↓
MLflow Experiment
        ↓
MLflow Model Registry
        ↓
FastAPI Health
        ↓
FastAPI Prediction
        ↓
Docker
        ↓
Render
        ↓
Prometheus
        ↓
Grafana
        ↓
GitHub Actions

📊 Final Results

Machine Learning

Accuracy        : 97.50%
Macro Precision : 97.85%
Macro Recall    : 97.50%
Macro F1        : 97.53%

Production Model

Architecture : Wav2Vec2
Format       : ONNX
Quantization : INT8
Runtime      : ONNX Runtime
Classes      : 12
Size         : ~117 MB

MLOps Components

Pipeline      : ZenML
Tracking      : MLflow
Registry      : MLflow Model Registry
API           : FastAPI
Container     : Docker
Deployment    : Render
Monitoring    : Prometheus
Dashboard     : Grafana
CI/CD         : GitHub Actions

Model Registry

Name    : audio-classification-wav2vec2
Version : 1
Status  : READY

⚠️ Challenges and Solutions

1. PyTorch Dependency During Production Startup

The initial production environment failed because PyTorch was unnecessarily imported during API startup.

Solution

Unnecessary PyTorch dependencies were removed from the production inference path.

The API now uses ONNX Runtime for CPU inference.

2. ONNX Model Optimization

The original model was not ideal for lightweight CPU-based production inference.

Solution

The production model was converted to ONNX and optimized using INT8 quantization.

3. MLflow Artifact Configuration

The first model registration attempt encountered an artifact configuration issue because the MLflow client was using a local SQLite tracking URI while the artifact server expected HTTP-based tracking.

Solution

The MLflow client was configured to use:

http://127.0.0.1:5001

with MLflow server-side artifact storage.

The final registration completed successfully.

4. Grafana Initially Showing No Data

Grafana initially displayed no monitoring data.

Solution

The API was exercised to generate Prometheus metrics and the correct PromQL queries were configured.

The dashboard subsequently displayed live request, prediction, error, and latency metrics.

🎓 Key Learnings

This project provided practical experience with:

Audio classification

Wav2Vec2

ONNX model conversion

INT8 optimization

ONNX Runtime

FastAPI

Docker

Render

ZenML

MLflow

MLflow Model Registry

Prometheus

Grafana

GitHub

GitHub Actions

CI/CD

Production troubleshooting

MLOps architecture

The project demonstrates that deploying an ML model requires more than achieving good model accuracy. A production ML system must also provide reproducibility, versioning, deployment, monitoring, and automation.

🔮 Future Scope

Future improvements may include:

Real-time microphone-based classification

Additional speech command classes

Improved _unknown_ and _silence_ detection

Automated data drift detection

Automated model retraining

Automated model promotion

Automated model rollback

Cloud-based MLflow artifact storage

Kubernetes deployment

Advanced latency benchmarking

Canary deployment

Blue-green deployment

Automated production model monitoring

Continuous model evaluation

🏁 Conclusion

This project successfully implements an end-to-end Audio Classification MLOps system for 12-class speech keyword classification.

The Wav2Vec2-based model achieved:

Accuracy : 97.50%
Macro F1 : 97.53%

The production model was converted to an INT8 ONNX model for optimized CPU inference.

ZenML provides workflow orchestration, while MLflow provides experiment tracking and model lifecycle management. The final production candidate is registered in the MLflow Model Registry as:

audio-classification-wav2vec2
Version 1
Status READY

FastAPI provides the prediction service, Docker provides reproducible packaging, and Render provides cloud deployment.

Prometheus collects production API and inference metrics, while Grafana provides real-time monitoring dashboards.

GitHub and GitHub Actions provide source control and CI/CD automation.

The resulting system demonstrates a complete transition from an experimental audio classification model to a deployable, versioned, monitored, and production-oriented MLOps application.

👨‍💻 Author

Meet Sakhiya

M.Sc. Data Science

GitHub:

https://github.com/MeetSakhiya234

Project Repository:

https://github.com/MeetSakhiya234/audio-classification-mlops

⭐ Project Stack

                    AUDIO CLASSIFICATION MLOps

                            Python
                              │
                       ┌──────┴──────┐
                       │   Wav2Vec2  │
                       └──────┬──────┘
                              │
                       ONNX + INT8
                              │
                       ONNX Runtime
                              │
                 ┌────────────┴────────────┐
                 │                         │
               ZenML                    MLflow
                 │                         │
          Pipeline Tracking         Experiment Tracking
                                           │
                                    Model Registry
                                           │
                                           ▼
                                      FastAPI
                                           │
                                         Docker
                                           │
                                        Render
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                         Prometheus                 Grafana
                              │                         │
                              └────────────┬────────────┘
                                           │
                                    Production Monitoring

                         GitHub + GitHub Actions
                              CI/CD Automation

🚀 End-to-End MLOps

Audio Data → Wav2Vec2 → Evaluation → ZenML → MLflow → Model Registry → ONNX INT8 → FastAPI → Docker → Render → Prometheus → Grafana → GitHub Actions
