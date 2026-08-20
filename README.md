# 🎙️ Audio Classification MLOps — End-to-End Production Pipeline

An end-to-end **MLOps system for 12-class audio keyword classification** using a Wav2Vec2-based speech classification model.
The project covers the complete machine learning lifecycle, from audio data preparation and model evaluation to **ZenML pipeline orchestration, 
MLflow experiment tracking and model registration, ONNX INT8 inference, FastAPI serving, Docker containerization, Render deployment, Prometheus monitoring, Grafana visualization, and GitHub Actions CI/CD**.
---
## 📌 Project Overview
Speech and keyword classification systems need to be reliable, reproducible, and efficient when deployed in production.
This project implements a production-oriented MLOps workflow for classifying short speech commands into 12 classes.
The trained Wav2Vec2 model is converted into an **INT8 ONNX model** for efficient CPU inference. The machine learning workflow is orchestrated using **ZenML**, 
experiments and model artifacts are tracked using **MLflow**, 
and the final model is registered using the **MLflow Model Registry**.
The model is exposed through a **FastAPI REST API**, packaged using **Docker**, deployed to **Render**, and monitored using **Prometheus and Grafana**.
GitHub and GitHub Actions provide source control and CI/CD automation.
---
## 🎯 Aim
To develop a complete, reproducible, and production-ready **MLOps pipeline for audio keyword classification** that:
- Classifies speech commands into 12 categories.
- Uses a Wav2Vec2-based speech classification model.
- Converts the model to ONNX format.
- Uses INT8 quantization for optimized CPU inference.
- Evaluates model performance using standard classification metrics.
- Orchestrates the ML workflow using ZenML.
- Tracks experiments, parameters, metrics, and model artifacts using MLflow.
- Registers the production candidate model using the MLflow Model Registry.
- Serves predictions through a FastAPI REST API.
- Packages the application using Docker.
- Deploys the API to Render.
- Exposes application and inference metrics using Prometheus.
- Visualizes production monitoring metrics using Grafana.
- Uses GitHub Actions for CI/CD automation.

---

## 📋 Problem Statement
Audio classification models can perform well during development but require additional engineering to become reliable production systems.
A production audio classification application must address several challenges:
- Reproducible ML workflows
- Experiment tracking
- Model versioning
- Efficient inference
- API-based model serving
- Containerized deployment
- Cloud deployment
- Application monitoring
- Inference latency monitoring
- Error monitoring
- CI/CD automation

This project addresses these requirements by integrating the machine learning model with a complete MLOps stack.
The resulting workflow is:
```text
Audio Dataset
     ↓
Audio Preprocessing
     ↓
Wav2Vec2 Model
     ↓
Evaluation
     ↓
ZenML Pipeline
     ↓
MLflow Tracking
     ↓
MLflow Model Registry
     ↓
ONNX INT8 Model
     ↓
FastAPI
     ↓
Docker
     ↓
Render
     ↓
Prometheus
     ↓
Grafana
