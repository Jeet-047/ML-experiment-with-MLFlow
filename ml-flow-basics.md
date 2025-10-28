# 🧠 MLflow Comprehensive Guide

MLflow is an **open-source platform** to manage the *Machine Learning lifecycle*, including:
- Experiment tracking
- Model versioning
- Model packaging & reproducibility
- Model deployment

---

## 📦 1. What Is MLflow?

MLflow provides tools for four key functionalities:

| Component | Purpose |
|------------|----------|
| **MLflow Tracking** | Log parameters, metrics, artifacts, and models during training. |
| **MLflow Projects** | Package data science code in a reusable and reproducible form. |
| **MLflow Models** | Manage and deploy models in a consistent format. |
| **MLflow Registry** | Centralized model store with version control and stage transitions. |

---

## 🧩 2. MLflow Tracking

MLflow Tracking is the **most commonly used** part of MLflow — it records all information about model training runs.

### 🔹 Basic Setup

Install MLflow:
```bash
pip install mlflow

---
## Common Logging Functions Summary

| Function                     | Description                          |
| ---------------------------- | ------------------------------------ |
| `mlflow.start_run()`         | Start a new MLflow run               |
| `mlflow.log_param()`         | Log a single parameter               |
| `mlflow.log_params()`        | Log multiple parameters              |
| `mlflow.log_metric()`        | Log a single metric                  |
| `mlflow.log_artifact()`      | Log a single output file             |
| `mlflow.log_artifacts()`     | Log all files in a directory         |
| `mlflow.sklearn.log_model()` | Log a model (framework-specific)     |
| `mlflow.set_experiment()`    | Create or switch to a new experiment |
| `mlflow.get_artifact_uri()`  | Get artifact storage path            |
