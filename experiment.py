"""Shared data contract and binary evaluation for the training script and app."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Gaussian Naive Bayes": "gaussian_nb",
    "Support Vector Machine": "svm",
}
FEATURES = list(load_breast_cancer().feature_names)
TARGET = "malignant"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def read_test_csv(stream):
    """Read a labeled UTF-8 CSV with the same features as the bundled test set."""
    try:
        header = next(csv.reader(iter([stream.readline().decode("utf-8-sig")])))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Use a UTF-8 CSV with a header row.") from exc
    if len(header) != len(set(header)):
        raise ValueError("CSV column names must be unique.")
    stream.seek(0)
    frame = pd.read_csv(stream, encoding="utf-8-sig", float_precision="round_trip")
    missing = [name for name in [*FEATURES, TARGET] if name not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Include at least one labeled test row.")
    try:
        features = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
        labels = pd.to_numeric(frame[TARGET], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All required columns must contain numeric values.") from exc
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Every feature value must be finite and present.")
    if not labels.isin([0, 1]).all():
        raise ValueError("malignant must be 0 for benign or 1 for malignant.")
    return features, labels.astype(int)


def evaluate(model, features, labels):
    predictions = model.predict(features)
    auc = np.nan
    if labels.nunique() == 2:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(features)
        else:
            positive_index = list(model.classes_).index(1)
            scores = model.predict_proba(features)[:, positive_index]
        auc = roc_auc_score(labels, scores)

    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return metrics, matrix, report
