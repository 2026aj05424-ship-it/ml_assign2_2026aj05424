"""Rebuild all five trained pipelines and the shared held-out evaluation."""

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from experiment import MODELS, MODEL_DIR, ROOT, TARGET, TEST_CSV, evaluate, read_test_csv

SEED = 42


def main():
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    labels = (dataset.target == 0).astype(int).rename(TARGET)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=SEED,
    )

    pipelines = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Gaussian Naive Bayes": make_pipeline(GaussianNB()),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    x_test.assign(**{TARGET: y_test}).to_csv(TEST_CSV, index=False)
    with TEST_CSV.open("rb") as stream:
        x_test, y_test = read_test_csv(stream)

    rows = []
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / f"{MODELS[name]}.joblib", compress=3)
        metrics, matrix, _ = evaluate(pipeline, x_test, y_test)
        rows.append({"Model": name, **metrics})
        print(f"{name}: TN={matrix[0, 0]} FP={matrix[0, 1]} "
              f"FN={matrix[1, 0]} TP={matrix[1, 1]}")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "holdout_metrics.csv", index=False, float_format="%.10f")
    print(f"Training: {len(x_train)} rows, {y_train.sum()} malignant")
    print(f"Test: {len(x_test)} rows, {y_test.sum()} malignant, {x_test.shape[1]} features")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
