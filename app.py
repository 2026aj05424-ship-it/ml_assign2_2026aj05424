"""Explore model performance on a labeled held-out CSV."""

from io import BytesIO

import joblib
import pandas as pd
import streamlit as st

from experiment import CLASS_NAMES, MODEL_DIR, MODELS, TEST_CSV, evaluate, read_test_csv


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / f"{MODELS[name]}.joblib")


def main():
    st.set_page_config(page_title="Wisconsin Classification Bench", layout="wide")
    st.title("Wisconsin Classification Bench")
    st.write("Compare five classifiers on measurements of breast cell nuclei.")
    st.caption("Instructional reference · Malignant = 1 · Benign = 0")

    with st.sidebar:
        st.header("Evaluation settings")
        selected = st.selectbox("Classifier", list(MODELS))
        uploaded = st.file_uploader(
            "Upload labeled test CSV", type=["csv"],
            help="Include the 30 feature columns and the malignant target from the sample.",
        )
        st.download_button("Download example test CSV", TEST_CSV.read_bytes(), TEST_CSV.name, "text/csv")

    try:
        stream = BytesIO(TEST_CSV.read_bytes()) if uploaded is None else uploaded
        stream.seek(0)
        features, labels = read_test_csv(stream)
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the test data. {exc}")
        return

    source = "Bundled holdout" if uploaded is None else uploaded.name
    st.text(f"{source} · {len(labels)} rows · {int(labels.sum())} malignant")
    st.subheader(selected)
    try:
        results = {
            name: evaluate(load_model(name), features, labels)
            for name in MODELS
        }
    except FileNotFoundError:
        st.error("A saved model is missing. Run python train.py and restart the app.")
        return

    metrics, matrix, report = results[selected]
    for column, (label, value) in zip(st.columns(6), metrics.items()):
        column.metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")
    if labels.nunique() == 1:
        st.info("ROC AUC requires both classes. The other metrics use the rows supplied.")

    errors, per_class = st.tabs(["Confusion matrix", "Classification report"])
    with errors:
        st.caption("Rows show actual labels. Columns show predicted labels.")
        st.dataframe(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES))
        st.caption(f"Missed malignant cases: {matrix[1, 0]} · Benign cases flagged: {matrix[0, 1]}")
    with per_class:
        st.dataframe(pd.DataFrame(report).T.loc[CLASS_NAMES].round(4))

    st.subheader("All five models on the same rows")
    comparison = pd.DataFrame.from_dict(
        {name: result[0] for name, result in results.items()}, orient="index",
    )
    st.dataframe(comparison.round(4))
    st.caption("Precision, recall and F1 use malignant as the positive class. "
               "Undefined precision or recall is displayed as zero.")


if __name__ == "__main__":
    main()
