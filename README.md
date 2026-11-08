# Wisconsin Classification Bench

BITS Pilani WILP · Machine Learning Assignment 2 · Instructional reference

Use this project to learn the workflow and check your understanding. Build and review your own version in BITS Virtual Lab. Write your own interpretation from your run, and follow the assignment's rules on AI assistance and originality.

## A. Problem statement

Compare five supervised classifiers on a single binary classification task. Given 30 measurements of cell nuclei, predict whether the dataset labels a sample as malignant or benign. An interactive Streamlit app evaluates saved models on a labeled test CSV.

## B. Dataset description

Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, via the copy bundled with scikit-learn. The 569 records include 357 benign and 212 malignant cases. The 30 numeric predictors describe the mean, standard error and worst values of ten cell-nucleus characteristics, such as radius, texture and area. The loader omits the original identifier. This project encodes the target as `malignant`, where 1 means malignant and 0 means benign.

The stratified 80/20 split uses seed 42 and yields 455 training rows (170 malignant) and 114 test rows (42 malignant). Every classifier uses the same split. The scaler, where used, is fitted on the training rows inside the saved pipeline. The reference uses fixed hyperparameters and evaluates on one split.

Data credit and license: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5DW2B) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled CSV is a subset with descriptive feature names, an omitted ID, and recoded labels.

## C. GitHub repository link

Add your verified GitHub repository URL here after publishing your reviewed project.

## D. Models and evaluation

| Model | Configuration |
| --- | --- |
| Logistic Regression | Standard scaling, L2 regularization, maximum 2,000 iterations |
| Decision Tree | Maximum depth 4 |
| KNN | Standard scaling, 7 neighbors, uniform voting |
| Gaussian Naive Bayes | Gaussian feature likelihoods with default variance smoothing |
| Support Vector Machine | Standard scaling, RBF kernel, C = 1, gamma = scale |

Accuracy is the fraction of correct predictions. Precision is the fraction of predicted malignant cases labeled malignant. Recall is the fraction of labeled malignant cases detected. F1 is the harmonic mean of precision and recall. ROC AUC uses positive-class decision scores for Logistic Regression and SVM, and positive-class probabilities for the other models. MCC uses all four confusion-matrix cells and ranges from −1 to +1. For one-class uploads, the app shows AUC as unavailable. Undefined precision, recall and F1 use zero.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.8770 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations from the reference run

| Model | Observation |
| --- | --- |
| Logistic Regression | Highest ROC AUC (0.9960). It missed three malignant cases and flagged one benign case. |
| Decision Tree | Lowest ROC AUC and MCC on this split. It missed eight malignant cases, the most of the five. |
| KNN | It missed four malignant cases and flagged one benign case. Its MCC was 0.9058. |
| Gaussian Naive Bayes | It flagged no benign cases and achieved 0.9934 AUC, while its default predictions missed seven malignant cases. |
| Support Vector Machine | It flagged no benign cases and missed three malignant cases, tying Logistic Regression for highest recall. |
| Overall winner on this holdout | SVM by MCC (0.9442). It also led on accuracy and F1. Logistic Regression led on AUC. A one-error gap in 114 cases warrants a fresh evaluation before generalizing the ranking. |

## Run the project

Use Python 3.12. From the project folder, create a virtual environment and install the pinned dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. The app can also run immediately after installing dependencies using the included models and test CSV. Restart the app after regenerating models. Only load trusted Joblib files from your repository.

In the sidebar, choose a classifier or upload a UTF-8 test CSV. Download the example CSV to see the schema. Use the same 30 feature names and the binary `malignant` column. Column order can vary. All required values must be present and finite. The comparison table evaluates all five models on the current rows. The dropdown controls the detailed metrics, confusion matrix and per-class report. The app performs inference without fitting.

| Path | Purpose |
| --- | --- |
| `train.py` | Rebuild the split, train the models and write evaluation artifacts |
| `experiment.py` | Define the CSV schema and shared evaluation logic |
| `app.py` | Run interactive evaluation with saved pipelines |
| `requirements.txt` | Install the tested package versions |
| `test_data.csv` | Shared labeled holdout |
| `holdout_metrics.csv` | Six metrics for each classifier on the bundled holdout |
| `model/*.joblib` | Five saved classifiers and their fitted preprocessing |

## BITS Virtual Lab and submission

1. Open BITS Virtual Lab, transfer your reviewed project to your lab workspace and use the lab terminal to install requirements and run `python train.py`. Launch the app in the lab if the environment supports its browser preview. Capture one genuine screenshot of assignment execution showing the lab context, training command and five-model output.
2. Create your GitHub repository and push the reviewed project, including the test CSV, metrics, five Joblib files and requirements. Exclude your virtual environment and credentials. Update the README with your repository URL and your own results and observations.
3. In Streamlit Community Cloud, create an app from that repository and branch, set `app.py` as the entry point, select Python 3.12 under Advanced settings and deploy. Open the resulting URL in a signed-out window and exercise the CSV upload and model dropdown. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
4. Assemble a single PDF in this order: clickable GitHub link, clickable live Streamlit link, one genuine BITS Virtual Lab screenshot, and your updated README content. Verify both links in the exported PDF and submit it through the course portal.
