# Multiple Disease Prediction System Using Machine Learning

Minor project — Department of ECE, Maharaja Surajmal Institute of Technology
Group: Shesh Narayan, Vivek Pant, Parambrata Majumdar · Supervisor: Prof. (Dr.) Neeru Rathee

A single, web-deployed screening tool that predicts risk for **Diabetes**,
**Heart Disease**, and **Parkinson's Disease** from one interface, and
explains every prediction in plain language using SHAP — built to match the
synopsis's proposed methodology end to end.

## What's included

```
mdps/
├── app.py                    # Streamlit web app (the deployed product)
├── requirements.txt
├── data/                     # Raw public datasets (see Datasets below)
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.data
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis for all three datasets
├── src/
│   ├── preprocess.py         # Cleaning, missing-value handling, feature metadata
│   ├── train.py               # Train/compare/select models, SMOTE, save artifacts
│   └── explain.py             # SHAP explainer selection + plain-language summaries
├── models/                   # Saved best model + scaler + metadata per disease
└── results/                  # Per-disease comparison tables, confusion matrices,
                               # ROC curve data, and summary.json
```

## How it maps to the synopsis

| Synopsis stage | Implementation |
|---|---|
| Step 1 — Data Acquisition & Preprocessing | `src/preprocess.py`: loads Pima Diabetes, Cleveland Heart Disease, UCI Parkinson's; imputes biologically-impossible zeros (diabetes) with the median; scales features with `StandardScaler`. Explored in `notebooks/eda.ipynb`. |
| Step 2 — Class Imbalance Correction | `src/train.py`: SMOTE applied to the **training split only** (never the test set, to avoid leakage) |
| Step 3 — Model Development & Comparison | Logistic Regression, Random Forest, SVM (RBF), XGBoost trained and compared per disease on Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Step 4 — Model Interpretability | `src/explain.py`: SHAP (`TreeExplainer` for RF/XGBoost, `LinearExplainer` for Logistic Regression, `KernelExplainer` for SVM), surfaced as plain-language "this factor is pushing risk up/down" statements |
| Step 5 — Deployment | `app.py`: Streamlit app — pick a disease, fill a form, get an instant risk prediction + explanation |

## Datasets

Standard, public, well-documented benchmark datasets, exactly as named in the
synopsis's literature review:

- **Diabetes**: Pima Indians Diabetes Dataset (768 records, 8 features)
- **Heart Disease**: Cleveland Heart Disease Dataset (UCI, 303 records, 13 features)
- **Parkinson's Disease**: UCI Parkinson's voice-biomarker dataset (195 records, 22 features — note: ~6 voice recordings per patient, not 195 independent patients; see `notebooks/eda.ipynb`)

`notebooks/eda.ipynb` walks through missing/zero-value checks, class balance,
feature distributions, and correlation structure for all three datasets
before any cleaning or training happens.

## Results (held-out 20% test split, after SMOTE-balanced training)

Best model selected per disease by highest F1-score. Full comparison across
all four algorithms is in `results/*_comparison.csv` and `results/summary.json`,
and is also viewable live in the app's **Model Performance** tab — which
includes an interactive grouped-bar chart, a radar chart, a per-model
confusion matrix, and an ROC curve overlay (all four models, AUC in the
legend), each with hover tooltips via Plotly. Raw confusion matrix and ROC
curve data for every model/disease combination is in `results/*_curves.json`.

| Disease | Best model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Diabetes | Random Forest | 74.7% | 62.3% | 70.4% | 66.1% | 81.5% |
| Heart Disease | Random Forest | 83.6% | 78.0% | 97.0% | 86.5% | 90.8% |
| Parkinson's Disease | XGBoost | 92.3% | 96.4% | 93.1% | 94.7% | 97.6% |

**Honest note on the ≥85% accuracy target:** the synopsis's Expected Outcomes
section targets 85%+ accuracy. Heart Disease and Parkinson's clear that bar
comfortably. Diabetes (Pima dataset) tops out around 74–75% accuracy with
these four classical/ensemble algorithms — this is consistent with published
literature on this specific dataset (it's a small, noisy dataset with only 8
features and known label ambiguity), not a bug in the pipeline. This is worth
stating explicitly in your presentation/viva rather than hiding it — target
metrics in a synopsis are goals, not guarantees, and reviewers generally
respond well to a team that reports honest numbers with a reasoned
explanation over one that just states "achieved 85%+" without evidence.
Legitimate ways to try to close the gap if you want to push further: engineer
additional features, try stacking/ensembling the four models together, or use
a more recent/larger diabetes dataset alongside Pima.

## Running it yourself

```bash
pip install -r requirements.txt

# (Re-)train all three models from scratch — writes to models/ and results/
python -m src.train

# Launch the web app
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`),
pick a disease in the sidebar, fill in the clinical parameters, and click
**Predict Risk**.

To explore the raw datasets interactively:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Design notes / decisions worth mentioning in your viva

- **Why F1-score to pick the "best" model**, not raw accuracy: these are
  imbalanced clinical datasets (e.g. only 34.9% positive in the diabetes set).
  A model can look "accurate" by mostly predicting the majority class while
  missing real positive cases. F1-score balances precision and recall, which
  matters far more for a screening tool where missing a true positive
  (false negative) has a real cost.
- **Why SMOTE is applied only to the training split**: applying it before the
  train/test split would leak synthetic copies of test-set patterns into
  training, silently inflating the reported metrics. This pipeline splits
  first, then resamples only the training data — the test set stays
  untouched and represents real-world class distribution.
- **Why the SHAP explainer differs by model**: `TreeExplainer` is exact and
  fast for the tree-based models (Random Forest, XGBoost); `LinearExplainer`
  is the appropriate exact method for Logistic Regression; SVM has no
  closed-form SHAP method, so `KernelExplainer` is used against a
  k-means-summarized background for tractable runtime.
- **Confusion matrix and ROC curve, not just accuracy**: a single accuracy
  number hides *how* a model fails. The Model Performance tab's confusion
  matrix shows the actual false-positive/false-negative trade-off per model,
  and the ROC curve shows performance across all classification thresholds,
  not just the default 0.5 cutoff — both computed once during training
  (`src/train.py`) and persisted to `results/*_curves.json`, since the
  test-set labels aren't available at inference time in the deployed app.
- **Not a diagnostic tool**: this is explicitly framed in the app UI as an
  educational / preliminary-screening demonstration, consistent with the
  synopsis's own problem statement (a screening aid, not a replacement for a
  specialist).

## Suggested next steps for the full project (beyond this synopsis-stage prototype)

- Hyperparameter tuning (GridSearchCV / Optuna) per disease/model
- Cross-validation instead of a single train/test split, for more robust metrics
- Targeted feature engineering for the Diabetes model specifically, to close
  the accuracy gap noted above
- Optionally: containerize (Dockerfile) for the final demo/deployment
