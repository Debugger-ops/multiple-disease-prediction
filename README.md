<<<<<<< HEAD
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
├── src/
│   ├── preprocess.py         # Cleaning, missing-value handling, feature metadata
│   ├── train.py              # Train/compare/select models, SMOTE, save artifacts
│   └── explain.py            # SHAP explainer selection + plain-language summaries
├── models/                   # Saved best model + scaler + metadata per disease
└── results/                  # Per-disease algorithm comparison tables + summary.json
```

## How it maps to the synopsis

| Synopsis stage | Implementation |
|---|---|
| Step 1 — Data Acquisition & Preprocessing | `src/preprocess.py`: loads Pima Diabetes, Cleveland Heart Disease, UCI Parkinson's; imputes biologically-impossible zeros (diabetes) with the median; scales features with `StandardScaler` |
| Step 2 — Class Imbalance Correction | `src/train.py`: SMOTE applied to the **training split only** (never the test set, to avoid leakage) |
| Step 3 — Model Development & Comparison | Logistic Regression, Random Forest, SVM (RBF), XGBoost trained and compared per disease on Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Step 4 — Model Interpretability | `src/explain.py`: SHAP (`TreeExplainer` for RF/XGBoost, `LinearExplainer` for Logistic Regression, `KernelExplainer` for SVM), surfaced as plain-language "this factor is pushing risk up/down" statements |
| Step 5 — Deployment | `app.py`: Streamlit app — pick a disease, fill a form, get an instant risk prediction + explanation |

## Datasets

Standard, public, well-documented benchmark datasets, exactly as named in the
synopsis's literature review:

- **Diabetes**: Pima Indians Diabetes Dataset (768 records, 8 features)
- **Heart Disease**: Cleveland Heart Disease Dataset (UCI, 303 records, 13 features)
- **Parkinson's Disease**: UCI Parkinson's voice-biomarker dataset (195 records, 22 features)

## Results (held-out 20% test split, after SMOTE-balanced training)

Best model selected per disease by highest F1-score. Full comparison across
all four algorithms is in `results/*_comparison.csv` and `results/summary.json`,
and is also viewable live in the app's "Model Performance" tab.

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
- **Not a diagnostic tool**: this is explicitly framed in the app UI as an
  educational / preliminary-screening demonstration, consistent with the
  synopsis's own problem statement (a screening aid, not a replacement for a
  specialist).

## Suggested next steps for the full project (beyond this synopsis-stage prototype)

- Hyperparameter tuning (GridSearchCV / Optuna) per disease/model
- Cross-validation instead of a single train/test split, for more robust metrics
- A short evaluation write-up per disease (confusion matrix, ROC curve) for the final report
- Optionally: containerize (Dockerfile) for the final demo/deployment
=======
# Multiple Disease Prediction System

A web application that predicts the likelihood of multiple diseases — **Diabetes**, **Heart Disease**, and **Parkinson's Disease** — from patient health parameters, using machine learning models served through an interactive Streamlit interface.

> Note: This README is scaffolded for a project built with **Python, scikit-learn, and Streamlit**, covering the standard three-disease set. Swap in additional diseases, datasets, or a different stack as the project takes shape — see [Customizing This Project](#customizing-this-project).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Performance](#model-performance)
- [Customizing This Project](#customizing-this-project)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

Early risk detection can help patients seek timely medical advice. This project brings together three independently trained classification models — one per disease — behind a single Streamlit dashboard, so a user can select a disease, enter the relevant clinical parameters, and get an instant prediction.

Each disease module is self-contained: its own dataset, its own trained model (pickled), and its own input form, all wired into one multi-page Streamlit app via a sidebar menu.

## Features

- Single dashboard covering three disease-prediction modules
- Sidebar navigation between disease modules (via `streamlit-option-menu`)
- Real-time predictions from pre-trained, pickled scikit-learn models
- Input validation on clinical parameter fields
- Clear positive/negative result messaging for each module
- Easily extensible to add new diseases or swap models

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML / Modeling | scikit-learn, pandas, NumPy |
| Web App / UI | Streamlit, streamlit-option-menu |
| Model Serialization | pickle |
| Notebook / Training | Jupyter Notebook |

## Project Structure

```
multiple-disease-prediction/
├── app.py                      # Main Streamlit app (sidebar + routing)
├── models/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── datasets/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── notebooks/
│   ├── diabetes_training.ipynb
│   ├── heart_disease_training.ipynb
│   └── parkinsons_training.ipynb
├── requirements.txt
└── README.md
```

## Datasets

| Disease | Dataset | Source |
|---|---|---|
| Diabetes | PIMA Indians Diabetes Dataset | Kaggle / UCI ML Repository |
| Heart Disease | Cleveland Heart Disease Dataset | UCI ML Repository |
| Parkinson's Disease | Parkinson's Disease Data Set | UCI ML Repository |

Update this table with the actual dataset links and versions once finalized.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/multiple-disease-prediction.git
cd multiple-disease-prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt (starting point)

```
streamlit
streamlit-option-menu
scikit-learn
pandas
numpy
```

### Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. Launch the app with `streamlit run app.py`.
2. Choose a disease module from the sidebar: **Diabetes Prediction**, **Heart Disease Prediction**, or **Parkinson's Prediction**.
3. Enter the requested clinical parameters (see below).
4. Click **Predict** / **Test Result** to view the outcome.

### Input Parameters by Module

**Diabetes Prediction**
Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

**Heart Disease Prediction**
Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, Slope, Major Vessels, Thalassemia

**Parkinson's Prediction**
MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), Jitter, Shimmer, NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE

Fill in the exact fields used by your trained models once finalized.

## Model Training

Each model is trained in its own notebook under `notebooks/`, following the same general workflow:

1. Load and clean the dataset
2. Handle missing values / outliers
3. Split into train/test sets
4. Train a classifier (e.g. Logistic Regression, SVM, or Random Forest — record the actual algorithm used per disease)
5. Evaluate on the test set
6. Serialize the trained model with `pickle` into `models/`

```python
import pickle

with open('models/diabetes_model.sav', 'wb') as f:
    pickle.dump(trained_model, f)
```

## Model Performance

| Disease | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Diabetes | TBD | — | — | — | — |
| Heart Disease | TBD | — | — | — | — |
| Parkinson's | TBD | — | — | — | — |

Fill this table in once training is complete — it's one of the first things reviewers look for.

## Customizing This Project

- **Add a disease module**: create a new dataset + training notebook, save the model to `models/`, add a new sidebar option and input form in `app.py`.
- **Swap the framework**: the same models can be served through Flask, FastAPI, or Django instead of Streamlit if you need more control over the UI.
- **Deploy**: Streamlit Community Cloud, Render, or Hugging Face Spaces are common low-friction options for a portfolio deployment.

## Roadmap

- [ ] Finalize datasets and confirm feature sets per disease
- [ ] Train and evaluate baseline models
- [ ] Build Streamlit UI and wire up predictions
- [ ] Fill in the Model Performance table with real metrics
- [ ] Deploy a live demo and link it here
- [ ] Add unit tests for input validation and prediction functions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-disease-x`)
3. Commit your changes
4. Open a pull request describing the change and any new dependencies

## License

Specify a license (e.g. MIT) once you've decided — this determines how others can use and modify the project.

---

**Disclaimer**: This tool is built for educational/portfolio purposes and is not a substitute for professional medical diagnosis. Predictions should not be used for actual clinical decision-making.
>>>>>>> origin/main
