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
