"""
Data loading and preprocessing for the Multiple Disease Prediction System.

Each `load_<disease>()` function returns:
    X (pd.DataFrame) : cleaned feature matrix
    y (pd.Series)    : binary target (1 = disease present / high risk)
    feature_info (dict) : metadata used by the Streamlit app to build input
                          forms (display name, min/max/typical value, help text)

Cleaning steps applied here follow the synopsis's Step 1 (Data Acquisition &
Preprocessing): missing-value handling and feature scaling. Feature scaling
itself (StandardScaler) is fit later, inside the train/test split, to avoid
data leakage -- see train.py.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# --------------------------------------------------------------------------- #
# Diabetes -- Pima Indians Diabetes Dataset
# --------------------------------------------------------------------------- #
def load_diabetes():
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))

    # In this dataset, a physiological value of 0 is not biologically possible
    # for these columns and actually encodes a missing measurement.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)

    # Median imputation (robust to the skew/outliers typical of clinical data).
    for col in zero_as_missing:
        df[col] = df[col].fillna(df[col].median())

    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])

    feature_info = {
        "Pregnancies": dict(label="Number of Pregnancies", min=0, max=17, step=1, default=1, help="Total number of pregnancies"),
        "Glucose": dict(label="Plasma Glucose Concentration (mg/dL)", min=40.0, max=250.0, step=1.0, default=120.0, help="2-hour oral glucose tolerance test result"),
        "BloodPressure": dict(label="Diastolic Blood Pressure (mm Hg)", min=30.0, max=140.0, step=1.0, default=70.0, help=""),
        "SkinThickness": dict(label="Triceps Skinfold Thickness (mm)", min=5.0, max=100.0, step=1.0, default=25.0, help=""),
        "Insulin": dict(label="2-Hour Serum Insulin (mu U/mL)", min=10.0, max=850.0, step=1.0, default=80.0, help=""),
        "BMI": dict(label="Body Mass Index (kg/m^2)", min=10.0, max=70.0, step=0.1, default=28.0, help=""),
        "DiabetesPedigreeFunction": dict(label="Diabetes Pedigree Function", min=0.05, max=2.5, step=0.01, default=0.4, help="Likelihood of diabetes based on family history"),
        "Age": dict(label="Age (years)", min=18, max=100, step=1, default=33, help=""),
    }
    return X, y, feature_info


# --------------------------------------------------------------------------- #
# Heart Disease -- Cleveland (UCI) Heart Disease Dataset
# --------------------------------------------------------------------------- #
def load_heart():
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
    df.columns = [c.strip() for c in df.columns]

    y = df["target"].astype(int)
    X = df.drop(columns=["target"])

    feature_info = {
        "age": dict(label="Age (years)", min=18, max=100, step=1, default=54, help=""),
        "sex": dict(label="Sex", min=0, max=1, step=1, default=1, help="1 = Male, 0 = Female", choices={0: "Female", 1: "Male"}),
        "cp": dict(label="Chest Pain Type", min=0, max=3, step=1, default=0, help="", choices={0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}),
        "trestbps": dict(label="Resting Blood Pressure (mm Hg)", min=80.0, max=220.0, step=1.0, default=130.0, help=""),
        "chol": dict(label="Serum Cholesterol (mg/dL)", min=100.0, max=600.0, step=1.0, default=245.0, help=""),
        "fbs": dict(label="Fasting Blood Sugar > 120 mg/dL", min=0, max=1, step=1, default=0, help="", choices={0: "No", 1: "Yes"}),
        "restecg": dict(label="Resting ECG Result", min=0, max=2, step=1, default=1, help="", choices={0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}),
        "thalach": dict(label="Max Heart Rate Achieved", min=60.0, max=220.0, step=1.0, default=150.0, help=""),
        "exang": dict(label="Exercise-Induced Angina", min=0, max=1, step=1, default=0, help="", choices={0: "No", 1: "Yes"}),
        "oldpeak": dict(label="ST Depression (exercise vs rest)", min=0.0, max=7.0, step=0.1, default=1.0, help=""),
        "slope": dict(label="Slope of Peak Exercise ST Segment", min=0, max=2, step=1, default=1, help="", choices={0: "Upsloping", 1: "Flat", 2: "Downsloping"}),
        "ca": dict(label="Number of Major Vessels Colored by Fluoroscopy", min=0, max=4, step=1, default=0, help=""),
        "thal": dict(label="Thalassemia", min=0, max=3, step=1, default=2, help="", choices={0: "Unknown", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}),
    }
    return X, y, feature_info


# --------------------------------------------------------------------------- #
# Parkinson's Disease -- UCI Parkinson's Disease Dataset (voice biomarkers)
# --------------------------------------------------------------------------- #
def load_parkinsons():
    df = pd.read_csv(os.path.join(DATA_DIR, "parkinsons.data"))

    y = df["status"].astype(int)
    X = df.drop(columns=["status", "name"])

    labels = {
        "MDVP:Fo(Hz)": "Avg Vocal Fundamental Frequency (Hz)",
        "MDVP:Fhi(Hz)": "Max Vocal Fundamental Frequency (Hz)",
        "MDVP:Flo(Hz)": "Min Vocal Fundamental Frequency (Hz)",
        "MDVP:Jitter(%)": "Jitter %",
        "MDVP:Jitter(Abs)": "Jitter (Absolute)",
        "MDVP:RAP": "Relative Amplitude Perturbation",
        "MDVP:PPQ": "Five-point Period Perturbation Quotient",
        "Jitter:DDP": "Avg Absolute Difference of Differences (Jitter)",
        "MDVP:Shimmer": "Shimmer",
        "MDVP:Shimmer(dB)": "Shimmer (dB)",
        "Shimmer:APQ3": "Three-point Amplitude Perturbation Quotient",
        "Shimmer:APQ5": "Five-point Amplitude Perturbation Quotient",
        "MDVP:APQ": "Amplitude Perturbation Quotient",
        "Shimmer:DDA": "Avg Absolute Differences of Amplitudes (Shimmer)",
        "NHR": "Noise-to-Harmonics Ratio",
        "HNR": "Harmonics-to-Noise Ratio",
        "RPDE": "Recurrence Period Density Entropy",
        "DFA": "Detrended Fluctuation Analysis",
        "spread1": "Fundamental Frequency Variation 1",
        "spread2": "Fundamental Frequency Variation 2",
        "D2": "Correlation Dimension",
        "PPE": "Pitch Period Entropy",
    }

    feature_info = {}
    for col in X.columns:
        col_min = float(X[col].min())
        col_max = float(X[col].max())
        col_med = float(X[col].median())
        span = col_max - col_min
        step = round(span / 200, 6) if span > 0 else 0.001
        feature_info[col] = dict(
            label=labels.get(col, col),
            min=round(col_min - 0.1 * abs(col_min), 6),
            max=round(col_max + 0.1 * abs(col_max), 6),
            step=step,
            default=round(col_med, 6),
            help="Voice biomarker extracted from sustained vowel phonation",
        )
    return X, y, feature_info


LOADERS = {
    "diabetes": load_diabetes,
    "heart": load_heart,
    "parkinsons": load_parkinsons,
}

DISEASE_DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "heart": "Heart Disease",
    "parkinsons": "Parkinson's Disease",
}
