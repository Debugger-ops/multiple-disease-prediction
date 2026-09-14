"""
Train, compare, and select the best model for each disease.

Pipeline per disease (matches the synopsis's proposed methodology):
    1. Load + clean data                              (src/preprocess.py)
    2. Stratified 80/20 train/test split
    3. Feature scaling (StandardScaler, fit on train only)
    4. Class-imbalance correction on the TRAINING split only, via SMOTE
    5. Train Logistic Regression, Random Forest, SVM, XGBoost
    6. Evaluate all four on the untouched (non-SMOTE) test split:
       accuracy, precision, recall, F1-score
    7. Select the best-performing model per disease (ranked by F1-score,
       which balances precision/recall -- appropriate for imbalanced
       clinical data -- ties broken by accuracy)
    8. Persist: best model, scaler, feature list, and comparison table

Run:
    python -m src.train
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocess import LOADERS, DISEASE_DISPLAY_NAMES

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42


def build_models():
    """Fresh, unfitted model instances -- rebuilt per disease to avoid state leakage."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def train_one_disease(disease_key: str):
    loader = LOADERS[disease_key]
    X, y, feature_info = loader()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Class imbalance correction -- training data only, never the test set.
    minority_count = y_train.value_counts().min()
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    models = build_models()
    rows = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted[best_name]

    # Persist artifacts for this disease.
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{disease_key}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{disease_key}_scaler.pkl"))
    joblib.dump(
        {
            "feature_names": feature_names,
            "feature_info": feature_info,
            "best_model_name": best_name,
            "X_train_res_sample": X_train_res[np.random.RandomState(RANDOM_STATE).choice(
                len(X_train_res), size=min(100, len(X_train_res)), replace=False
            )],  # background sample for SHAP, already scaled
        },
        os.path.join(MODELS_DIR, f"{disease_key}_meta.pkl"),
    )
    comparison.to_csv(os.path.join(RESULTS_DIR, f"{disease_key}_comparison.csv"), index=False)

    print(f"\n=== {DISEASE_DISPLAY_NAMES[disease_key]} ===")
    print(comparison.to_string(index=False))
    print(f"Best model: {best_name}  (F1={comparison.iloc[0]['F1-score']:.3f}, "
          f"Accuracy={comparison.iloc[0]['Accuracy']:.3f})")

    return comparison, best_name


def main():
    summary = {}
    for disease_key in LOADERS:
        comparison, best_name = train_one_disease(disease_key)
        summary[disease_key] = {
            "best_model": best_name,
            "metrics": comparison.iloc[0].to_dict(),
            "all_models": comparison.to_dict(orient="records"),
        }

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nAll models trained. Artifacts saved to models/ and results/.")


if __name__ == "__main__":
    main()
