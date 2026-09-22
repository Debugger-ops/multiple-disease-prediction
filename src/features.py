"""
Leak-free preprocessing + feature engineering transformers.

Everything that *learns* from data (median imputation, scaling) now lives
inside a scikit-learn pipeline that is fit on the training fold only. The
previous version imputed diabetes zeros with the median of the WHOLE dataset
before the train/test split, which leaks a tiny amount of test-set
information into training -- small, but exactly the kind of thing a viva
examiner will ask about.

The fitted preprocessor is saved as `models/<disease>_preprocessor.pkl` and is
applied to raw form input in the app, so training and inference share the
exact same transformations.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Columns where a value of 0 is physiologically impossible in the Pima
# dataset and actually means "not measured".
DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in the given columns (stateless)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns or []:
            if col in X.columns:
                X[col] = X[col].astype(float).replace(0.0, np.nan)
        return X

    def set_output(self, *, transform=None):
        return self


class DiabetesFeatures(BaseEstimator, TransformerMixin):
    """
    Domain-motivated features for the Pima dataset (stateless, so no leakage).

    Runs BEFORE imputation so the missing-value indicators see the real gaps.
      - Insulin_missing / SkinThickness_missing : ~49% / ~30% of rows lack
        these measurements; whether a test was ordered can itself carry signal.
      - Glucose_x_BMI  : interaction of the two strongest single predictors.
      - Glucose_x_Insulin : glucose x insulin / 405 -- a HOMA-IR-style
        insulin-resistance proxy (Pima insulin is 2-hour, not fasting, so this
        is a proxy, not true HOMA-IR).
      - Glucose_ge_140 : 2-hour OGTT glucose in the WHO impaired-tolerance range.
      - BMI_ge_30      : WHO obesity threshold.
    """

    DERIVED_INFO = {
        "Insulin_missing": dict(label="Insulin not measured"),
        "SkinThickness_missing": dict(label="Skinfold thickness not measured"),
        "Glucose_x_BMI": dict(label="Glucose × BMI interaction"),
        "Glucose_x_Insulin": dict(label="Glucose × Insulin (insulin-resistance proxy)"),
        "Glucose_ge_140": dict(label="Glucose ≥ 140 mg/dL (impaired tolerance range)"),
        "BMI_ge_30": dict(label="BMI ≥ 30 (obese range)"),
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["Insulin_missing"] = X["Insulin"].isna().astype(float)
        X["SkinThickness_missing"] = X["SkinThickness"].isna().astype(float)
        X["Glucose_x_BMI"] = X["Glucose"] * X["BMI"]
        X["Glucose_x_Insulin"] = X["Glucose"] * X["Insulin"] / 405.0
        # Keep NaN where the source is missing so the imputer handles it.
        X["Glucose_ge_140"] = np.where(X["Glucose"].isna(), np.nan, (X["Glucose"] >= 140).astype(float))
        X["BMI_ge_30"] = np.where(X["BMI"].isna(), np.nan, (X["BMI"] >= 30).astype(float))
        return X

    def set_output(self, *, transform=None):
        return self


def build_preprocessor_steps(disease_key: str, feature_engineering: bool = False) -> list:
    """(name, transformer) steps: raw DataFrame in -> scaled numpy array out."""
    steps = []
    if disease_key == "diabetes":
        steps.append(("zero_as_missing", ZeroAsMissing(DIABETES_ZERO_AS_MISSING)))
        if feature_engineering:
            steps.append(("features", DiabetesFeatures()))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.set_output(transform="pandas")  # keep column names for SHAP labels
    steps.append(("impute", imputer))
    steps.append(("scaler", StandardScaler()))
    return steps


def build_preprocessor(disease_key: str, feature_engineering: bool = False) -> Pipeline:
    return Pipeline(build_preprocessor_steps(disease_key, feature_engineering))


def model_feature_names(fitted_preprocessor: Pipeline, X_sample: pd.DataFrame) -> list:
    """Column names as seen by the model (raw + any engineered features)."""
    return list(fitted_preprocessor[:-1].transform(X_sample.head(1)).columns)


def derived_feature_info(disease_key: str, feature_engineering: bool) -> dict:
    if disease_key == "diabetes" and feature_engineering:
        return dict(DiabetesFeatures.DERIVED_INFO)
    return {}
