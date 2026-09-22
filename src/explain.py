"""
SHAP-based interpretability layer.

Provides:
    get_explainer(model, background)  -> a fitted SHAP explainer, choosing
        the fastest exact/appropriate method for the model family:
            - Tree models (Random Forest, XGBoost) -> shap.TreeExplainer
            - Logistic Regression                   -> shap.LinearExplainer
            - Everything else (e.g. SVM)             -> shap.KernelExplainer
              on a k-means-summarized background for speed.

    explain_instance(...) -> per-feature SHAP contribution values for one
        input row, already resolved to "contributes toward disease" (+ve)
        vs "contributes toward healthy" (-ve) on the positive class.

    plain_language_summary(...) -> a short, human-readable explanation of
        the top contributing factors, per Objective #5 of the synopsis
        ("explains each prediction in plain language").
"""

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_explainer(model, background: np.ndarray):
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            # Some shap/xgboost version pairs can't parse the booster
            # (e.g. base_score "[5E-1]"); fall back to the model-agnostic path.
            pass
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    # Fallback (e.g. SVM / kernel models) -- summarize background for speed.
    n_summary = min(30, background.shape[0])
    summary = shap.kmeans(background, n_summary)
    return shap.KernelExplainer(model.predict_proba, summary)


def _positive_class_values(shap_values, model):
    """Normalize the various shapes SHAP can return into a 1D array for the positive class."""
    if isinstance(shap_values, list):
        # KernelExplainer / older TreeExplainer API: list of arrays per class.
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.array(vals).reshape(-1)
    vals = np.array(shap_values)
    if vals.ndim == 3:
        # shape (n_samples, n_features, n_classes)
        return vals[0, :, 1] if vals.shape[2] > 1 else vals[0, :, 0]
    if vals.ndim == 2:
        return vals[0]
    return vals.reshape(-1)


def explain_instance(explainer, model, x_scaled_row: np.ndarray, feature_names):
    """
    x_scaled_row: shape (1, n_features), already scaled the same way training data was.
    Returns: dict[feature_name] -> shap contribution (float), sorted by |value| descending.
    """
    raw = explainer.shap_values(x_scaled_row)
    values = _positive_class_values(raw, model)
    contributions = dict(zip(feature_names, values.tolist()))
    return dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))


def plain_language_summary(contributions: dict, feature_info: dict, top_k: int = 4) -> list:
    """
    Turn signed SHAP contributions into plain-language bullet strings, e.g.
    "Glucose is pushing the risk UP the most" / "BMI is pushing the risk DOWN".
    """
    items = list(contributions.items())[:top_k]
    lines = []
    for feat, val in items:
        label = feature_info.get(feat, {}).get("label", feat)
        direction = "increasing" if val > 0 else "decreasing"
        strength = abs(val)
        lines.append({
            "feature": feat,
            "label": label,
            "direction": direction,
            "shap_value": val,
            "strength": strength,
        })
    return lines
