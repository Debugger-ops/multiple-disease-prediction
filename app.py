"""
Multiple Disease Prediction System -- Streamlit web app.

Single-interface app (per synopsis Step 5 / objective #6): user picks a
disease, fills in a form of clinical parameters, and receives an instant
ML-based risk prediction with a SHAP-based plain-language explanation.

Run:
    streamlit run app.py
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocess import LOADERS, DISEASE_DISPLAY_NAMES
from src.explain import get_explainer, explain_instance, plain_language_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.set_page_config(page_title="Multiple Disease Prediction System", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Caching: load model artifacts once per disease
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts(disease_key: str):
    model = joblib.load(os.path.join(MODELS_DIR, f"{disease_key}_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, f"{disease_key}_scaler.pkl"))
    meta = joblib.load(os.path.join(MODELS_DIR, f"{disease_key}_meta.pkl"))
    explainer = get_explainer(model, meta["X_train_res_sample"])
    return model, scaler, meta, explainer


@st.cache_data
def load_comparison(disease_key: str):
    path = os.path.join(RESULTS_DIR, f"{disease_key}_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def render_input_form(disease_key: str, feature_info: dict, feature_names: list):
    st.subheader("Enter Clinical Parameters")
    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        info = feature_info[feat]
        col = cols[i % 2]
        with col:
            if "choices" in info:
                choice_keys = list(info["choices"].keys())
                idx = choice_keys.index(info["default"]) if info["default"] in choice_keys else 0
                sel = st.selectbox(
                    info["label"],
                    options=choice_keys,
                    index=idx,
                    format_func=lambda k, ch=info["choices"]: ch[k],
                    key=f"{disease_key}_{feat}",
                    help=info.get("help", ""),
                )
                values[feat] = sel
            else:
                is_int = isinstance(info["default"], int) and float(info["step"]).is_integer()
                if is_int:
                    val = st.number_input(
                        info["label"],
                        min_value=int(info["min"]),
                        max_value=int(info["max"]),
                        value=int(info["default"]),
                        step=int(info["step"]),
                        key=f"{disease_key}_{feat}",
                        help=info.get("help", ""),
                    )
                else:
                    val = st.number_input(
                        info["label"],
                        min_value=float(info["min"]),
                        max_value=float(info["max"]),
                        value=float(info["default"]),
                        step=float(info["step"]),
                        key=f"{disease_key}_{feat}",
                        help=info.get("help", ""),
                        format="%.4f",
                    )
                values[feat] = val
    return values


def render_shap_chart(summary_items, positive_label, negative_label):
    labels = [item["label"] for item in reversed(summary_items)]
    shap_vals = [item["shap_value"] for item in reversed(summary_items)]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(6, 0.6 * len(labels) + 1))
    ax.barh(labels, shap_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (→ higher risk | → lower risk ←)")
    ax.set_title("Top factors influencing this prediction")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("🩺 Multiple Disease Prediction System")
    st.caption(
        "A single web interface that screens for **Diabetes**, **Heart Disease**, and "
        "**Parkinson's Disease** from user-entered clinical parameters, with SHAP-based "
        "plain-language explanations for every prediction."
    )

    with st.sidebar:
        st.header("Select Disease")
        disease_key = st.radio(
            "Choose a screening module:",
            options=list(DISEASE_DISPLAY_NAMES.keys()),
            format_func=lambda k: DISEASE_DISPLAY_NAMES[k],
        )
        st.markdown("---")
        st.markdown(
            "**Methodology:** SMOTE-balanced training data · "
            "Logistic Regression / Random Forest / SVM / XGBoost compared · "
            "best model selected per disease · SHAP interpretability."
        )
        st.markdown(
            "⚠️ This tool is for **educational / preliminary screening demonstration "
            "purposes only** and is **not a medical diagnosis**. Always consult a "
            "qualified healthcare professional."
        )

    model, scaler, meta, explainer = load_artifacts(disease_key)
    feature_names = meta["feature_names"]
    feature_info = meta["feature_info"]
    best_model_name = meta["best_model_name"]

    tab_predict, tab_performance = st.tabs(["🔍 Predict", "📊 Model Performance"])

    with tab_predict:
        left, right = st.columns([3, 2])

        with left:
            values = render_input_form(disease_key, feature_info, feature_names)
            predict_clicked = st.button("Predict Risk", type="primary", width='stretch')

        with right:
            st.subheader("Result")
            if predict_clicked:
                x_row = pd.DataFrame([values])[feature_names]
                x_scaled = scaler.transform(x_row)

                proba = model.predict_proba(x_scaled)[0].astype(float)
                pred = int(np.argmax(proba))
                risk_pct = float(proba[1] * 100)

                disease_display = DISEASE_DISPLAY_NAMES[disease_key]
                if pred == 1:
                    st.error(f"**High risk indicator for {disease_display}**")
                else:
                    st.success(f"**Low risk indicator for {disease_display}**")
                st.metric("Predicted probability of disease", f"{risk_pct:.1f}%")
                st.progress(float(min(max(proba[1], 0.0), 1.0)))
                st.caption(f"Model used: {best_model_name} (best-performing model for this disease)")

                st.markdown("#### Why this prediction? (SHAP explanation)")
                contributions = explain_instance(explainer, model, x_scaled, feature_names)
                summary_items = plain_language_summary(contributions, feature_info, top_k=6)

                for item in summary_items:
                    arrow = "⬆️" if item["direction"] == "increasing" else "⬇️"
                    st.write(
                        f"{arrow} **{item['label']}** is {item['direction']} the predicted risk "
                        f"(contribution: {item['shap_value']:+.3f})"
                    )

                render_shap_chart(summary_items, "Higher risk", "Lower risk")
            else:
                st.info("Fill in the parameters and click **Predict Risk** to see a result.")

    with tab_performance:
        st.subheader(f"Algorithm comparison — {DISEASE_DISPLAY_NAMES[disease_key]}")
        comparison = load_comparison(disease_key)
        if comparison is not None:
            st.dataframe(
                comparison.style.format({
                    "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                    "Recall": "{:.3f}", "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}",
                }).highlight_max(subset=["F1-score"], color="#c6f5c6"),
                width='stretch',
            )
            st.caption(
                f"Best model selected: **{best_model_name}** (highest F1-score on held-out test data, "
                "computed after SMOTE-based class-imbalance correction on the training split only)."
            )
        else:
            st.warning("No comparison results found. Run `python -m src.train` first.")


if __name__ == "__main__":
    main()
