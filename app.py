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
import plotly.express as px
import plotly.graph_objects as go

from src.preprocess import LOADERS, DISEASE_DISPLAY_NAMES
from src.explain import get_explainer, explain_instance, plain_language_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

st.set_page_config(page_title="Multiple Disease Prediction System", page_icon="🩺", layout="wide")

RESULT_CARD_CSS = """
<style>
.risk-card {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 1.75rem;
    border-radius: 16px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.18);
    margin-bottom: 0.75rem;
}
.risk-gauge {
    position: relative;
    width: 110px;
    height: 110px;
    min-width: 110px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.risk-gauge::before {
    content: "";
    position: absolute;
    inset: 11px;
    border-radius: 50%;
    background: var(--background-color);
}
.risk-gauge-value {
    position: relative;
    z-index: 1;
    text-align: center;
    line-height: 1.15;
}
.risk-gauge-value .risk-num {
    display: block;
    font-size: 1.55rem;
    font-weight: 700;
}
.risk-gauge-value .risk-unit {
    display: block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.6;
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.92rem;
    line-height: 1.4;
}
.risk-disease-name {
    margin-top: 0.4rem;
    font-size: 0.85rem;
    opacity: 0.7;
}
.risk-model-caption {
    margin-top: 0.65rem;
    font-size: 0.8rem;
    opacity: 0.55;
}
</style>
"""


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


@st.cache_data
def load_curves(disease_key: str):
    path = os.path.join(RESULTS_DIR, f"{disease_key}_curves.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
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


def render_result_card(risk_pct: float, pred: int, disease_display: str, best_model_name: str):
    is_high = pred == 1
    accent = "#dc2626" if is_high else "#16a34a"
    accent_soft = "rgba(220, 38, 38, 0.12)" if is_high else "rgba(22, 163, 74, 0.12)"
    label = "High risk indicator" if is_high else "Low risk indicator"
    icon = "⚠️" if is_high else "✅"
    pct = max(0.0, min(100.0, risk_pct))

    st.markdown(
        f"""
        <div class="risk-card">
            <div class="risk-gauge" style="background: conic-gradient({accent} {pct:.1f}%, rgba(128,128,128,0.18) {pct:.1f}% 100%);">
                <div class="risk-gauge-value">
                    <span class="risk-num">{pct:.1f}%</span>
                    <span class="risk-unit">risk</span>
                </div>
            </div>
            <div>
                <span class="risk-badge" style="background: {accent_soft}; color: {accent};">{icon} {label}</span>
                <div class="risk-disease-name">{disease_display}</div>
                <div class="risk-model-caption">Model used: {best_model_name} (best-performing model for this disease)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(curves: dict, best_model_name: str, disease_key: str):
    model_names = list(curves.keys())
    default_idx = model_names.index(best_model_name) if best_model_name in model_names else 0
    selected_model = st.selectbox(
        "Model",
        options=model_names,
        index=default_idx,
        key=f"{disease_key}_cm_model",
    )

    cm = np.array(curves[selected_model]["confusion_matrix"])
    axis_labels = ["No disease (0)", "Disease (1)"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=axis_labels,
        y=axis_labels,
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont={"size": 22},
        colorscale="Blues",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width='stretch', theme="streamlit")

    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    total = tn + fp + fn + tp
    st.caption(
        f"**{selected_model}** on the held-out test set ({total} patients): "
        f"{tp} true positives, {tn} true negatives, {fp} false positives, {fn} false negatives."
    )


def render_roc_curve(curves: dict, best_model_name: str, disease_key: str):
    fig = go.Figure()
    for name, data in curves.items():
        is_best = name == best_model_name
        fig.add_trace(go.Scatter(
            x=data["roc"]["fpr"],
            y=data["roc"]["tpr"],
            mode="lines",
            name=f"{name} (AUC={data['roc_auc']:.3f})",
            line=dict(width=3.5 if is_best else 1.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>" + name + "</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random classifier",
        line=dict(dash="dash", color="gray", width=1),
        hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=460,
        margin=dict(t=20, b=10, l=10, r=10),
        legend=dict(x=0.42, y=0.08),
    )
    st.plotly_chart(fig, width='stretch', theme="streamlit")
    st.caption(
        "Curves closer to the top-left corner indicate stronger separation between classes — "
        "AUC = 1.0 is a perfect classifier, AUC = 0.5 is random guessing."
    )


def render_performance_charts(comparison: "pd.DataFrame", curves: dict, best_model_name: str, disease_key: str):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    long_df = comparison.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")

    chart_options = ["Grouped bar", "Radar"]
    if curves:
        chart_options += ["Confusion Matrix", "ROC Curve"]

    chart_type = st.radio(
        "Chart type",
        options=chart_options,
        horizontal=True,
        key=f"{disease_key}_perf_chart_type",
    )

    if chart_type == "Confusion Matrix":
        render_confusion_matrix(curves, best_model_name, disease_key)
        return
    if chart_type == "ROC Curve":
        render_roc_curve(curves, best_model_name, disease_key)
        return

    if chart_type == "Grouped bar":
        fig = px.bar(
            long_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text=long_df["Score"].map(lambda v: f"{v:.3f}"),
            hover_data={"Score": ":.3f"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis=dict(range=[0, 1.08], title="Score"),
            xaxis_title=None,
            legend_title_text="Model",
            height=440,
            margin=dict(t=20, b=10, l=10, r=10),
            hovermode="closest",
        )
        for trace in fig.data:
            if trace.name == best_model_name:
                trace.marker.line = dict(width=2, color="black")
    else:
        fig = go.Figure()
        theta = metric_cols + [metric_cols[0]]
        for _, row in comparison.iterrows():
            values = [row[m] for m in metric_cols] + [row[metric_cols[0]]]
            is_best = row["Model"] == best_model_name
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                name=row["Model"],
                fill="toself",
                opacity=0.85 if is_best else 0.4,
                line=dict(width=3 if is_best else 1.5),
                hovertemplate="%{theta}: %{r:.3f}<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=460,
            margin=dict(t=30, b=10, l=10, r=10),
            legend_title_text="Model",
        )

    st.plotly_chart(fig, width='stretch', theme="streamlit")
    st.caption(f"**{best_model_name}** is highlighted — hover any point/bar for exact scores.")


def main():
    st.markdown(RESULT_CARD_CSS, unsafe_allow_html=True)
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
                render_result_card(risk_pct, pred, disease_display, best_model_name)

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
        curves = load_curves(disease_key)
        if comparison is not None:
            render_performance_charts(comparison, curves, best_model_name, disease_key)

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
