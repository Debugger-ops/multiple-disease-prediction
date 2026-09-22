"""
Tune, cross-validate, compare, and select the best model for each disease.

Pipeline per disease:
    1. Load raw data                                   (src/preprocess.py)
    2. Hold out a stratified 20% test set -- never touched until the very end.
       Parkinson's is split by *subject* (stratified at subject level) because each
       person contributes ~6 voice recordings.
    3. Build a leak-free imblearn pipeline, re-fit inside every CV fold:
           [zero->NaN, feature engineering]  (diabetes only)
           -> median imputation -> StandardScaler -> SMOTE -> model
       SMOTE lives inside the pipeline, so it only ever touches the training
       part of each fold -- validation folds and the test set stay real.
    4. Hyperparameter tuning with Optuna (TPE sampler), maximising the mean
       5-fold CV selection metric on the training split -- F1 for diabetes
       and heart, balanced accuracy for Parkinson's (see SELECTION_METRIC) -- for
       Logistic Regression, Random Forest, SVM (RBF) and XGBoost.
    5. Re-evaluate each tuned model with repeated CV (3 x 5-fold, fresh
       shuffles) -> mean ± std for Accuracy / Precision / Recall / F1 / ROC-AUC.
    6. Pick the best model by the *mean CV selection metric* (not by test-set score, which
       would turn the test set into a second validation set).
    7. Refit the tuned pipelines on the full training split and report
       metrics, confusion matrices and ROC curves on the untouched test set.
    8. Diabetes only: run steps 3-6 with and without engineered features and
       keep whichever variant has the higher best-model CV F1.

Run:
    python -m src.train                  # default: 50 Optuna trials per model
    python -m src.train --trials 100     # longer search
    python -m src.train --disease diabetes
"""

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.features import build_preprocessor, build_preprocessor_steps, derived_feature_info, model_feature_names
from src.preprocess import DISEASE_DISPLAY_NAMES, GROUP_LOADERS, LOADERS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
EVAL_REPEATS = 3          # repeated CV for the reported mean ± std
MODEL_NAMES = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
SCORING = {
    "accuracy": "accuracy", "precision": "precision", "recall": "recall",
    "f1": "f1", "roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy",
}

# Metric used to tune and select models, per disease.
# F1 (on the disease class) suits diabetes (35% positive) and heart (54%).
# In Parkinson's, 75% of subjects HAVE the disease, so a model that predicts
# "Parkinson's" for everyone already scores F1 ~0.86 -- F1 would reward that.
# Balanced accuracy (mean of sensitivity and specificity) scores such a
# model at 0.50, so it forces the model to also recognise healthy people.
SELECTION_METRIC = {"parkinsons": ("balanced_accuracy", "Balanced Accuracy")}
DEFAULT_SELECTION = ("f1", "F1")


def selection_metric(disease_key):
    return SELECTION_METRIC.get(disease_key, DEFAULT_SELECTION)


# --------------------------------------------------------------------------- #
# Models: defaults (the old untuned baseline) and Optuna search spaces
# --------------------------------------------------------------------------- #
def default_params(name, n_features):
    """The previous untuned configuration, expressed in search-space terms.

    It is enqueued as Optuna's first trial, so tuning can never end up worse
    than the untuned baseline on the tuning folds.
    """
    return {
        "Logistic Regression": {"C": 1.0, "penalty": "l2"},
        "Random Forest": {"n_estimators": 300, "max_depth": None, "min_samples_split": 2,
                          "min_samples_leaf": 1, "max_features": "sqrt"},
        # gamma="scale" == 1 / (n_features * var(X)) == 1 / n_features after StandardScaler
        "SVM": {"C": 1.0, "gamma": 1.0 / n_features},
        "XGBoost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0,
                    "colsample_bytree": 1.0, "min_child_weight": 1, "gamma": 0.0, "reg_lambda": 1.0},
    }[name]


def suggest_params(trial, name):
    if name == "Logistic Regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    if name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 16]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(name)


def make_model(name, params):
    if name == "Logistic Regression":
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=RANDOM_STATE, **params)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    if name == "SVM":
        # SVC(probability=True) uses an internal Platt fit whose probabilities
        # can disagree with (or even invert) its own predict(). Wrapping in
        # CalibratedClassifierCV makes predict == argmax(predict_proba), and is
        # used identically in CV and in the saved model.
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE, **params),
                                      method="sigmoid", cv=3)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1,
                             random_state=RANDOM_STATE, **params)
    raise ValueError(name)


def make_pipeline(disease_key, name, params, fe, k_neighbors):
    # imblearn forbids nested Pipelines, so preprocessing steps are inlined.
    return ImbPipeline(build_preprocessor_steps(disease_key, fe) + [
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ("model", make_model(name, params)),
    ])


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _subject_labels(y, groups):
    """One label per subject (Parkinson's status is constant per person)."""
    lab = pd.Series(np.asarray(y)).groupby(np.asarray(groups)).first()
    return lab.index.to_numpy(), lab.to_numpy()


def holdout_split(X, y, groups):
    if groups is None:
        return train_test_split(np.arange(len(y)), test_size=0.2,
                                random_state=RANDOM_STATE, stratify=y)
    # Stratify at the *subject* level so both classes appear in the test set.
    subjects, labels = _subject_labels(y, groups)
    tr_s, te_s = train_test_split(subjects, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
    g = np.asarray(groups)
    return np.where(np.isin(g, tr_s))[0], np.where(np.isin(g, te_s))[0]


def cv_splits(X, y, groups, seed):
    """Materialised fold indices so every model/trial sees identical folds."""
    if groups is None:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        return list(cv.split(X, y))
    # Subject-level stratified K-fold: every recording of a person lands in the
    # same fold, and healthy subjects are spread evenly across folds.
    subjects, labels = _subject_labels(y, groups)
    g = np.asarray(groups)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return [(np.where(np.isin(g, subjects[a]))[0], np.where(np.isin(g, subjects[b]))[0])
            for a, b in cv.split(subjects, labels)]


def safe_k_neighbors(y_train):
    # Smallest minority count any CV training fold can have, minus one.
    minority = int(y_train.value_counts().min() * (CV_FOLDS - 1) / CV_FOLDS)
    return max(1, min(5, minority - 1))


# --------------------------------------------------------------------------- #
# Tuning + evaluation
# --------------------------------------------------------------------------- #
def cv_scores(pipe, X, y, folds, scoring=SCORING):
    res = cross_validate(pipe, X, y, cv=folds, scoring=scoring, n_jobs=-1, error_score="raise")
    return {k.replace("test_", ""): v for k, v in res.items() if k.startswith("test_")}


def tune(disease_key, name, X, y, folds, fe, k, n_trials, baseline_params):
    metric, _ = selection_metric(disease_key)

    def objective(trial):
        params = suggest_params(trial, name)
        pipe = make_pipeline(disease_key, name, params, fe, k)
        return cv_scores(pipe, X, y, folds, scoring={metric: metric})[metric].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.enqueue_trial(baseline_params)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def repeated_cv(disease_key, name, params, X, y, groups, fe, k):
    """3 x 5-fold with shuffles different from the tuning folds -> mean ± std."""
    collected = {m: [] for m in SCORING}
    for r in range(EVAL_REPEATS):
        folds = cv_splits(X, y, groups, seed=1000 + r)
        s = cv_scores(make_pipeline(disease_key, name, params, fe, k), X, y, folds)
        for m in SCORING:
            collected[m].extend(s[m].tolist())
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in collected.items()}


def run_variant(disease_key, X_tr, y_tr, g_tr, fe, n_trials):
    """Tune + CV-evaluate all four models for one preprocessing variant."""
    metric, label = selection_metric(disease_key)
    k = safe_k_neighbors(y_tr)
    folds = cv_splits(X_tr, y_tr, g_tr, seed=RANDOM_STATE)
    out = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        n_feat = build_preprocessor(disease_key, fe).fit(X_tr).transform(X_tr.head(1)).shape[1]
        base_params = default_params(name, n_feat)
        best_params, tuned_f1 = tune(disease_key, name, X_tr, y_tr, folds, fe, k, n_trials, base_params)
        # Untuned vs tuned, both measured on the same repeated-CV folds.
        baseline = repeated_cv(disease_key, name, base_params, X_tr, y_tr, g_tr, fe, k)[metric][0]
        cv = repeated_cv(disease_key, name, best_params, X_tr, y_tr, g_tr, fe, k)
        out[name] = {
            "params": best_params,
            "baseline_cv_score": float(baseline),
            "tuning_cv_score": float(tuned_f1),
            "cv": cv,
        }
        print(f"  {name:<20} default CV {label}={baseline:.3f}  tuned CV {label}={cv[metric][0]:.3f} "
              f"± {cv[metric][1]:.3f}  ({time.time() - t0:.0f}s)")
    return out, k


def best_by_cv(variant, metric):
    return max(variant, key=lambda n: variant[n]["cv"][metric][0])


# --------------------------------------------------------------------------- #
# Main per-disease routine
# --------------------------------------------------------------------------- #
def train_one_disease(disease_key, n_trials):
    print(f"\n=== {DISEASE_DISPLAY_NAMES[disease_key]} ===")
    X, y, feature_info = LOADERS[disease_key]()
    groups = GROUP_LOADERS[disease_key]() if disease_key in GROUP_LOADERS else None
    feature_names = list(X.columns)

    train_idx, test_idx = holdout_split(X, y, groups)
    X_tr, X_te = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    g_tr = groups.iloc[train_idx].reset_index(drop=True) if groups is not None else None
    if groups is not None:
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]), "subject leak"
        print(f"  Group split: {g_tr.nunique()} train subjects / "
              f"{groups.iloc[test_idx].nunique()} test subjects")

    variants_to_try = [False, True] if disease_key == "diabetes" else [False]
    variant_results = {}
    for fe in variants_to_try:
        if len(variants_to_try) > 1:
            print(f"  -- feature engineering {'ON' if fe else 'OFF'} --")
        variant_results[fe] = run_variant(disease_key, X_tr, y_tr, g_tr, fe, n_trials)

    metric, label = selection_metric(disease_key)
    use_fe = max(variant_results, key=lambda fe: variant_results[fe][0][best_by_cv(variant_results[fe][0], metric)]["cv"][metric][0])
    results, k = variant_results[use_fe]
    best_name = best_by_cv(results, metric)

    fe_experiment = None
    if len(variants_to_try) > 1:
        fe_experiment = {
            ("with_features" if fe else "raw_features"): {
                name: {"cv_f1_mean": r["cv"]["f1"][0], "cv_f1_std": r["cv"]["f1"][1],
                       "cv_accuracy_mean": r["cv"]["accuracy"][0]}
                for name, r in variant_results[fe][0].items()
            }
            for fe in variants_to_try
        }
        fe_experiment["feature_engineering_adopted"] = bool(use_fe)
        print(f"  Feature engineering adopted: {use_fe}")

    # Refit every tuned pipeline on the full training split; score on test.
    rows, curves, fitted = [], {}, {}
    for name in MODEL_NAMES:
        r = results[name]
        pipe = make_pipeline(disease_key, name, r["params"], use_fe, k)
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
        cv = r["cv"]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, zero_division=0),
            "Recall": recall_score(y_te, y_pred, zero_division=0),
            "F1-score": f1_score(y_te, y_pred, zero_division=0),
            "ROC-AUC": auc,
            "CV F1 (mean)": cv["f1"][0],
            "CV F1 (std)": cv["f1"][1],
            "CV Balanced Accuracy (mean)": cv["balanced_accuracy"][0],
            "CV Balanced Accuracy (std)": cv["balanced_accuracy"][1],
            "CV Accuracy (mean)": cv["accuracy"][0],
            "CV Accuracy (std)": cv["accuracy"][1],
            "CV ROC-AUC (mean)": cv["roc_auc"][0],
            "CV ROC-AUC (std)": cv["roc_auc"][1],
            f"Untuned CV {label}": r["baseline_cv_score"],
        })
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        curves[name] = {
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(auc),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        }

    comparison = (pd.DataFrame(rows).sort_values(f"CV {label} (mean)", ascending=False)
                  .reset_index(drop=True))
    best_pipe = fitted[best_name]
    # Everything before SMOTE = the fitted preprocessor used at inference.
    preprocessor = SkPipeline(best_pipe.steps[:-2])
    model = best_pipe.named_steps["model"]

    # SHAP background: scaled + SMOTE-resampled training data (what the model saw).
    X_tr_scaled = preprocessor.transform(X_tr)
    X_res, _ = SMOTE(random_state=RANDOM_STATE, k_neighbors=k).fit_resample(X_tr_scaled, y_tr)
    bg = X_res[np.random.RandomState(RANDOM_STATE).choice(len(X_res), size=min(100, len(X_res)), replace=False)]

    cv_scheme = (f"{CV_FOLDS}-fold stratified K-fold grouped by subject" if groups is not None
                 else f"{CV_FOLDS}-fold StratifiedKFold") + f", repeated {EVAL_REPEATS}x for reporting"

    joblib.dump(model, os.path.join(MODELS_DIR, f"{disease_key}_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"{disease_key}_preprocessor.pkl"))
    joblib.dump(
        {
            "feature_names": feature_names,                       # raw form inputs
            "model_feature_names": model_feature_names(preprocessor, X_tr),
            "feature_info": feature_info,
            "derived_feature_info": derived_feature_info(disease_key, use_fe),
            "feature_engineering": bool(use_fe),
            "best_model_name": best_name,
            "best_params": results[best_name]["params"],
            "cv_scheme": cv_scheme,
            "selection_metric": metric,
            "selection_label": label,
            "X_train_res_sample": bg,
        },
        os.path.join(MODELS_DIR, f"{disease_key}_meta.pkl"),
    )
    comparison.to_csv(os.path.join(RESULTS_DIR, f"{disease_key}_comparison.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{disease_key}_curves.json"), "w") as f:
        json.dump(curves, f, indent=2)
    tuning = {
        "cv_scheme": cv_scheme,
        "n_trials_per_model": n_trials,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "feature_engineering": bool(use_fe),
        "selection_metric": metric,
        "models": {n: {"best_params": results[n]["params"],
                       f"untuned_cv_{metric}": results[n]["baseline_cv_score"],
                       "tuned_cv": {m: {"mean": v[0], "std": v[1]} for m, v in results[n]["cv"].items()}}
                   for n in MODEL_NAMES},
    }
    if fe_experiment:
        tuning["feature_engineering_experiment"] = fe_experiment
    with open(os.path.join(RESULTS_DIR, f"{disease_key}_tuning.json"), "w") as f:
        json.dump(tuning, f, indent=2, default=str)

    show_cols = ["Model", f"CV {label} (mean)", f"CV {label} (std)"]
    if label != "F1":
        show_cols += ["CV F1 (mean)"]
    show = comparison[show_cols + ["Accuracy", "F1-score", "ROC-AUC"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Best model (by CV {label}): {best_name}  params={results[best_name]['params']}")
    return comparison, best_name, use_fe


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model (default 50)")
    parser.add_argument("--disease", choices=list(LOADERS), help="Train only one disease")
    args = parser.parse_args()

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    summary = {}
    if args.disease and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    for disease_key in ([args.disease] if args.disease else LOADERS):
        comparison, best_name, use_fe = train_one_disease(disease_key, args.trials)
        summary[disease_key] = {
            "best_model": best_name,
            "selection": f"highest mean cross-validated {selection_metric(disease_key)[1]} on the training split",
            "feature_engineering": bool(use_fe),
            "metrics": comparison.iloc[0].to_dict(),
            "all_models": comparison.to_dict(orient="records"),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nDone. Artifacts saved to models/ and results/.")


if __name__ == "__main__":
    main()
