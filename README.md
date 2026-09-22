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
├── requirements.txt          # Full dev/training environment
├── requirements-app.txt      # Pinned runtime deps for the Docker image
├── Dockerfile / .dockerignore
├── data/                     # Raw public datasets (see Datasets below)
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.data
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis for all three datasets
├── src/
│   ├── preprocess.py         # Data loading, feature metadata, Parkinson's subject IDs
│   ├── features.py           # Leak-free preprocessing + diabetes feature engineering
│   ├── train.py               # Optuna tuning, cross-validation, model selection, artifacts
│   └── explain.py             # SHAP explainer selection + plain-language summaries
├── models/                   # Saved best model + preprocessor + metadata per disease
└── results/                  # Per-disease comparison tables (test + CV), tuning
                               # logs (*_tuning.json), confusion matrices/ROC data,
                               # and summary.json
```

## How it maps to the synopsis

| Synopsis stage | Implementation |
|---|---|
| Step 1 — Data Acquisition & Preprocessing | `src/preprocess.py` loads Pima Diabetes, Cleveland Heart Disease, UCI Parkinson's. `src/features.py` turns biologically-impossible zeros (diabetes) into missing values, median-imputes and scales them **inside the training pipeline** (fit on training folds only). Explored in `notebooks/eda.ipynb`. |
| Step 2 — Class Imbalance Correction | SMOTE is a step in an `imblearn` pipeline, so it only ever resamples the training part of each CV fold — validation folds and the test set stay real |
| Step 3 — Model Development & Comparison | Logistic Regression, Random Forest, SVM (RBF), XGBoost, each **tuned with Optuna** (50 trials, maximising 5-fold CV F1), compared on Accuracy, Precision, Recall, F1-score, ROC-AUC — both cross-validated (mean ± std) and on a held-out test set |
| Step 4 — Model Interpretability | `src/explain.py`: SHAP (`TreeExplainer` for RF/XGBoost, `LinearExplainer` for Logistic Regression, `KernelExplainer` for SVM), surfaced as plain-language "this factor is pushing risk up/down" statements |
| Step 5 — Deployment | `app.py`: Streamlit app — pick a disease, fill a form, get an instant risk prediction + explanation |

## Datasets

Standard, public, well-documented benchmark datasets, exactly as named in the
synopsis's literature review:

- **Diabetes**: Pima Indians Diabetes Dataset (768 records, 8 features)
- **Heart Disease**: Cleveland Heart Disease Dataset (UCI, 303 records, 13 features)
- **Parkinson's Disease**: UCI Parkinson's voice-biomarker dataset (195 records, 22 features — note: ~6 voice recordings each from **32 people**, not 195 independent patients; splits and CV folds are therefore grouped by subject, see below)

The Cleveland file contains one exact duplicate row, which is dropped (302 rows used).

`notebooks/eda.ipynb` walks through missing/zero-value checks, class balance,
feature distributions, and correlation structure for all three datasets
before any cleaning or training happens.

## Evaluation protocol

1. **Hold-out test set (20%)** — stratified; for Parkinson's, split **by
   subject** so no person has recordings in both train and test. It is used
   exactly once, at the end.
2. **Hyperparameter tuning** — Optuna (TPE sampler, seeded) searches each
   model's hyperparameters to maximise a selection metric averaged over
   5-fold stratified CV on the training split: **F1** for diabetes and heart,
   **balanced accuracy** for Parkinson's (see below). The previous default configuration is always the first
   trial, so tuning can never make a model worse on the tuning folds.
3. **Cross-validated metrics** — each tuned model is re-scored with 3 × 5-fold
   CV (fresh shuffles) → mean ± std for every metric. The untuned defaults are
   scored on the same folds, so "tuned vs untuned" is a like-for-like comparison.
4. **Model selection** — the deployed model per disease is the one with the
   **highest mean CV selection metric**, not the highest test-set score (choosing on the test
   set would quietly turn it into a second validation set).
5. **Diabetes feature engineering** — steps 2–4 run twice, with and without
   engineered features; the variant with the higher best CV F1 is kept. The
   comparison is saved in `results/diabetes_tuning.json`
   (`feature_engineering_experiment`).

Engineered diabetes features (`src/features.py → DiabetesFeatures`):
missing-value indicators for Insulin and SkinThickness (~49% / ~30% of rows
lack them), Glucose × BMI, a Glucose × Insulin / 405 insulin-resistance proxy
(HOMA-IR-style; Pima insulin is 2-hour, so it's a proxy), and clinical
threshold flags for Glucose ≥ 140 mg/dL (WHO impaired tolerance) and BMI ≥ 30.
Every engineered feature is computed per row with no fitted state, so nothing
leaks across folds. In the app, entering **0** for Insulin or Skinfold
thickness means "not measured", matching how the dataset encodes it.

## Results

> **Regenerate before presenting.** After `python -m src.train` finishes it
> prints a table per disease; copy those numbers here. Everything is also in
> `results/summary.json`, `results/*_comparison.csv` and
> `results/*_tuning.json`, and is shown live in the app's **Model
> Performance** tab (new **Cross-validation** chart: tuned vs untuned CV F1
> with ± std error bars).

Final run: 50 Optuna trials per model. Test = untouched 20% hold-out (61
patients for heart, 154 for diabetes, 43 recordings from 7 people for Parkinson's).

| Disease | Best model | Selected by | CV score (mean ± std) | CV Accuracy | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| Diabetes | Random Forest | F1 | 0.695 ± 0.040 | 77.4% | 74.7% | 67.8% | 82.1% |
| Heart Disease | Random Forest | F1 | 0.852 ± 0.033 | 83.8% | 82.0% | 84.9% | 91.5% |
| Parkinson's Disease | Logistic Regression | Balanced accuracy | 0.702 ± 0.056 | 62.0% | 95.3% | 96.9% | 99.7% |

**Reading the Parkinson's row:** the 95% test accuracy comes from only 7
people (43 recordings), so it is mostly luck of the split. The CV numbers,
averaged over 15 folds of unseen people, are the honest estimate: the model
reliably flags Parkinson's but often misreads healthy voices, because there
are only 6 healthy people to learn from in the training data. Tuning lifted CV
balanced accuracy from 0.580 (untuned) to 0.702. More healthy subjects would
be the single biggest improvement.

Tuning gains (CV F1, untuned → tuned, same folds): diabetes Random Forest
0.668 → 0.695; heart Random Forest 0.840 → 0.852, Logistic Regression
0.828 → 0.848. Diabetes feature engineering was **not** adopted: best CV F1
0.686 with the engineered features vs 0.695 without — within one standard
deviation, so the extra features add complexity without a measurable gain.
That's a legitimate negative result to report.

**What to expect compared with the earlier single-split numbers:**

- **Parkinson's scores will drop**, and that is the correct result. The
  earlier 92–95% came from a random split where the same person's voice
  recordings sat in both train and test, so the model could partly recognise
  the *speaker*. With subject-grouped splits the estimates are much lower
  and vary a lot between folds (only 8 healthy people in the whole dataset,
  ~1–2 per fold).
- **Why Parkinson's is selected by balanced accuracy, not F1:** 75% of the
  subjects have Parkinson's, so a model that says "Parkinson's" for everyone
  already scores F1 ≈ 0.86. A first run selected by F1 picked an SVM with CV
  F1 0.885, and on the test set it correctly identified only 1 of 12 healthy
  recordings. Balanced accuracy (the average of sensitivity and specificity)
  scores an always-positive model at 0.50, so it rewards models that also
  recognise healthy voices.
- **Heart Disease** should stay roughly where it was (low-to-mid 80s), with
  a ± std that shows how much a single 61-patient test split can swing.
- **Diabetes** — tuning helps a little (Random Forest CV F1 rose ~0.67 →
  ~0.70 in quick runs). Feature engineering may or may not be adopted: CV
  decides. Expect accuracy to stay in the mid-70s. Many online notebooks
  report 85–90%+ on Pima, but they usually impute with class-wise medians
  (which uses the label) or oversample before splitting — both leak the
  answer. Being able to explain that is a strong viva point.

**On the synopsis's ≥85% accuracy target:** it's a goal, not a guarantee.
Report the cross-validated numbers as they come out and explain the gap
(small, noisy Pima data; strict leak-free evaluation). Reviewers generally
trust a team that does this more than one that reports "85%+" without evidence.

## Running it yourself

```bash
pip install -r requirements.txt

# Tune + cross-validate + train all three diseases — writes to models/ and results/
python -m src.train                      # 50 Optuna trials per model (~10–20 min)
python -m src.train --trials 10          # quick run
python -m src.train --disease diabetes   # one disease only

# Launch the web app
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`),
pick a disease in the sidebar, fill in the clinical parameters, and click
**Predict Risk**.

### Docker

```bash
# Train first (on the same library versions pinned in requirements-app.txt),
# because the image copies the models/ and results/ folders as they are.
docker build -t mdps .
docker run --rm -p 8501:8501 mdps
# open http://localhost:8501
```

To explore the raw datasets interactively:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Design notes / decisions worth mentioning in your viva

- **Why F1-score to pick the "best" model** (diabetes, heart), not raw accuracy: these are
  imbalanced clinical datasets (e.g. only 34.9% positive in the diabetes set).
  A model can look "accurate" by mostly predicting the majority class while
  missing real positive cases. F1-score balances precision and recall, which
  matters far more for a screening tool where missing a true positive
  (false negative) has a real cost. The exception is Parkinson's, where the
  disease is the *majority* class. There, F1 rewards predicting "disease" for
  everyone, so balanced accuracy is used instead.
- **Why SMOTE, imputation and scaling live inside the pipeline**: applying
  any of them to the full dataset before splitting leaks test-set
  information into training (synthetic neighbours of test points, or medians
  and means computed from them), which inflates the scores. Inside an
  `imblearn` pipeline they are re-fit on the training part of every CV fold,
  so each validation fold is scored the way new patients would be.
- **Why cross-validation, not one split**: with 61 test patients (heart) or
  ~7 test *people* (Parkinson's), one lucky or unlucky split can move
  accuracy by 5–10 points. Mean ± std over 15 folds shows both the typical
  performance and how stable it is.
- **Why Parkinson's is split by subject**: each person has ~6 recordings.
  Grouping keeps all of a person's recordings on the same side of every split,
  so the model is tested on voices it has never heard, as it would be in real use.
- **Why the SVM is wrapped in `CalibratedClassifierCV`**: `SVC(probability=True)`
  fits its own internal Platt scaling, whose probabilities can disagree with
  (or even invert against) its own `predict()`. Calibrating explicitly keeps
  the predicted class, the risk %, and the ROC curve consistent.
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

## Suggested next steps

- Done: hyperparameter tuning (Optuna), cross-validation (with subject
  grouping for Parkinson's), a diabetes feature-engineering experiment, and a Dockerfile.
- Nested cross-validation, if you want a CV estimate that is also free of
  tuning bias (the held-out test set already covers this).
- Decision-threshold tuning per disease: a screening tool might prefer a
  lower threshold (higher recall), chosen on CV folds.
- A stacking or soft-voting ensemble of the four tuned models.
