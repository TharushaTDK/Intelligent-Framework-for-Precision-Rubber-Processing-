# RubberSense — Component 1 (Estate-wise DRC, Tapping Advisory, Price Estimation)

A numbered, staged ML research workflow for Component 1, organised in the style of the supplied
Component 3 project. It **adapts the organisational pattern only**; no Component 3 domain logic or
data is copied. See `STRUCTURE_MAPPING.md` and
`reference_structure/component_3_smokehouse_structure.md`.

> **Dataset (October 2026):** `data/raw/Combined_Rubber_Latex_Dataset_Condition_Matched.xlsx`.
> Models train on its **610 farmer records** (Estate 01-05, June-September 2026) with Open-Meteo
> hourly weather. The other 600 rows are the earlier synthetic set and have no weather.
> - DRC/Metrolac values were normalized by the dataset author to Kudaligama chart entries, and the
>   visible latex condition was aligned with DRC; they are not original field measurements.
> - Latex price is filled by stage 00 as **farm-gate latex price x DRC / 100** (field latex is paid
>   on dry rubber content, so watery latex is worth less). Monthly farm-gate levels and their sources
>   are in `data/reference/latex_farmgate_price_reference.csv`; June-September 2026 are estimates.
> - Tapping labels come from a rainfall rule (no added noise) and are a proxy, not expert advice.
> The previous synthetic workbook and the duplicate stage scripts were removed after this retrain
> (a backup of the earlier version is kept outside the repository).

## Included tasks
1. DRC regression (`latex_drc_pct`) from weather, farmer soil moisture, soil pH, visible latex
   condition, estate and date. The Metrolac reading and latex temperature are excluded because
   they give DRC exactly through the Kudaligama chart.
2. Tapping suitability classification (`tapping_suitability_proxy`, rainfall-rule proxy labels).
3. Field latex price regression (`price_lkr_per_kg`, price per kg of field latex).

Collection-time prediction is **removed** — no collection-time target exists and no such model,
comparison, report or artefact is produced anywhere.

## Protocol (leakage-safe)
- One deterministic 80/20 split per task is stored under `data/splits/` (seed 340).
- Preprocessing (imputation / scaling / one-hot) lives inside sklearn `Pipeline`s and is fitted on
  the **training partition only**.
- Candidate selection uses **k-fold cross-validation on the training partition** (regression: RMSE;
  classification: stratified macro-F1). The reference approach of using a validation/CV metric — not
  the test set — is followed.
- The held-out **test partition is loaded once, in stage 31**.
- `30_estate_group_cross_validation.py` adds a group-aware (`estate_id`) estimate because the stored
  split is row-wise.

## Numbered workflow
- `00_prepare_realistic_dataset.py`: reads the workbook (unchanged), maps column names, marks the
  rows usable for weather models, fills the latex price, and writes
  `data/validated/component1_realistic_prepared.csv` plus `00_dataset_preparation_report.md`.
- `01_data_understanding.py`: raw data schema and overview.
- `02_target_validation.py`, `02_data_quality_analysis.py`: target and quality checks.
- `03_preprocessing.py`: preprocessing report and cleaned copy.
- `04_build_targets_and_splits.py`: builds the stored per-task splits.
- `04b_estimate_drc_for_price.py`: leakage-safe estimated DRC that the price model takes as input
  (out-of-fold on train, train-fitted on test), because in the app price is always worked out from
  an estimated DRC, never a lab value.
- `05_eda.py`: numeric summaries, cardinality, distributions (+ figures).
- `06_temporal_estate_analysis.py`: monthly and estate-level summaries.
- `07_feature_engineering.py`: derived time/weather features.
- `08_feature_analysis_selection.py`: correlations and feature diagnostics.
- `09_split_integrity_audit.py`: split file audit.
- `10_train_only_preprocessing.py`: leakage-safe preprocessing protocol.
- `src/regression/11–16`: fit the Dummy Mean, Random Forest, Gradient Boosting, Extra Trees, XGBoost
  and HistGradientBoosting candidates (train split only). `11_compare_regression_models.py` runs the
  CV comparison of all of them, and `19` selects the winner by CV and writes `final/`.
- `src/classification/20–26`: fit the tapping baseline, Logistic Regression, Random Forest, Extra
  Trees, Gradient Boosting and XGBoost candidates. `21_class_imbalance_analysis.py` reports class
  distribution, `21_compare_classification_models.py` runs the CV comparison, `29` selects by CV.
- Tuning for both kinds of task is done by `40_tune_candidate_models.py`.
- `30_cv_data_integrity_audit.py`: split integrity, index disjointness, estate-overlap, leakage notes.
- `30_estate_group_cross_validation.py`: grouped (unseen-estate) CV.
- `30_summarize_model_comparison.py`: CV-based consolidated summary.
- `31_final_test_evaluation.py`: one-time held-out test evaluation + required figures.
- `32_model_explainability.py`: permutation + impurity importance (SHAP not used).
- `33_export_final_models.py`: deployment pipelines, artefact manifest, summary/methodology docs.
- `40_tune_candidate_models.py`: greedy CV tuning of the shortlisted families.

## Artefact layout
```
artifacts/
├── models/
│   ├── regression/<task>/{baseline,candidates,tuned,final}/   # task ∈ drc_prediction, price_estimation
│   └── classification/tapping_advisory/{baseline,candidates,tuned,final}/
├── pipelines/        # component1_<task>_pipeline.pkl (+ metadata)
└── preprocessing/    # component1_preprocessor_metadata.json
```
Every trained model is a `.joblib` plus a sibling `*_metadata.json` recording scope flags
(`fit_scope`, `test_loaded`, `hyperparameter_tuning_performed`), explicit hyperparameters, CV
metrics and SHA-256 hashes. The `final/` metadata also records the selection rule, runner-up and the
source→final copy hash comparison. The exported pipelines are the **train-fitted, CV-selected**
models — not full-data refits.

## Outputs
```
outputs/
├── reports/   # numbered CSV/JSON/TXT + component1_final_ml_model_summary.md,
│              # methodology_and_limitations.md, data_provenance_and_limitations.md,
│              # component1_final_ml_artifact_manifest.csv
└── figures/   # 05_eda/, 31_regression_final_test/, 31_classification_final_test/, 32_explainability/
```

## Run
On Windows the project path is too long for a virtual environment inside it, so create one at a
short path, using the same library versions as the backend so the exported models load there:

```bash
python -m venv C:\Users\<you>\.venvs\rubbersense-c1
C:\Users\<you>\.venvs\rubbersense-c1\Scripts\python.exe -m pip install pandas==3.0.6 numpy==2.5.3 scikit-learn==1.9.1 xgboost==3.4.1 joblib==1.6.0 "openpyxl>=3.1" "matplotlib>=3.7"
```

```bash
pip install -r requirements.txt

# data stages
python src/00_prepare_realistic_dataset.py
python src/01_data_understanding.py
python src/02_target_validation.py
python src/02_data_quality_analysis.py
python src/03_preprocessing.py
python src/04_build_targets_and_splits.py
python src/04b_estimate_drc_for_price.py
python src/05_eda.py
python src/06_temporal_estate_analysis.py
python src/07_feature_engineering.py
python src/08_feature_analysis_selection.py
python src/09_split_integrity_audit.py
python src/10_train_only_preprocessing.py

# candidate comparison (fits every family incl. xgboost when installed)
python src/regression/11_compare_regression_models.py
python src/classification/21_compare_classification_models.py
# tune the shortlisted families
python src/40_tune_candidate_models.py
# final selection by cross-validation (baseline + candidates + tuned)
python src/regression/19_final_selection.py
python src/classification/29_final_selection.py

# audits + finalisation
python src/30_cv_data_integrity_audit.py
python src/30_estate_group_cross_validation.py
python src/30_summarize_model_comparison.py
python src/31_final_test_evaluation.py
python src/32_model_explainability.py
python src/33_export_final_models.py
```

The individual `11–16` / `20,22–26` family scripts are thin wrappers over `src/train_candidate.py`
and can be run individually to inspect artefact organisation.

### Training-only mode
Setting `RUBBERSENSE_TRAIN_ONLY_DIR` redirects every generated processed copy, report, figure and
model artefact under that directory, leaving the project's `outputs/` and `artifacts/` untouched:

```bash
RUBBERSENSE_TRAIN_ONLY_DIR=/tmp/component1 python src/train_only_checkpoint.py
```

## Evaluation and limitations
Regression metrics include MAE, RMSE, R² and a declared within-tolerance percentage (DRC ±1
percentage point; price ±LKR 5/kg of field latex) — **within-tolerance % is not accuracy**. Classification reports
accuracy, macro precision, macro recall and macro-F1. For valid conclusions, collect measured DRC by
estate, expert-verified tapping labels, and dated verified price records. For claims about
unseen **estates**, use estate-grouped splits/CV and keep a final holdout untouched.
