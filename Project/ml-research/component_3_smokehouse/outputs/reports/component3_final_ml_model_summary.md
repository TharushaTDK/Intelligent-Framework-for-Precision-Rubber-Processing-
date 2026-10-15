# Component 3 — Final ML Model Summary

**This document supplements, and does not replace, `component3_final_methodology_and_limitations.md`.**
It records the end state of the ML development lifecycle after classifier finalization.

## Final Models

**Prediction 1 — Remaining_Drying_Time:**
**Gradient Boosting GB_02** (unchanged throughout this task). Stage-30 development GroupKFold:
MAE 3.1283h, RMSE 3.9558h, R² 0.9860. Historical Stage-31 locked test: MAE 3.756103h, RMSE
4.299451h, R² 0.9835396. Artifact: `backend/app/ml/component_3_smokehouse/drying_time_pipeline.pkl`
(SHA-256 `5bc446c3802d3d1d8f4eb997253eee1095410c4ed43edac743990dc8f928fdd6`, unchanged).

**Prediction 2 — Fire_Risk_Label:**
**Ordinal XGBoost Classifier with Argmax Decision Rule** (newly promoted this task). Development
GroupKFold: Accuracy 0.9661, Balanced Accuracy 0.7405, Macro F1 0.6748, Warning Recall 0.8117,
Critical Recall 0.4348, Critical Precision 0.4027, Critical F1 0.4181, Critical F2 0.4280.
Artifact: `backend/app/ml/component_3_smokehouse/fire_risk_pipeline.pkl` (SHA-256
`61b4304b082672e9b7e0f91620e9854cac1abaabefd345a190e375110983083b`).

## Why Ordinal XGBoost Was Selected Over XGB_03

**Improved:** Critical Recall (0.3551→0.4348), Critical F2 (0.3684→0.4280), Macro F1
(0.6453→0.6748), Balanced Accuracy (0.7250→0.7405), overall Accuracy (0.9558→0.9661), while
Warning Recall (0.8117) remained above the 0.80 guardrail.

**Cost:** slightly lower Critical Precision (0.4336→0.4027), slightly lower Warning Recall
(0.8565→0.8117), and +25 false Critical alerts (64→89).

**Unresolved:** two of six Critical-containing development batches (`SH02_B01`, `SH02_B02`)
still produce zero Critical Recall — unchanged from XGB_03. This was true under *every* method
tested across two complete experimentation rounds (cost-sensitive weighting, threshold tuning,
two-stage classification, dedicated Critical detector, ordinal classification), strongly
indicating a structural data-scarcity limitation rather than a model-architecture deficiency.

**Dataset is synthetic/simulated.** Independent field validation has not been completed. This
promotion is based entirely on leakage-safe, development-only, group-aware (Batch_ID) GroupKFold
evidence — never on the historical Stage-31 Test holdout, which was observed once already during
the original XGB_03 research process and cannot be presented as an untouched test for a newly
promoted model.

## Final ML Lineage

```
Frozen Raw (24,463 rows, SHA 229139d2b4...)
  |
  v
Validated (24,463 rows, SHA 229139d2b4... - identical, pre-imputation snapshot)
  |
  v
Preprocessed (24,463 rows, SHA 957c9c150b...)
  |
  v
Feature Engineered (24,463 rows, SHA 4990acb69d...)
  |
  v
Existing Batch-Aware Split (Stage 09, unchanged)
  Train 16,246  |  Validation 4,159  |  Test 4,058 (locked, never used for Ordinal selection)
  |
  v
Development = Train + Validation = 20,405 rows, 10 batches
  |
  +--------------------------------+--------------------------------------------+
  |                                                                              |
  v                                                                              v
REGRESSION BRANCH (frozen)                                    CLASSIFICATION BRANCH
  Stage 11-19: model comparison -> GB_02 selected                Stage 20-29: XGB_03 baseline selected
  Stage 30: GroupKFold validation                                Stage 30: GroupKFold validation (historical)
  Stage 31: locked final test (historical, preserved)            Stage 31: locked final test (historical, preserved)
  Stage 33: export -> drying_time_pipeline.pkl                    |
  |                                                                v
  v                                                    Controlled improvement research (2 rounds):
Final regression pipeline (UNCHANGED)                    cost-sensitive / threshold / two-stage /
                                                           Critical-vs-Rest detector / ordinal
                                                                  |
                                                                  v
                                                     Ordinal XGBoost Classifier (Argmax)
                                                                  |
                                                                  v
                                                     Promotion-readiness gate (11/11 PASS)
                                                                  |
                                                                  v
                                                     Final Development refit (20,405 rows)
                                                                  |
                                                                  v
                                                     Export -> fire_risk_pipeline.pkl (ACTIVE)
                                                     XGB_03 archived with verified matching hash
```

**Test never appears as a model-selection input for the Ordinal classifier** — it is checked
for SHA-256 integrity only (`2ff12dc0ff30b673b6ad26cafc03cda8d9d665425c2eb2381f793991fbed8c80`,
unchanged) and never loaded into memory by any `classification_v2` script.

## Verification Performed

- Dataset/split/regression hashes re-verified identical before and after this task (14 artifacts,
  all matched; see `component3_final_ml_artifact_manifest.csv`).
- Baseline (XGB_03) and Ordinal GroupKFold evidence both independently re-reproduced this session
  (exact match to prior reference values, within floating-point/reporting tolerance).
- Promotion-readiness gate: 11/11 PASS, including a fresh-process reload test (0 label
  mismatches across all 20,405 Development rows, max probability difference 2.98e-08).
- Probability integrity: all reconstructed probabilities non-negative, rows sum to 1 within
  1.2e-07, argmax matches stored predictions exactly, 580/20,405 (2.84%) raw monotonicity
  violations corrected deterministically and documented (not hidden).
- Explainability computed separately for O1 and O2 (SHAP, TreeExplainer) — consistent with the
  historical Stage-32 physical story (temperature drives Normal/Warning; humidity variability,
  drying stage, and sheet load drive Critical).

## Remaining Limitations

- Synthetic/simulated dataset — not field-validated.
- Only 6 of 10 development batches (and only 1 of 2 historical Test batches) contain any
  Critical example; 2 of 6 remain undetectable under every method tried.
- No further modeling experimentation on this same frozen dataset is expected to meaningfully
  change this picture (five distinct architectures tested across two rounds, consistent ceiling).
- This classifier is a **research/prototype decision-support tool**, not a certified fire alarm,
  guaranteed fire detector, or production safety device.

## Status

**ML development lifecycle on the current frozen dataset is CLOSED.** Regression finalized
(unchanged). Classification finalized (Ordinal XGBoost promoted). No further model experiments,
dataset changes, or threshold/architecture tuning should be performed on this dataset without a
new, separately-scoped research round. FastAPI/backend integration has not been started and
awaits user + external reviewer approval.
