# Component 3 — Critical-Class Recall Improvement Experiment (V2, Classification Only)

**Status: EXPERIMENTAL EVIDENCE ONLY. XGB_03 has NOT been replaced. Nothing described in
this document has been deployed, exported, or used to rerun Stage 31/32/33.**

Scope: classification-only (`Fire_Risk_Label`). The regression side (GradientBoostingRegressor
GB_02, `drying_time_pipeline.pkl`) was never touched — not loaded, not refit, not re-evaluated —
anywhere in this experiment. All work used **Development data only** (Train + Validation,
20,405 rows, 10 batches); `data/splits/Component3_Test.csv` was never read.

---

## Phase 0 — Baseline Safety Snapshot

`git status` at the start of this task showed the same 34 pre-existing modified files and
untracked directories already present from earlier sessions (not attributable to this task).
Protected-artifact hashes were recorded before any V2 code was written and re-verified identical
after all experiments (see **Protected Artifact Check** below) — all 14 matched before/after.

---

## Phase 1 — Read-Only Classification Audit (confirmed facts used to design this experiment)

Read in full: `src/10_train_only_preprocessing.py`, `src/classification/20_classification_baseline.py`,
`21_class_imbalance_analysis.py`, `26_xgboost.py`, `28_hyperparameter_tuning.py`,
`29_final_selection.py`, `src/30_group_cross_validation.py` (full), plus the corresponding
`outputs/reports/*` artifacts and `21_class_imbalance_strategy.json` / `21_class_weights.csv`.

1. **Class-weighting method already used:** `sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=...)`, computed independently inside every training fold from that fold's own labels (Stage 30, confirmed by direct code read). Balanced weight values from Stage 21: Normal ≈0.346, Warning ≈11.28, Critical ≈45.13 (computed from the Stage-09 Train split's row distribution, *not* reused by Stage 30 — Stage 30 recomputes per fold, which this experiment's harness also does).
2. **Balanced sample weights are active:** Yes, for every XGBoost fit used in V1 production (`29_classification_final_selection_report.txt`, `30_group_cross_validation.py` lines 1301-1312).
3. **Threshold tuning previously performed:** No. `29_classification_final_selection_report.txt` selects among whole-model trials using standard `.predict()` (argmax); no probability-threshold logic exists anywhere in Stages 20-33.
4. **Argmax used for final prediction:** Yes — confirmed in Stage 30 (`classification_model.predict(X_fold_validation)`, XGBoost's default `.predict()` for `multi:softprob` returns the argmax class).
5. **Stage-30 baseline reproducibility:** **Confirmed exact.** See Phase 3.
6. **SMOTE:** explicitly considered and rejected at Stage 21 ("Grouped temporal dataset with only 8 Train batches ... Synthetic row generation could violate temporal/process structure"). Not revisited in this round, per your instruction.

**17 classification input features** (from `08_classification_selected_features.csv`): `Smokehouse_ID`, `Layer_ID` (categorical, one-hot encoded) + 15 numeric features (`Temperature_C`, `Humidity_%`, `Elapsed_Drying_Hours`, `Number_of_Sheets`, `Layer_Area_m2`, `Temperature_Rate_per_10min`, `Humidity_Rate_per_10min`, `Temperature_Change_30m`, `Humidity_Change_30m`, `Temperature_Change_60m`, `Humidity_Change_60m`, `Temperature_RollStd_30m`, `Humidity_RollStd_30m`, `Temperature_RollStd_60m`, `Humidity_RollStd_60m`), median-imputed + standard-scaled. **Unchanged in every experiment below**, per your instruction not to touch features this round.

---

## Phase 2 — Isolated V2 Experiment Area

```
src/classification_v2/
  common.py                       (shared, leakage-safe utilities; never imports/reruns Stage 30)
  01_reproduce_baseline.py        (Phase 3)
  02_cost_sensitive_experiments.py(Phase 4 / Experiment A)
  03_threshold_experiments.py     (Phase 5 / Experiment B, nested CV)
  04_two_stage_experiments.py     (Phase 6 / Experiment C)
  05_compare_candidates.py        (candidate comparison + plots)

outputs/classification_v2/
  01_..., 02_..., 03_..., 04_..., 05_candidate_comparison_table.csv
  figures/*.png
  component3_critical_recall_improvement_report.md   <- this file
```

No existing Stage 20-33 script, report, model artifact, split file, or backend file was modified.

---

## Phase 3 — Baseline Reproduction

The V2 harness independently re-implements the Stage-30 methodology (5-fold `GroupKFold` on
`Batch_ID`, fold-local preprocessing, fold-local balanced sample weights, the exact recovered
XGB_03 hyperparameters) and was run against the frozen Development split.

| Metric | Stage-30 Reference | V2 Reproduced | Abs. Difference |
|---|---:|---:|---:|
| Accuracy | 0.9558441558 | 0.9558441558 | 0.00e+00 |
| Balanced Accuracy | 0.7249685993 | 0.7249685993 | 0.00e+00 |
| Macro F1 | 0.6452671122 | 0.6452671122 | 0.00e+00 |
| Warning Recall | 0.8564814815 | 0.8564814815 | 0.00e+00 |
| Critical Recall | 0.3550724638 | 0.3550724638 | 0.00e+00 |

**Exact bit-for-bit match on all five reference metrics**, and the pooled confusion matrix is
identical to `30_classification_group_cv_confusion_matrix.csv`. The harness is confirmed faithful
to the V1 methodology; all subsequent experiments build on this same harness.

Extended baseline metrics (not previously reported at this granularity): Critical Precision
0.4336, Critical F1 ≈0.3929, Critical F2 0.3684, Normal Recall ≈0.9732. Predicted Critical count
= 113 vs. actual 138 (64 false Critical alerts, 49 true positives).

---

## Phase 4 — Experiment A: Cost-Sensitive XGBoost

Predefined Critical-weight multipliers applied **on top of** the existing balanced weighting
(not instead of it), computed strictly inside each fold's training portion: **1.0, 1.5, 2.0, 3.0**.
1.0 reproduces the exact baseline (included as an internal consistency check — confirmed 0.00e+00
difference against Phase 3).

| Candidate | Critical Recall | Critical Precision | Critical F2 | Warning Recall | Macro F1 | Balanced Acc. | Predicted Critical (actual=138) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CS ×1.0 (= baseline) | 0.3551 | 0.4336 | 0.3684 | 0.8565 | 0.6453 | 0.7250 | 113 |
| CS ×1.5 | 0.3478 | 0.4000 | 0.3571 | 0.8488 | 0.6404 | 0.7204 | 120 |
| CS ×2.0 | 0.3623 | 0.3846 | 0.3666 | 0.8410 | 0.6386 | 0.7224 | 130 |
| CS ×3.0 | 0.3841 | 0.3605 | 0.3791 | 0.8380 | 0.6374 | 0.7284 | 147 |

CS ×3.0 catches 53/138 Critical rows vs. baseline's 49/138 (+4 rows), at the cost of 94 false
Critical alerts vs. baseline's 64 (+30 false alerts: +19 Normal→Critical, +11 Warning→Critical).

**Per-batch detail (CS ×3.0 vs. baseline Critical Recall), full tables in
`02_cost_sensitive_batch_metrics.csv`:**

| Batch | Actual Critical | Baseline Recall | CS ×3.0 Recall |
|---|---:|---:|---:|
| SH01_B04 | 66 | 0.258 | 0.303 |
| SH01_B06 | 12 | 0.833 | 0.833 |
| SH02_B01 | 6 | 0.000 | 0.333 |
| SH02_B02 | 18 | 0.000 | 0.000 |
| SH02_B03 | 18 | 0.944 | **1.000** |
| SH02_B06 | 18 | 0.278 | **0.167 (worse)** |

**This is not a uniform improvement.** 3 of 6 Critical-containing batches improved, 1 is
unchanged, 1 (`SH02_B02`) remains a complete miss in both versions, and 1 (`SH02_B06`) got
measurably *worse*. The pooled-level gain (+2.9pp Critical Recall) masks this batch-level mix.

---

## Phase 5 — Experiment B: Critical Probability Threshold (Nested, Leakage-Safe)

**Decision rule:** `if P(Critical) >= threshold: predict Critical; else argmax(P(Normal), P(Warning))`.

**Leakage control:** Outer 5-fold `GroupKFold` (identical to Stage 30) provides the unbiased
evaluation fold. Inside each outer-training set only, an inner `GroupKFold` (4 splits over the
outer-training batches) produces pooled inner-OOF probabilities; the threshold is selected from
**only** this inner-pooled evidence by maximizing Critical F2 (tie-break: higher Critical
Precision; tie-break 2: larger/more conservative threshold) — declared before any outer-fold
result was inspected. The selected threshold is then applied to a model refit on the *full*
outer-training set and scored on the untouched outer-test fold. No outer-validation row ever
influences its own threshold.

**Threshold selected per outer fold** (full detail in `03_threshold_selected_per_outer_fold.csv`):

| Outer Fold | Validation batches | Selected threshold |
|---|---|---:|
| 1 | SH01_B04, SH01_B05 | 0.10 |
| 2 | SH01_B01, SH01_B02 | 0.10 |
| 3 | SH02_B02, SH02_B06 | 0.15 |
| 4 | SH02_B01, SH02_B05 | 0.10 |
| 5 | SH01_B06, SH02_B03 | 0.10 |

Every outer fold's inner-CV selected the **most aggressive end of the grid** — F2 (which weights
recall more than precision) is maximized at low thresholds given this classifier's generally
low/uncertain Critical-class probabilities.

**Pooled OOF result (nested candidate):**

| Metric | Baseline | Threshold-Tuned (Nested) | Change |
|---|---:|---:|---:|
| Critical Recall | 0.3551 | **0.5652** | +0.2101 |
| Critical Precision | 0.4336 | **0.2149** | −0.2187 |
| Critical F2 | 0.3684 | 0.4262 | +0.0578 |
| Warning Recall | 0.8565 | **0.7130** | −0.1435 |
| Macro F1 | 0.6453 | 0.6023 | −0.0430 |
| Balanced Accuracy | 0.7250 | 0.7462 | +0.0212 |
| False Critical alerts | 64 | **285** | +221 |

A separate, clearly-labeled **descriptive (non-selected) threshold sweep**
(`03_threshold_descriptive_sweep_NOT_A_CANDIDATE.csv`) was produced purely to visualize the
full Recall/Precision curve (0.10→0.60 Recall as threshold drops from 0.50→0.10, Precision
falling from 0.452→0.223) — this sweep was **not** used to pick a threshold; it exists only for
the required plot.

**Interpretation:** this candidate buys a large Critical Recall gain almost entirely by flooding
the Warning/Normal boundary with false Critical alerts — 285 false alerts to catch ~37 more true
Critical rows than baseline (78 vs. 49). Warning Recall drops by 14 points and Macro F1 drops by
4.3 points. This is close to the exact failure mode the task warned about: recall bought by
indiscriminate over-prediction.

---

## Phase 6 — Experiment C: Two-Stage Classification

Stage A: Normal vs. Abnormal (Warning+Critical). Stage B: Warning vs. Critical, trained only on
fold-training rows whose **true** label is Abnormal, applied only to validation rows Stage A
*predicts* as Abnormal. Both stages use the same core XGBoost hyperparameters as XGB_03,
binarized (`objective="binary:logistic"`), with balanced sample weights computed independently
per stage from that stage's own fold-training labels.

**Stage-A pooled (Normal vs. Abnormal):** Abnormal Recall 0.9046, Abnormal Precision 0.5847,
Normal Recall 0.9743. Stage A is a fairly strong anomaly detector — it misses only 75 of 786
Warning+Critical rows, but over-flags 505 Normal rows as Abnormal.

**Stage-B pooled (Warning vs. Critical, evaluated only on true-Abnormal rows):** Critical Recall
0.3043, Critical Precision 0.5385, Warning Recall 0.9444. Stage B is precision-leaning: when it
does commit to "Critical", it's right more often than the single-stage baseline (precision 0.54
vs. 0.43), but it recovers fewer of the true Critical rows (42/138) than baseline (49/138).

**Combined 3-class pooled result:**

| Metric | Baseline | Two-Stage | Change |
|---|---:|---:|---:|
| Critical Recall | 0.3551 | 0.3043 | −0.0507 |
| Critical Precision | 0.4336 | **0.4615** | +0.0279 |
| Critical F2 | 0.3684 | 0.3266 | −0.0418 |
| Warning Recall | 0.8565 | 0.8580 | +0.0015 |
| Normal Recall | ≈0.9732 | **0.9743** | +0.0011 |
| Macro F1 | 0.6453 | **0.6597** | +0.0144 |
| Balanced Accuracy | 0.7250 | 0.7122 | −0.0128 |
| False Critical alerts | 64 | **49** | −15 |

**Interpretation:** the two-stage architecture trades Critical Recall for Critical Precision and
fewer false alerts, with Warning/Normal performance essentially unchanged and a small Macro-F1
gain. It is a legitimate, different point on the precision/recall trade-off curve, not a strict
improvement or a strict regression.

---

## Phase 7 — Ordinal Research Note (not implemented, per instructions)

`Normal < Warning < Critical` does have a defensible physical ordering (increasing fire risk
severity), which could in principle support an ordinal classifier (e.g., proportional-odds /
cumulative-link models, or an ordinal-aware boosting objective). However:

- The three classes are **not evenly spaced** in a measurable risk sense — we have no calibrated
  continuous severity score underlying the three labels, only the hourly expert/keeper
  categorical assessment. Treating the gap "Normal→Warning" as equal to "Warning→Critical" would
  be an assumption we cannot currently justify from the data.
- Implementing a genuine ordinal model (e.g., `mord`, or a custom cumulative-threshold XGBoost
  wrapper) would be a non-trivial architecture change, essentially a fourth experiment track with
  its own leakage-safety design, not a small variant of the current harness.
- Given this round's explicit scope limits (no feature changes, no SMOTE, close to XGBoost), a
  rigorous ordinal implementation does not fit cleanly this round.

**Recommendation:** document as a candidate direction for a future, separately-scoped experiment,
not implemented now.

---

## Candidate Comparison Table

(Full machine-readable version: `outputs/classification_v2/05_candidate_comparison_table.csv`)

| Candidate | Crit. Recall | Crit. Precision | Crit. F2 | Warning Recall | Macro F1 | Bal. Acc. | Fold Crit. Recall (mean±std, min–max) | Folds w/ Crit. support | Classification |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| Baseline XGB_03 | 0.3551 | 0.4336 | 0.3684 | 0.8565 | 0.6453 | 0.7250 | 0.324 ± 0.345 (0.00–0.90) | 4/5 | REFERENCE |
| CS ×1.0 | 0.3551 | 0.4336 | 0.3684 | 0.8565 | 0.6453 | 0.7250 | (identical) | 4/5 | sanity-check duplicate |
| CS ×1.5 | 0.3478 | 0.4000 | 0.3571 | 0.8488 | 0.6404 | 0.7204 | 0.314 ± 0.352 (0.00–0.90) | 4/5 | A — worse |
| CS ×2.0 | 0.3623 | 0.3846 | 0.3666 | 0.8410 | 0.6386 | 0.7224 | 0.326 ± 0.366 (0.00–0.93) | 4/5 | B — trade-off |
| **CS ×3.0** | **0.3841** | 0.3605 | 0.3791 | 0.8380 | 0.6374 | 0.7284 | 0.413 ± 0.315 (0.08–0.93) | 4/5 | **C — promising (see caveat)** |
| Threshold-Tuned (nested) | **0.5652** | 0.2149 | 0.4262 | 0.7130 | 0.6023 | 0.7462 | 0.634 ± 0.232 (0.42–1.00) | 4/5 | B — trade-off (severe) |
| Two-Stage XGB | 0.3043 | **0.4615** | 0.3266 | 0.8580 | **0.6597** | 0.7122 | 0.276 ± 0.335 (0.00–0.83) | 4/5 | B — trade-off |

**Best Critical Recall:** Threshold-Tuned (Nested), 0.5652 — but at the cost of 285 false alerts
and a 14-point Warning Recall drop.
**Best Critical F2:** Threshold-Tuned (Nested), 0.4262 — same caveat applies; CS ×3.0 is a distant
second (0.3791) with a far smaller cost profile.
**Best overall trade-off (Macro F1 / Balanced Accuracy without a severe Warning/precision
collapse):** No candidate strictly dominates baseline. CS ×3.0 gives the only genuinely "mostly
positive" pooled profile, and even that is **undermined by its batch-level inconsistency**
(Phase 4): it helped 3 of 6 Critical-containing batches, left 1 unchanged, left 1 at zero, and
made 1 measurably worse.

**Is any candidate clearly superior to XGB_03? No.** Every candidate that meaningfully raises
pooled Critical Recall does so by sacrificing Warning Recall and/or Critical Precision, in some
cases severely (Threshold-Tuned). CS ×3.0 is the mildest, most defensible trade-off, but its
batch-level instability (including one Critical-containing batch getting *worse*) means it should
be read as a modest, unstable nudge rather than a validated improvement.

---

## Protected Artifact Check

All 14 protected artifacts were hashed before Phase 0 and re-hashed after every experiment
(Phases 3-6). **Every hash is identical, before and after:**

| Artifact | Status |
|---|---|
| `data/raw/Component3_Dataset.csv` | UNCHANGED |
| `data/processed/Component3_Dataset_Preprocessed.csv` | UNCHANGED |
| `data/processed/Component3_Dataset_FeatureEngineered.csv` | UNCHANGED |
| `data/splits/Component3_Train.csv` | UNCHANGED |
| `data/splits/Component3_Validation.csv` | UNCHANGED |
| `data/splits/Component3_Test.csv` | UNCHANGED |
| `backend/.../drying_time_pipeline.pkl` | UNCHANGED |
| `backend/.../fire_risk_pipeline.pkl` | UNCHANGED |
| `backend/.../component3_model_metadata.json` | UNCHANGED |
| `backend/.../component3_inference_contract.json` | UNCHANGED |
| `outputs/reports/30_group_cross_validation_summary.json` | UNCHANGED |
| `outputs/reports/31_final_test_evaluation_summary.json` | UNCHANGED |
| `outputs/reports/32_shap_explainability_summary.json` | UNCHANGED |
| `outputs/reports/33_export_final_models_summary.json` | UNCHANGED |

No protected artifact changed. The STOP condition in §"Protected Artifact Check" was never
triggered.

---

## Research Impact Questions

1. Did the raw dataset change? **No.**
2. Did preprocessing change? **No** (Stage 04/10 untouched; V2 reimplements, does not modify, Stage-10-equivalent preprocessing fold-locally).
3. Did feature engineering change? **No.**
4. Did Train/Validation/Test split change? **No.**
5. Did the regression model change? **No** — never loaded, fit, or evaluated in this task.
6. Did regression metrics change? **No.**
7. Did the existing XGB_03 model change? **No** — it was reproduced independently, not modified.
8. Did Stage-31 results change? **No.**
9. Did Stage-32 SHAP change? **No.**
10. Did Stage-33 exported pipelines change? **No.**
11. Was Test data used for candidate tuning? **No** — `Component3_Test.csv` was never opened by any V2 script (existence-checked only via `common.py`).
12. Best Critical Recall candidate: **Threshold-Tuned (Nested)**, 0.5652.
13. Best Critical F2 candidate: **Threshold-Tuned (Nested)**, 0.4262.
14. Best overall trade-off: **No candidate clearly wins**; CS ×3.0 is the mildest positive-leaning trade-off but is batch-unstable; Two-Stage XGB trades recall for precision/Macro-F1 with the smallest downside elsewhere.
15. Was improvement consistent across Critical-containing batches? **No.** For CS ×3.0 specifically: 3/6 batches improved, 1/6 unchanged, 1/6 stayed at zero, 1/6 got worse.
16. Is there enough evidence to replace XGB_03? **No.**
17. Would more Critical-containing batches still be recommended? **Yes** — every experiment this round operated on the same 6 Critical-containing development batches; no amount of weighting/threshold/architecture change increased the number of *independent* batches carrying Critical signal, which remains the structural bottleneck identified in the prior Stage-09 audit.
