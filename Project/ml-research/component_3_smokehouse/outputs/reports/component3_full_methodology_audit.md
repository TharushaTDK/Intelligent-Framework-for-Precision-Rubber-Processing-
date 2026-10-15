# Component 3 — Full Methodology Audit

**Group:** J26-DS-340 — Smart Layer-Wise Rubber Smokehouse Drying-Time & Fire-Risk Prediction System
**Audit type:** Read-only. No project file was modified during this audit.
**Scope:** `src/04` through `src/33`, the three pre-existing `src/30_*_audit.py` scripts, their generated reports in `outputs/reports/`, `data/validated/`, and the exported backend inference contract.
**Date:** 2026-10-13

---

## 1. Executive Summary

| # | Finding | Severity |
|---|---|---|
| 1 | Raw dataset integrity (SHA-256, row/column count) | PASS |
| 2 | Stage 04 preprocessing (causal LOCF, no leakage) | PASS |
| 3 | Stage 07 feature engineering (no future data, `Actual_Total_Drying_Time` excluded) | PASS |
| 4 | Stage 08 feature-selection **mechanism** is a hardcoded, non-target-driven list | PASS |
| 5 | Stage 05/06/08 descriptive/EDA statistics were computed on the **full 24,463-row dataset**, before Stage 09 froze the test batches | MODERATE |
| 6 | Stage 09 batch→split assignment was **deliberately chosen using per-batch Fire_Risk_Label (target) distribution** from the full dataset | **MAJOR** |
| 7 | Stage 10 train-only preprocessing | PASS |
| 8 | `30_cv_data_integrity_audit.py` self-reports **"CLASS MAPPING COULD NOT BE INFERRED"** — an unresolved audit flag | MODERATE (tooling bug, not a data defect — see §4.8) |
| 9 | Stage 30 GroupKFold, Stage 31 final test, Stage 32 SHAP, Stage 33 export — methodology is sound and test isolation is real | PASS |
| 10 | `data/validated/` is empty and is never written to by any stage | MINOR |
| 11 | Backend live feature-construction code (Stage-07 equivalent) does not yet exist | INFORMATIONAL (expected — Phase G not started) |

**Headline conclusion:** There is **no classical train/test leakage** in the sense of target values or test rows entering model fitting, preprocessing fitting, or feature-selection algorithms. However, Finding #6 is a real and quantifiable methodology concern: the final-test batches were not chosen blind to the classification target. This is very likely the primary cause of the 0.0 Critical recall reported at Stage 31, and it means that number should be read as a **small-sample, single-batch worst case**, not as a stable estimate of real-world Critical-class performance. This nuance changes how the limitation should be worded in the thesis, but it does **not** mean Stage 31's reported numbers are fabricated, mishandled, or need to be hidden — they are exactly what the frozen pipeline produces.

---

## 2. Raw Dataset Integrity — PASS

`src/04_preprocessing.py` hard-checks the raw file's SHA-256 against a frozen constant before anything else runs. I independently recomputed the hash:

```
sha256sum data/raw/Component3_Dataset.csv
229139d2b4413fe15a8d322023b3b42574f9dccd5df6a91550bc72f05faaa5cb
```

This matches the value hard-coded in `04_preprocessing.py` (`EXPECTED_SHA256`) and in `30_raw_to_preprocessed_identity_audit.txt`. Row count (24,463) and the 17-column schema also match. **No drift detected.**

---

## 3. Stage 04 — Preprocessing — PASS

File: `src/04_preprocessing.py`

- Missing sensor values are imputed with **last-observation-carried-forward strictly within the same `Smokehouse_ID+Batch_ID+Layer_ID` sequence**, using only earlier timestamps (`get_past_trend_reference` walks backward only; the main imputation loop tracks `last_valid_value`/`last_valid_timestamp` forward in time and never looks ahead).
- The script independently re-verifies afterward that every *originally observed* sensor value is byte-identical to the raw file (`changed_observed_temperature`, `changed_observed_humidity` checks), that no duplicate observation keys were introduced, and that the 12 genuine 20-minute packet-loss gaps are preserved exactly (`ten_minute_intervals == 24415`, `twenty_minute_intervals == 12`).
- No scaling, encoding, global-statistic imputation, model fitting, or split occurs in this stage (confirmed by reading the full script — there is no `fit`, no `StandardScaler`, no `train_test_split` call anywhere in it).

All 13 pass/fail checks in `04_preprocessing_checks.csv` are self-reported PASS, and the independent `30_raw_to_preprocessed_identity_audit.txt` cross-check confirms row count (24,463→24,463), Fire-Risk class counts (Normal 23,371 / Warning 936 / Critical 156) and target values are unchanged from raw to preprocessed.

---

## 4. Stage 07 — Feature Engineering — PASS

File: `src/07_feature_engineering.py`

- `Actual_Total_Drying_Time` is deliberately dropped from the output schema (`EXCLUDED_LEAKAGE_REFERENCE` manifest entry) — confirmed absent from `Component3_Dataset_FeatureEngineered.csv`'s column list in code.
- Exact-timestamp lag/rolling features are built via `df.merge(..., how="left")` on `Timestamp + lag_minutes`, not via blind `.shift(n)` row offsets — this is the correct technique to avoid the 12 packet-loss gaps silently fabricating a false "future-looking" 10-minute lag.
- Section 15 of the script (`NO-FUTURE-DATA AUDIT FOR EXACT LAGS`) explicitly validates that no lag/rolling value exists before its required history window has elapsed since sequence start (`future_lag_violations` check, reported as 0).
- Rolling windows use `closed="both"` with the **current** timestamp as the index, which includes the current observation plus strictly earlier ones inside the window — no forward-looking data possible given `rolling()` on a sorted, monotonic per-sequence index.

**No model, scaler, encoder, or split occurs in this stage.** All checks PASS.

---

## 5. Stage 08 — Feature Analysis & Selection

File: `src/08_feature_analysis_selection.py`

### 5.1 How the 17 features were actually chosen

The final `SELECTED_SHARED_FEATURES` list (lines 718–745) is a **literal, hardcoded Python list**. It is not computed from any ranking, threshold, or model output. The decision table (`decision_reason` dict, built in Section 11 of the script) assigns KEEP/DROP labels by a fixed dictionary lookup — there is no code path where `REGRESSION_SELECTED` or `CLASSIFICATION_SELECTED` is derived from `regression_association`, `classification_association`, or `high_corr_pairs`. Those three tables are written to CSV and explicitly labeled in their own `Selection_Use` column as:

```
DESCRIPTIVE_ONLY_NOT_USED_FOR_PRE_SPLIT_SELECTION
```

A dedicated pass/fail check (`"Selection policy did not use target-ranking output"`) is hard-wired to PASS with that same justification. **Verdict: the selection mechanism itself did not consume target-ranked or test-exposed statistics.** This directly answers the prompt's highest-priority question: no algorithmic/target-aware leakage was found in the Stage-08 selection logic.

### 5.2 What the full-dataset descriptive stats *were* used for — MODERATE

This nuance matters and should not be glossed over. The redundancy analysis in Section 7 (`numeric_corr = df[NUMERIC_CANDIDATES].corr()`) and the two target-association tables in Sections 8–9 are computed on **`df`, the complete 24,463-row Stage-07 output — i.e., all 12 batches, including the two batches (`SH01_B03`, `SH02_B04`) that Stage 09 later assigns to the final test.** The written report then uses *empirical* (not purely algebraic) justifications for some DROP decisions — e.g., "Rolling mean is highly redundant with the current sensor value **in this smooth time series**" is a claim about the actual data's correlation structure, not a structural/algebraic fact like `Lag = Current − Change`.

Separately, the Stage-08 report explicitly cites "temporal/domain findings from Stages 05–06" as part of the policy's basis, and Stages 05 (EDA) and 06 (temporal/layer analysis) both ran on the same full, pre-split 24,463-row dataset.

**Why this matters:** even though no algorithm converted target information into a feature decision, a human reviewing full-dataset correlation heatmaps and EDA plots that already include the eventual test batches is a form of indirect "test awareness" during feature-set design — a well-known soft form of data snooping in research practice, distinct from (and less severe than) classical leakage. It does not necessarily invalidate the 16 selected features (the stated reasons are plausible and mostly structural/algebraic), but it means the research cannot currently claim the feature set was designed **completely blind** to the final-test batches.

- **Does this affect reported metrics?** Unlikely to meaningfully inflate them — the selection criteria are physically/structurally motivated (algebraic redundancy, runtime availability, near-constant indicators), not test-performance-tuned.
- **Retraining required?** Not urgently. Recommended low-priority fix: regenerate the Stage-08 descriptive tables (Sections 7–9) using **Train-only** rows in a corrected v2, to remove any residual appearance of test-aware reporting. The hardcoded feature list would very likely be unchanged.
- **Earliest stage needing rerun if corrected:** Stage 08 reporting only; no propagation to Stage 09+ expected since the 17-feature list itself is not expected to change.

---

## 6. Stage 09 — Batch-Aware Split — **MAJOR FINDING**

File: `src/09_batch_aware_split.py`, manifest: `outputs/reports/09_batch_split_manifest.csv`

### 6.1 Mechanics are sound

- Splitting is done by **complete `Smokehouse_ID+Batch_ID` groups**, never by individual rows (`TRAIN_BATCHES`/`VALIDATION_BATCHES`/`TEST_BATCHES` are fixed sets of batch IDs).
- The script independently verifies zero batch-group overlap and zero observation-key overlap across splits (`assignment_overlap_count`, `observation_overlap_count` — both checked and reported as 0).
- No scaling/encoding/resampling/model-fitting happens here (verified by reading the full script — Stage 09 only filters rows and writes CSVs).

### 6.2 The batch assignment itself is target-informed

The code's own comments (lines 34–53, 195–199) state the rationale directly:

> "TEST deliberately includes: SH01_B03 → long drying regime, SH02_B04 → shorter drying regime + substantial Warning... **TRAIN retains the majority of Critical assessment events.**"
> "A purely chronological split was rejected because the corresponding B05 validation batches contain no Warning/Critical hourly assessment events."

I verified this quantitatively from `09_batch_split_manifest.csv` (`Hourly_Critical_Assessments` column, unique hourly keeper-assessed events, not 10-minute carried-forward rows):

| Split | Batch | Critical (hourly events) |
|---|---|---|
| Train | SH01_B04 | 11 |
| Train | SH01_B06 | 2 |
| Train | SH02_B01 | 1 |
| Train | SH02_B03 | 3 |
| Train | SH02_B06 | 3 |
| **Train total** | | **20 / 26 (76.9%)** |
| Validation | SH02_B02 | 3 |
| **Test** | **SH01_B03** | **0** |
| **Test** | **SH02_B04** | **3** |
| **Test total** | | **3 / 26 (11.5%)** |

(3 hourly Critical events in Test × ~6 ten-minute rows/hour ≈ 18 rows — this reconciles exactly with the "18 actual Critical rows in the final test" figure already reported at Stage 31.)

**This confirms, with evidence, that the batch→split assignment was chosen using knowledge of the per-batch Fire_Risk_Label (target) distribution computed over the complete dataset — before the test set was "frozen."** This is a different and more concrete issue than the Stage-08 question the brief asked about.

### 6.3 Why this matters

- This is **not** classical leakage (no test row or test label was ever used to fit a preprocessor, model, or hyperparameter — Stages 10, 11–29, 31 all verifiably respect the frozen split).
- It **does** compromise the framing of the final test as a "blind, representative holdout." The test set's Critical examples are not a random/representative sample of Critical conditions across batches — they are concentrated into a single batch (`SH02_B04`), with the other test batch (`SH01_B03`) contributing zero Critical examples by construction.
- **This plausibly explains, at least in part, why Stage 31's locked Critical recall is exactly 0.0**, while Stage 30's GroupKFold pooled OOF Critical recall (where Critical events are spread across multiple folds drawn from all 10 development batches) is a much higher 0.355. The disparity is consistent with Critical-class behavior being highly batch/regime-specific, and with the final test's Critical sample being a single, small (3-event / 18-row), out-of-distribution batch rather than a statistically stable population estimate.
- **Practical implication for the thesis:** the 0.0 Critical recall finding is real and must stay reported (per the project's own rule, it must not be hidden or re-tuned away) — but it should now be *framed* as "0 of 18 Critical rows detected, drawn entirely from one held-out batch, under a test-batch assignment that was deliberately designed to concentrate Critical events in training" rather than as an unqualified, generalizable 0% Critical-detection rate. That framing is more defensible and more accurate.

### 6.4 Severity, retraining, and recommended fix

- **Severity: MAJOR.** Not leakage into model fitting; it is a compromised test-set-design assumption that affects how the headline Critical-recall limitation should be interpreted and reported.
- **Does it make reported metrics optimistic?** No — if anything it is more likely to have made the Critical-class test metric pessimistic/high-variance (worst case from a single unseen batch), while the Warning/Normal metrics and the regression metrics are less directly affected by this particular design choice (though the long/short-regime batch choice for `SH01_B03`/`SH02_B04` was also informed by known drying-duration characteristics, which has a smaller but non-zero analogous effect on the regression test's representativeness).
- **Is retraining required?** Not immediately — this is a methodology-framing issue in the existing frozen result, not a bug that silently corrupted a number. If the team wants a **true, blind, unbiased** test set for a "corrected v1," the batch assignment would need to be rebuilt using a constraint that does **not** consult `Fire_Risk_Label` counts (e.g., constrained random assignment that only enforces "both smokehouses + all 3 layers represented," with Warning/Critical representation accepted as whatever falls out, or an explicitly pre-registered stratification rule disclosed as such rather than discovered after inspecting the full target distribution).
- **Earliest stage requiring rerun if corrected:** **Stage 09** — and because Stages 10 through 33 all key off the Stage-09 split files, a corrected split would require rerunning the entire downstream chain (10 → 11-29 → 30 → 31 → 32 → 33) and regenerating the exported backend pipelines. This is a significant effort and should only be undertaken after explicit user sign-off (Phase C/D decision), not assumed.

---

## 7. Stage 10 — Train-Only Preprocessing — PASS

File: `src/10_train_only_preprocessing.py`

- `Component3_Test.csv`'s existence is checked but **its contents are never read** (`test_content_read` is never set; the file is opened nowhere after the existence check — verified by reading the full script, there is no `pd.read_csv(TEST_PATH)` call).
- `SimpleImputer(strategy="median")`, `StandardScaler`, and `OneHotEncoder` are all fit via `preprocessor.fit(X_train)` — Train rows only — then applied via `.transform()` to Validation.
- The script independently re-derives Train medians directly from the raw Train columns and diffs them against the fitted imputer's `.statistics_` (`median_stat_max_error`, required `< 1e-12`) as a cross-check that the imputer truly only saw Train data.
- Validation's transformed numeric means are checked to be **non-zero** (`validation_mean_not_artificially_zero`) — supporting evidence that Validation was transformed with the Train-fitted scaler rather than being independently re-standardized (which would hide a mistaken refit).

All 14 checks PASS. **No leakage.**

---

## 8. Stage 30 — Three Pre-Existing Audit Scripts

Three separate `30_*.py` scripts already exist in the repo. I ran none of them myself (read-only), but inspected their code and their already-generated report outputs.

### 8.1 `30_data_lineage_audit.py` → `30_full_data_lineage_audit.txt` — PASS
Confirms `Fire_Risk_Label` is unchanged across Preprocessed → Feature-Engineered → Train+Validation, with 0 missing-row mismatches and 0 label changes. Test contents were not loaded.

### 8.2 `30_raw_to_preprocessed_identity_audit.py` → report — PASS
Confirms 24,463 raw rows, 0 raw duplicates, identical Fire-Risk counts (Normal 23,371 / Warning 936 / Critical 156) before and after Stage 04.

### 8.3 `30_cv_data_integrity_audit.py` → report — **INCONCLUSIVE (tooling bug, not a data defect)**

This script's own report says:

```
Audit status: CLASS MAPPING COULD NOT BE INFERRED
Could not infer the numeric class mapping from Validation. Review Stage-10/Stage-20 target encoding before continuing.
```

I traced the cause. The script's `infer_numeric_mapping()` (line 71) calls `pd.to_numeric(model_ready_series, errors="coerce")` on `Component3_y_classification_Train.csv` / `..._Validation.csv`, expecting those files to contain encoded integers (0/1/2). I read the actual file:

```
Component3_y_classification_Train.csv (head):
Fire_Risk_Label
Normal
Normal
Normal
Normal
```

Stage 10 (`10_train_only_preprocessing.py`, Section 23) saves `y_cls_train`/`y_cls_validation` as the **raw string labels** ("Normal"/"Warning"/"Critical"), not integers. So `pd.to_numeric(...)` on an all-string column correctly returns all-NaN, `mr.isna().any()` is `True`, and the function returns `None` by design — producing the "could not infer" message. **This is a pre-existing bug in the audit script's assumption about the model-ready file format, not evidence of actual class-mapping corruption or leakage.** The real classification pipeline's class order is independently and correctly documented elsewhere (`component3_inference_contract.json`: `class_id 0=Normal, 1=Warning, 2=Critical`), and Stage 31/32/33 all reproduce consistent, cross-validated results using that mapping.

- **Severity: MODERATE** as an unresolved/unaddressed audit flag (the project rule says never hide failed checks, so this must be surfaced), but **MINOR** in terms of actual research risk once traced to its root cause.
- **Recommended fix:** update `read_single_target()`/`infer_numeric_mapping()` in `30_cv_data_integrity_audit.py` to handle string-labeled target files directly (compare string value_counts to string value_counts) instead of assuming a numeric encoding. No retraining required — this is a fix to a diagnostic script only.

### 8.4 `30_group_cross_validation.py` → report — PASS
5-fold GroupKFold by `Batch_ID` over the 10 development batches; preprocessing refit inside every fold; 0 batch-overlap folds; Test never loaded. Reported pooled OOF: Regression MAE 3.128h / R² 0.986; Classification Warning recall 0.856, **Critical recall 0.355** (notably higher than Stage 31's 0.0 — see §6.3 for why this gap is expected given the Stage-09 batch design).

---

## 9. Stage 31 — Locked Final Test Evaluation — PASS (methodology), limitation re-framed per §6

File reports: `31_final_test_evaluation_report.txt`

- Development (Train+Validation, 20,405 rows / 10 batches) preprocessing and the frozen Stage-19/29 model configurations are refit on Development **only**; the Test file is opened **after** all fitting is complete.
- Batch overlap and observation-key overlap between Development and Test are both independently re-verified as 0.
- Reported numbers match exactly what was given in the task brief (MAE 3.756h, RMSE 4.299h, R² 0.9835; Warning recall 0.986, Critical recall 0.0, confusion matrix with 18/18 actual-Critical rows predicted Warning).

**No tuning/reselection/threshold-adjustment occurred after Test was opened** — confirmed by the absence of any such code following the test-load point in the script's reported methodology section, consistent with Section 10 of the project's working rules.

---

## 10. Stage 32 — SHAP Explainability — PASS

- Reproduces Stage-31 predictions exactly before computing SHAP (`Regression predictions reproduce Stage 31: True`, `Classification predictions reproduce Stage 31: True`).
- Regression SHAP: `Elapsed_Drying_Hours` and `Humidity_%` dominate, consistent with the brief.
- Classification SHAP: Normal/Warning driven by temperature-related features; Critical driven more by `Humidity_RollStd_60m`, `Humidity_%`, `Number_of_Sheets`, `Elapsed_Drying_Hours` — consistent with the brief's "humidity variability, drying stage, environmental variability" description.
- The report explicitly disclaims causal interpretation and labels the actual-Critical diagnostic as "post-hoc and must not be used to alter Stage-31 test results" — correct safeguard language already in place.

---

## 11. Stage 33 — Export Final Pipelines — PASS

- Final pipelines are refit on the full Development set (Train+Validation), **not** on Test.
- Both exported pipelines (`drying_time_pipeline.pkl`, `fire_risk_pipeline.pkl`) are independently re-verified to reproduce the exact Stage-31 predictions (`Regression Stage-31 reproduction: True`, `Classification Stage-31 reproduction: True`).
- The exported `component3_inference_contract.json` correctly documents: 17 required post-feature-engineering inputs, 20 transformed features after the pipeline's internal OneHotEncoder/StandardScaler, class-ID mapping (0=Normal/1=Warning/2=Critical), and an explicit safety disclaimer that the classifier "must NOT be treated as a proven standalone Critical fire alarm."
- The contract correctly states **Stage-07 feature engineering is NOT inside the sklearn pipeline** and must be reconstructed by the backend — this is the deployment-consistency risk flagged in the brief (Section 14). As of this audit, `backend/app/ml/component_3_smokehouse/` contains only the exported artifacts (pipelines + metadata + contract) — the actual backend feature-construction code that would replicate Stage 07's lag/rate/rolling-window logic does not yet exist in `backend/app/services/` or `backend/app/api/routes/smokehouse.py`. **This is expected at this point in the project (Phase G has not started)** and is not a defect, but it is the single highest-priority item to get exactly right before any live inference is trusted, given how specific Stage 07's exact-timestamp-join and timestamp-aware rolling-window logic is (ordinary `.shift(n)`-based reimplementations would silently diverge from training-time feature definitions across the 12 packet-loss gaps).

---

## 12. `data/validated/` — Empty Directory — MINOR

```
find data -type f    → no files under data/validated/
grep "validated" src/*.py src/**/*.py   → no matches
```

No script in `src/` reads from or writes to `data/validated/` at any stage — not just Stage 04. The directory appears to be scaffolding (possibly created in anticipation of a "Stage 03 quality-gated snapshot" that was never implemented, or created manually) rather than a broken pipeline step. It carries zero current data-integrity or reproducibility risk because nothing depends on it, but its existence alongside `raw/`, `processed/`, `splits/`, and `model_ready/` implies a lineage stage that isn't real, which is misleading for anyone reading the repo structure. **No retraining implication.** Recommended fix (pending your approval, not yet made): either (a) implement it as the intended Stage-03/04 boundary artifact (e.g., save the data-quality-checked-but-not-yet-imputed snapshot there), or (b) remove the empty directory and stop implying a lineage step exists.

---

## 13. Summary Table

| Area | File(s) | Severity | Leakage into model/metrics? | Retrain needed? | Earliest stage to rerun |
|---|---|---|---|---|---|
| Raw dataset hash/schema | `data/raw/Component3_Dataset.csv` | PASS | No | No | — |
| Stage 04 preprocessing | `04_preprocessing.py` | PASS | No | No | — |
| Stage 07 feature engineering | `07_feature_engineering.py` | PASS | No | No | — |
| Stage 08 selection mechanism | `08_feature_analysis_selection.py` | PASS | No | No | — |
| Stage 05/06/08 full-dataset descriptive stats pre-split | `05_eda.py`, `06_temporal_layer_analysis.py`, `08_feature_analysis_selection.py` | MODERATE | Indirect/soft only | Optional, low priority | Stage 08 (reporting only) |
| **Batch→split assignment uses target distribution** | `09_batch_aware_split.py` | **MAJOR** | No (not model-fitting leakage); compromises blind-holdout framing | Only if a "corrected v1" blind split is desired | **Stage 09** (full downstream rerun 09→33) |
| Stage 10 train-only preprocessing | `10_train_only_preprocessing.py` | PASS | No | No | — |
| `30_cv_data_integrity_audit.py` inconclusive result | `30_cv_data_integrity_audit.py` | MODERATE (tooling bug) | No | No (fix script only) | — |
| Stage 30/31/32/33 | respective scripts | PASS | No | No | — |
| `data/validated/` empty | `data/validated/` | MINOR | No | No | — |
| Backend feature construction missing | `backend/app/ml/component_3_smokehouse/` | INFORMATIONAL | N/A (not built yet) | N/A | Phase G |

---

## 14. Recommended Next Steps (for your decision — nothing has been changed)

1. Decide whether the Stage-09 batch-assignment issue (§6) warrants a "corrected v1" split that is blind to `Fire_Risk_Label` counts, accepting the downstream cost of rerunning Stages 09–33, or whether the existing frozen result should be kept and the thesis narrative updated to correctly characterize the Critical-recall limitation as above (small-sample, single-batch, target-informed-split artifact) rather than re-running anything.
2. Low-priority: fix `30_cv_data_integrity_audit.py`'s string/numeric target-mapping assumption so it stops reporting a false "could not infer" status.
3. Low-priority: decide the fate of the empty `data/validated/` directory (implement or remove).
4. No action needed yet on Stages 04/07/08(mechanism)/10/30(other two scripts)/31/32/33 — they passed their own self-checks and my independent re-derivation of key statistics (raw hash, Critical-event batch counts) agrees with the reports.

I have not modified any file. Awaiting your approval before any fix is applied.

