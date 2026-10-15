# Component 3 — Final Critical-Recall Improvement Experiment (Experiments D & E)

**Status: EXPERIMENTAL EVIDENCE ONLY. This is the final controlled model-improvement round on
the current frozen dataset. XGB_03 has NOT been replaced. No production artifact was touched.**

---

## 1. Research Question

Can either (D) a dedicated Critical-vs-Rest binary detector combined with the existing
multiclass logic, or (E) an ordinal severity formulation that exploits `Normal < Warning <
Critical`, improve Critical-class detection beyond frozen XGB_03 while preserving acceptable
Normal/Warning behavior and batch-level stability?

## 2. Why This Final Experiment Was Conducted

Two prior rounds of controlled experimentation (first V2 round: cost-sensitive weighting,
nested probability-threshold tuning, two-stage classification) found no candidate clearly
superior to XGB_03. This final round tests the two remaining, scientifically distinct
directions — a dedicated rare-event detector and an ordinal formulation — before closing
ML experimentation on this frozen dataset, per explicit protocol.

## 3. Frozen Baseline (XGB_03, reproduced fresh this round)

| Metric | Value | Reproduction diff vs. Stage 30 |
|---|---:|---:|
| Accuracy | 0.9558441558 | 0.00e+00 |
| Balanced Accuracy | 0.7249685993 | 0.00e+00 |
| Macro F1 | 0.6452671122 | 0.00e+00 |
| Warning Recall | 0.8564814815 | 0.00e+00 |
| Critical Recall | 0.3550724638 | 0.00e+00 |
| Critical Precision | 0.4336 | (newly computed, not in Stage 30's original report) |
| Critical F1 | 0.3929 | — |
| Critical F2 | 0.3684 | — |

**Exact bit-for-bit reproduction confirmed again** before any new experiment was attempted, per
the STOP-if-mismatch protocol (it did not trigger).

## 4. Data-Freeze Statement

No dataset file was created, deleted, or modified. Row counts, labels, `Batch_ID` values, and
timestamps are untouched everywhere. Verified by SHA-256 before and after (§12). Specifically:
`data/raw/Component3_Dataset.csv` (24,463 rows), `data/validated/...`, `data/processed/...`
(both preprocessed and feature-engineered), and all three `data/splits/...` files are byte-for-
byte identical to their pre-experiment hashes.

## 5. Test-Set Isolation Statement

`data/splits/Component3_Test.csv` was never opened by any script in this round. `common.py`'s
`load_development()` only checks its existence (for the SHA-256 integrity record below) and
never calls `pd.read_csv` on it. No candidate's threshold, weight, architecture, or selection
decision used Test data, Test labels, or Stage-31 predictions in any way.

## 6. Development Dataset

20,405 rows, 10 batches (Train 16,246 + Validation 4,159). **Only 6 of these 10 batches contain
any Critical example** (`SH01_B04`, `SH01_B06`, `SH02_B01`, `SH02_B02`, `SH02_B03`, `SH02_B06`);
the other 4 (`SH01_B01`, `SH01_B05`, `SH02_B05`, and `SH01_B02`) are Critical-free. This scarcity
is the central constraint shaping every result below (consistent with the prior Stage-09
robustness audit).

## 7. Experiment D — Methodology: Dedicated Critical-vs-Rest Detector

- Binary labels (`Critical=1` / `Rest=0`) created **in memory only**, inside the script; never
  written to any dataset file.
- Binary detector: XGB_03's tree hyperparameters unchanged; `objective="binary:logistic"`
  (removes `num_class`); `eval_metric="aucpr"` — the **one** documented, predefined deviation
  from XGB_03, chosen because PR-AUC-oriented training is more appropriate than log-loss at
  ~0.68% positive prevalence. Not tuned, not searched.
- Imbalance handling: `scale_pos_weight = n_negative / n_positive`, computed strictly from the
  applicable training partition (inner-train for threshold selection, full outer-train for the
  final fit). No SMOTE, no duplicated rows.
- The multiclass XGB_03 configuration is **retrained inside every fold** (inner and outer) using
  the exact Stage-30 recipe (fold-local preprocessing, fold-local balanced sample weights) — the
  production `fire_risk_pipeline.pkl` is never loaded for fold predictions, preventing
  development leakage through the frozen artifact.
- Combination rule: `if P_critical >= threshold: Critical; else: multiclass argmax`.

### Nested threshold selection (leakage control)

Outer `GroupKFold(5, Batch_ID)` → outer-test fold = unbiased evaluation set. Inside each
outer-training set, inner `GroupKFold(≤4, Batch_ID)` pools inner-validation predictions from
both models; for each grid threshold the **combined system's** pooled inner metrics are
computed. Selection: maximize Critical F2, tie-break Critical Precision, tie-break combined
Warning Recall, tie-break larger threshold — declared before any outer-fold result was seen.
Grid: 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50 (used unmodified; the observed probability
distribution did not require adjustment).

### Experiment D — Results

**Selected thresholds per outer fold:** 0.10, 0.10, 0.15, 0.10, 0.35 — four of five folds again
selected the aggressive end of the grid (consistent with the earlier Experiment B pattern).

**Standalone binary detector (Critical-vs-Rest), pooled OOF:**

| Metric | Value |
|---|---:|
| Recall | 0.6087 |
| Precision | 0.2228 |
| F1 | 0.3262 |
| F2 | 0.4521 |
| ROC-AUC | 0.9437 |
| PR-AUC (Average Precision) | 0.3620 |
| Predicted positive | 377 |
| Actual positive | 138 |
| False positives | 293 |
| False negatives | 54 |

ROC-AUC (0.94) looks strong but is **not trustworthy alone** given ~0.68% prevalence — PR-AUC
(0.36) is the honest picture: most of the detector's positive predictions are false alarms. This
is exactly the imbalance caveat the brief warned against relying on ROC-AUC for.

**Combined 3-class system, pooled OOF:**

| Metric | Baseline | Critical-vs-Rest Combined | Change |
|---|---:|---:|---:|
| Critical Recall | 0.3551 | **0.6159** | +0.2608 (highest of any candidate tested, either round) |
| Critical Precision | 0.4336 | 0.2208 | −0.2128 |
| Critical F2 | 0.3684 | **0.4536** | +0.0852 (highest of any candidate tested) |
| Warning Recall | 0.8565 | **0.6620** | −0.1945 (**below the 0.80 guardrail**) |
| Macro F1 | 0.6453 | 0.5980 | −0.0473 |
| Balanced Accuracy | 0.7250 | 0.7464 | +0.0215 |
| False Critical alerts | 64 | **300** | +236 |

**Batch-level detail (notable finding):** this is the **only candidate across both experiment
rounds** that moves batch `SH02_B02` off its persistent 0.0 Critical Recall — to **0.611**.
However, Warning Recall collapses on nearly every batch simultaneously (e.g. `SH01_B02`:
0.899→0.250; `SH02_B06`: 0.722→0.333; `SH02_B03`: 0.467→0.433). The detector is not
indiscriminately flagging Critical everywhere — it does genuinely find signal in `SH02_B02` that
no other candidate found — but it pays for that with a severe, broad Warning-detection cost.

**Interpretation:** this is the strongest decision-dimension-leading candidate (best Recall,
best F2) but falls under the explicit "extremely high Critical Recall... severe Warning
degradation... NOT automatically better" rule. Warning Recall below the 0.80 guardrail is a
**major, clearly-flagged trade-off**, not a disqualification by itself, but it means this
candidate cannot be called an uncomplicated improvement.

## 8. Experiment E — Methodology: Ordinal Severity Classification

Cumulative ordinal design: `O1: P(Y≥Warning)` (target Normal=0, Warning=Critical=1), `O2:
P(Y≥Critical)` (target Normal=Warning=0, Critical=1). Both are independent binary XGBoost
classifiers (binarized XGB_03 hyperparameters: `objective="binary:logistic"`,
`eval_metric="logloss"`), with `compute_sample_weight("balanced")` computed independently per
fold from that fold's own binary-target distribution.

**Probability reconstruction (exact):**
```
p_ge_warning  = O1.predict_proba[:, 1]
p_ge_critical = O2.predict_proba[:, 1]
p_ge_critical_corrected = min(p_ge_critical, p_ge_warning)   # monotonicity fix
P(Normal)   = 1 - p_ge_warning
P(Warning)  = p_ge_warning - p_ge_critical_corrected
P(Critical) = p_ge_critical_corrected
```
All three clipped ≥0 and verified to sum to 1 within 1e-6 for every evaluated row (verified
programmatically; the run would have raised an error otherwise — it did not).

**Monotonicity violations** (raw `P(Y≥Critical) > P(Y≥Warning)`, before correction): **580 of
20,405 rows (2.84%)**, corrected deterministically as specified. This is a modest, expected rate
of ordinal inconsistency for two independently-trained binary models and is handled
transparently, not hidden.

Two candidates reported:
1. **Ordinal-Argmax**: `argmax(P(Normal), P(Warning), P(Critical))`. No tuning.
2. **Ordinal-Threshold** (exactly one nested threshold experiment, same grid/inner-CV/tie-break
   convention as Experiment D, applied to `p_ge_critical_corrected`).

### Experiment E — Results

**Ordinal-Argmax, pooled OOF:**

| Metric | Baseline | Ordinal-Argmax | Change |
|---|---:|---:|---:|
| Critical Recall | 0.3551 | **0.4348** | +0.0797 |
| Critical Precision | 0.4336 | 0.4027 | −0.0309 |
| Critical F2 | 0.3684 | **0.4280** | +0.0596 |
| Warning Recall | 0.8565 | 0.8117 | −0.0448 (still above the 0.80 guardrail) |
| Macro F1 | 0.6453 | **0.6748** | **+0.0295 (improved)** |
| Balanced Accuracy | 0.7250 | **0.7405** | **+0.0155 (improved)** |
| False Critical alerts | 64 | 89 | +25 |

**This is the only candidate, across both experiment rounds, that improves Critical Recall,
Critical F2, Macro F1, AND Balanced Accuracy simultaneously**, while Warning Recall stays above
the 0.80 guardrail and Critical Precision only drops modestly (3 points).

**Batch-level detail (6 Critical-containing batches):**

| Batch | Actual Critical | Baseline Recall | Ordinal-Argmax Recall | Change |
|---|---:|---:|---:|---|
| SH01_B04 | 66 | 0.258 | **0.379** | Improved |
| SH01_B06 | 12 | 0.833 | **1.000** | Improved |
| SH02_B01 | 6 | 0.000 | 0.000 | Unchanged (still 0) |
| SH02_B02 | 18 | 0.000 | 0.000 | Unchanged (still 0) |
| SH02_B03 | 18 | 0.944 | 0.889 | Slightly worse (−0.056; precision improved 0.405→0.500) |
| SH02_B06 | 18 | 0.278 | **0.389** | Improved |

**3 of 6 batches improved, 2 remain unchanged at zero, 1 is marginally worse (with a precision
offset).** This is the **most stable** batch-level profile of any candidate tested in either
round — no batch suffered a severe regression, unlike `CostSensitive_x3.0` (which made
`SH02_B06` measurably worse by 11pp with no offsetting gain) or either threshold/detector
candidate (broad Warning collapse).

**Ordinal-Threshold, pooled OOF:**

| Metric | Baseline | Ordinal-Threshold | Change |
|---|---:|---:|---:|
| Critical Recall | 0.3551 | 0.5000 | +0.1449 |
| Critical Precision | 0.4336 | 0.2072 | −0.2264 |
| Critical F2 | 0.3684 | 0.3898 | +0.0214 |
| Warning Recall | 0.8565 | 0.6698 | −0.1867 (**below guardrail**) |
| Macro F1 | 0.6453 | 0.6117 | −0.0336 |
| Balanced Accuracy | 0.7250 | 0.7143 | −0.0107 |
| False Critical alerts | 64 | 264 | +200 |

Selected thresholds per fold: 0.10, 0.10, 0.30, 0.25, 0.15 — again clustering toward the
aggressive end. This candidate repeats the now-familiar pattern (Experiments B and D): adding a
hard threshold override buys recall at a severe, guardrail-breaking Warning cost. It is **not**
an improvement on the Ordinal-Argmax candidate — the un-thresholded ordinal reconstruction is
strictly the better of the two ordinal variants.

## 9. Final Comparison Table

(Full machine-readable table: `08_final_advanced_candidate_comparison.csv` / `.json`)

| Candidate | Crit. Recall | Crit. Precision | Crit. F2 | Warning Recall | Macro F1 | Bal. Acc. | False Crit. Alerts | Mean Fold Recall ± Std | Below 0.80 Guardrail | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Baseline XGB_03 | 0.3551 | 0.4336 | 0.3684 | 0.8565 | 0.6453 | 0.7250 | 64 | 0.324 ± 0.345 | No | REFERENCE |
| CostSensitive ×3.0 (prior round) | 0.3841 | 0.3605 | 0.3791 | 0.8380 | 0.6374 | 0.7284 | 94 | 0.413 ± 0.315 | No | PROMISING |
| Threshold-Tuned Nested (prior round) | 0.5652 | 0.2149 | 0.4262 | 0.7130 | 0.6023 | 0.7462 | 285 | 0.634 ± 0.232 | **Yes** | TRADE-OFF (severe) |
| Two-Stage XGB (prior round) | 0.3043 | 0.4615 | 0.3266 | 0.8580 | 0.6597 | 0.7122 | 49 | 0.276 ± 0.335 | No | TRADE-OFF |
| **Critical-vs-Rest Combined (new)** | **0.6159** | 0.2208 | **0.4536** | 0.6620 | 0.5980 | 0.7464 | 300 | 0.640 ± 0.200 | **Yes** | TRADE-OFF (severe) |
| **Ordinal-Argmax (new)** | 0.4348 | 0.4027 | 0.4280 | 0.8117 | **0.6748** | 0.7405 | 89 | 0.377 ± 0.348 | No | **PROMISING_IMPROVEMENT** |
| Ordinal-Threshold (new) | 0.5000 | 0.2072 | 0.3898 | 0.6698 | 0.6117 | 0.7143 | 264 | 0.426 ± 0.368 | **Yes** | TRADE-OFF (severe) |

**No candidate was classified CLEARLY_SUPERIOR.** Ordinal-Argmax is the strongest result found
in this entire two-round experimentation effort — it is the only candidate improving Macro F1
and Balanced Accuracy alongside Critical Recall/F2, with the most stable batch-level profile and
no guardrail breach — but it still leaves 2 of 6 Critical-containing batches at zero recall and
costs 4.5 points of Warning Recall, so it is reported as **PROMISING_IMPROVEMENT**, not a
decisive win.

## 10. Trade-offs, Summarized

- **Every candidate that pushes Critical Recall meaningfully above baseline does so by adding a
  hard probability-threshold override** (Experiments B, D's combination, E's thresholded
  variant). In all three cases this breaches the 0.80 Warning guardrail and roughly
  quadruples-to-quintuples false Critical alerts.
- **The only candidates that avoid a guardrail breach** (CostSensitive ×3.0, Ordinal-Argmax) are
  also the only candidates with modest, defensible gains — and even they do not fix the two
  batches (`SH02_B01`, `SH02_B02`) that have never produced a single true Critical detection
  under any method tried in either round.
- `SH02_B02` was moved off zero by exactly one candidate (Critical-vs-Rest Combined), at the cost
  of that candidate's systemic Warning collapse — suggesting there may be *some* learnable signal
  for that batch, but only extractable by an architecture that is otherwise too aggressive to
  recommend.

## 11. Limitations

- Development data still contains only 6 Critical-containing batches out of 10; every
  experiment in this round operates on the same underlying signal as the prior round. No
  architecture change manufactures new independent Critical-batch evidence.
- Ordinal monotonicity violations (2.84% of rows) indicate O1 and O2 are not perfectly
  consistent independent estimators; the deterministic `min()` correction is a reasonable,
  transparent fix but a jointly-trained ordinal model (not attempted here, per scope) might
  reduce this rate.
- This is still Development-only evidence (GroupKFold, 10 batches). No candidate's number here
  should be read as a prediction of final-test (Stage 31-style) performance.

## 12. Final Recommendation

**No candidate is clearly superior to XGB_03.** Per protocol, this means: **retain XGB_03** as
the production classifier and close ML experimentation on the current frozen dataset. Ordinal-
Argmax is flagged as the single most promising direction for any future, separately-scoped
round (it would need, at minimum, evaluation on more Critical-containing batches before any
replacement could be justified — the current 2-of-6-batches-still-at-zero result is too thin a
basis for a production change).

## 13. Future Work

- More Critical-containing batches (the structural fix identified across all three audits/
  experiment rounds to date).
- A jointly-trained ordinal model (e.g. a proper cumulative-link loss) to remove the 2.84%
  monotonicity violation rate rather than correcting it post hoc.
- Batch-level (not row-level) resampling strategies, explicitly out of scope this round.
- Revisiting Stage-07/08 feature engineering if more data does not resolve the `SH02_B01`/
  `SH02_B02` zero-recall batches.

---

**Explicit statements required by protocol:**
- **No dataset was modified.** (Verified: all hashes identical before/after, §14.)
- **Regression was untouched.** GB_02 / `drying_time_pipeline.pkl` was never loaded, fit, or
  evaluated anywhere in this task.
- **The Test split was not used for candidate development.** `Component3_Test.csv` was never
  opened by any script in `src/classification_v2/`.
