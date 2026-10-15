# Component 3 — Final Classifier Selection (Promotion Decision)

**This document does not rewrite or replace Stage-29's historical selection report
(`outputs/reports/29_classification_final_selection_report.txt`). It records the
subsequent, separate promotion decision made after two rounds of controlled
improvement research.**

## Chronology

1. **Original selection (Stage 29):** XGBoost, trial XGB_03, selected via the Stage-29
   guardrail policy (Warning Recall ≥ 0.80, Critical Recall ≥ 0.50 among candidates,
   highest Macro-F1 among those passing). Exported at Stage 33.
2. **Observed limitation:** Stage-31 locked-test Critical Recall = 0.0 (0/18 rows, all from
   one batch, `SH02_B04`). Stage-30 development GroupKFold showed a broader, batch-sensitive
   Critical Recall of 0.355, with fold-level values ranging 0.0–0.90.
3. **Improvement research, round 1 (V2):** cost-sensitive weighting, nested probability
   threshold tuning, two-stage (Normal-vs-Abnormal / Warning-vs-Critical) classification. No
   candidate was clearly superior.
4. **Improvement research, round 2 (final):** dedicated Critical-vs-Rest detector combined
   with retrained multiclass logic; ordinal severity classification (cumulative O1/O2 with
   argmax). The Critical-vs-Rest combination achieved the highest raw Critical Recall/F2 of
   any candidate tested but collapsed Warning Recall below the 0.80 guardrail. The Ordinal
   XGBoost Classifier with Argmax Decision Rule was the only candidate, across both rounds, to
   improve Critical Recall, Critical F2, Macro F1, and Balanced Accuracy simultaneously while
   remaining above the Warning guardrail.
5. **Finalization (this task):** both the frozen XGB_03 baseline and the Ordinal candidate
   were independently re-reproduced from the existing classification_v2 harness (exact match
   on all reference metrics). A promotion-readiness gate was evaluated.

## Final Evidence (Development-only, 5-fold GroupKFold by Batch_ID, Test never loaded)

| Metric | XGB_03 (historical baseline) | Ordinal XGBoost Classifier | Change |
|---|---:|---:|---:|
| Accuracy | 0.9558 | 0.9661 | +0.0103 |
| Balanced Accuracy | 0.7250 | 0.7405 | +0.0155 |
| Macro F1 | 0.6453 | 0.6748 | +0.0295 |
| Warning Recall | 0.8565 | 0.8117 | −0.0448 (still ≥ 0.80) |
| Critical Recall | 0.3551 | 0.4348 | +0.0797 |
| Critical Precision | 0.4336 | 0.4027 | −0.0309 |
| Critical F1 | 0.3929 | 0.4181 | +0.0252 |
| Critical F2 | 0.3684 | 0.4280 | +0.0596 |
| False Critical alerts | 64 | 89 | +25 |

Batch-level (6 Critical-containing development batches): **3 improved, 2 unchanged
(both at zero recall before and after), 1 marginally worse** (precision improved on that
batch, partially offsetting the recall dip). No batch was severely regressed — the most
stable batch profile of any candidate tested in either research round.

## Promotion-Readiness Gate — Result: **PASS (11/11)**

| # | Criterion | Result |
|---|---|---|
| 1 | Baseline reproduces | PASS (0.00e+00 diff, all 5 reference metrics) |
| 2 | Ordinal results reproduce | PASS (all 6 gate metrics within 4e-5 of prior citation; see note below) |
| 3 | Warning Recall ≥ 0.80 | PASS (0.8117) |
| 4 | Critical Recall > baseline | PASS (0.4348 > 0.3551) |
| 5 | Critical F2 > baseline | PASS (0.4280 > 0.3684) |
| 6 | Macro F1 > baseline | PASS (0.6748 > 0.6453) |
| 7 | Balanced Accuracy > baseline | PASS (0.7405 > 0.7250) |
| 8 | No dataset changed | PASS (all 7 dataset-lineage files hash-identical) |
| 9 | No batch leakage | PASS (zero overlap, enforced structurally) |
| 10 | Probability reconstruction integrity | PASS (all P≥0; max\|sum−1\|=1.02e-07; argmax matches stored predictions; 580/20,405 = 2.84% raw monotonicity violations, corrected deterministically, documented) |
| 11 | Backend packaging reproduces research predictions | PASS (fresh-process reload: 0 label mismatches / 20,405 rows; max prob. diff 2.98e-08) |

**Note on item 2:** Accuracy reproduces at 0.9661, not the ≈0.9645 informally cited earlier in
the research conversation. This is a correction of an imprecise verbal citation — all 6
gate-relevant metrics (Balanced Accuracy, Macro F1, Warning Recall, Critical
Recall/Precision/F2) and the full confusion matrix match their previously-reported values to
reporting precision. The corrected Accuracy is still higher than XGB_03's.

## Final Decision

**PROMOTED.** The Ordinal XGBoost Classifier with Argmax Decision Rule is now the active
production classifier (`backend/app/ml/component_3_smokehouse/fire_risk_pipeline.pkl`). XGBoost
XGB_03 is retained as the historical baseline, archived with a verified byte-identical copy at
`backend/app/ml/component_3_smokehouse/archive/fire_risk_xgb03_baseline_pipeline.pkl` (SHA-256
`8f375f81b16213e5bde09a6f4519e65db2bb872f4b1f076f4565ae4af6a601aa`, matching the pre-promotion
production hash exactly) together with its metadata and inference contract.

This is **not** a claim of proven Critical-class reliability. Two of six Critical-containing
development batches still produce zero Critical Recall under the promoted model, exactly as
under every other method tested across two full experimentation rounds. The promotion reflects
a measured, development-evidence-based improvement in the overall classification trade-off —
not a resolution of the underlying Critical-class data scarcity (only 6 of 10 development
batches contain any Critical example).

The historical Stage-31 locked-test result for XGB_03 (Warning Recall 0.986, Critical Recall
0.0) remains preserved, unmodified, and is **not** reused or reinterpreted as an evaluation of
the Ordinal classifier. No post-hoc evaluation of the Ordinal classifier on the historical Test
split was performed in this task (default: skip, per protocol — no scientific need was
identified that would justify it).
