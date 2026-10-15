# Component 3 — Final Methodology & Limitations

**Smart Layer-Wise Rubber Smokehouse Drying-Time & Fire-Risk Prediction System**
Group J26-DS-340 — "Intelligent Framework for Precision Rubber Processing and Supply Chain Optimization"

**Status of this document:** New documentation artifact only. It does not overwrite any thesis
text, and it does not alter the frozen V1 research results described below. It consolidates the
scientifically correct interpretation established across two read-only methodology audits
(`component3_full_methodology_audit.md`, `component3_stage09_split_robustness_audit.md`) into a
single citable summary.

---

## 1. Dataset Status

- Component 3's model development data (`data/raw/Component3_Dataset.csv`) is a **synthetic /
  simulated development dataset**, generated to represent plausible layer-wise smokehouse drying
  behavior. It is **not** physically field-collected sensor data and must never be described as
  such.
- **24,463 rows**, spanning **12 complete drying batches** across **2 smokehouses** (`SH01`,
  `SH02`; 6 batches each), 3 physical layers per batch (`L1`/`L2`/`L3`), sampled at a nominal
  10-minute interval (with 12 genuine 20-minute packet-loss gaps preserved as-is).
- Frozen raw identity: 17 columns, SHA-256
  `229139d2b4413fe15a8d322023b3b42574f9dccd5df6a91550bc72f05faaa5cb`, independently re-verified
  during both audits and unchanged throughout this repair.

## 2. Evaluation Design

- Splitting is performed at the **complete `Smokehouse_ID + Batch_ID` group level** — individual
  10-minute rows are never treated as independently splittable units. This prevents the same
  physical drying event from leaking across Train/Validation/Test via adjacent rows.
- **Stage 30** evaluates with **5-fold `GroupKFold`** over the 10 development batches (Train +
  Validation combined), refitting preprocessing independently inside every fold and never loading
  the final-test file.
- **Stage 31** evaluates once on the frozen two-batch holdout (`SH01_B03`, `SH02_B04`), using
  preprocessing and model configurations refit on the full Development set only, opened after
  fitting was complete.
- Across two independent audits, **no confirmed model-training leakage** was found: no test row,
  test label, or test-derived statistic was ever used to fit a preprocessor, model, or
  hyperparameter at any stage (04 through 33 inspected directly at the code level).

## 3. Stage-09 Split Disclosure

The Stage-09 `Smokehouse_ID+Batch_ID` → Train/Validation/Test assignment was **manually
constructed with awareness of the per-batch Fire_Risk_Label distribution**, specifically so that
Training retained adequate representation of the rare Critical class and no split ended up with
zero Warning/Critical examples (a real risk independently reproduced during the Stage-09
robustness audit: a purely positional target-blind rule was shown to be capable of producing an
all-Normal Validation set).

**This is not model-training leakage.** It did not provide the model, the preprocessor, or any
hyperparameter-selection step with access to test data or test labels — that was independently
re-verified at the code level in both audits. What it does mean is that the final-test holdout
cannot be described as a **fully blind** holdout with respect to class distribution: the people
who fixed the split had already seen the full dataset's per-batch Critical/Warning counts,
including those of the two batches that became Test. This reduces the strength of a "completely
blind holdout" claim and is disclosed here transparently rather than left implicit. A structural
analysis (the Stage-09 robustness audit) found that this specific outcome — one Critical-containing
batch landing in the 2-batch test set — is also the single most likely result (~53%) under a
purely random, target-blind group split given that only 7 of 12 batches contain any Critical
example at all; the scarcity of Critical-containing batches is substantially a property of the
dataset, not solely of the manual split choice.

## 4. Critical-Class Limitation

The classifier showed unstable Critical-class generalization across unseen batches. Group-based
cross-validation (Stage 30) produced an overall pooled Critical recall of approximately **0.355**,
with fold-level Critical recall varying substantially between held-out batch groups — from **0.0**
(6 Critical rows held out) to **0.90** (30 Critical rows held out) across the 5 folds. On the
locked final holdout (Stage 31), all **18 Critical rows** originated from **one** Critical-containing
batch (`SH02_B04`) and were classified as Warning, resulting in a Critical recall of **0.000**.
Because the final Critical cases represent only one independent batch context, this result should
be interpreted as a **high-uncertainty, batch-specific estimate** rather than a universal measure
of Critical-detection capability. The broader, multi-batch Stage-30 evidence — not the single
Stage-31 number — is the primary basis for characterizing this classifier's Critical-class
behavior; Stage 31 is reported as one additional, fragile, corroborating data point.

## 5. Warning Performance

| Metric | Stage 30 (pooled OOF, GroupKFold) | Stage 31 (locked final test) |
|---|---:|---:|
| Warning Recall | ≈ 0.8565 | ≈ 0.9861 |

Both figures come from the frozen, unmodified Stage-30/31 result files and were not recomputed or
altered by this repair.

## 6. Regression Performance

Both results are on the synthetic/simulated development data described in §1 and must not be
presented as validated field accuracy.

| Metric | Stage 30 (pooled OOF, GroupKFold) | Stage 31 (locked final test) |
|---|---:|---:|
| MAE | 3.1283 h | 3.756103 h |
| RMSE | 3.9558 h | 4.299451 h |
| R² | 0.9860 | 0.983540 |

`Elapsed_Drying_Hours` is a strong predictor in this synthetic target construction (confirmed by
Stage-32 SHAP as the dominant regression feature); the very high R² should be read in that light
rather than as proof of transferable real-world precision.

## 7. External Validation Requirement

A stronger future validation should use **genuinely new unseen batches** — ideally from additional
smokehouse runs not present in this 12-batch dataset — and, where feasible, **real
operational/field/expert-validated observations**. This has **not yet been completed**. Once the
batches in this dataset have been inspected (as they now have been, across two audits and this
repair), any new split drawn from the same 12 batches cannot be described as a completely untouched
holdout (see the Stage-09 robustness audit, Task 5, point 6) — a genuinely blind evaluation
requires data collected and held out before inspection.

## 8. Safety Interpretation

The exported classifier (`backend/app/ml/component_3_smokehouse/fire_risk_pipeline.pkl`) is a
**research/prototype decision-support artifact**. It is **not** a certified standalone fire alarm
and must not be deployed or described as one. This is consistent with the existing safety language
already present in `component3_inference_contract.json`
(`"safety_limitation": "Locked final-test Critical recall was 0.0... must NOT be treated as a
proven standalone Critical fire alarm."`), which this document reaffirms rather than supersedes.

---

## Provenance

This document synthesizes, without altering any figures, the findings of:
- `component3_full_methodology_audit.md` (first audit — Stages 04–33 code-level review)
- `component3_stage09_split_robustness_audit.md` (second audit — Stage-09 split forensics,
  Critical-holdout reliability, target-blind alternative-split simulation, Stage-30 context)
- The frozen `outputs/reports/30_*`, `31_*`, `32_*`, `33_*` result files (unmodified by this
  repair; see the accompanying change report for hash verification).
