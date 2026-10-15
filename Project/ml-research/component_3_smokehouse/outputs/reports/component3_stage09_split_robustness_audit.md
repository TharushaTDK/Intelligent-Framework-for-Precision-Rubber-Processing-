# Component 3 — Stage 09 Split Robustness & Rare-Class Audit (Second Audit)

**Scope:** Read-only forensic follow-up to the first methodology audit, focused on Stage 09's
batch-to-split assignment and Critical-class holdout reliability.
**Method note:** All class-count statistics below are either (a) taken directly from
already-existing, frozen Stage-09/30/31 reports, independently re-derived from
`data/processed/Component3_Dataset_FeatureEngineered.csv` for verification, or (b) computed
in a standalone scratch script that reads only the feature-engineered CSV and writes only to a
local scratch directory outside the project. **No project file (source, data, or model) was
modified.** No model was retrained. The raw dataset and Stage-09 split were not touched.

---

## TASK 1 — Stage 09 Split Forensics

### 1.1 Per-Batch Master Table

Row-level counts (10-minute carried-forward `Fire_Risk_Label`) were independently recomputed
from `Component3_Dataset_FeatureEngineered.csv` and cross-checked against the already-frozen
`30_classification_group_cv_by_batch.csv` and `31_classification_final_test_by_batch.csv` —
they match exactly (no drift). Hourly-assessment-level counts (unique keeper/expert events,
`Timestamp == Fire_Risk_Assessment_Timestamp`) were independently recomputed and match
`09_batch_split_manifest.csv` exactly.

| Smokehouse | Batch | Rows | Normal | Warning | Critical | Crit % of batch rows | Hourly Normal | Hourly Warning | Hourly Critical | Stage-09 Split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SH01 | B01 | 2,267 | 2,267 | 0 | 0 | 0.00% | 381 | 0 | 0 | Train |
| SH01 | B02 | 1,899 | 1,731 | 168 | 0 | 0.00% | 291 | 28 | 0 | Validation |
| SH01 | B03 | 2,224 | 2,224 | 0 | 0 | 0.00% | 372 | 0 | 0 | **Test** |
| SH01 | B04 | 1,763 | 1,325 | 372 | 66 | 3.74% | 223 | 62 | 11 | Train |
| SH01 | B05 | 2,271 | 2,271 | 0 | 0 | 0.00% | 381 | 0 | 0 | Train |
| SH01 | B06 | 2,021 | 1,985 | 24 | 12 | 0.59% | 332 | 4 | 2 | Train |
| SH02 | B01 | 1,931 | 1,889 | 36 | 6 | 0.31% | 316 | 6 | 1 | Train |
| SH02 | B02 | 2,260 | 2,242 | 0 | 18 | 0.80% | 376 | 0 | 3 | Validation |
| SH02 | B03 | 1,992 | 1,944 | 30 | 18 | 0.90% | 325 | 5 | 3 | Train |
| SH02 | B04 | 1,834 | 1,528 | 288 | 18 | 0.98% | 256 | 48 | 3 | **Test** |
| SH02 | B05 | 2,082 | 2,082 | 0 | 0 | 0.00% | 348 | 0 | 0 | Train |
| SH02 | B06 | 1,919 | 1,883 | 18 | 18 | 0.94% | 315 | 3 | 3 | Train |
| **Total** | | **24,463** | **23,371** | **936** | **156** | | | | | 12 batches |

Column totals reconcile exactly with the known frozen dataset identity (Normal 23,371 / Warning
936 / Critical 156; 24,463 rows), confirming no drift between the feature-engineered file and
the earlier-audited preprocessed/raw files.

### 1.2 How TRAIN_BATCHES / VALIDATION_BATCHES / TEST_BATCHES were actually chosen

Reading `src/09_batch_aware_split.py` lines 24–53 and 195–221 directly (not paraphrased):

> "Why fixed group-stratified rather than purely chronological? A purely chronological split
> would put both B05 batches into validation, but both are all-Normal at the unique hourly
> fire-risk assessment level. That would make validation unsuitable for Warning/Critical
> evaluation... TEST deliberately includes: SH01_B03 → long drying regime, SH02_B04 → shorter
> drying regime + substantial Warning... TRAIN retains the majority of Critical assessment
> events."

This is confirmed quantitatively in §1.1: Train holds 5 of 7 Critical-containing batches (120 of
156 Critical rows, 76.9%); Validation holds 1 (18 rows); Test holds 1 (18 rows). The three sets
are **hand-picked, hard-coded `Batch_ID` literals** — there is no randomization, hashing, or
`sklearn` splitter call anywhere in `09_batch_aware_split.py`. The assignment was made by a
person who had already seen the full-dataset per-batch Fire-Risk distribution (and, per the
Stage-08 report, the Stage-05/06 EDA) before freezing it.

### 1.3 Classification: **D — Manually target-aware grouped split**

| Option | Verdict | Why |
|---|---|---|
| A. Random grouped split | No | No random seed, no shuffler; assignment is hard-coded literals. |
| B. Target-blind deterministic grouped split | No | The code's own comments justify the choice by citing per-batch Warning/Critical counts. |
| C. Grouped stratified split (formal/algorithmic) | No | There is no stratification algorithm (e.g. no iterative-stratification library, no "match target proportions" routine) — it is manual, not formulaic. |
| **D. Manually target-aware grouped split** | **Yes** | A human fixed the batch→split assignment after inspecting the batches' Fire_Risk_Label distribution, explicitly to (a) avoid an all-Normal Validation batch and (b) keep most Critical events in Train. |

### 1.4 Distinguishing what this is — and is not

This distinction matters and should not be collapsed into one label:

- **Model-training leakage: NOT PRESENT.** No test row, test label, or test statistic was ever
  used to fit a preprocessor, model, or hyperparameter. Stage 10 never opens `Component3_Test.csv`;
  Stages 11–29 only see Train/Validation; Stage 31 refits on Development only before opening Test
  once. This was independently verified in the first audit and nothing here contradicts it.
- **Test-set tuning: NOT PRESENT.** Stage 31's report states no reselection/tuning/threshold
  change occurred after Test was opened, and Stage 32/33 only reproduce — never alter — the
  Stage-31 predictions.
- **Grouped stratification: PARTIALLY, INFORMALLY.** The intent (ensure every split has some
  Warning/Critical representation) is a legitimate, common goal in imbalanced grouped-data
  research. What is non-standard is that it was done **by hand after inspecting the full target
  distribution**, rather than via a disclosed, pre-registered, or algorithmic stratification rule
  that could be independently reproduced without having seen the labels.
- **Post-hoc test awareness: PRESENT.** By the time the split was frozen, the researcher had
  already seen the per-batch Critical/Warning counts for all 12 batches, including the two that
  became Test. This is the central finding — see Task 3 for how much this plausibly mattered in
  practice.
- **Weak/unstable rare-class representation: PRESENT, AND STRUCTURAL.** Only 7 of 12 batches
  contain any Critical row at all (§2). With a mandatory 2-batch test size, *any* splitting
  strategy — blind or not — was always going to leave Test with at most 1–2 Critical-containing
  batches out of a pool of 7. This is a property of the dataset's rarity, not only of Stage 09's
  specific choice (quantified in Task 3).

**Do not overstate:** this is a disclosure/framing problem about how "blind" the holdout is, not
evidence that the model was trained on test data or that reported Stage-31 numbers were fabricated
or manipulated after the fact.

---

## TASK 2 — Critical-Class Holdout Reliability

| Quantity | Value |
|---|---|
| Critical rows per batch (row-level) | SH01_B04=66, SH01_B06=12, SH02_B01=6, SH02_B03=18, SH02_B06=18, SH02_B02=18, SH02_B04=18; all other 5 batches = 0 |
| Critical-containing batches in full dataset | **7 / 12** |
| Critical-containing batches in Train | **5** (SH01_B04, SH01_B06, SH02_B01, SH02_B03, SH02_B06) |
| Critical-containing batches in Validation | **1** (SH02_B02) |
| Critical-containing batches in Test | **1** (SH02_B04) |
| Stage-31 Critical test rows | **18** (confirmed via `31_classification_final_test_metrics.csv` confusion matrix: Actual_Critical row = [0, 18, 0]) |
| Independent batches those 18 rows came from | **1** (`SH02_B04` only — `31_classification_final_test_by_batch.csv` shows `SH01_B03` has `Critical_Support=0`) |

### Why this makes the reported 0.0 Critical recall statistically fragile

- The 18 Critical test rows are **not 18 independent trials**. They are 10-minute carried-forward
  labels from what the manifest shows is **3 unique hourly keeper-assessment events**, all inside
  one physical drying batch (`SH02_B04`), observed on consecutive/nearby timestamps from the same
  layer/smokehouse under one continuous thermal episode. Rows within that window are highly
  autocorrelated (same underlying physical event, same sensor drift, same batch-specific
  conditions) — they do not carry 18 independent bits of information about "can this model detect
  Critical conditions."
- **Effective independent sample size for the Critical class in Stage 31 is closer to 1 batch**
  (one physical regime, one set of environmental idiosyncrasies) than to 18 i.i.d. rows. A 0%
  recall computed from one batch's worth of Critical examples is a single realization of a
  batch-level random variable, not a stable population estimate — any confidence interval
  constructed by treating the 18 rows as independent Bernoulli trials would understate the true
  uncertainty.
- Stage 30's own GroupKFold evidence (Task 4) shows Critical recall swinging from 0.0 to 0.90
  depending on *which* batch is held out — direct, measured confirmation that batch identity, not
  just "the model," drives a large share of the observed variance in this metric.

**What this evidence does *not* establish:** it does not prove that the 0.0 Critical recall would
have been different with a different test batch, nor that the classifier is secretly competent at
detecting Critical conditions. It only establishes that **a point estimate built from one batch's
Critical examples carries much higher uncertainty than its raw row-count (n=18) suggests**, and
that the true generalization performance on Critical conditions should be read primarily from
Stage 30's multi-batch pooled evidence, with Stage 31 treated as one additional (and fragile)
single-batch data point rather than the definitive number.

---

## TASK 3 — Target-Blind Alternative Split Analysis (diagnostic only — no split was changed)

### 3.1 Structural context: how likely is "1 Critical batch in Test" by pure chance?

Treating the 2 Test batches as drawn uniformly at random from the 12 batches (7 of which contain
Critical rows), independent of any Stage-09 decision:

| Outcome | Probability |
|---|---:|
| Test contains 0 Critical-containing batches | 15.15% |
| **Test contains exactly 1 Critical-containing batch (the actual Stage-09 outcome)** | **53.03%** |
| Test contains 2 Critical-containing batches | 31.82% |

**The actual Stage-09 test composition (1 Critical-containing batch) is the single most likely
outcome under a purely random, target-blind draw — it is not an unusual or extreme result.**

### 3.2 Five target-blind candidate splits (group identity only — labels never inspected during assignment)

Each candidate assigns the 12 `(Smokehouse_ID, Batch_ID)` groups to Train/Validation/Test
(8/2/2) using only identifiers (a stable hash of the batch key, `sklearn.GroupShuffleSplit` on
`Batch_ID` with different seeds, or a fixed positional rule). Class counts below were computed
**after** assignment, purely for reporting — they played no role in deciding the assignment.

| Candidate | Rule | Test batches | Test rows | Test Critical rows | Test Critical-containing batches | Validation Critical-containing batches |
|---|---|---|---:|---:|---:|---:|
| A | SHA-256 hash of batch key, sorted | SH01_B02, SH01_B06 | 3,920 | 12 | 1 | 1 |
| B (seed 42) | `GroupShuffleSplit`, seed=42 | SH02_B04, SH02_B05 | 3,916 | 18 | 1 | 1 |
| B (seed 7) | `GroupShuffleSplit`, seed=7 | SH02_B02, SH02_B05 | 4,342 | 18 | 1 | 2 |
| B (seed 123) | `GroupShuffleSplit`, seed=123 | SH01_B01, SH01_B06 | 4,288 | 12 | 1 | 1 |
| C | Positional rule: per-smokehouse batch #5→Val, #6→Test | SH01_B06, SH02_B06 | 3,940 | 30 | **2** | **0** |
| — | **Actual Stage-09 (for comparison)** | SH01_B03, SH02_B04 | 4,058 | 18 | 1 | 1 |

(Full per-split Train/Validation/Test row and class breakdowns are in
`outputs/reports/component3_stage09_split_candidates.csv`, generated alongside this report.)

### 3.3 Interpretation

- **Every single target-blind candidate produced a test set with at least 1 Critical-containing
  batch** — none produced the "0 Critical batches in test" scenario that the hypergeometric
  analysis says is still a 15% possibility. In that narrow sense, the actual Stage-09 test is not
  worse than typical blind alternatives.
- **Candidate C independently reproduces the exact failure mode Stage 09's own comments warned
  about**: its Validation split (`SH01_B05`, `SH02_B05`) ends up with **zero** Critical (and zero
  Warning) rows — an all-Normal validation set, precisely the scenario the Stage-09 author said
  they were trying to avoid by rejecting "purely chronological" splitting. This independently
  corroborates that the concern driving the manual intervention was a real, reproducible risk of
  this dataset's group structure — not a post-hoc excuse.
- **No candidate matched Stage-09's specific choice of concentrating 77% of all Critical events
  into Train.** The blind candidates' Train Critical shares ranged roughly 67–100% of total
  Critical rows depending on which batches landed in Test/Validation — overlapping with, but not
  identical to, the actual 76.9%. This means the *qualitative* outcome (thin Critical
  representation in a 2-batch test) was likely regardless of blindness, but the *specific* 77%
  figure was a chosen, not emergent, number.
- **Bottom line:** the Stage-09 test's Critical fragility is substantially a **structural property
  of this dataset** (only 7 of 12 batches ever see a Critical condition, and a 2-batch test is a
  mandatory design constraint), not primarily an artifact manufactured by the manual
  intervention. The manual intervention changed *which* batch ended up in Test and *how much*
  of the total Critical signal went to Train, but it did not create the underlying scarcity.

**No new split was adopted. This section is diagnostic only, per your instruction.**

---

## TASK 4 — Stage 30 Context

### 4.1 Pooled OOF metrics (already-frozen Stage-30 results, not rerun)

**Regression (Gradient_Boosting / GB_02), pooled OOF, 5-fold GroupKFold, 20,405 development rows:**

| MAE | RMSE | R² |
|---:|---:|---:|
| 3.1283 h | 3.9558 h | 0.9860 |

**Classification (XGBoost / XGB_03), pooled OOF:**

| Warning Recall | Critical Recall | Macro F1 | Balanced Accuracy |
|---:|---:|---:|---:|
| 0.8565 | 0.3551 | 0.6453 | 0.7250 |

### 4.2 Fold-level Critical support and recall (5-fold GroupKFold)

| Fold | Validation batches | Critical support | Critical recall | Warning recall |
|---|---|---:|---:|---:|
| 1 | SH01_B04, SH01_B05 | 66 | 0.2576 | 0.8844 |
| 2 | SH01_B01, SH01_B02 | 0 | — (no Critical rows in fold) | 0.8988 |
| 3 | SH02_B02, SH02_B06 | 36 | 0.1389 | 0.7222 |
| 4 | SH02_B01, SH02_B05 | 6 | 0.0000 | 0.9167 |
| 5 | SH01_B06, SH02_B03 | 30 | 0.9000 | 0.5370 |

### 4.3 How much confidence should Stage 30 carry relative to Stage 31?

- Stage 30 pools evidence across **all 10 development batches** via 5 independent held-out folds,
  with preprocessing refit fold-by-fold and zero batch overlap (`Batch-overlap folds: 0`,
  independently re-confirmed in the first audit). It is the broader-based, lower-variance estimate
  of the two.
- Stage 31 is a **single realization**: one fixed partition, one 2-batch test set, with only one
  of those two batches containing any Critical examples.
- **Evidence Stage 30 provides that the Critical problem is broader than one batch:** fold-level
  Critical recall ranges from **0.0 (fold 4, support=6) to 0.90 (fold 5, support=30)** — a swing of
  90 percentage points depending purely on which batches were held out that fold. This is direct,
  measured evidence that Critical-class generalization is highly **batch/regime-dependent across
  the whole development set**, not a phenomenon confined to the specific Stage-31 test batch. Low
  recall is not unique to `SH02_B04`/Stage-31 — fold 4 (batches `SH02_B01`, `SH02_B05`) also shows
  0.0 Critical recall, and fold 3 shows a comparatively weak 0.139.
- At the same time, the pooled Stage-30 Critical recall (0.355) is **substantially higher** than
  Stage-31's 0.0, and fold 5 shows the model *can* reach 0.90 recall on some held-out batches. This
  means the evidence is mixed, not uniformly pessimistic: the classifier's Critical-detection
  ability is real but highly inconsistent across physical batches, and Stage 31 happened to land on
  one of the harder cases (consistent with, though not proof of, fold 4's similarly poor 0.0
  result).

**Conclusion for Task 4 (stated at the appropriate confidence level):** Stage 30 provides genuine
evidence that Critical-class unreliability is a broader, batch-dependent pattern across the
development set — it is not solely a Stage-31 artifact. It does **not** prove that Stage 31's
specific 0.0 was inevitable, only that 0.0 (and worse) are recurring, plausible outcomes for this
classifier on held-out batches in general.

---

## TASK 5 — Recommendation

**1. Is the current Stage-31 test technically invalid?**
**NO.** No model-fitting leakage occurred; Development/Test batch and observation-key overlap are
independently verified as zero at every stage. The test is methodologically *valid* as a leakage-free
holdout. It is, however, **not a blind holdout** with respect to how its batches were chosen (Task 1),
and its Critical-class estimate specifically carries very high uncertainty (Task 2) because it rests
on a single batch. "Valid" and "fully representative/low-variance" are different properties — Stage 31
has the first but only partially the second for the Critical class.

**2. Is there confirmed leakage into model training?**
**NO.** Re-confirmed in this audit: Stage 10 never reads `Component3_Test.csv`'s contents; Stages
11–29 only ever see Train/Validation; Stage 31 refits on Development only and opens Test exactly
once, after fitting. No target value or test row entered any fitting step at any stage.

**3. Is the Critical holdout statistically weak?**
**YES.** 18 test rows reduce to effectively 1 independent batch / ~3 correlated hourly events. This
is a small, clustered, non-i.i.d. sample for a rare class, and §4 shows per-batch/per-fold Critical
recall swinging between 0.0 and 0.90 elsewhere in the same dataset — strong corroborating evidence
that single-batch Critical estimates are inherently high-variance here.

**4. Is rerunning Stage 09–33 scientifically REQUIRED?**
**NO, not required.** The current results are valid evidence of what the frozen pipeline produces
and contain no leakage; they can be reported as-is with the correct caveats (as recommended in the
first audit and reinforced in Task 2 here). Task 3 shows that target-blind alternatives would very
likely face a similar — not necessarily better — Critical-representation problem in Test, so a rerun
is not guaranteed to produce a materially more informative Critical-class estimate; it would mainly
improve the *disclosure/blindness* property of the holdout, not necessarily its statistical power for
the rare class.

**5. Would a corrected Version 2 methodology be beneficial?**
**YES, for disclosure/rigor, with tempered expectations.** A Version 2 with a pre-registered,
target-blind (or formally/algorithmically stratified and disclosed-as-such) batch assignment would
let the project state "the test set was chosen without consulting the labels," which is a stronger
methodological claim than can be made today. Task 3's simulations show this would not necessarily
fix the rare-class sample-size problem (several candidates still land near 1 Critical-containing
test batch, and achieving better Critical coverage in Test would itself require *some* target
awareness, reintroducing the same tension) — so Version 2 should be framed as a rigor/transparency
improvement, not a fix guaranteed to raise Critical recall.

**6. If Version 2 is created using the same dataset, can a newly chosen holdout still be called
completely untouched?**
**NO, not completely — with explanation.** The dataset's full per-batch Fire-Risk distribution has
now been seen, documented, and published in two audit reports (this one and the first). Any new
split chosen from the same 12 batches is being chosen by people who already know which batches
contain Critical/Warning events, even if the *selection rule* for Version 2 is made mechanical/blind
(e.g., hash- or seed-based). This is an irreducible property of re-splitting a fixed, already-analyzed
dataset — true blindness only existed, if at all, before Stage 05's EDA ran. The honest framing for
Version 2 is "a disclosed, mechanical, reproducible rule applied to known groups," not "a blind
holdout" in the strongest sense. A truly unseen test would require new data collection (consistent
with the first audit's note that field validation would need new unseen data).

**7. What is the safest next action?**
Do not rerun any ML stage yet. Recommended order, each requiring your separate sign-off:
   a. Update the thesis/report narrative for the existing Stage-31 Critical-recall limitation using
      the precise framing established in Task 2 (0/18 rows, 1 independent batch, corroborated by
      Stage-30 fold-level swings of 0.0–0.90) rather than an unqualified "the model detects 0% of
      Critical cases."
   b. Decide, as a separate and explicit decision, whether the value of a disclosed/mechanical
      Version 2 split (point 5) justifies the cost of rerunning Stages 09–33 — this audit shows that
      decision should be made for **methodological transparency**, not in expectation of a better
      Critical-recall number.
   c. Continue treating Stage 30's pooled and fold-level results as the primary evidence base for
      Critical-class claims, with Stage 31 cited as a single corroborating (fragile) data point.
   d. Low-priority items from the first audit (fixing `30_cv_data_integrity_audit.py`'s string/
      numeric bug, resolving the empty `data/validated/`) remain open and unaffected by this audit.
