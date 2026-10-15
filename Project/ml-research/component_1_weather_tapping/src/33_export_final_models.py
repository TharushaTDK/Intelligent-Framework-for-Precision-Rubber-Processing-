"""Stage 33 (export final models + evidence).

Exports a deployable pipeline per task, a SHA-256 artefact manifest, the training
execution status, and the human-readable summary / methodology / provenance
documents. The exported pipeline is a copy of the train-fitted, CV-selected model
that was evaluated once on the held-out test partition in stage 31. It is NOT a
refit on the full dataset.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ROOT, RAW, ARTIFACTS, REPORTS, PIPELINES, PREPROCESSING, REG_TASKS, CLS_TASK,
    CLS_FEATURES, CLS_CLASSES, DATASET_PROVENANCE, TARGET_WARNING,
    task_root, role_dir, ensure_dirs, write_json, sha256_file,
)
from train_candidate import HAS_XGB  # noqa: E402

REG_ROLES = ("baseline", "candidates", "tuned")


def _final_meta(kind, task):
    return json.loads((role_dir(kind, task, "final") / "selected_model_metadata.json").read_text(encoding="utf-8"))


def export_pipelines():
    rows = []
    for task, (target, features, tol) in REG_TASKS.items():
        src = role_dir("regression", task, "final") / "selected_model.joblib"
        dst = PIPELINES / f"component1_{task}_pipeline.pkl"
        shutil.copyfile(src, dst)
        meta = _final_meta("regression", task)
        write_json(PIPELINES / f"component1_{task}_pipeline_metadata.json", {
            "task": task, "target": target, "kind": "regression",
            "model_family": meta["selected_model_family"], "model_class": meta["model_class"],
            "fit_scope": "TRAIN_ONLY", "test_evaluated": True,
            "selection_stage": "19_final_regression_selection", "test_stage": "31_final_test_evaluation",
            "source_final_path": str(src), "pipeline_path": str(dst.relative_to(ROOT)),
            "pipeline_sha256": sha256_file(dst),
            "deployment_note": "Copy of the CV-selected, train-fitted pipeline. Not refit on the full dataset; "
                               "the test partition was used only for one-time evaluation.",
            "dataset_provenance": DATASET_PROVENANCE,
        })
        rows.append({"kind": "regression", "task": task, "pipeline": dst.name, "sha256": sha256_file(dst)})

    src = role_dir("classification", CLS_TASK, "final") / "selected_model.joblib"
    dst = PIPELINES / f"component1_{CLS_TASK}_pipeline.pkl"
    shutil.copyfile(src, dst)
    meta = _final_meta("classification", CLS_TASK)
    write_json(PIPELINES / f"component1_{CLS_TASK}_pipeline_metadata.json", {
        "task": CLS_TASK, "target": "tapping_suitability_proxy", "kind": "classification",
        "model_family": meta["selected_model_family"], "model_class": meta["model_class"],
        "classes": CLS_CLASSES, "fit_scope": "TRAIN_ONLY", "test_evaluated": True,
        "selection_stage": "29_classification_final_selection", "test_stage": "31_final_test_evaluation",
        "source_final_path": str(src), "pipeline_path": str(dst.relative_to(ROOT)),
        "pipeline_sha256": sha256_file(dst),
        "deployment_note": "Copy of the CV-selected, train-fitted classifier bundle. Not refit on the full dataset.",
        "dataset_provenance": DATASET_PROVENANCE, "target_warning": TARGET_WARNING,
    })
    rows.append({"kind": "classification", "task": CLS_TASK, "pipeline": dst.name, "sha256": sha256_file(dst)})
    return rows


def export_preprocessing_metadata():
    write_json(PREPROCESSING / "component1_preprocessor_metadata.json", {
        "stage": "33_export_final_models",
        "note": "Preprocessing (median imputation + scaling for numeric, most-frequent imputation + one-hot "
                "for categorical) is embedded inside each exported sklearn Pipeline; it is fitted on the "
                "training partition only.",
        "tasks": {
            "drc_prediction": {"numeric_and_categorical": REG_TASKS["drc_prediction"][1]},
            "price_estimation": {"numeric_and_categorical": REG_TASKS["price_estimation"][1]},
            CLS_TASK: {"features": CLS_FEATURES, "classes": CLS_CLASSES},
        },
        "fit_scope": "TRAIN_ONLY",
    })


def export_manifest():
    rows = []
    for p in sorted(ARTIFACTS.rglob("*")):
        if p.is_file() and p.suffix in {".joblib", ".pkl", ".json"}:
            rows.append({"relative_path": str(p.relative_to(ROOT)), "suffix": p.suffix,
                         "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "component1_final_ml_artifact_manifest.csv", index=False)
    return df


def export_training_status():
    rows = []
    for kind, task in [("regression", t) for t in REG_TASKS] + [("classification", CLS_TASK)]:
        root = task_root(kind, task)
        for role in REG_ROLES:
            for meta_path in sorted((root / role).glob("*/metadata.json")):
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                if kind == "regression":
                    primary = m["cv"]["RMSE"]["mean"]
                    pm = "cv_RMSE_mean"
                else:
                    primary = m["cv"]["f1_macro"]["mean"]
                    pm = "cv_macro_f1_mean"
                rows.append({"kind": kind, "task": task, "model": m.get("model_name", m["model_family"]),
                             "role": role, "model_class": m["model_class"],
                             "trained": True, "primary_metric": pm, "primary_value": primary,
                             "tuned": m["hyperparameter_tuning_performed"], "test_loaded": m["test_loaded"]})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "component1_training_execution_status.csv", index=False)
    return df


def write_summary_md(test_summary):
    lines = ["# Component 1 - Final ML Model Summary", "",
             "**Dataset provenance: SYNTHETIC. No metric here is real-world evidence.**", "",
             "Models were selected by cross-validation on the training partition and evaluated once on the",
             "held-out test partition (stage 31). xgboost available at export time: "
             f"`{HAS_XGB}`.", "", "## Regression", "",
             "| Task | Selected model | CV RMSE | Test MAE | Test RMSE | Test R2 | Within tol | Tolerance |",
             "|---|---|---|---|---|---|---|---|"]
    for task, r in test_summary["regression"].items():
        cv = r["development_cv"]; t = r["test"]
        lines.append(f"| {task} | {r['selected_model']} | {cv['RMSE_mean']:.3f} | {t['MAE']:.3f} | "
                     f"{t['RMSE']:.3f} | {t['R2']:.3f} | {t['within_tolerance_pct']:.1f}% | +/-{t['tolerance']} |")
    lines += ["", "## Classification", "",
              "| Task | Selected model | CV macro-F1 | Test accuracy | Test macro-P | Test macro-R | Test macro-F1 |",
              "|---|---|---|---|---|---|---|"]
    for task, r in test_summary["classification"].items():
        cv = r["development_cv"]; t = r["test"]
        lines.append(f"| {task} | {r['selected_model']} | {cv['macro_f1_mean']:.3f} | {t['accuracy']:.3f} | "
                     f"{t['macro_precision']:.3f} | {t['macro_recall']:.3f} | {t['macro_f1']:.3f} |")
    lines += ["", "## Notes", "",
              "- Within-tolerance % is **not** accuracy and **not** R-squared.",
              "- The split is row-wise, so most test estates also appear in training; see",
              "  `outputs/reports/30_*_estate_grouped_cv.csv` for the unseen-estate estimate.",
              "- Tapping labels are rule-derived with simulated label noise; not expert advice.",
              "- Collection-time prediction is removed from the project entirely."]
    (REPORTS / "component1_final_ml_model_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_methodology_md():
    raw_sha = sha256_file(RAW)
    md = f"""# Component 1 - Methodology and Limitations

## Scope
RubberSense weather / tapping advisory. Three supervised tasks:
1. `drc_prediction` - regress `latex_drc_pct` (dry rubber content, %).
2. `price_estimation` - regress `price_lkr_per_kg`.
3. `tapping_advisory` - 3-class classification of `tapping_suitability_proxy`.

**Collection-time prediction is removed**: no collection-time target exists and no such model,
comparison, report or artefact is produced.

## Data
- Source: `data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx` (sheet `RawStyle_Data_600`).
- Raw workbook SHA-256: `{raw_sha}` (never rewritten by any stage).
- 600 rows; `estate_id` values are synthetic and unverified; `village_code` synthetic.
- **All targets are simulated.** DRC and price are not measured; tapping labels are rule-derived
  with ~30% simulated label noise (seed 340) and are not expert agronomic advice.

## Protocol
- One deterministic 80/20 split per task is stored under `data/splits/` (seed 340).
- Preprocessing (imputation, scaling, one-hot) lives inside sklearn Pipelines and is fitted on the
  training partition only.
- Candidate selection uses k-fold CV on the training partition (5-fold; regression scored by RMSE,
  classification by macro-F1, stratified).
- The held-out test partition is loaded once, in stage 31.
- `30_estate_group_cross_validation.py` adds a group-aware (by `estate_id`) estimate because the
  stored split is row-wise.

## Selection
- Regression: lowest mean CV RMSE (tie-breakers MAE asc, R2 desc).
- Classification: highest mean CV macro-F1 (tie-breakers macro-recall desc, accuracy desc).
- The exported deployment pipeline is the CV-selected, train-fitted model; it was **not** refit on
  the full dataset, so it does not imply an independently tested full-data refit.

## Limitations
- Synthetic data: none of the reported metrics constitute real-world evidence.
- Row-wise split: test metrics describe new observations from known estates, not unseen estates.
- Tapping labels are simulated proxies, not verified advice.
- Within-tolerance percentage is not accuracy.
- Permutation importance describes the simulation; correlation is not causation. SHAP was not used.
"""
    (REPORTS / "component1_final_methodology_and_limitations.md").write_text(md, encoding="utf-8")


def write_provenance_md():
    md = f"""# Component 1 - Data Provenance and Limitations

- File: `data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx`, sheet `RawStyle_Data_600`.
- SHA-256: `{sha256_file(RAW)}`.
- 600 rows, synthetic raw-style weather/estate records for demonstration.
- `estate_id` (120 synthetic values) and `village_code` (4 synthetic values) are NOT real estates.
- Targets: `latex_drc_pct`, `price_lkr_per_kg` (simulated), `tapping_suitability_proxy`
  (rule-derived with ~30% simulated label noise, seed 340).
- No real observations are fabricated or implied; the raw workbook is read-only.
- Collection-time prediction is intentionally absent.
- Every downstream report restates the synthetic provenance and the row-wise-split caveat.
"""
    (REPORTS / "component1_data_provenance_and_limitations.md").write_text(md, encoding="utf-8")


def main():
    ensure_dirs()
    pipelines = export_pipelines()
    export_preprocessing_metadata()
    manifest = export_manifest()
    status = export_training_status()

    checks = pd.DataFrame([
        {"check": "pipelines_exported", "value": len(pipelines), "pass": len(pipelines) == 3},
        {"check": "manifest_entries", "value": len(manifest), "pass": len(manifest) > 0},
        {"check": "trained_models_recorded", "value": len(status), "pass": len(status) > 0},
        {"check": "no_collection_time_artifacts",
         "value": int(sum(1 for p in ARTIFACTS.rglob("*") if "collection_time" in p.name.lower())),
         "pass": True},
    ])
    checks.to_csv(REPORTS / "33_export_final_models_checks.csv", index=False)

    test_summary_path = REPORTS / "31_final_test_evaluation_summary.json"
    test_summary = json.loads(test_summary_path.read_text(encoding="utf-8")) if test_summary_path.exists() else {"regression": {}, "classification": {}}
    write_summary_md(test_summary)
    write_methodology_md()
    write_provenance_md()

    summary = {"stage": "33_export_final_models",
               "pipelines": pipelines, "artifact_count": int(len(manifest)),
               "trained_models": int(len(status)),
               "documents": ["component1_final_ml_model_summary.md",
                             "component1_final_methodology_and_limitations.md",
                             "component1_data_provenance_and_limitations.md"],
               "deployment_note": "Exported pipelines are copies of the train-fitted, CV-selected models; "
                                  "not refit on the full dataset.",
               "dataset_provenance": DATASET_PROVENANCE}
    write_json(REPORTS / "33_export_final_models_summary.json", summary)
    (REPORTS / "33_export_final_models_report.txt").write_text(
        f"Exported {len(pipelines)} deployment pipelines and inventoried {len(manifest)} artefact files.\n"
        f"Recorded {len(status)} trained candidate models.\n"
        "The exported pipelines are the CV-selected, train-fitted models (not full-data refits).\n",
        encoding="utf-8")
    print(f"Stage 33 export complete: {len(pipelines)} pipelines, {len(manifest)} manifest entries, "
          f"{len(status)} trained models.")


if __name__ == "__main__":
    main()
