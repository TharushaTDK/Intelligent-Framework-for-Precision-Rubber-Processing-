"""Stage 29 (final tapping-advisory selection).

Selects the classifier with the highest mean 5-fold stratified CV macro-F1 on
the **training partition** across baseline + candidates + tuned models. The
selected bundle is copied to ``.../tapping_advisory/final/selected_model.joblib``.
The test partition is not loaded.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLS_TASK, REPORTS, DATASET_PROVENANCE, TARGET_WARNING, task_root, role_dir, sha256_file  # noqa: E402

ROLES = ("baseline", "candidates", "tuned")


def _gather():
    rows = []
    root = task_root("classification", CLS_TASK)
    for role in ROLES:
        for meta_path in sorted((root / role).glob("*/metadata.json")):
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            model_path = meta_path.parent / "model.joblib"
            rows.append({
                "task": CLS_TASK, "model": m.get("model_name", m["model_family"]), "role": role,
                "cv_accuracy_mean": m["cv"]["accuracy"]["mean"],
                "cv_macro_precision_mean": m["cv"]["precision_macro"]["mean"],
                "cv_macro_recall_mean": m["cv"]["recall_macro"]["mean"],
                "cv_macro_f1_mean": m["cv"]["f1_macro"]["mean"],
                "model_class": m["model_class"], "tuned": m["hyperparameter_tuning_performed"],
                "model_path": model_path, "model_sha256": m["model_sha256"],
            })
    return pd.DataFrame(rows)


def main():
    df = _gather()
    if df.empty:
        raise SystemExit("no candidates found; run the comparison/tuning stages first")
    df = df.sort_values(["cv_macro_f1_mean", "cv_macro_recall_mean", "cv_accuracy_mean"],
                        ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS / "29_classification_selection_ranking.csv", index=False)

    best = df.iloc[0]
    runner = df.iloc[1] if len(df) > 1 else None
    final_dir = role_dir("classification", CLS_TASK, "final")
    final_path = final_dir / "selected_model.joblib"
    shutil.copyfile(best["model_path"], final_path)
    identical = sha256_file(best["model_path"]) == sha256_file(final_path)

    meta = {
        "stage": "29_classification_final_selection", "task": CLS_TASK,
        "target": "tapping_suitability_proxy", "selection_scope": "DEVELOPMENT_ONLY",
        "selected_model_family": best["model"], "selected_role": best["role"],
        "model_class": best["model_class"],
        "selection_rule": {"primary": "cv_macro_f1_mean descending",
                           "tie_breaker_1": "cv_macro_recall_mean descending",
                           "tie_breaker_2": "cv_accuracy_mean descending"},
        "development_cv": {"accuracy_mean": float(best["cv_accuracy_mean"]),
                           "macro_precision_mean": float(best["cv_macro_precision_mean"]),
                           "macro_recall_mean": float(best["cv_macro_recall_mean"]),
                           "macro_f1_mean": float(best["cv_macro_f1_mean"])},
        "runner_up": None if runner is None else {
            "model": runner["model"], "role": runner["role"],
            "cv_macro_f1_mean": float(runner["cv_macro_f1_mean"]),
            "cv_accuracy_mean": float(runner["cv_accuracy_mean"]),
        },
        "artifact": {"source_path": str(best["model_path"]), "final_path": str(final_path),
                     "source_sha256": sha256_file(best["model_path"]),
                     "final_sha256": sha256_file(final_path), "copy_identical": identical},
        "test_loaded": False, "test_evaluated": False,
        "test_evaluation_stage": 31, "explainability_stage": 32, "export_stage": 33,
        "dataset_provenance": DATASET_PROVENANCE,
        "target_warning": TARGET_WARNING,
        "warnings": ["Row-wise split: most test estates also appear in training."],
    }
    (final_dir / "selected_model_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    summary = {"project": "RubberSense Component 1", "stage": "29_classification_final_selection",
               "selection_scope": "DEVELOPMENT_ONLY (stratified CV on the training partition)",
               "selection_rule": meta["selection_rule"], "candidates_considered": int(len(df)),
               "tapping_advisory": {"selected_model": best["model"], "selected_role": best["role"],
                                    "cv_macro_f1_mean": float(best["cv_macro_f1_mean"]),
                                    "cv_accuracy_mean": float(best["cv_accuracy_mean"])},
               "dataset_provenance": DATASET_PROVENANCE,
               "test_loaded": False, "test_evaluated": False}
    (REPORTS / "29_classification_selection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[{CLS_TASK}] selected {best['model']} ({best['role']}) CV macro-F1={best['cv_macro_f1_mean']:.4f}")
    print("Stage 29 final classification selection complete.")


if __name__ == "__main__":
    main()
