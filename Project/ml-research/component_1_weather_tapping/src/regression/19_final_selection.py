"""Stage 19 (final regression selection).

Selects, per regression task, the model with the lowest mean 5-fold CV RMSE on
the **training partition** across baseline + candidates + tuned models. The
selected pipeline is copied to ``.../<task>/final/selected_model.joblib`` with
full selection metadata. The test partition is not loaded.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REG_TASKS, REPORTS, DATASET_PROVENANCE, task_root, role_dir, sha256_file  # noqa: E402

ROLES = ("baseline", "candidates", "tuned")


def _gather(task):
    rows = []
    root = task_root("regression", task)
    for role in ROLES:
        for meta_path in sorted((root / role).glob("*/metadata.json")):
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            model_path = meta_path.parent / "model.joblib"
            rows.append({
                "task": task, "model": m.get("model_name", m["model_family"]), "role": role,
                "cv_MAE_mean": m["cv"]["MAE"]["mean"], "cv_RMSE_mean": m["cv"]["RMSE"]["mean"],
                "cv_R2_mean": m["cv"]["R2"]["mean"],
                "cv_within_tolerance_pct_mean": m["cv"]["within_tolerance_pct"]["mean"],
                "model_class": m["model_class"], "tuned": m["hyperparameter_tuning_performed"],
                "model_path": model_path, "model_sha256": m["model_sha256"],
            })
    return pd.DataFrame(rows)


def main():
    summary = {"project": "RubberSense Component 1", "stage": "19_final_regression_selection",
               "selection_scope": "DEVELOPMENT_ONLY (cross-validation on the training partition)",
               "selection_rule": {"primary": "cv_RMSE_mean ascending",
                                  "tie_breaker_1": "cv_MAE_mean ascending",
                                  "tie_breaker_2": "cv_R2_mean descending"},
               "dataset_provenance": DATASET_PROVENANCE,
               "test_loaded": False, "test_evaluated": False, "tasks": {}}

    for task in REG_TASKS:
        df = _gather(task)
        if df.empty:
            raise SystemExit(f"no candidates found for {task}; run the comparison/tuning stages first")
        df = df.sort_values(["cv_RMSE_mean", "cv_MAE_mean", "cv_R2_mean"],
                            ascending=[True, True, False]).reset_index(drop=True)
        df.to_csv(REPORTS / f"19_{task}_selection_ranking.csv", index=False)
        best = df.iloc[0]
        runner = df.iloc[1] if len(df) > 1 else None
        final_dir = role_dir("regression", task, "final")
        final_path = final_dir / "selected_model.joblib"
        shutil.copyfile(best["model_path"], final_path)
        identical = sha256_file(best["model_path"]) == sha256_file(final_path)
        meta = {
            "stage": "19_final_regression_selection", "task": task,
            "target": REG_TASKS[task][0], "selection_scope": "DEVELOPMENT_ONLY",
            "selected_model_family": best["model"], "selected_role": best["role"],
            "model_class": best["model_class"],
            "selection_rule": summary["selection_rule"],
            "development_cv": {
                "RMSE_mean": float(best["cv_RMSE_mean"]), "MAE_mean": float(best["cv_MAE_mean"]),
                "R2_mean": float(best["cv_R2_mean"]),
                "within_tolerance_pct_mean": float(best["cv_within_tolerance_pct_mean"]),
            },
            "runner_up": None if runner is None else {
                "model": runner["model"], "role": runner["role"],
                "cv_RMSE_mean": float(runner["cv_RMSE_mean"]), "cv_MAE_mean": float(runner["cv_MAE_mean"]),
            },
            "artifact": {"source_path": str(best["model_path"]), "final_path": str(final_path),
                         "source_sha256": sha256_file(best["model_path"]),
                         "final_sha256": sha256_file(final_path), "copy_identical": identical},
            "test_loaded": False, "test_evaluated": False,
            "test_evaluation_stage": 31, "explainability_stage": 32, "export_stage": 33,
            "dataset_provenance": DATASET_PROVENANCE,
            "warnings": [
                "Row-wise split: most test estates also appear in training. Use stage-30 estate-grouped CV for the unseen-estate estimate.",
                "Within-tolerance % is not accuracy.",
            ],
        }
        (final_dir / "selected_model_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        summary["tasks"][task] = {
            "selected_model": best["model"], "selected_role": best["role"],
            "cv_RMSE_mean": float(best["cv_RMSE_mean"]), "cv_MAE_mean": float(best["cv_MAE_mean"]),
            "cv_R2_mean": float(best["cv_R2_mean"]),
            "candidates_considered": int(len(df)),
        }
        print(f"[{task}] selected {best['model']} ({best['role']}) CV RMSE={best['cv_RMSE_mean']:.4f}")

    (REPORTS / "19_regression_selection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Stage 19 final regression selection complete.")


if __name__ == "__main__":
    main()
