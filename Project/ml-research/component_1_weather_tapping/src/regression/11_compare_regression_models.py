"""Stage 17 (comparison): rank regression candidates by cross-validation on the
training partition for every regression task.

Ensures each declared family has been fitted (via :mod:`train_candidate`), then
aggregates the CV metrics stored in each candidate metadata JSON into
``outputs/reports/regression_model_comparison.csv``.

The held-out test partition is **never** touched here.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REG_TASKS, REPORTS, task_root  # noqa: E402
from train_candidate import (  # noqa: E402
    REG_CANDIDATES, BASELINE_REG, save_regression_candidate, HAS_XGB,
)


def _collect(task):
    rows = []
    for name in REG_CANDIDATES:
        role = "baseline" if name == BASELINE_REG else "candidates"
        meta_path = task_root("regression", task) / role / name / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing candidate metadata: {meta_path}")
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        cv = m["cv"]
        rows.append({
            "task": task, "model": name, "role": role,
            "cv_MAE_mean": cv["MAE"]["mean"], "cv_MAE_std": cv["MAE"]["std"],
            "cv_RMSE_mean": cv["RMSE"]["mean"], "cv_RMSE_std": cv["RMSE"]["std"],
            "cv_R2_mean": cv["R2"]["mean"], "cv_R2_std": cv["R2"]["std"],
            "cv_within_tolerance_pct_mean": cv["within_tolerance_pct"]["mean"],
            "cv_within_tolerance_pct_std": cv["within_tolerance_pct"]["std"],
            "parameters": str(m["parameters"]),
            "model_sha256": m["model_sha256"],
        })
    return rows


def main():
    all_rows = []
    for task in REG_TASKS:
        for name in REG_CANDIDATES:
            save_regression_candidate(name, stage=f"17_{name}_regression")
        rows = _collect(task)
        df = pd.DataFrame(rows).sort_values("cv_RMSE_mean").reset_index(drop=True)
        df.insert(1, "cv_rank", range(1, len(df) + 1))
        df.to_csv(REPORTS / f"{task}_model_comparison.csv", index=False)
        all_rows.extend(df.to_dict("records"))
        best = df.iloc[0]
        print(f"[{task}] CV-best (RMSE): {best['model']}  RMSE={best['cv_RMSE_mean']:.4f}")
    pd.DataFrame(all_rows).to_csv(REPORTS / "regression_model_comparison.csv", index=False)
    print(f"Regression CV comparison written ({len(all_rows)} rows). xgboost_available={HAS_XGB}")


if __name__ == "__main__":
    main()
