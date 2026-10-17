"""Stage 27 (comparison): rank tapping-advisory classifiers by stratified
cross-validation on the training partition.

Ensures each declared family has been fitted (via :mod:`train_candidate`), then
aggregates the CV metrics stored in each candidate metadata JSON into
``outputs/reports/classification_model_comparison.csv``.

The held-out test partition is **never** touched here.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLS_TASK, REPORTS, task_root  # noqa: E402
from train_candidate import (  # noqa: E402
    CLS_CANDIDATES, BASELINE_CLS, save_classification_candidate, HAS_XGB,
)


def main():
    for name in CLS_CANDIDATES:
        save_classification_candidate(name, stage=f"27_{name}_classification")

    rows = []
    for name in CLS_CANDIDATES:
        role = "baseline" if name == BASELINE_CLS else "candidates"
        meta_path = task_root("classification", CLS_TASK) / role / name / "metadata.json"
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        cv = m["cv"]
        rows.append({
            "task": CLS_TASK, "model": name, "role": role,
            "cv_accuracy_mean": cv["accuracy"]["mean"], "cv_accuracy_std": cv["accuracy"]["std"],
            "cv_macro_precision_mean": cv["precision_macro"]["mean"],
            "cv_macro_recall_mean": cv["recall_macro"]["mean"],
            "cv_macro_f1_mean": cv["f1_macro"]["mean"], "cv_macro_f1_std": cv["f1_macro"]["std"],
            "parameters": str(m["parameters"]),
            "model_sha256": m["model_sha256"],
        })
    df = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    df.insert(1, "cv_rank", range(1, len(df) + 1))
    df.to_csv(REPORTS / "classification_model_comparison.csv", index=False)
    best = df.iloc[0]
    print(f"[{CLS_TASK}] CV-best (macro-F1): {best['model']}  F1={best['cv_macro_f1_mean']:.4f}")
    print(f"Classification CV comparison written ({len(df)} rows). xgboost_available={HAS_XGB}")


if __name__ == "__main__":
    main()
