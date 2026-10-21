"""Stage 30 (consolidated model comparison summary).

Builds a single CV-based summary of every task from the comparison tables and
the final-selection summaries produced by stages 18/19 and 28/29. The selection
reported here is based on **cross-validation on the training partition**, never
on the held-out test set.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, REG_TASKS, DATASET_PROVENANCE, TARGET_WARNING, ensure_dirs, write_json  # noqa: E402


def _read_json(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ensure_dirs()
    reg_cmp = pd.read_csv(REPORTS / "regression_model_comparison.csv") if (REPORTS / "regression_model_comparison.csv").exists() else pd.DataFrame()
    cls_cmp = pd.read_csv(REPORTS / "classification_model_comparison.csv") if (REPORTS / "classification_model_comparison.csv").exists() else pd.DataFrame()

    payload = {
        "project": "RubberSense Component 1",
        "stage": "30_consolidated_model_summary",
        "dataset_provenance": DATASET_PROVENANCE,
        "selection_basis": "Cross-validation on the training partition (k-fold regression RMSE; "
                           "stratified macro-F1 classification). The held-out test set is used once, in stage 31.",
        "regression_accuracy_definition": "Percentage of predictions whose absolute error falls within a "
                                          f"declared tolerance (DRC +/-{REG_TASKS['drc_prediction'][2]} pp; "
                                          f"price +/-LKR {REG_TASKS['price_estimation'][2]}/kg of field latex). "
                                          "This is NOT accuracy and NOT R-squared.",
        "candidates": {
            "regression": "dummy_mean, linear_regression, random_forest, extra_trees, gradient_boosting, "
                          "hist_gradient_boosting, xgboost (when installed), plus tuned shortlist",
            "classification": "dummy_most_frequent, logistic_regression, random_forest, extra_trees, "
                              "gradient_boosting, xgboost (when installed), plus tuned shortlist",
        },
        "cv_regression_ranking_top": {},
        "cv_classification_ranking_top": {},
        "selected": {},
        "limitations": [
            "DRC values were normalized to Kudaligama chart entries by the dataset author; they are "
            "not original field measurements.",
            "Price is derived as farm-gate latex price x DRC / 100, so a model with DRC as an input is "
            "close to relearning that formula; Jun-Sep 2026 farm-gate levels are estimates.",
            "Collection-time prediction is excluded; no collection-time training data exists.",
            TARGET_WARNING,
            "Only 5 estates and 4 months (Jun-Sep 2026) of farmer records; results may not hold for "
            "other estates or seasons.",
            "The stored split is row-wise, so most test estates also appear in training; see the "
            "estate-grouped CV for the unseen-estate estimate.",
            "Within-tolerance percentage is not the same as classification accuracy or R-squared.",
        ],
    }
    if not reg_cmp.empty:
        for task, group in reg_cmp.groupby("task"):
            top = group.sort_values("cv_RMSE_mean").iloc[0]
            payload["cv_regression_ranking_top"][task] = {
                "model": top["model"], "cv_RMSE_mean": float(top["cv_RMSE_mean"]),
                "cv_MAE_mean": float(top["cv_MAE_mean"]), "cv_R2_mean": float(top["cv_R2_mean"]),
            }
    if not cls_cmp.empty:
        top = cls_cmp.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]
        payload["cv_classification_ranking_top"]["tapping_advisory"] = {
            "model": top["model"], "cv_macro_f1_mean": float(top["cv_macro_f1_mean"]),
            "cv_accuracy_mean": float(top["cv_accuracy_mean"]),
        }
    reg_sel = _read_json("19_regression_selection_summary.json")
    cls_sel = _read_json("29_classification_selection_summary.json")
    if reg_sel:
        payload["selected"].update({t: v for t, v in reg_sel.get("tasks", {}).items()})
    if cls_sel:
        payload["selected"]["tapping_advisory"] = cls_sel.get("tapping_advisory", {})
    write_json(REPORTS / "final_model_selection.json", payload)
    write_json(REPORTS / "30_consolidated_model_summary.json", payload)
    print(json.dumps(payload["selected"], indent=2, default=str))
    print("Stage 30 consolidated summary complete (CV-based selection).")


if __name__ == "__main__":
    main()
