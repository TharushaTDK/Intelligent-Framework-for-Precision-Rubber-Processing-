"""Stage 31 (one-time final test evaluation).

Loads the frozen selections from ``.../<task>/final/selected_model.joblib`` and
evaluates them on the stored held-out test partition exactly once. Also scores
every other saved candidate on the test partition for transparency (clearly
labelled as exploratory), and compares the stage-30 CV estimate with the final
test result.

Produces the required evidence: regression metrics, actual-vs-predicted and
residual figures, tapping confusion matrix and per-class report, per-model test
predictions, and the estate-wise DRC test summary.

This is the ONLY stage that touches the test targets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIGURES, REPORTS, REG_TASKS, CLS_TASK, CLS_TARGET, CLS_FEATURES, CLS_CLASSES,
    task_root, role_dir, ensure_dirs, load_raw, clean_basic, write_json, sha256_frame,
)
from train_candidate import _reg_data, _cls_data, split_indices  # noqa: E402

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

REG_ROLES = ("baseline", "candidates", "tuned")
FIGDIR_REGRESSION = FIGURES / "31_regression_final_test"
FIGDIR_CLASSIFICATION = FIGURES / "31_classification_final_test"


def _iter_models(kind, task):
    root = task_root(kind, task)
    for role in REG_ROLES:
        for meta_path in sorted((root / role).glob("*/metadata.json")):
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            name = m.get("model_name", m["model_family"])
            model_path = meta_path.parent / "model.joblib"
            yield role, name, model_path, m


def _load_regression_model(path):
    return joblib.load(path)


def _load_classification_bundle(path):
    return joblib.load(path)


def evaluate_regression_task(task, summary):
    target, features, tol = REG_TASKS[task]
    X, y, _ = _reg_data(task)
    tr, te = split_indices(task)
    te = [i for i in te if i in X.index]
    Xte, yte = X.loc[te], y.loc[te]
    estate = clean_basic(load_raw()).loc[te, "estate_id"].astype(str).to_numpy()

    # selected model
    final_meta = json.loads((role_dir("regression", task, "final") / "selected_model_metadata.json").read_text(encoding="utf-8"))
    selected_name = final_meta["selected_model_family"]
    model = _load_regression_model(role_dir("regression", task, "final") / "selected_model.joblib")
    pred = model.predict(Xte)
    yv = np.asarray(yte, float)
    selected_test = {
        "MAE": float(mean_absolute_error(yv, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yv, pred))),
        "R2": float(r2_score(yv, pred)),
        "within_tolerance_pct": float(100 * np.mean(np.abs(yv - pred) <= tol)),
        "tolerance": tol,
    }

    # all candidates on test (exploratory comparison)
    all_rows = []
    for role, name, model_path, m in _iter_models("regression", task):
        mdl = _load_regression_model(model_path)
        p = mdl.predict(Xte)
        all_rows.append({
            "task": task, "model": name, "role": role,
            "test_MAE": float(mean_absolute_error(yv, p)),
            "test_RMSE": float(np.sqrt(mean_squared_error(yv, p))),
            "test_R2": float(r2_score(yv, p)),
            "test_within_tolerance_pct": float(100 * np.mean(np.abs(yv - p) <= tol)),
            "cv_RMSE_mean": m["cv"]["RMSE"]["mean"],
            "selected": name == selected_name,
        })
        pd.DataFrame({"row_index": Xte.index, "estate_id": estate, "actual": yv, "predicted": p}
                     ).to_csv(REPORTS / f"31_{task}_{name}_test_predictions.csv", index=False)
    test_df = pd.DataFrame(all_rows).sort_values("test_RMSE").reset_index(drop=True)
    test_df.to_csv(REPORTS / f"31_{task}_all_models_test_metrics.csv", index=False)
    pd.DataFrame([selected_test | {"task": task, "model": selected_name, "tolerance": tol}]
                 ).to_csv(REPORTS / f"31_{task}_final_test_metrics.csv", index=False)

    # stage30 CV vs final test for the selected model
    cv_rmse = final_meta["development_cv"]["RMSE_mean"]
    pd.DataFrame([{"task": task, "model": selected_name,
                   "stage30_cv_RMSE_mean": cv_rmse, "stage31_test_RMSE": selected_test["RMSE"],
                   "stage30_cv_MAE_mean": final_meta["development_cv"]["MAE_mean"],
                   "stage31_test_MAE": selected_test["MAE"]}
                  ]).to_csv(REPORTS / f"31_{task}_stage30_vs_final_test.csv", index=False)

    # estate-wise DRC summary
    if task == "drc_prediction":
        est = pd.DataFrame({"estate_id": estate, "actual_drc_pct": yv, "predicted_drc_pct": pred})
        est["absolute_error_drc_pct"] = np.abs(est["actual_drc_pct"] - est["predicted_drc_pct"])
        est.groupby("estate_id").agg(test_observations=("actual_drc_pct", "count"),
                                     mean_actual_drc_pct=("actual_drc_pct", "mean"),
                                     mean_predicted_drc_pct=("predicted_drc_pct", "mean"),
                                     mae_drc_pct=("absolute_error_drc_pct", "mean")
                                     ).reset_index().to_csv(REPORTS / "31_drc_estate_wise_test_summary.csv", index=False)

    if plt is not None:
        FIGDIR_REGRESSION.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(yv, pred, alpha=.6, edgecolor="k", linewidth=.3)
        lo = min(yv.min(), pred.min()); hi = max(yv.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="crimson")
        ax.set_xlabel("Actual (synthetic)"); ax.set_ylabel("Predicted")
        ax.set_title(f"{task}: selected {selected_name} (test)")
        fig.tight_layout(); fig.savefig(FIGDIR_REGRESSION / f"{task}_actual_vs_predicted.png", dpi=140); plt.close(fig)

        resid = yv - pred
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.scatter(pred, resid, alpha=.6, edgecolor="k", linewidth=.3)
        ax.axhline(0, color="crimson", linestyle="--")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{task}: residuals (selected {selected_name}, test)")
        fig.tight_layout(); fig.savefig(FIGDIR_REGRESSION / f"{task}_residuals.png", dpi=140); plt.close(fig)

        cmp_df = test_df.sort_values("cv_RMSE_mean")
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.bar(cmp_df["model"], cmp_df["cv_RMSE_mean"])
        ax.set_ylabel("CV RMSE (train partition)")
        ax.set_xticks(range(len(cmp_df)))
        ax.set_xticklabels(list(cmp_df["model"]), rotation=30, ha="right")
        ax.set_title(f"{task}: cross-validated candidate comparison (synthetic)")
        fig.tight_layout(); fig.savefig(FIGDIR_REGRESSION / f"{task}_cv_model_comparison.png", dpi=140); plt.close(fig)

    print(f"[{task}] selected {selected_name}: test RMSE={selected_test['RMSE']:.4f} "
          f"MAE={selected_test['MAE']:.4f} R2={selected_test['R2']:.4f} "
          f"within±{tol}={selected_test['within_tolerance_pct']:.1f}%")
    summary["regression"][task] = {"selected_model": selected_name,
                                   "selection_role": final_meta["selected_role"],
                                   "development_cv": final_meta["development_cv"],
                                   "test": selected_test,
                                   "test_estates": int(len(set(estate))),
                                   "test_rows": int(len(yv)),
                                   "x_test_sha256": sha256_frame(Xte),
                                   "y_test_sha256": sha256_frame(yte.to_frame())}
    return test_df


def evaluate_classification_task(summary):
    task = CLS_TASK
    X, y = _cls_data()
    tr, te = split_indices(task)
    te = [i for i in te if i in X.index]
    Xte, yte = X.loc[te], y.loc[te]
    estate = clean_basic(load_raw()).loc[te, "estate_id"].astype(str).to_numpy()

    final_meta = json.loads((role_dir("classification", task, "final") / "selected_model_metadata.json").read_text(encoding="utf-8"))
    selected_name = final_meta["selected_model_family"]
    bundle = _load_classification_bundle(role_dir("classification", task, "final") / "selected_model.joblib")
    pipe, enc = bundle["pipeline"], bundle["label_encoder"]
    pred = enc.inverse_transform(pipe.predict(Xte).astype(int))
    yv = yte.to_numpy()

    def _cls_metrics(y_true, y_pred):
        return {"accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}

    selected_test = _cls_metrics(yv, pred)
    labels = list(enc.classes_)

    all_rows = []
    for role, name, model_path, m in _iter_models("classification", task):
        b = _load_classification_bundle(model_path)
        p = b["label_encoder"].inverse_transform(b["pipeline"].predict(Xte).astype(int))
        all_rows.append({"task": task, "model": name, "role": role,
                         **_cls_metrics(yv, p),
                         "cv_macro_f1_mean": m["cv"]["f1_macro"]["mean"],
                         "selected": name == selected_name})
        pd.DataFrame({"row_index": Xte.index, "estate_id": estate, "actual": yv, "predicted": p}
                     ).to_csv(REPORTS / f"31_{task}_{name}_test_predictions.csv", index=False)
    test_df = pd.DataFrame(all_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    test_df.to_csv(REPORTS / f"31_{task}_all_models_test_metrics.csv", index=False)
    pd.DataFrame([selected_test | {"task": task, "model": selected_name}]
                 ).to_csv(REPORTS / f"31_{task}_final_test_metrics.csv", index=False)

    cm = confusion_matrix(yv, pred, labels=labels)
    pd.DataFrame(cm, index=[f"actual_{c}" for c in labels], columns=[f"predicted_{c}" for c in labels]
                 ).to_csv(REPORTS / f"31_{task}_final_test_confusion_matrix.csv")
    rep = pd.DataFrame(classification_report(yv, pred, labels=labels, output_dict=True, zero_division=0)).T
    rep.to_csv(REPORTS / f"31_{task}_final_test_per_class.csv")

    pd.DataFrame([{"task": task, "model": selected_name,
                   "stage30_cv_macro_f1_mean": final_meta["development_cv"]["macro_f1_mean"],
                   "stage31_test_macro_f1": selected_test["macro_f1"],
                   "stage30_cv_accuracy_mean": final_meta["development_cv"]["accuracy_mean"],
                   "stage31_test_accuracy": selected_test["accuracy"]}
                  ]).to_csv(REPORTS / f"31_{task}_stage30_vs_final_test.csv", index=False)

    if plt is not None:
        FIGDIR_CLASSIFICATION.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4.4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Tapping advisory confusion matrix: {selected_name}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout()
        fig.savefig(FIGDIR_CLASSIFICATION / "tapping_advisory_confusion_matrix.png", dpi=140); plt.close(fig)

        cmp_df = test_df.sort_values("cv_macro_f1_mean")
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.bar(cmp_df["model"], cmp_df["cv_macro_f1_mean"])
        ax.set_ylabel("CV macro-F1 (train partition)")
        ax.set_xticks(range(len(cmp_df)))
        ax.set_xticklabels(list(cmp_df["model"]), rotation=30, ha="right")
        ax.set_title("Tapping advisory: cross-validated classifier comparison (synthetic proxy)")
        fig.tight_layout(); fig.savefig(FIGDIR_CLASSIFICATION / "tapping_advisory_cv_model_comparison.png", dpi=140); plt.close(fig)

        per = rep.loc[[c for c in labels if c in rep.index], ["precision", "recall", "f1-score"]]
        fig, ax = plt.subplots(figsize=(7, 4))
        per.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Score")
        ax.set_title(f"Per-class metrics: {selected_name} (test)")
        fig.tight_layout(); fig.savefig(FIGDIR_CLASSIFICATION / "tapping_advisory_per_class_metrics.png", dpi=140); plt.close(fig)

    print(f"[{task}] selected {selected_name}: test macro-F1={selected_test['macro_f1']:.4f} "
          f"accuracy={selected_test['accuracy']:.4f}")
    summary["classification"][task] = {"selected_model": selected_name,
                                       "selection_role": final_meta["selected_role"],
                                       "development_cv": final_meta["development_cv"],
                                       "test": selected_test,
                                       "confusion_matrix_labels": labels,
                                       "confusion_matrix": cm.tolist(),
                                       "test_estates": int(len(set(estate))),
                                       "test_rows": int(len(yv)),
                                       "x_test_sha256": sha256_frame(Xte),
                                       "y_test_sha256": sha256_frame(yte.to_frame())}
    return test_df


def main():
    ensure_dirs()
    summary = {
        "project": "RubberSense Component 1", "stage": "31_final_test_evaluation",
        "protocol": "Selected models were chosen by CV on the training partition. The test partition "
                    "was loaded once here. Other candidates are also scored on test for transparency, "
                    "but those numbers did not influence selection.",
        "test_loaded": True, "test_evaluated": True,
        "dataset_provenance": "synthetic; not real-world evidence",
        "regression": {}, "classification": {},
        "figures": {"matplotlib_available": plt is not None},
        "warnings": [
            "Row-wise split: most test estates also appear in training.",
            "Within-tolerance % is not accuracy.",
        ],
    }
    for task in REG_TASKS:
        evaluate_regression_task(task, summary)
    evaluate_classification_task(summary)
    write_json(REPORTS / "31_final_test_evaluation_summary.json", summary)
    report = [
        "FINAL TEST EVALUATION REPORT (stage 31)",
        "=" * 40,
        "The held-out test partition is evaluated here exactly once.",
        "Selection was performed by cross-validation on the training partition (stages 18/19, 28/29).",
        "",
        "Regression (selected model -> test):",
    ]
    for task, r in summary["regression"].items():
        t = r["test"]
        report.append(f"  [{task}] {r['selected_model']}: MAE={t['MAE']:.4f} RMSE={t['RMSE']:.4f} "
                      f"R2={t['R2']:.4f} within+/-{t['tolerance']}={t['within_tolerance_pct']:.1f}%")
    report += ["", "Classification (selected model -> test):"]
    for task, r in summary["classification"].items():
        t = r["test"]
        report.append(f"  [{task}] {r['selected_model']}: accuracy={t['accuracy']:.4f} "
                      f"macro-P={t['macro_precision']:.4f} macro-R={t['macro_recall']:.4f} macro-F1={t['macro_f1']:.4f}")
    report += ["", "Reminder: synthetic data. Not real-world evidence. Within-tolerance % is not accuracy."]
    (REPORTS / "31_final_test_evaluation_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nStage 31 complete. Figures available: {plt is not None}")


if __name__ == "__main__":
    main()
