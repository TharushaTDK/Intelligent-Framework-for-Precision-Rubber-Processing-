"""Stage 32 (model explainability).

Computes permutation importance (model-agnostic, on the test partition) and, for
tree-based models, impurity-based feature importance for the selected models.
SHAP is intentionally not used (not a project dependency); this is stated in the
report. Correlational diagnostics are also kept.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIGURES, REPORTS, REG_TASKS, CLS_TASK, role_dir, ensure_dirs, write_json,
)
from train_candidate import _reg_data, _cls_data, split_indices  # noqa: E402

from sklearn.inspection import permutation_importance

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

FIGDIR = FIGURES / "32_explainability"


def _tree_importance(pipe, feature_names):
    model = pipe.named_steps.get("model")
    if not hasattr(model, "feature_importances_"):
        return None
    try:
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        names = feature_names
    imp = model.feature_importances_
    if len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]
    return pd.DataFrame({"transformed_feature": names, "impurity_importance": imp}
                        ).sort_values("impurity_importance", ascending=False)


def _perm(pipe, X, y, scoring, tag, kind):
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=10, random_state=340, n_jobs=1)
    df = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
                      ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS / f"32_{kind}_{tag}_permutation_importance.csv", index=False)
    return df


def main():
    ensure_dirs()
    summary = {"stage": "32_model_explainability",
               "method": "permutation importance on the held-out test partition + impurity importance for tree models",
               "shap_used": False,
               "shap_note": "SHAP is not installed and was deliberately not used. Permutation and impurity "
                            "importance are computed with scikit-learn only. Correlation is not causation.",
               "dataset_provenance": "synthetic; not real-world evidence",
               "regression": {}, "classification": {}}

    # regression
    for task, (target, features, tol) in REG_TASKS.items():
        X, y, _ = _reg_data(task)
        tr, te = split_indices(task)
        te = [i for i in te if i in X.index]
        Xte, yte = X.loc[te], y.loc[te]
        pipe = joblib.load(role_dir("regression", task, "final") / "selected_model.joblib")
        perm = _perm(pipe, Xte, yte, "neg_root_mean_squared_error", task, "regression")
        imp = _tree_importance(pipe, features)
        if imp is not None:
            imp.to_csv(REPORTS / f"32_regression_{task}_impurity_importance.csv", index=False)
        summary["regression"][task] = {"top_permutation_features": perm.head(8).to_dict("records"),
                                       "has_impurity_importance": imp is not None}
        if plt is not None:
            FIGDIR.mkdir(parents=True, exist_ok=True)
            top = perm.head(10).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 4.6))
            ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
            ax.set_xlabel("Permutation importance (drop in score)"); ax.set_title(f"{task}: permutation importance")
            fig.tight_layout(); fig.savefig(FIGDIR / f"regression_{task}_permutation_importance.png", dpi=140); plt.close(fig)

    # classification
    X, y = _cls_data()
    tr, te = split_indices(CLS_TASK)
    te = [i for i in te if i in X.index]
    Xte, yte = X.loc[te], y.loc[te]
    bundle = joblib.load(role_dir("classification", CLS_TASK, "final") / "selected_model.joblib")
    pipe = bundle["pipeline"]
    enc = bundle["label_encoder"]
    yte_enc = enc.transform(yte)
    perm = _perm(pipe, Xte, yte_enc, "f1_macro", CLS_TASK, "classification")
    imp = _tree_importance(pipe, list(X.columns))
    if imp is not None:
        imp.to_csv(REPORTS / f"32_classification_{CLS_TASK}_impurity_importance.csv", index=False)
    summary["classification"][CLS_TASK] = {"top_permutation_features": perm.head(8).to_dict("records"),
                                           "has_impurity_importance": imp is not None}
    if plt is not None:
        FIGDIR.mkdir(parents=True, exist_ok=True)
        top = perm.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_xlabel("Permutation importance (drop in macro-F1)"); ax.set_title("tapping_advisory: permutation importance")
        fig.tight_layout(); fig.savefig(FIGDIR / "classification_tapping_advisory_permutation_importance.png", dpi=140); plt.close(fig)

    write_json(REPORTS / "32_model_explainability_summary.json", summary)
    (REPORTS / "32_explainability_report.txt").write_text(
        "Stage 32 explainability\n" + "=" * 24 + "\n"
        "Permutation importance was computed on the held-out test partition for the selected models.\n"
        "Impurity importance is reported for tree-based selected models.\n"
        "SHAP was NOT used (not installed); correlation and importance do not prove causation.\n"
        "All data are synthetic; importances describe the simulation, not real agronomy.\n",
        encoding="utf-8")
    print("Stage 32 explainability complete. SHAP used: False")


if __name__ == "__main__":
    main()
