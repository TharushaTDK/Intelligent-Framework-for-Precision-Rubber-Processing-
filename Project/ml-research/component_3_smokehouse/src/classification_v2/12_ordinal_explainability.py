"""
CLASSIFICATION V2 - FINALIZATION PHASE 25
ORDINAL CLASSIFIER EXPLAINABILITY (O1 and O2, separately)
====================================================================

The final Ordinal XGBoost Classifier is NOT one ordinary multiclass model -
it is two independent binary XGBoost models (O1, O2) feeding a deterministic
probability reconstruction. Therefore SHAP is computed and reported
SEPARATELY for each stage:

  O1: Normal vs Warning-or-Higher  (P(Y >= Warning))
  O2: Normal/Warning vs Critical   (P(Y >= Critical)) - the stage that
      corresponds directly to Critical-severity behavior.

Uses the already-exported, reloaded production artifact (fire_risk_pipeline.pkl)
so the explained model is exactly the one that was packaged - not a
freshly refit duplicate.

SHAP values explain model behavior/association only. They are NOT evidence
of physical causation.

Historical Stage-32 SHAP evidence (for XGB_03) is NOT overwritten - these
outputs are saved under a new, clearly-labeled location.

Test split is never read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BACKEND_ML_DIR = (
    Path(__file__).resolve().parents[4] / "backend" / "app" / "ml" / "component_3_smokehouse"
)
sys.path.insert(0, str(BACKEND_ML_DIR))
import ordinal_fire_risk_pipeline  # noqa: E402,F401
import joblib

FINAL_DIR = common.ROOT / "outputs" / "final_classification"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def run():
    print("=" * 78)
    print("ORDINAL CLASSIFIER EXPLAINABILITY - O1 AND O2 (SEPARATE SHAP)")
    print("=" * 78)

    model = joblib.load(BACKEND_ML_DIR / "fire_risk_pipeline.pkl")
    print(f"\nLoaded production artifact: {type(model).__name__} from {BACKEND_ML_DIR / 'fire_risk_pipeline.pkl'}")

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    X_raw = development[selected_features].copy()

    X_transformed = model.preprocessor.transform(X_raw)
    feature_names = model.preprocessor.get_feature_names_out().tolist()
    print(f"Transformed feature space: {len(feature_names)} features")
    print(f"Development rows explained: {len(X_transformed):,}")

    # ---- O1 SHAP ----
    print("\nComputing SHAP for O1 (Normal vs Warning-or-Higher)...")
    explainer_o1 = shap.TreeExplainer(model.model_o1)
    shap_values_o1 = explainer_o1.shap_values(X_transformed)
    mean_abs_o1 = np.abs(shap_values_o1).mean(axis=0)
    o1_importance = (
        pd.DataFrame({"Transformed_Feature": feature_names, "Mean_Absolute_SHAP": mean_abs_o1})
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    o1_importance.to_csv(FINAL_DIR / "ordinal_o1_shap_importance.csv", index=False)

    print("\nTop 10 O1 features (Normal vs Warning-or-Higher):")
    print(o1_importance.head(10).to_string(index=False))

    # ---- O2 SHAP ----
    print("\nComputing SHAP for O2 (Normal/Warning vs Critical)...")
    explainer_o2 = shap.TreeExplainer(model.model_o2)
    shap_values_o2 = explainer_o2.shap_values(X_transformed)
    mean_abs_o2 = np.abs(shap_values_o2).mean(axis=0)
    o2_importance = (
        pd.DataFrame({"Transformed_Feature": feature_names, "Mean_Absolute_SHAP": mean_abs_o2})
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    o2_importance.to_csv(FINAL_DIR / "ordinal_o2_shap_importance.csv", index=False)

    print("\nTop 10 O2 features (Normal/Warning vs Critical - Critical-severity stage):")
    print(o2_importance.head(10).to_string(index=False))

    # ---- Diagnostic: actual-Critical rows, same style as historical Stage-32 ----
    y_true = development["Fire_Risk_Int"].to_numpy()
    critical_mask = y_true == common.CLASS_TO_INT["Critical"]
    n_critical = int(critical_mask.sum())
    if n_critical > 0:
        mean_abs_o2_on_critical = np.abs(shap_values_o2[critical_mask]).mean(axis=0)
        critical_diag = (
            pd.DataFrame({
                "Transformed_Feature": feature_names,
                "Mean_Absolute_SHAP_O2_on_Actual_Critical_Rows": mean_abs_o2_on_critical,
            })
            .sort_values("Mean_Absolute_SHAP_O2_on_Actual_Critical_Rows", ascending=False)
            .reset_index(drop=True)
        )
        critical_diag.to_csv(FINAL_DIR / "ordinal_o2_shap_on_actual_critical_rows.csv", index=False)
        print(f"\nO2 SHAP restricted to the {n_critical} actual-Critical development rows "
              f"(top 10):")
        print(critical_diag.head(10).to_string(index=False))

    with open(FINAL_DIR / "ordinal_explainability_notes.txt", "w", encoding="utf-8") as f:
        f.write("ORDINAL CLASSIFIER EXPLAINABILITY NOTES\n")
        f.write("=" * 78 + "\n\n")
        f.write("The final classifier is two independent binary XGBoost models (O1, O2), "
                "not one multiclass model. SHAP was computed separately for each.\n\n")
        f.write("O1: Normal vs Warning-or-Higher (P(Y>=Warning))\n")
        f.write("O2: Normal/Warning vs Critical (P(Y>=Critical)) - corresponds directly to "
                "Critical-severity behavior.\n\n")
        f.write("SHAP values explain model behavior/association only. They are NOT evidence "
                "of physical causation.\n\n")
        f.write("Historical Stage-32 SHAP evidence for XGB_03 (outputs/reports/32_*) was NOT "
                "modified or overwritten by this analysis.\n")

    print(f"\nOutputs saved to: {FINAL_DIR}")
    return {"o1_importance": o1_importance, "o2_importance": o2_importance}


if __name__ == "__main__":
    run()
