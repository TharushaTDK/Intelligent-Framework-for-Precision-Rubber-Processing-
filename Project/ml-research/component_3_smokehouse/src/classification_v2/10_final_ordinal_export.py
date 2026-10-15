"""
CLASSIFICATION V2 - FINALIZATION PHASE 16/17/22
FINAL ORDINAL XGBOOST TRAINING AND EXPORT
====================================================================

ONLY runs after the promotion-readiness gate (09_promotion_readiness_gate.py)
passed all 10 checkable criteria.

Fits the final Ordinal XGBoost Classifier (shared preprocessor + O1 + O2)
on the full allowed Development partition (Train + Validation, 20,405 rows)
using EXACTLY the frozen research methodology (same preprocessing recipe,
same XGBoost hyperparameters, same balanced-weighting rule, same probability
reconstruction/monotonicity-correction/argmax decision rule already validated
in 07_ordinal_severity_classifier.py). No new fitting behavior is introduced.

Does NOT use Component3_Test.csv for fitting (existence-checked only).

Packages the result as an OrdinalFireRiskPipeline instance (see
backend/app/ml/component_3_smokehouse/ordinal_fire_risk_pipeline.py) and
writes it to the active production path:
  backend/app/ml/component_3_smokehouse/fire_risk_pipeline.pkl
(The prior XGB_03 artifact was already archived, with verified matching
SHA256, to backend/app/ml/component_3_smokehouse/archive/ before this
script ran.)
"""

from pathlib import Path
import sys
import hashlib

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BACKEND_ML_DIR = (
    Path(__file__).resolve().parents[4] / "backend" / "app" / "ml" / "component_3_smokehouse"
)
sys.path.insert(0, str(BACKEND_ML_DIR))
import ordinal_fire_risk_pipeline as ofrp  # noqa: E402

FINAL_DIR = common.ROOT / "outputs" / "final_classification"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def make_ordinal_binary_params(base_xgb_params):
    params = dict(base_xgb_params)
    params.pop("num_class", None)
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    return params


def run():
    print("=" * 78)
    print("FINAL ORDINAL XGBOOST TRAINING AND EXPORT")
    print("=" * 78)

    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    numeric_features = data["numeric_features"]
    categorical_features = data["categorical_features"]

    if selected_features != ofrp.SELECTED_FEATURES:
        raise RuntimeError(
            "Feature-list mismatch between Stage-08 selected features and "
            "the packaged OrdinalFireRiskPipeline.SELECTED_FEATURES. "
            "STOP - do not export."
        )

    print(f"\nDevelopment rows: {len(development):,} (Test NOT loaded for fitting)")
    print(f"Selected features ({len(selected_features)}): {selected_features}")

    _, _, xgb_params_multiclass = common.recover_selected_xgb_configuration()
    ordinal_params = make_ordinal_binary_params(xgb_params_multiclass)

    X_raw = development[selected_features].copy()
    y_cls = development["Fire_Risk_Int"].to_numpy(dtype=int)

    print("\nFitting shared preprocessor on full Development...")
    preprocessor = common.make_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_raw)
    X_transformed = preprocessor.transform(X_raw)

    print("Fitting O1 (P(Y>=Warning)) on full Development...")
    y_ge_warning = (y_cls >= common.CLASS_TO_INT["Warning"]).astype(int)
    w1 = compute_sample_weight(class_weight="balanced", y=y_ge_warning)
    model_o1 = XGBClassifier(**ordinal_params)
    model_o1.fit(X_transformed, y_ge_warning, sample_weight=w1)

    print("Fitting O2 (P(Y>=Critical)) on full Development...")
    y_ge_critical = (y_cls >= common.CLASS_TO_INT["Critical"]).astype(int)
    w2 = compute_sample_weight(class_weight="balanced", y=y_ge_critical)
    model_o2 = XGBClassifier(**ordinal_params)
    model_o2.fit(X_transformed, y_ge_critical, sample_weight=w2)

    print("\nAssembling OrdinalFireRiskPipeline...")
    pipeline = ofrp.OrdinalFireRiskPipeline(
        preprocessor=preprocessor,
        model_o1=model_o1,
        model_o2=model_o2,
        selected_features=selected_features,
    )

    # ---- In-memory reference predictions on full Development ----
    proba_memory = pipeline.predict_proba(X_raw)
    pred_memory = pipeline.predict(X_raw)
    pred_int_memory = pipeline.predict_int(X_raw)

    sums = proba_memory.sum(axis=1)
    max_sum_error = float(np.max(np.abs(sums - 1.0)))
    nonneg_ok = bool(np.all(proba_memory >= -1e-9))
    print(f"\nIn-memory refit sanity check: max|sum(P)-1|={max_sum_error:.2e}, "
          f"all P>=0: {nonneg_ok}")
    if max_sum_error >= 1e-6 or not nonneg_ok:
        raise RuntimeError("In-memory final refit failed probability integrity checks. STOP.")

    memory_df = pd.DataFrame({
        "Original_Row_Index": np.arange(len(development)),
        "Batch_ID": development["Batch_ID"].to_numpy(),
        "y_true_int": y_cls,
        "y_pred_label": pred_memory,
        "y_pred_int": pred_int_memory,
        "P_Normal": proba_memory[:, 0],
        "P_Warning": proba_memory[:, 1],
        "P_Critical": proba_memory[:, 2],
    })
    memory_df.to_csv(FINAL_DIR / "final_ordinal_development_refit_predictions.csv", index=False)

    # Full-development (training-set) accuracy is NOT a generalization metric -
    # it is recorded only as the reference for the reload-reproduction test.
    train_accuracy = float(np.mean(pred_int_memory == y_cls))
    print(f"Full-Development (in-sample, NOT a generalization metric) accuracy: {train_accuracy:.6f}")

    # -------------------- EXPORT --------------------
    ACTIVE_PATH = BACKEND_ML_DIR / "fire_risk_pipeline.pkl"
    print(f"\nExporting final pipeline to: {ACTIVE_PATH}")
    joblib.dump(pipeline, ACTIVE_PATH)

    with open(ACTIVE_PATH, "rb") as f:
        new_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"New fire_risk_pipeline.pkl SHA-256: {new_sha256}")

    with open(FINAL_DIR / "final_ordinal_export_manifest.txt", "w", encoding="utf-8") as f:
        f.write("FINAL ORDINAL XGBOOST EXPORT MANIFEST\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Development rows used for final fit: {len(development):,}\n")
        f.write(f"Selected features: {selected_features}\n")
        f.write(f"Active artifact path: {ACTIVE_PATH}\n")
        f.write(f"Active artifact SHA-256: {new_sha256}\n")
        f.write(f"Max probability sum error: {max_sum_error:.2e}\n")
        f.write(f"Full-Development (in-sample) accuracy: {train_accuracy:.6f}\n")
        f.write("Test split NOT used for fitting (existence-checked only).\n")

    print(f"\nOutputs saved to: {FINAL_DIR}")
    print(f"Manifest: {FINAL_DIR / 'final_ordinal_export_manifest.txt'}")

    return {
        "pipeline": pipeline,
        "new_sha256": new_sha256,
        "memory_df": memory_df,
        "active_path": ACTIVE_PATH,
    }


if __name__ == "__main__":
    run()
