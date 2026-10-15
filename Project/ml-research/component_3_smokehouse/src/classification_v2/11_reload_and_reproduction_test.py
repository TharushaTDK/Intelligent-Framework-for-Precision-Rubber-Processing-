"""
CLASSIFICATION V2 - FINALIZATION PHASE 23/24
SERIALIZATION RELOAD TEST + FULL DEVELOPMENT REPRODUCTION TEST
====================================================================

Run as a standalone `python` invocation (a genuinely fresh process - the
in-memory trained objects from 10_final_ordinal_export.py do NOT exist in
this process's memory). Loads the serialized artifact from disk, recomputes
predictions on all 20,405 Development rows, and compares against the
reference predictions saved by 10_final_ordinal_export.py.

Test split is never read.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BACKEND_ML_DIR = (
    Path(__file__).resolve().parents[4] / "backend" / "app" / "ml" / "component_3_smokehouse"
)
sys.path.insert(0, str(BACKEND_ML_DIR))
import ordinal_fire_risk_pipeline  # noqa: E402,F401  (registers class for unpickling)

FINAL_DIR = common.ROOT / "outputs" / "final_classification"
ARTIFACT_PATH = BACKEND_ML_DIR / "fire_risk_pipeline.pkl"
REFERENCE_PATH = FINAL_DIR / "final_ordinal_development_refit_predictions.csv"


def run():
    print("=" * 78)
    print("SERIALIZATION RELOAD + FULL DEVELOPMENT REPRODUCTION TEST")
    print("=" * 78)
    print("\nThis process did NOT train anything; the model below was loaded fresh from disk.")

    # ---- Reload test ----
    model = joblib.load(ARTIFACT_PATH)
    print(f"\nLoaded artifact: {ARTIFACT_PATH}")
    print(f"Loaded object type: {type(model)} (module: {type(model).__module__})")
    reload_ok = isinstance(model, ordinal_fire_risk_pipeline.OrdinalFireRiskPipeline)
    print(f"Reload test - correct class instance: {reload_ok}")

    # ---- Full development reproduction test ----
    data = common.load_development()
    development = data["development"]
    selected_features = data["selected_features"]
    X_raw = development[selected_features].copy()

    reloaded_proba = model.predict_proba(X_raw)
    reloaded_pred = model.predict(X_raw)
    reloaded_pred_int = model.predict_int(X_raw)

    reference = pd.read_csv(REFERENCE_PATH)

    n_rows = len(development)
    rows_match = len(reference) == n_rows
    print(f"\nRow count: {n_rows:,} (reference has {len(reference):,}) -> {'MATCH' if rows_match else 'MISMATCH'}")

    label_mismatches = int(np.sum(reloaded_pred != reference["y_pred_label"].to_numpy()))
    int_mismatches = int(np.sum(reloaded_pred_int != reference["y_pred_int"].to_numpy()))

    ref_proba = reference[["P_Normal", "P_Warning", "P_Critical"]].to_numpy()
    abs_diff = np.abs(reloaded_proba - ref_proba)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Also verify against TRUE labels directly (sanity, not just vs reference).
    y_true = development["Fire_Risk_Int"].to_numpy()
    reloaded_accuracy = float(np.mean(reloaded_pred_int == y_true))

    print(f"\nClass-label mismatches (string labels) vs in-memory reference: {label_mismatches}")
    print(f"Class-label mismatches (int labels) vs in-memory reference: {int_mismatches}")
    print(f"Maximum absolute probability difference: {max_abs_diff:.3e}")
    print(f"Mean absolute probability difference: {mean_abs_diff:.3e}")
    print(f"Reloaded-model in-sample accuracy (sanity check, not a generalization metric): {reloaded_accuracy:.6f}")

    # XGBoost's predict_proba returns float32-derived values internally;
    # independent predict_proba() calls on identical inputs/model can differ
    # by ~1e-7-1e-8 due to ordinary floating-point reconstruction noise, not
    # non-reproducibility. 1e-6 is strict (10-30x the observed noise floor)
    # while not flagging float32 rounding as a false failure.
    PROB_TOLERANCE = 1e-6
    test_pass = (
        rows_match
        and label_mismatches == 0
        and int_mismatches == 0
        and max_abs_diff < PROB_TOLERANCE
    )

    print(f"\n{'='*78}")
    print(f"RELOAD + REPRODUCTION TEST: {'PASS' if test_pass else 'FAIL'}")
    print(f"{'='*78}")

    with open(FINAL_DIR / "final_ordinal_reload_reproduction_test.txt", "w", encoding="utf-8") as f:
        f.write("SERIALIZATION RELOAD + FULL DEVELOPMENT REPRODUCTION TEST\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Artifact: {ARTIFACT_PATH}\n")
        f.write(f"Reload test - correct class instance: {reload_ok}\n")
        f.write(f"Development rows: {n_rows:,}\n")
        f.write(f"Row count match: {rows_match}\n")
        f.write(f"Class-label mismatches (string): {label_mismatches}\n")
        f.write(f"Class-label mismatches (int): {int_mismatches}\n")
        f.write(f"Maximum absolute probability difference: {max_abs_diff:.6e}\n")
        f.write(f"Mean absolute probability difference: {mean_abs_diff:.6e}\n")
        f.write(f"Reloaded-model in-sample accuracy: {reloaded_accuracy:.6f}\n")
        f.write(f"\nTEST RESULT: {'PASS' if test_pass else 'FAIL'}\n")

    if not test_pass:
        raise RuntimeError("Reload/reproduction test FAILED. Do not finalize promotion.")

    print(f"\nOutputs saved to: {FINAL_DIR}")
    return {
        "reload_ok": reload_ok,
        "label_mismatches": label_mismatches,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "test_pass": test_pass,
    }


if __name__ == "__main__":
    run()
