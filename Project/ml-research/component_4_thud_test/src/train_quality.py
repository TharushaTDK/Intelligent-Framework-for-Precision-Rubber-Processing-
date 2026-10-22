"""Train and evaluate a preliminary, sheet-grouped quality classifier."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.app.ml.component_4_thud_test.feature_contract import FEATURE_NAMES
from backend.app.ml.component_4_thud_test.preprocessing import (
    FEATURE_VERSION, extract_features, CONFIG, RULE_VERSION, NEAR_FULL_SCALE_LIMIT,
)


COMPONENT = PROJECT / "backend" / "app" / "ml" / "component_4_thud_test"
RESEARCH = PROJECT / "ml-research" / "component_4_thud_test"
DATA = RESEARCH / "data"
CODE_FILES = ("preprocessing.py", "feature_contract.py", "model_loader.py", "inference.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=3000, random_state=42
        )),
    ])


def matrix(rows):
    values = np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite training input.")
    return values


def scored(y, prediction):
    return {
        "n_original_recordings": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "invalid_f1": float(f1_score(y, prediction, pos_label="INVALID", zero_division=0)),
        "label_order": ["INVALID", "VALID"],
        "confusion_matrix": confusion_matrix(y, prediction, labels=["INVALID", "VALID"]).tolist(),
        "false_acceptance_rate": sum(a == "INVALID" and b == "VALID" for a, b in zip(y, prediction)) / max(1, y.count("INVALID")),
        "false_rejection_rate": sum(a == "VALID" and b == "INVALID" for a, b in zip(y, prediction)) / max(1, y.count("VALID")),
    }


def feature_row(annotation, path, source_type, parent=""):
    features, checks = extract_features(path)
    row = {
        "recording_id": annotation["recording_id"],
        "sheet_id": annotation["sheet_id"],
        "condition_id": annotation["condition_id"],
        "quality_label": annotation["quality_label"],
        "source_type": source_type,
        "parent_recording_id": parent,
        "technical_status": checks["technical_status"],
        "has_features": features is not None,
        "sha256": sha256(path),
    }
    row.update(features or {name: None for name in FEATURE_NAMES})
    return row


def main():
    manifest_path = DATA / "metadata" / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        annotations = list(csv.DictReader(file))
    if not annotations:
        raise ValueError("No dataset manifest rows. Run preprocessing.py first.")
    ids = [r["recording_id"] for r in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate recording IDs.")
    collected = {r["recording_id"]: r for r in annotations if r["source_type"] == "COLLECTED"}
    originals, expanded, manifest = [], [], []
    print("Reading existing input recordings and verifying provenance...")
    for annotation in annotations:
        identifier = annotation["recording_id"]
        if annotation["quality_label"] not in ("VALID", "INVALID"):
            raise ValueError(f"Missing/invalid label: {identifier}")
        if annotation["source_type"] not in ("COLLECTED", "DERIVED"):
            raise ValueError(f"Unknown input origin: {identifier}")
        if not annotation.get("sheet_id") or not annotation.get("condition_id"):
            raise ValueError("Sheet and condition identifiers are required.")
        if (Path(annotation["filename"]).name != annotation["filename"]
                or "/" in annotation["filename"] or "\\" in annotation["filename"]):
            raise ValueError("Metadata filename must be a basename.")
        if not identifier.replace("_", "").isalnum():
            raise ValueError("Recording IDs must contain letters, numbers and underscores.")
        parent_id = annotation["parent_recording_id"]
        if annotation["source_type"] == "DERIVED":
            parent = collected[parent_id]
            if any(annotation[key] != parent[key] for key in ("sheet_id", "condition_id", "quality_label")):
                raise ValueError("Derived recording must remain with its parent sheet/label.")
            if annotation["parent_sha256"] != parent["sha256"]:
                raise ValueError("Derived recording parent hash mismatch.")
            manifest.append(annotation)
        elif parent_id:
            raise ValueError("Collected recording cannot have a derived parent reference.")
        path = DATA / "raw" / annotation["filename"]
        feature = feature_row(annotation, path, annotation["source_type"], parent_id)
        if feature["sha256"] != annotation["sha256"]:
            raise ValueError(f"Input WAV changed after dataset preparation: {identifier}")
        expanded.append(feature)
        if annotation["source_type"] == "COLLECTED":
            originals.append(feature)
    hashes = [r["sha256"] for r in originals]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate original files. Resolve provenance before evaluation.")

    groups = np.array([r["sheet_id"] for r in originals])
    if len(set(groups)) < 3:
        raise ValueError("At least three independent sheets are required for this pilot procedure.")
    predictions, folds = [], []
    splitter = LeaveOneGroupOut()
    for fold, (train_indices, test_indices) in enumerate(splitter.split(originals, groups=groups), 1):
        training_sheets = set(groups[train_indices])
        held_sheets = set(groups[test_indices])
        training = [r for r in expanded if r["sheet_id"] in training_sheets
                    and r["technical_status"] == "TECHNICAL_PASSED" and r["has_features"]]
        if set(r["quality_label"] for r in training) != {"VALID", "INVALID"}:
            raise ValueError(f"Fold {fold}: classifier training requires both labels after rules.")
        estimator = model()
        estimator.fit(matrix(training), [r["quality_label"] for r in training])
        assert not training_sheets.intersection(held_sheets)
        folds.append({"fold": fold, "training_sheets": sorted(training_sheets),
                      "held_out_sheets": sorted(held_sheets), "training_rows": len(training)})
        for index in test_indices:
            original = originals[index]
            ml_prediction = None
            if original["technical_status"] == "TECHNICAL_REJECTED":
                full_prediction = "INVALID"
            else:
                ml_prediction = str(estimator.predict(matrix([original]))[0])
                full_prediction = ml_prediction
            predictions.append({
                "recording_id": original["recording_id"], "sheet_id": original["sheet_id"],
                "fold": fold, "true_label": original["quality_label"],
                "technical_status": original["technical_status"],
                "ml_prediction": ml_prediction, "workflow_prediction": full_prediction,
            })
    y = [r["true_label"] for r in predictions]
    workflow = [r["workflow_prediction"] for r in predictions]
    ml_rows = [r for r in predictions if r["ml_prediction"] is not None]
    rules_prediction = ["INVALID" if r["technical_status"] == "TECHNICAL_REJECTED" else "VALID" for r in predictions]
    metrics = {
        "status": "PRELIMINARY_DEMO",
        "evaluation": "leave_one_sheet_out; score original recordings only",
        "independent_sheets": len(set(groups)), "original_recordings": len(originals),
        "derived_recordings": len(manifest), "combined_feature_rows": len(expanded),
        "rules_only": scored(y, rules_prediction),
        "complete_workflow": scored(y, workflow),
        "classifier_on_rule_passed_originals": scored(
            [r["true_label"] for r in ml_rows], [r["ml_prediction"] for r in ml_rows]
        ),
        "folds": folds,
        "limitations": [
            "Small pilot dataset, not final independent validation.",
            "Derived recording labels are inherited; variants are not new sheets.",
            "Some invalid examples have protocol deviations that may not be acoustically identifiable.",
            "No moisture model or calibrated reliability guarantee is included.",
        ],
    }
    final_training = [r for r in expanded if r["technical_status"] == "TECHNICAL_PASSED" and r["has_features"]]
    final = model()
    final.fit(matrix(final_training), [r["quality_label"] for r in final_training])
    artifacts = COMPONENT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    model_path = artifacts / "quality_pipeline.joblib"
    joblib.dump(final, model_path, compress=3)
    metadata = {
        "model_version": "quality-logistic-pilot-v1", "status": "PRELIMINARY_DEMO",
        "runtime_layout_version": "preprocessing-inference-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": model_path.name, "artifact_sha256": sha256(model_path),
        "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
        "rule_version": RULE_VERSION, "near_full_scale_limit": NEAR_FULL_SCALE_LIMIT,
        "segmentation_config": CONFIG, "decision_threshold_invalid": 0.5,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "classes": list(final.classes_),
        "original_recordings": len(originals), "independent_sheets": len(set(groups)),
        "derived_recordings": len(manifest), "fit_rows": len(final_training),
        "code_hashes": {name: sha256(COMPONENT / name) for name in CODE_FILES},
        "dataset_manifest_sha256": sha256(manifest_path),
    }
    save_json(artifacts / "quality_model_metadata.json", metadata)
    save_csv(DATA / "processed" / "quality_features.csv", expanded)
    save_csv(DATA / "model_ready" / "quality_features.csv", final_training)
    save_csv(RESEARCH / "outputs" / "reports" / "pilot_quality_oof_predictions.csv", predictions)
    save_json(RESEARCH / "outputs" / "reports" / "pilot_quality_metrics.json", metrics)
    reloaded = joblib.load(model_path)
    if not np.array_equal(final.predict(matrix(final_training)), reloaded.predict(matrix(final_training))):
        raise RuntimeError("Saved-model reload check failed.")
    print(f"Original recordings: {len(originals)}")
    print(f"Derived recordings: {len(manifest)}")
    print(f"Combined feature rows: {len(expanded)}")
    print(f"Independent sheets: {len(set(groups))}")
    print(f"Final classifier fit rows after rules: {len(final_training)}")
    print("Pilot evaluation:", json.dumps(metrics["complete_workflow"]))
    print("Rules-only baseline:", json.dumps(metrics["rules_only"]))
    print(f"Saved model: {model_path}")
    print("Saved-model reload check: PASSED")


if __name__ == "__main__":
    main()
