"""Rule-first Component 4 recording quality inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import model_loader as loader
from .preprocessing import check_recording, extract_features


def assess_recording(path):
    checks = check_recording(path)
    result = dict(checks, model_version=None, invalid_probability_uncalibrated=None)
    if checks["technical_status"] == "TECHNICAL_REJECTED":
        result.update(quality_status="INVALID", decision_source="TECHNICAL_RULES")
        return result
    try:
        estimator, metadata = loader.get_quality_model()
    except loader.ModelLoadError as exc:
        result.update(quality_status="UNAVAILABLE", classifier_status="QUALITY_MODEL_UNAVAILABLE", error=str(exc))
        return result
    features, _ = extract_features(path)
    if features is None:
        raise ValueError("No features available for classifier inference.")
    x = np.array([[features[name] for name in metadata["feature_names"]]], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite model features.")
    probabilities = estimator.predict_proba(x)[0]
    invalid_score = float(probabilities[list(estimator.classes_).index("INVALID")])
    label = "INVALID" if invalid_score >= metadata["decision_threshold_invalid"] else "VALID"
    result.update(
        quality_status=label, classifier_status="MODEL_EVALUATED", decision_source="ML_CLASSIFIER",
        model_version=metadata["model_version"], model_status=metadata["status"],
        invalid_probability_uncalibrated=invalid_score,
    )
    if label == "INVALID":
        result["rejection_reasons"] = ["ML_RECORDING_QUALITY_REJECTION"]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    files = sorted(args.path.glob('*.wav')) if args.path.is_dir() else [args.path]
    if not files:
        parser.error('No WAV files found.')
    for path in files:
        print(json.dumps(assess_recording(path), indent=2, allow_nan=False))
