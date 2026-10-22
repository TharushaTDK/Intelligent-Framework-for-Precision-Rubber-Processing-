"""Verified loading of the fitted Component 4 quality pipeline."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import sklearn

from .feature_contract import FEATURE_NAMES
from .preprocessing import FEATURE_VERSION, CONFIG, RULE_VERSION, NEAR_FULL_SCALE_LIMIT


COMPONENT = Path(__file__).resolve().parent


class ModelLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_quality_model():
    try:
        metadata = json.loads((COMPONENT / "artifacts" / "quality_model_metadata.json").read_text(encoding="utf-8"))
        artifact_name = metadata["artifact"]
        if Path(artifact_name).name != artifact_name:
            raise ModelLoadError("Invalid artifact path.")
        path = COMPONENT / "artifacts" / artifact_name
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
            raise ModelLoadError("Model artifact SHA-256 mismatch.")
        for name, expected in metadata["code_hashes"].items():
            if Path(name).name != name or hashlib.sha256((COMPONENT / name).read_bytes()).hexdigest() != expected:
                raise ModelLoadError("Processing code changed. Review and retrain the pilot model.")
        if metadata["sklearn_version"] != sklearn.__version__:
            raise ModelLoadError("scikit-learn version differs from the training environment.")
        if (metadata["feature_version"] != FEATURE_VERSION or metadata["rule_version"] != RULE_VERSION
                or metadata["segmentation_config"] != CONFIG
                or metadata["near_full_scale_limit"] != NEAR_FULL_SCALE_LIMIT):
            raise ModelLoadError("Inference configuration differs from training.")
        if metadata["feature_names"] != FEATURE_NAMES:
            raise ModelLoadError("Model feature order differs from the runtime contract.")
        estimator = joblib.load(path)
        if set(estimator.classes_) != {"VALID", "INVALID"}:
            raise ModelLoadError("Unexpected model classes.")
        return estimator, metadata
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError("Quality model is unavailable or invalid. Run pilot training successfully first.") from exc
