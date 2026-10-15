import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from backend.app.ml.component_3_smokehouse import (
    ordinal_fire_risk_pipeline as ordinal_fire_risk_module,
)


# ------------------------------------------------------------------
# Pickle compatibility
# ------------------------------------------------------------------
#
# The finalized fire-risk artifact was serialized when the custom
# class was available under the module name:
#
#     ordinal_fire_risk_pipeline
#
# In the backend package the same file is imported as:
#
#     backend.app.ml.component_3_smokehouse.ordinal_fire_risk_pipeline
#
# Registering this alias allows joblib/pickle to reconstruct the
# frozen artifact without modifying the model or its custom
# implementation.
sys.modules.setdefault(
    "ordinal_fire_risk_pipeline",
    ordinal_fire_risk_module,
)


# ============================================================
# FINALIZED COMPONENT 3 FILES
# ============================================================


MODEL_DIR = Path(__file__).resolve().parent

DRYING_TIME_MODEL_PATH = (
    MODEL_DIR / "drying_time_pipeline.pkl"
)

FIRE_RISK_MODEL_PATH = (
    MODEL_DIR / "fire_risk_pipeline.pkl"
)

MODEL_METADATA_PATH = (
    MODEL_DIR / "component3_model_metadata.json"
)


# ============================================================
# TRUSTED FROZEN ARTIFACT HASHES
# ============================================================
#
# These remain hard-coded intentionally.
#
# They are the trusted integrity anchors for the finalized
# research artifacts.
#
# The metadata file must agree with these hashes before its
# version information is accepted.
# ============================================================


EXPECTED_DRYING_TIME_SHA256 = (
    "5bc446c3802d3d1d8f4eb997253eee1095410c4ed43edac743990dc8f928fdd6"
)

EXPECTED_FIRE_RISK_SHA256 = (
    "61b4304b082672e9b7e0f91620e9854cac1abaabefd345a190e375110983083b"
)


# ============================================================
# EXCEPTION
# ============================================================


class ModelLoadError(RuntimeError):
    """
    Raised when finalized Component 3 model resources cannot
    be loaded or verified safely.
    """


# ============================================================
# FINALIZED MODEL METADATA
# ============================================================


@dataclass(frozen=True)
class Component3ModelMetadata:
    """
    Minimal authoritative runtime metadata extracted from
    component3_model_metadata.json.

    This prevents backend services from manually duplicating
    model-version strings.
    """

    regression_model_version: str
    classification_model_version: str

    classification_model_name: str

    regression_sha256: str
    classification_sha256: str


# ============================================================
# LOADED MODEL BUNDLE
# ============================================================


@dataclass(frozen=True)
class Component3Models:
    """
    Finalized Component 3 runtime models together with their
    verified authoritative metadata.
    """

    drying_time: Any
    fire_risk: Any

    metadata: Component3ModelMetadata


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _calculate_sha256(path: Path) -> str:
    """
    Calculate SHA-256 without modifying the file.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def _verify_model_file(
    path: Path,
    expected_sha256: str,
) -> None:
    """
    Verify that a model file exists and still matches the
    frozen finalized research artifact.
    """

    if not path.is_file():
        raise ModelLoadError(
            "Required model artifact was not found: "
            f"{path.name}"
        )

    try:
        actual_sha256 = _calculate_sha256(
            path
        )

    except OSError as exc:
        raise ModelLoadError(
            "Could not calculate integrity hash for "
            f"{path.name}."
        ) from exc

    if (
        actual_sha256.lower()
        != expected_sha256.lower()
    ):
        raise ModelLoadError(
            f"Integrity check failed for {path.name}. "
            "The model artifact does not match the "
            "finalized research model."
        )


def _require_mapping(
    value: object,
    field_name: str,
) -> dict:
    """
    Require a metadata field to be a JSON object.
    """

    if not isinstance(value, dict):
        raise ModelLoadError(
            "Invalid Component 3 metadata: "
            f"'{field_name}' must be an object."
        )

    return value


def _require_non_empty_string(
    mapping: dict,
    key: str,
    context: str,
) -> str:
    """
    Read one required non-empty string value from metadata.
    """

    value = mapping.get(key)

    if not isinstance(value, str):
        raise ModelLoadError(
            "Invalid Component 3 metadata: "
            f"'{context}.{key}' must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ModelLoadError(
            "Invalid Component 3 metadata: "
            f"'{context}.{key}' must not be empty."
        )

    return normalized


# ============================================================
# METADATA LOADING + VERIFICATION
# ============================================================


@lru_cache(maxsize=1)
def load_model_metadata() -> Component3ModelMetadata:
    """
    Load and verify the authoritative finalized Component 3
    model metadata.

    Runtime version identifiers are taken from this file rather
    than duplicated manually in backend service code.

    Authoritative fields:

        regression.trial
            -> regression model version

        model_version
            -> classification model version

        classification.model_name
            -> human-readable classifier name

    The artifact hashes recorded in metadata must also match
    the trusted frozen hashes declared in this loader.
    """

    if not MODEL_METADATA_PATH.is_file():
        raise ModelLoadError(
            "Required Component 3 model metadata was not found: "
            f"{MODEL_METADATA_PATH.name}"
        )

    try:
        with MODEL_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_metadata = json.load(file)

    except json.JSONDecodeError as exc:
        raise ModelLoadError(
            "Component 3 model metadata is not valid JSON."
        ) from exc

    except OSError as exc:
        raise ModelLoadError(
            "Component 3 model metadata could not be read."
        ) from exc

    if not isinstance(raw_metadata, dict):
        raise ModelLoadError(
            "Invalid Component 3 metadata: "
            "top-level JSON value must be an object."
        )

    # --------------------------------------------------------
    # Top-level classification version
    # --------------------------------------------------------

    classification_model_version = (
        _require_non_empty_string(
            raw_metadata,
            "model_version",
            "metadata",
        )
    )

    # --------------------------------------------------------
    # Regression metadata
    # --------------------------------------------------------

    regression = _require_mapping(
        raw_metadata.get("regression"),
        "regression",
    )

    regression_model_version = (
        _require_non_empty_string(
            regression,
            "trial",
            "regression",
        )
    )

    regression_sha256 = (
        _require_non_empty_string(
            regression,
            "sha256",
            "regression",
        )
    )

    # --------------------------------------------------------
    # Classification metadata
    # --------------------------------------------------------

    classification = _require_mapping(
        raw_metadata.get("classification"),
        "classification",
    )

    classification_model_name = (
        _require_non_empty_string(
            classification,
            "model_name",
            "classification",
        )
    )

    classification_sha256 = (
        _require_non_empty_string(
            classification,
            "sha256",
            "classification",
        )
    )

    # --------------------------------------------------------
    # Metadata must agree with trusted frozen artifact hashes
    # --------------------------------------------------------

    if (
        regression_sha256.lower()
        != EXPECTED_DRYING_TIME_SHA256.lower()
    ):
        raise ModelLoadError(
            "Component 3 metadata regression SHA-256 "
            "does not match the trusted finalized "
            "drying-time artifact hash."
        )

    if (
        classification_sha256.lower()
        != EXPECTED_FIRE_RISK_SHA256.lower()
    ):
        raise ModelLoadError(
            "Component 3 metadata classification SHA-256 "
            "does not match the trusted finalized "
            "fire-risk artifact hash."
        )

    return Component3ModelMetadata(
        regression_model_version=(
            regression_model_version
        ),
        classification_model_version=(
            classification_model_version
        ),
        classification_model_name=(
            classification_model_name
        ),
        regression_sha256=regression_sha256,
        classification_sha256=(
            classification_sha256
        ),
    )


# ============================================================
# MODEL LOADING
# ============================================================


@lru_cache(maxsize=1)
def load_models() -> Component3Models:
    """
    Load the two finalized Component 3 models and their
    authoritative metadata.

    lru_cache means the files are loaded once per Python
    process instead of being reloaded for every API request.
    """

    # --------------------------------------------------------
    # 1. Load + verify authoritative metadata
    # --------------------------------------------------------

    metadata = load_model_metadata()

    # --------------------------------------------------------
    # 2. Verify actual artifact files
    # --------------------------------------------------------

    _verify_model_file(
        DRYING_TIME_MODEL_PATH,
        metadata.regression_sha256,
    )

    _verify_model_file(
        FIRE_RISK_MODEL_PATH,
        metadata.classification_sha256,
    )

    # --------------------------------------------------------
    # 3. Deserialize finalized models
    # --------------------------------------------------------

    try:
        drying_time_model = joblib.load(
            DRYING_TIME_MODEL_PATH
        )

        fire_risk_model = joblib.load(
            FIRE_RISK_MODEL_PATH
        )

    except Exception as exc:
        raise ModelLoadError(
            "Component 3 ML models could not be loaded."
        ) from exc

    # --------------------------------------------------------
    # 4. Runtime interface verification
    # --------------------------------------------------------

    if not hasattr(
        drying_time_model,
        "predict",
    ):
        raise ModelLoadError(
            "Drying-time artifact does not provide predict()."
        )

    if not hasattr(
        fire_risk_model,
        "predict",
    ):
        raise ModelLoadError(
            "Fire-risk artifact does not provide predict()."
        )

    if not hasattr(
        fire_risk_model,
        "predict_proba",
    ):
        raise ModelLoadError(
            "Fire-risk artifact does not provide predict_proba()."
        )

    # --------------------------------------------------------
    # 5. Return verified runtime bundle
    # --------------------------------------------------------

    return Component3Models(
        drying_time=drying_time_model,
        fire_risk=fire_risk_model,
        metadata=metadata,
    )


# ============================================================
# PUBLIC ACCESSORS
# ============================================================


def get_models() -> Component3Models:
    """
    Public accessor used by backend inference services.
    """

    return load_models()


def get_model_metadata() -> Component3ModelMetadata:
    """
    Public accessor for verified authoritative runtime
    model-version metadata.
    """

    return load_model_metadata()
