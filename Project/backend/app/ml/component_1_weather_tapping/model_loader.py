import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


# ============================================================
# FINALIZED COMPONENT 1 FILES
# ============================================================


MODEL_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = MODEL_DIR / "artifacts"

DRC_PIPELINE_PATH = (
    ARTIFACTS_DIR / "drc_pipeline.pkl"
)

PRICE_PIPELINE_PATH = (
    ARTIFACTS_DIR / "price_pipeline.pkl"
)

TAPPING_PIPELINE_PATH = (
    ARTIFACTS_DIR / "tapping_pipeline.pkl"
)

ARTIFACT_MANIFEST_PATH = (
    ARTIFACTS_DIR / "artifact_manifest.json"
)


# ============================================================
# TRUSTED RUNTIME ARTIFACT HASHES
# ============================================================
#
# These remain hard-coded intentionally.
#
# They are the trusted integrity anchors for the
# scikit-learn 1.9.1-compatible re-serialized runtime
# artifacts produced by the verified Phase 1 migration.
#
# Each hash was verified against the Phase 0 transformation
# evidence (transformation_record.json) before the artifact
# was copied into this repository, and the manifest must
# agree with these hashes before its contents are accepted.
#
# A verification failure must NEVER update these values.
# The artifact or this loader must be reviewed by a human.
# ============================================================


EXPECTED_DRC_PIPELINE_SHA256 = (
    "4f08f7bde48f158bb71c034ec9f887e19ec26a6982af8891d623e938d80eeba1"
)

EXPECTED_PRICE_PIPELINE_SHA256 = (
    "3046401e9d06427823f344f2361bd0cbed19a21ad569764760c2e25dd42707dd"
)

EXPECTED_TAPPING_PIPELINE_SHA256 = (
    "ae87ebff1314d10961da74f9e121d15402c8fb642d658c53a75df26cd7a5f612"
)


# ============================================================
# AUTHORITATIVE TAPPING CLASS ORDER
# ============================================================
#
# The fitted label encoder's classes are exactly:
#
#     index 0 -> High
#     index 1 -> Low
#     index 2 -> Moderate
#
# predict_proba() column order always follows this order.
# ============================================================


TAPPING_CLASSES_IN_PROBABILITY_ORDER = (
    "High",
    "Low",
    "Moderate",
)


# ============================================================
# EXCEPTION
# ============================================================


class ModelLoadError(RuntimeError):
    """
    Raised when finalized Component 1 model resources cannot
    be loaded or verified safely.
    """


# ============================================================
# FINALIZED MODEL BUNDLES
# ============================================================


@dataclass(frozen=True)
class Component1TappingBundle:
    """
    The frozen Component 1 tapping-advisory artifact.

    The research export is a dictionary bundle with the keys
    'pipeline', 'label_encoder', 'classes' and 'warning'.
    This dataclass exposes the same frozen objects with the
    class order made explicit.
    """

    pipeline: Any
    label_encoder: Any
    classes: tuple[str, ...]
    warning: str


@dataclass(frozen=True)
class Component1Models:
    """
    Finalized Component 1 runtime models together with the
    verified runtime manifest.
    """

    drc: Any
    price: Any
    tapping: Component1TappingBundle
    manifest: dict


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
    Verify that a runtime artifact file exists and still
    matches its trusted frozen hash.

    On failure this function only raises. It never rewrites
    a trusted hash and never modifies the artifact.
    """

    if not path.is_file():
        raise ModelLoadError(
            "Required Component 1 model artifact was not "
            f"found: {path.name}"
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
            "The runtime artifact does not match the "
            "trusted verified Component 1 artifact."
        )


def _require_mapping(
    value: object,
    field_name: str,
) -> dict:
    """
    Require a manifest field to be a JSON object.
    """

    if not isinstance(value, dict):
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            f"'{field_name}' must be an object."
        )

    return value


def _require_non_empty_string(
    mapping: dict,
    key: str,
    context: str,
) -> str:
    """
    Read one required non-empty string value from a manifest
    object.
    """

    value = mapping.get(key)

    if not isinstance(value, str):
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            f"'{context}.{key}' must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            f"'{context}.{key}' must not be empty."
        )

    return normalized


def _require_ordered_features(
    mapping: dict,
    context: str,
) -> list[str]:
    """
    Read the exact ordered feature-name list from a manifest
    model entry.
    """

    features = mapping.get(
        "features_in_pipeline_order"
    )

    if (
        not isinstance(features, list)
        or not features
    ):
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            f"'{context}.features_in_pipeline_order' must "
            "be a non-empty list."
        )

    normalized: list[str] = []

    for entry in features:
        if not isinstance(entry, str) or not entry.strip():
            raise ModelLoadError(
                "Invalid Component 1 artifact manifest: "
                f"'{context}.features_in_pipeline_order' "
                "must contain only non-empty strings."
            )

        normalized.append(entry)

    return normalized


# ============================================================
# MANIFEST LOADING + VERIFICATION
# ============================================================


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    """
    Load and verify the authoritative runtime artifact
    manifest.

    The manifest is accepted only when every model entry
    agrees with the trusted hard-coded runtime artifact
    hashes declared in this loader.
    """

    if not ARTIFACT_MANIFEST_PATH.is_file():
        raise ModelLoadError(
            "Required Component 1 artifact manifest was "
            f"not found: {ARTIFACT_MANIFEST_PATH.name}"
        )

    try:
        with ARTIFACT_MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_manifest = json.load(file)

    except json.JSONDecodeError as exc:
        raise ModelLoadError(
            "Component 1 artifact manifest is not valid "
            "JSON."
        ) from exc

    except OSError as exc:
        raise ModelLoadError(
            "Component 1 artifact manifest could not be "
            "read."
        ) from exc

    if not isinstance(raw_manifest, dict):
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            "top-level JSON value must be an object."
        )

    models = raw_manifest.get("models")

    if (
        not isinstance(models, list)
        or not models
    ):
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            "'models' must be a non-empty list."
        )

    expected_hashes = {
        "drc_prediction": (
            EXPECTED_DRC_PIPELINE_SHA256
        ),
        "price_estimation": (
            EXPECTED_PRICE_PIPELINE_SHA256
        ),
        "tapping_advisory": (
            EXPECTED_TAPPING_PIPELINE_SHA256
        ),
    }

    seen: set[str] = set()

    for entry in models:
        model = _require_mapping(
            entry,
            "models[]",
        )

        name = _require_non_empty_string(
            model,
            "name",
            "models[]",
        )

        if name in seen:
            raise ModelLoadError(
                "Invalid Component 1 artifact manifest: "
                f"duplicate model entry '{name}'."
            )

        seen.add(name)

        if name not in expected_hashes:
            raise ModelLoadError(
                "Invalid Component 1 artifact manifest: "
                f"unknown model entry '{name}'."
            )

        runtime_sha256 = _require_non_empty_string(
            model,
            "runtime_sha256",
            name,
        )

        if (
            runtime_sha256.lower()
            != expected_hashes[name].lower()
        ):
            raise ModelLoadError(
                "Component 1 artifact manifest SHA-256 "
                f"for '{name}' does not match the trusted "
                "verified runtime artifact hash."
            )

        _require_non_empty_string(
            model,
            "runtime_artifact_relative_path",
            name,
        )

        _require_non_empty_string(
            model,
            "target",
            name,
        )

        _require_ordered_features(
            model,
            name,
        )

        if name == "tapping_advisory":
            classes = model.get(
                "classes_in_probability_order"
            )

            if (
                not isinstance(classes, list)
                or tuple(classes)
                != TAPPING_CLASSES_IN_PROBABILITY_ORDER
            ):
                raise ModelLoadError(
                    "Invalid Component 1 artifact manifest: "
                    "'tapping_advisory."
                    "classes_in_probability_order' must be "
                    f"{list(TAPPING_CLASSES_IN_PROBABILITY_ORDER)}."
                )

    missing = set(expected_hashes) - seen

    if missing:
        raise ModelLoadError(
            "Invalid Component 1 artifact manifest: "
            f"missing model entries {sorted(missing)}."
        )

    return raw_manifest


# ============================================================
# MODEL LOADING
# ============================================================


@lru_cache(maxsize=1)
def load_models() -> Component1Models:
    """
    Load the three finalized Component 1 models and the
    verified runtime manifest.

    lru_cache means the files are loaded once per Python
    process instead of being reloaded for every API request.
    """

    # --------------------------------------------------------
    # 1. Load + verify authoritative manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    # --------------------------------------------------------
    # 2. Verify actual artifact files against trusted hashes
    # --------------------------------------------------------

    _verify_model_file(
        DRC_PIPELINE_PATH,
        EXPECTED_DRC_PIPELINE_SHA256,
    )

    _verify_model_file(
        PRICE_PIPELINE_PATH,
        EXPECTED_PRICE_PIPELINE_SHA256,
    )

    _verify_model_file(
        TAPPING_PIPELINE_PATH,
        EXPECTED_TAPPING_PIPELINE_SHA256,
    )

    # --------------------------------------------------------
    # 3. Deserialize finalized models
    #
    # The runtime artifacts are the verified scikit-learn
    # 1.9.1-compatible re-serializations and load natively
    # under the pinned runtime stack without any pickle
    # compatibility aliases.
    # --------------------------------------------------------

    try:
        drc_model = joblib.load(
            DRC_PIPELINE_PATH
        )

        price_model = joblib.load(
            PRICE_PIPELINE_PATH
        )

        raw_tapping_bundle = joblib.load(
            TAPPING_PIPELINE_PATH
        )

    except Exception as exc:
        raise ModelLoadError(
            "Component 1 ML models could not be loaded."
        ) from exc

    # --------------------------------------------------------
    # 4. Runtime interface verification
    # --------------------------------------------------------

    if not hasattr(drc_model, "predict"):
        raise ModelLoadError(
            "DRC artifact does not provide predict()."
        )

    if not hasattr(price_model, "predict"):
        raise ModelLoadError(
            "Price artifact does not provide predict()."
        )

    if not isinstance(raw_tapping_bundle, dict):
        raise ModelLoadError(
            "Tapping artifact must be a dictionary bundle."
        )

    for key in (
        "pipeline",
        "label_encoder",
        "classes",
        "warning",
    ):
        if key not in raw_tapping_bundle:
            raise ModelLoadError(
                "Tapping artifact bundle is missing the "
                f"required key '{key}'."
            )

    tapping_pipeline = raw_tapping_bundle["pipeline"]
    tapping_label_encoder = raw_tapping_bundle[
        "label_encoder"
    ]
    tapping_classes = tuple(
        str(class_name)
        for class_name in raw_tapping_bundle["classes"]
    )
    tapping_warning = str(
        raw_tapping_bundle["warning"]
    )

    if not hasattr(tapping_pipeline, "predict"):
        raise ModelLoadError(
            "Tapping pipeline artifact does not provide "
            "predict()."
        )

    if not hasattr(tapping_pipeline, "predict_proba"):
        raise ModelLoadError(
            "Tapping pipeline artifact does not provide "
            "predict_proba()."
        )

    if not hasattr(
        tapping_label_encoder,
        "inverse_transform",
    ):
        raise ModelLoadError(
            "Tapping label encoder does not provide "
            "inverse_transform()."
        )

    if (
        tapping_classes
        != TAPPING_CLASSES_IN_PROBABILITY_ORDER
    ):
        raise ModelLoadError(
            "Tapping artifact class order does not match "
            "the trusted order "
            f"{list(TAPPING_CLASSES_IN_PROBABILITY_ORDER)}."
        )

    encoder_classes = tuple(
        str(class_name)
        for class_name in getattr(
            tapping_label_encoder,
            "classes_",
            (),
        )
    )

    if (
        encoder_classes
        != TAPPING_CLASSES_IN_PROBABILITY_ORDER
    ):
        raise ModelLoadError(
            "Tapping label encoder class order does not "
            "match the trusted order "
            f"{list(TAPPING_CLASSES_IN_PROBABILITY_ORDER)}."
        )

    # --------------------------------------------------------
    # 5. Return verified runtime bundle
    # --------------------------------------------------------

    return Component1Models(
        drc=drc_model,
        price=price_model,
        tapping=Component1TappingBundle(
            pipeline=tapping_pipeline,
            label_encoder=tapping_label_encoder,
            classes=tapping_classes,
            warning=tapping_warning,
        ),
        manifest=manifest,
    )


# ============================================================
# PUBLIC ACCESSORS
# ============================================================


def get_models() -> Component1Models:
    """
    Public accessor used by backend inference services.
    """

    return load_models()


def get_manifest() -> dict:
    """
    Public accessor for the verified runtime artifact
    manifest.
    """

    return load_manifest()
