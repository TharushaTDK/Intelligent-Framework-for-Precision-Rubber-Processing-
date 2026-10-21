import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.component_1_weather_tapping import model_loader
from backend.app.ml.component_1_weather_tapping.model_loader import (
    EXPECTED_DRC_PIPELINE_SHA256,
    EXPECTED_PRICE_PIPELINE_SHA256,
    EXPECTED_TAPPING_PIPELINE_SHA256,
    TAPPING_CLASSES_IN_PROBABILITY_ORDER,
    ModelLoadError,
    _calculate_sha256,
    _verify_model_file,
    get_manifest,
    get_models,
)


# ============================================================
# TEST HELPERS
# ============================================================


VALID_ROW = {
    "temperature_c": 27.5,
    "relative_humidity_pct": 82.0,
    "rainfall_last_6h_mm": 4.2,
    "forecast_rain_next_6h_mm": 0.0,
    "rainfall_last_3days_mm": 18.5,
    "wind_speed_kmh": 7.3,
    "soil_moisture_pct": 41.0,
    "soil_ph": 5.8,
    "visible_latex_condition": "Normal",
    "estate_id": "Estate 01",
    "latex_drc_pct": 34.5,
    "month": 10.0,
    "day_of_year": 291.0,
}


def frame_for(model_entry: dict, **overrides) -> pd.DataFrame:
    """
    Build a one-row feature frame using the exact ordered
    feature list recorded in the verified runtime manifest.
    """

    features = model_entry["features_in_pipeline_order"]
    row = {name: VALID_ROW[name] for name in features}
    row.update(overrides)
    return pd.DataFrame([row], columns=features)


def manifest_model(manifest: dict, name: str) -> dict:
    for entry in manifest["models"]:
        if entry["name"] == name:
            return entry
    raise AssertionError(f"manifest entry not found: {name}")


# ============================================================
# TEST 1 — ARTIFACTS AND MANIFEST EXIST
# ============================================================


def test_runtime_artifacts_and_manifest_exist():
    """
    All three runtime artifacts and the runtime manifest
    must exist in the approved Component 1 runtime folder.
    """

    artifacts_dir = model_loader.ARTIFACTS_DIR

    assert artifacts_dir.is_dir()
    assert model_loader.ARTIFACT_MANIFEST_PATH.is_file()
    assert model_loader.DRC_PIPELINE_PATH.is_file()
    assert model_loader.PRICE_PIPELINE_PATH.is_file()
    assert model_loader.TAPPING_PIPELINE_PATH.is_file()


# ============================================================
# TEST 2 — ARTIFACT HASHES MATCH TRUSTED ANCHORS
# ============================================================


def test_runtime_artifact_hashes_match_trusted_hashes():
    """
    Every runtime artifact file must still hash to its
    trusted hard-coded SHA-256.
    """

    assert (
        _calculate_sha256(model_loader.DRC_PIPELINE_PATH)
        == EXPECTED_DRC_PIPELINE_SHA256
    )

    assert (
        _calculate_sha256(model_loader.PRICE_PIPELINE_PATH)
        == EXPECTED_PRICE_PIPELINE_SHA256
    )

    assert (
        _calculate_sha256(model_loader.TAPPING_PIPELINE_PATH)
        == EXPECTED_TAPPING_PIPELINE_SHA256
    )


def test_manifest_hashes_agree_with_trusted_hashes():
    """
    The verified runtime manifest must record the same
    trusted hashes for all three models.
    """

    manifest = get_manifest()

    assert (
        manifest_model(manifest, "drc_prediction")[
            "runtime_sha256"
        ].lower()
        == EXPECTED_DRC_PIPELINE_SHA256.lower()
    )

    assert (
        manifest_model(manifest, "price_estimation")[
            "runtime_sha256"
        ].lower()
        == EXPECTED_PRICE_PIPELINE_SHA256.lower()
    )

    assert (
        manifest_model(manifest, "tapping_advisory")[
            "runtime_sha256"
        ].lower()
        == EXPECTED_TAPPING_PIPELINE_SHA256.lower()
    )


# ============================================================
# TEST 3 — MISSING ARTIFACT IS REJECTED
# ============================================================


def test_missing_artifact_raises_model_load_error(tmp_path: Path):
    """
    Verification of a nonexistent artifact must raise a
    controlled ModelLoadError.
    """

    with pytest.raises(
        ModelLoadError,
        match="was not found",
    ):
        _verify_model_file(
            tmp_path / "missing_pipeline.pkl",
            EXPECTED_DRC_PIPELINE_SHA256,
        )


# ============================================================
# TEST 4 — TAMPERED ARTIFACT IS REJECTED, HASH NOT UPDATED
# ============================================================


def test_tampered_artifact_is_rejected_and_trusted_hash_is_not_updated(
    tmp_path: Path,
):
    """
    A single changed byte must fail verification with a
    controlled ModelLoadError, and the trusted hash anchor
    must remain unchanged afterwards.
    """

    tampered = tmp_path / "tampered_drc_pipeline.pkl"
    tampered.write_bytes(
        model_loader.DRC_PIPELINE_PATH.read_bytes() + b"\x00"
    )

    trusted_before = model_loader.EXPECTED_DRC_PIPELINE_SHA256

    with pytest.raises(
        ModelLoadError,
        match="Integrity check failed",
    ):
        _verify_model_file(
            tampered,
            trusted_before,
        )

    assert (
        model_loader.EXPECTED_DRC_PIPELINE_SHA256
        == trusted_before
    )


def test_manifest_disagreeing_with_trusted_hashes_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    """
    A manifest whose recorded runtime hash disagrees with
    the trusted anchor must be rejected before any model is
    loaded, and the trusted anchor must remain unchanged.
    """

    manifest = json.loads(
        model_loader.ARTIFACT_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )
    manifest["models"][0]["runtime_sha256"] = "0" * 64

    bad_manifest = tmp_path / "artifact_manifest.json"
    bad_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    trusted_before = model_loader.EXPECTED_DRC_PIPELINE_SHA256

    monkeypatch.setattr(
        model_loader,
        "ARTIFACT_MANIFEST_PATH",
        bad_manifest,
    )
    model_loader.load_manifest.cache_clear()
    model_loader.load_models.cache_clear()

    try:
        with pytest.raises(
            ModelLoadError,
            match="does not match the trusted",
        ):
            get_manifest()
    finally:
        monkeypatch.undo()
        model_loader.load_manifest.cache_clear()
        model_loader.load_models.cache_clear()

    assert (
        model_loader.EXPECTED_DRC_PIPELINE_SHA256
        == trusted_before
    )


# ============================================================
# TEST 5 — MODELS LOAD AND ARE CACHED
# ============================================================


def test_models_load_and_are_cached():
    """
    All three finalized artifacts must load successfully.

    get_models() is cached, so repeated calls must return
    the same loaded model container and the same model
    objects.
    """

    first = get_models()
    second = get_models()

    assert first is second
    assert first.drc is second.drc
    assert first.price is second.price
    assert first.tapping is second.tapping

    assert hasattr(first.drc, "predict")
    assert hasattr(first.price, "predict")
    assert hasattr(first.tapping.pipeline, "predict")
    assert hasattr(
        first.tapping.pipeline,
        "predict_proba",
    )


# ============================================================
# TEST 6 — MANIFEST FEATURE CONTRACTS
# ============================================================


def test_manifest_records_exact_feature_contracts():
    """
    The manifest must carry the exact ordered feature lists
    and the exact tapping class order taken from the
    authoritative research metadata.
    """

    manifest = get_manifest()

    drc_features = manifest_model(
        manifest,
        "drc_prediction",
    )["features_in_pipeline_order"]

    assert drc_features[-4:] == [
        "visible_latex_condition",
        "estate_id",
        "month",
        "day_of_year",
    ]
    assert len(drc_features) == 12

    price_features = manifest_model(
        manifest,
        "price_estimation",
    )["features_in_pipeline_order"]

    assert price_features[0] == "latex_drc_pct"
    assert "estate_id" not in price_features
    assert len(price_features) == 8

    tapping_features = manifest_model(
        manifest,
        "tapping_advisory",
    )["features_in_pipeline_order"]

    assert "estate_id" not in tapping_features
    assert "visible_latex_condition" not in tapping_features
    assert len(tapping_features) == 10

    assert manifest_model(
        manifest,
        "tapping_advisory",
    )["classes_in_probability_order"] == list(
        TAPPING_CLASSES_IN_PROBABILITY_ORDER
    )


# ============================================================
# TEST 7 — PREDICTION OUTPUT SHAPES AND FINITENESS
# ============================================================


def test_regression_prediction_shapes_and_finite_outputs():
    """
    Both regression pipelines must return one finite
    prediction per input row, including a row where every
    numeric sensor value is missing (median-imputation
    path).
    """

    models = get_models()
    manifest = get_manifest()

    drc_frame = frame_for(
        manifest_model(manifest, "drc_prediction")
    )
    price_frame = frame_for(
        manifest_model(manifest, "price_estimation")
    )

    drc_predictions = models.drc.predict(drc_frame)
    price_predictions = models.price.predict(price_frame)

    assert drc_predictions.shape == (1,)
    assert price_predictions.shape == (1,)

    assert np.all(np.isfinite(drc_predictions))
    assert np.all(np.isfinite(price_predictions))

    missing_drc_frame = frame_for(
        manifest_model(manifest, "drc_prediction"),
        temperature_c=math.nan,
        relative_humidity_pct=math.nan,
        rainfall_last_6h_mm=math.nan,
        soil_moisture_pct=math.nan,
        soil_ph=math.nan,
    )

    missing_predictions = models.drc.predict(
        missing_drc_frame
    )

    assert missing_predictions.shape == (1,)
    assert np.all(np.isfinite(missing_predictions))


def test_tapping_prediction_shapes_probabilities_and_labels():
    """
    The tapping bundle must return one decoded label and an
    (n, 3) probability matrix whose rows sum to 1, with
    column order equal to the trusted class order.
    """

    models = get_models()
    manifest = get_manifest()

    frame = frame_for(
        manifest_model(manifest, "tapping_advisory")
    )

    encoded = models.tapping.pipeline.predict(frame)
    probabilities = models.tapping.pipeline.predict_proba(
        frame
    )
    decoded = models.tapping.label_encoder.inverse_transform(
        encoded
    )

    assert encoded.shape == (1,)
    assert probabilities.shape == (
        1,
        len(TAPPING_CLASSES_IN_PROBABILITY_ORDER),
    )
    assert decoded.shape == (1,)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert probabilities.sum(axis=1) == pytest.approx(
        [1.0],
        abs=1e-6,
    )

    assert tuple(models.tapping.classes) == (
        TAPPING_CLASSES_IN_PROBABILITY_ORDER
    )

    assert decoded[0] in TAPPING_CLASSES_IN_PROBABILITY_ORDER


# ============================================================
# TEST 8 — NON-FINITE INPUTS ARE REJECTED
# ============================================================


@pytest.mark.parametrize(
    ("model_name",),
    [
        ("drc_prediction",),
        ("price_estimation",),
        ("tapping_advisory",),
    ],
)
def test_non_finite_inputs_are_rejected(
    model_name: str,
):
    """
    All three pipelines must reject non-finite (infinity)
    sensor values with ValueError instead of silently
    producing a prediction.
    """

    models = get_models()
    manifest = get_manifest()

    frame = frame_for(
        manifest_model(manifest, model_name),
        temperature_c=math.inf,
    )

    if model_name == "drc_prediction":
        with pytest.raises(ValueError):
            models.drc.predict(frame)
    elif model_name == "price_estimation":
        with pytest.raises(ValueError):
            models.price.predict(frame)
    else:
        with pytest.raises(ValueError):
            models.tapping.pipeline.predict(frame)


# ============================================================
# TEST 9 — RUNTIME ARTIFACTS MATCH THE RESEARCH EXPORTS
# ============================================================


def test_original_research_artifacts_are_unchanged():
    """
    The research exports must still hash to the values the
    manifest records, and since the retrain used the backend's
    own library versions, each runtime copy is byte-identical
    to its research export.
    """

    research_pipelines = (
        model_loader.MODEL_DIR.parents[3]
        / "ml-research"
        / "component_1_weather_tapping"
        / "artifacts"
        / "pipelines"
    )

    manifest = get_manifest()

    expected = {
        "drc_prediction": (
            "component1_drc_prediction_pipeline.pkl",
            EXPECTED_DRC_PIPELINE_SHA256,
        ),
        "price_estimation": (
            "component1_price_estimation_pipeline.pkl",
            EXPECTED_PRICE_PIPELINE_SHA256,
        ),
        "tapping_advisory": (
            "component1_tapping_advisory_pipeline.pkl",
            EXPECTED_TAPPING_PIPELINE_SHA256,
        ),
    }

    for name, (file_name, trusted_hash) in expected.items():
        research_hash = _calculate_sha256(
            research_pipelines / file_name
        )
        entry = manifest_model(manifest, name)

        assert research_hash == entry["original_artifact"]["sha256"]
        assert research_hash == trusted_hash
