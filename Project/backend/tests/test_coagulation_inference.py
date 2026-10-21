from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.component_2_coagulation import model_loader
from backend.app.services.coagulation_service import (
    CoagulationValidationError,
    run_coagulation_inference,
)

RESEARCH = Path(__file__).resolve().parents[2] / "ml-research" / "component_2_coagulation"


@pytest.fixture(scope="module")
def research_test_data():
    split = pd.read_csv(RESEARCH / "data" / "splits" / "Component2_Test.csv")
    ref = pd.read_csv(RESEARCH / "artifacts" / "metrics" / "test_predictions_vs_actual.csv")
    assert (split["batch_id"].values == ref["batch_id"].values).all()
    return split, ref


def test_models_load_once_and_expose_predict():
    first = model_loader.get_models()
    second = model_loader.get_models()
    assert first is second
    assert hasattr(first.water_acid, "predict")
    assert hasattr(first.coagulation_time, "predict")


def test_metadata_versions_and_feature_order():
    meta = model_loader.get_metadata()
    assert meta.water_acid_version == "component2-water-acid-v2"
    assert meta.coagulation_time_version == "component2-coagulation-time-v2"
    assert meta.water_acid_features == ("drc", "latex_volume_L")
    assert meta.coagulation_time_features == (
        "drc",
        "latex_volume_L",
        "water_amount_L",
        "prepared_1pct_formic_acid_solution_mL",
    )


def test_artifact_hashes_match_trusted_values():
    assert model_loader._sha256(model_loader.WATER_ACID_MODEL_PATH) == model_loader.EXPECTED_WATER_ACID_SHA256
    assert (
        model_loader._sha256(model_loader.COAGULATION_TIME_MODEL_PATH)
        == model_loader.EXPECTED_COAGULATION_TIME_SHA256
    )


def test_tampered_hash_is_rejected(monkeypatch):
    monkeypatch.setattr(model_loader, "EXPECTED_WATER_ACID_SHA256", "0" * 64)
    with pytest.raises(model_loader.ModelLoadError):
        model_loader._verify_file(model_loader.WATER_ACID_MODEL_PATH, model_loader.EXPECTED_WATER_ACID_SHA256)


def test_water_acid_reproduces_research_test_predictions(research_test_data):
    split, ref = research_test_data
    models = model_loader.get_models()
    pred = models.water_acid.predict(split[["drc", "latex_volume_L"]])
    # XGBoost predicts in float32, so compare relative to the value
    np.testing.assert_allclose(pred[:, 0], ref["predicted_water_amount_L"], rtol=1e-6)
    np.testing.assert_allclose(pred[:, 1], ref["predicted_prepared_1pct_formic_acid_solution_mL"], rtol=1e-6)


def test_coagulation_time_reproduces_research_test_predictions(research_test_data):
    split, ref = research_test_data
    models = model_loader.get_models()
    cols = ["drc", "latex_volume_L", "water_amount_L", "prepared_1pct_formic_acid_solution_mL"]
    pred = models.coagulation_time.predict(split[cols])
    np.testing.assert_allclose(pred, ref["predicted_coagulation_time_minutes"], atol=1e-6)


def test_time_uses_water_acid_model_output():
    result = run_coagulation_inference(drc=35.26, latex_volume_L=162.58, include_time=True)
    models = model_loader.get_models()
    x = pd.DataFrame(
        [[35.26, 162.58, result.water_amount_L, result.prepared_1pct_formic_acid_solution_mL]],
        columns=list(model_loader.get_metadata().coagulation_time_features),
    )
    # model minutes come back as hours
    assert result.coagulation_time_hours == pytest.approx(float(models.coagulation_time.predict(x)[0]) / 60)
    assert result.coagulation_time_model_version == "component2-coagulation-time-v2"


def test_water_acid_only_skips_time_model():
    result = run_coagulation_inference(drc=35.26, latex_volume_L=162.58, include_time=False)
    assert result.coagulation_time_hours is None
    assert result.coagulation_time_model_version is None
    assert math.isfinite(result.water_amount_L)
    assert math.isfinite(result.prepared_1pct_formic_acid_solution_mL)


def test_same_input_gives_same_output():
    a = run_coagulation_inference(drc=30.0, latex_volume_L=50.0, include_time=True)
    b = run_coagulation_inference(drc=30.0, latex_volume_L=50.0, include_time=True)
    # trees are summed in parallel so only the last float digit may differ
    assert a.water_amount_L == pytest.approx(b.water_amount_L, rel=1e-12)
    assert a.prepared_1pct_formic_acid_solution_mL == pytest.approx(b.prepared_1pct_formic_acid_solution_mL, rel=1e-12)
    assert a.coagulation_time_hours == pytest.approx(b.coagulation_time_hours, rel=1e-12)


@pytest.mark.parametrize(
    "drc, volume, outside",
    [(43.8, 1.5, False), (40.9, 1.0, False), (46.9, 2.0, False), (38.6, 1.5, True), (43.8, 2.5, True)],
)
def test_training_range_flag(drc, volume, outside):
    result = run_coagulation_inference(drc=drc, latex_volume_L=volume, include_time=False)
    assert result.outside_training_range is outside


@pytest.mark.parametrize(
    "seconds, text",
    [(20, "1 min"), (60 * 30, "30 min"), (3600 * 2, "2 h"), (3600 * 7 + 60 * 36, "7 h 36 min")],
)
def test_alert_duration_text(seconds, text):
    from datetime import timedelta

    from backend.app.services.coagulation_service import _format_duration

    assert _format_duration(timedelta(seconds=seconds)) == text


@pytest.mark.parametrize(
    "drc, volume",
    [(0.0, 10.0), (120.0, 10.0), (float("nan"), 10.0), (30.0, 0.0), (30.0, float("inf"))],
)
def test_invalid_domain_values_are_rejected(drc, volume):
    with pytest.raises(CoagulationValidationError):
        run_coagulation_inference(drc=drc, latex_volume_L=volume, include_time=True)
