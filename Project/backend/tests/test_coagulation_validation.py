from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.coagulation import (
    CoagulationPreviewRequest,
    CoagulationRecommendationCreate,
    CoagulationStatusUpdate,
)
from backend.app.services.coagulation_service import hours_and_minutes


def valid_request(**overrides):
    data = {"location_id": 1, "latex_volume_L": 50.0, "outputs": ["water_acid"]}
    data.update(overrides)
    return data


def test_valid_request_and_estate_id_is_trimmed():
    req = CoagulationRecommendationCreate.model_validate(
        valid_request(location_id=None, estate_id="  EST-027  ", drc=40.0)
    )
    assert req.estate_id == "EST-027"


@pytest.mark.parametrize("estate_id", ["", "   ", "E" * 51])
def test_bad_estate_id_is_rejected(estate_id):
    with pytest.raises(ValidationError):
        CoagulationRecommendationCreate.model_validate(valid_request(location_id=None, estate_id=estate_id, drc=40.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"location_id": None},
        {"location_id": None, "estate_id": "EST-027"},
        {"location_id": None, "drc": 40.0},
        {"location_id": 0},
        {"location_id": -3},
    ],
)
def test_location_or_manual_drc_is_needed(overrides):
    with pytest.raises(ValidationError):
        CoagulationRecommendationCreate.model_validate(valid_request(**overrides))


@pytest.mark.parametrize("volume", [0, -5, float("nan"), float("inf"), "abc"])
def test_bad_latex_volume_is_rejected(volume):
    with pytest.raises(ValidationError):
        CoagulationRecommendationCreate.model_validate(valid_request(latex_volume_L=volume))


@pytest.mark.parametrize(
    "outputs",
    [[], ["water"], ["water_acid", "water_acid"], None],
)
def test_bad_outputs_are_rejected(outputs):
    with pytest.raises(ValidationError):
        CoagulationRecommendationCreate.model_validate(valid_request(outputs=outputs))


def test_both_outputs_are_allowed():
    req = CoagulationRecommendationCreate.model_validate(
        valid_request(outputs=["water_acid", "coagulation_time"])
    )
    assert req.outputs == ["water_acid", "coagulation_time"]


@pytest.mark.parametrize("drc", [0, -1, 100.5, float("nan"), float("inf")])
def test_preview_rejects_bad_drc(drc):
    with pytest.raises(ValidationError):
        CoagulationPreviewRequest.model_validate(
            {"drc": drc, "latex_volume_L": 50.0, "outputs": ["coagulation_time"]}
        )


def test_status_update_requires_timezone():
    with pytest.raises(ValidationError):
        CoagulationStatusUpdate.model_validate(
            {"status": "in_progress", "changed_at": "2026-10-17T08:00:00"}
        )
    ok = CoagulationStatusUpdate.model_validate(
        {"status": "in_progress", "changed_at": "2026-10-17T08:00:00+05:30"}
    )
    assert ok.changed_at.utcoffset() is not None


def test_status_update_rejects_unknown_status():
    with pytest.raises(ValidationError):
        CoagulationStatusUpdate.model_validate({"status": "recommended"})


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (7.7966, "7 hours 48 minutes"),
        (7.79, "7 hours 47 minutes"),
        (1.0, "1 hour"),
        (2.0, "2 hours"),
        (0.5, "30 minutes"),
        (1.0167, "1 hour 1 minute"),
        (0.0, "0 minutes"),
    ],
)
def test_hours_and_minutes(hours, expected):
    assert hours_and_minutes(hours) == expected
