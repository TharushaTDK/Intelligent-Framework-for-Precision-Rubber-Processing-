from __future__ import annotations

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import delete

import backend.app.api.routes.weather_tapping as weather_tapping_routes
from backend.app.database.models import TappingAdvisory
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services.advisory_record_service import (
    AdvisoryRecordStorageError,
)


# These tests use the real local PostgreSQL database, like the
# smokehouse API tests. Each test deletes only the rows it created.

client = TestClient(app)

URL = "/api/v1/weather-tapping/advisory-records"


@pytest.fixture
def created_ids():
    ids: list[int] = []
    yield ids
    if ids:
        with SessionLocal() as db:
            db.execute(delete(TappingAdvisory).where(TappingAdvisory.id.in_(ids)))
            db.commit()


def valid_record(**overrides) -> dict:
    record = {
        "latitude": 7.411142,
        "longitude": 79.822483,
        "location_source": "gps",
        "observation_timestamp": "2026-10-19T21:00:00+05:30",
        "forecast_based": False,
        "temperature_c": 27.5,
        "relative_humidity_pct": 80.0,
        "wind_speed_kmh": 3.1,
        "rainfall_last_6h_mm": 0.0,
        "rainfall_last_3days_mm": 8.1,
        "forecast_rain_next_6h_mm": 3.6,
        "predicted_latex_drc_pct": 30.26,
        "drc_model_name": "drc_prediction",
        "drc_runtime_artifact_sha256": "a" * 64,
        "tapping_label": "High",
        "tapping_probability_high": 0.597,
        "tapping_probability_low": 0.137,
        "tapping_probability_moderate": 0.267,
        "predicted_price_lkr_per_kg": 413.0,
    }
    record.update(overrides)
    return record


def test_save_and_read_back_an_advisory(created_ids):
    response = client.post(URL, json=valid_record())

    assert response.status_code == 201
    saved = response.json()
    created_ids.append(saved["id"])

    # Stored exactly as sent.
    assert saved["latitude"] == 7.411142
    assert saved["longitude"] == 79.822483
    assert saved["predicted_latex_drc_pct"] == 30.26
    assert saved["tapping_label"] == "High"
    assert saved["created_at"]

    one = client.get(f"{URL}/{saved['id']}")
    assert one.status_code == 200
    assert one.json() == saved


def test_list_returns_newest_first(created_ids):
    first = client.post(URL, json=valid_record(predicted_latex_drc_pct=29.0)).json()
    second = client.post(URL, json=valid_record(predicted_latex_drc_pct=31.0)).json()
    created_ids.extend([first["id"], second["id"]])

    listing = client.get(URL, params={"limit": 500}).json()
    ids = [r["id"] for r in listing["records"]]

    assert listing["count"] == len(listing["records"])
    assert ids.index(second["id"]) < ids.index(first["id"])


def test_tapping_and_price_are_optional(created_ids):
    # DRC is what Component 2 needs; the others can fail on their own.
    body = valid_record(
        tapping_label=None,
        tapping_probability_high=None,
        tapping_probability_low=None,
        tapping_probability_moderate=None,
        predicted_price_lkr_per_kg=None,
    )
    response = client.post(URL, json=body)

    assert response.status_code == 201
    created_ids.append(response.json()["id"])
    assert response.json()["predicted_price_lkr_per_kg"] is None


def test_forecast_based_flag_is_kept(created_ids):
    response = client.post(URL, json=valid_record(
        observation_timestamp="2026-10-22T21:00:00+05:30",
        forecast_based=True,
    ))

    assert response.status_code == 201
    created_ids.append(response.json()["id"])
    assert response.json()["forecast_based"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"predicted_latex_drc_pct": None},                       # DRC is required
        {"latitude": 95},                                        # out of range
        {"longitude": -181},
        {"observation_timestamp": "2026-10-19T21:00:00"},        # no timezone
        {"location_source": "guess"},
        {"tapping_label": "Maybe"},
        {"tapping_probability_high": 1.5},
        {"drc_runtime_artifact_sha256": "not-a-hash"},
        {"estate_name": "TBD_EST-001"},                          # unknown field
    ],
)
def test_invalid_records_are_rejected_before_saving(overrides):
    response = client.post(URL, json=valid_record(**overrides))

    assert response.status_code == 422


def test_missing_record_is_404():
    response = client.get(f"{URL}/999999999")

    assert response.status_code == 404


def test_database_failure_is_503_without_details(monkeypatch):
    def broken(db, request):
        raise AdvisoryRecordStorageError("connection refused at localhost:5432")

    monkeypatch.setattr(weather_tapping_routes, "create_advisory_record", broken)

    response = client.post(URL, json=valid_record())

    assert response.status_code == 503
    assert "5432" not in response.text
    assert "database" in response.json()["detail"]
