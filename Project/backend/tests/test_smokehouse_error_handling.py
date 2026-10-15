from __future__ import annotations

from fastapi.testclient import TestClient

import backend.app.api.routes.smokehouse as smokehouse_routes
from backend.app.main import app
from backend.app.services.smokehouse_service import (
    SmokehouseServiceError,
    build_fire_risk_guidance,
)


client = TestClient(app)


# ============================================================
# SHARED VALID PREVIEW PAYLOAD
# ============================================================


def valid_preview_payload() -> dict:
    return {
        "smokehouse_id": "SH01",
        "layer_id": "L1",
        "temperature_c": 40.0,
        "humidity_pct": 80.0,
        "recorded_at": "2026-10-15T01:00:00Z",
        "batch_started_at": "2026-10-15T00:00:00Z",
        "number_of_sheets": 40,
        "layer_area_m2": 8.5,
        "history": [],
    }


# ============================================================
# TEST 1 — REQUEST VALIDATION STILL RETURNS 422
# ============================================================


def test_preview_request_validation_returns_422():
    """
    Invalid client input must continue to produce HTTP 422.

    This error happens before the service/model layer runs.
    """

    payload = valid_preview_payload()

    payload["recorded_at"] = (
        "2026-10-15T01:00:00"
    )

    response = client.post(
        "/api/v1/smokehouse/predict-preview",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# TEST 2 — INTERNAL SERVICE FAILURE RETURNS 500
# ============================================================


def test_preview_internal_service_failure_returns_500(
    monkeypatch,
):
    """
    A genuine backend/model failure must not be reported as
    a client-side 422 validation problem.
    """

    def fail_prediction(**kwargs):
        raise SmokehouseServiceError(
            "Simulated internal model failure."
        )

    monkeypatch.setattr(
        smokehouse_routes,
        "predict_smokehouse_state",
        fail_prediction,
    )

    response = client.post(
        "/api/v1/smokehouse/predict-preview",
        json=valid_preview_payload(),
    )

    assert response.status_code == 500

    assert (
        "Simulated internal model failure"
        in response.json()["detail"]
    )


# ============================================================
# TEST 3 — PREVIEW RETURNS CLASSIFIER-DERIVED GUIDANCE
# ============================================================


def test_preview_returns_guidance_matching_fire_risk_label():
    """
    A successful stateless preview must return the same
    classifier-derived guidance produced by the service layer.

    This verifies API consistency with the persistent
    POST /readings prediction flow.

    The guidance is explanatory only. It does not implement
    Add Fire / Hold Fire / Reduce Fire decision thresholds.
    """

    response = client.post(
        "/api/v1/smokehouse/predict-preview",
        json=valid_preview_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["fire_risk_label"] in {
        "Normal",
        "Warning",
        "Critical",
    }

    assert isinstance(
        body["guidance"],
        str,
    )

    assert body["guidance"].strip()

    assert body["guidance"] == (
        build_fire_risk_guidance(
            body["fire_risk_label"]
        )
    )
