"""Component 4 HTTP contracts; no device capture or database writes."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import thud_test as routes
from backend.app.schemas.thud_test import AcousticMeasurementResponse


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client


def result():
    return AcousticMeasurementResponse(
        measurement_id=uuid4(), created_at=datetime.now(timezone.utc),
        filename="fixture.wav", source_sha256="a" * 64,
        technical_status="TECHNICAL_PASSED", quality_status="VALID",
        classifier_status="MODEL_EVALUATED", decision_source="ML_CLASSIFIER",
        rule_version="test-rule", model_version="test-model", candidate_count=1,
        guidance="Test fixture: recording accepted.",
    )


def test_upload_wav_uses_raw_bytes_and_returns_201(client, monkeypatch):
    expected = result()
    seen = []
    def create(content, metadata):
        seen.append((content, metadata))
        return expected
    monkeypatch.setattr(routes, "create_measurement", create)
    response = client.post("/api/v1/thud-test/measurements?filename=fixture.wav&sheet_id=S001",
                           content=b"raw fixture", headers={"Content-Type": "audio/wav"})
    assert response.status_code == 201
    assert response.json() == expected.model_dump(mode="json")
    assert seen[0][0] == b"raw fixture"
    assert seen[0][1].sheet_id == "S001"


def test_wrong_upload_type_returns_415(client):
    response = client.post("/api/v1/thud-test/measurements", json={"audio": "wrong transport"})
    assert response.status_code == 415


@pytest.mark.parametrize("method,path,dependency", [
    ("POST", "/api/v1/thud-test/measurements", "create_measurement"),
    ("GET", "/api/v1/thud-test/measurements/00000000-0000-0000-0000-000000000000", "retrieve_measurement"),
    ("POST", "/api/v1/thud-test/capture", "capture_from_device"),
])
def test_storage_unavailable_returns_503(client, monkeypatch, method, path, dependency):
    def unavailable(*args, **kwargs):
        raise routes.AcousticUnavailableError("Database unavailable.")
    monkeypatch.setattr(routes, dependency, unavailable)
    response = client.request(method, path, content=b"fixture" if dependency == "create_measurement" else None,
                              headers={"Content-Type": "audio/wav"})
    assert response.status_code == 503


def test_missing_measurement_returns_404(client, monkeypatch):
    def missing(*args):
        raise routes.AcousticNotFoundError("Measurement was not found.")
    monkeypatch.setattr(routes, "retrieve_measurement", missing)
    response = client.get(f"/api/v1/thud-test/measurements/{uuid4()}")
    assert response.status_code == 404


def test_capture_calls_backend_once_without_physical_device(client, monkeypatch):
    calls = []
    expected = result()
    def capture(sheet_id, condition_id):
        calls.append((sheet_id, condition_id))
        return expected
    monkeypatch.setattr(routes, "capture_from_device", capture)
    response = client.post("/api/v1/thud-test/capture?sheet_id=S001&condition_id=C01")
    assert response.status_code == 201
    assert calls == [("S001", "C01")]


def test_invalid_uuid_returns_422(client):
    response = client.get("/api/v1/thud-test/measurements/not-a-uuid")
    assert response.status_code == 422
