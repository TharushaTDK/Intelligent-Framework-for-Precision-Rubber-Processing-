"""Component 4 storage integration tests against the configured local database.

Use only the local development/test PostgreSQL database. Assessment is a
controlled test fixture: these tests verify persistence, not model accuracy.
Only rows created by these tests are deleted. WAV files use pytest tmp_path.
"""
import hashlib
import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.exc import OperationalError

from backend.app.database.models import AcousticMeasurement
from backend.app.database.session import SessionLocal
from backend.app.schemas.thud_test import AcousticUploadMetadata
from backend.app.services import thud_test_service as service


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "STORAGE_DIR", tmp_path)
    ids = []
    yield tmp_path, ids
    if ids:
        with SessionLocal() as db:
            db.execute(delete(AcousticMeasurement).where(AcousticMeasurement.measurement_id.in_(ids)))
            db.commit()


@pytest.fixture
def assessment(monkeypatch):
    def evaluate(path):
        return {
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "quality_status": "VALID", "technical_status": "TECHNICAL_PASSED",
            "classifier_status": "MODEL_EVALUATED", "decision_source": "ML_CLASSIFIER",
            "rejection_reasons": [], "warnings": ["HIGH_PEAK_REVIEW"],
            "rule_version": "test-rule", "model_version": "test-model",
            "invalid_probability_uncalibrated": 0.2,
            "candidate_count": 1, "onset_seconds": 1.0,
            "segment_duration_seconds": 0.8, "peak_percent_full_scale": 99.1,
            "near_full_scale_fraction": 0.0,
        }
    monkeypatch.setattr(service, "assess_recording", evaluate)
    return evaluate


def test_valid_measurement_database_roundtrip(storage, assessment):
    directory, ids = storage
    content = b"controlled storage fixture, not audio classification input"
    response = service.create_measurement(content, AcousticUploadMetadata(
        filename="fixture.wav", sheet_id="TEST-SHEET", condition_id="TEST-CONDITION"))
    ids.append(response.measurement_id)
    fetched = service.retrieve_measurement(response.measurement_id)
    assert fetched.model_dump(mode="json") == response.model_dump(mode="json")
    assert fetched.quality_status == "VALID"
    assert fetched.moisture_percent is None
    assert (directory / f"{response.measurement_id}.wav").read_bytes() == content
    assert not list(directory.glob("*.json"))
    with SessionLocal() as db:
        row = db.get(AcousticMeasurement, response.measurement_id)
        assert row.raw_file_name == f"{response.measurement_id}.wav"
        assert row.sheet_id == "TEST-SHEET"
        assert row.warnings == ["HIGH_PEAK_REVIEW"]
        assert row.rejection_reasons == []


def test_technical_rejection_is_saved_without_model_score(storage, assessment, monkeypatch):
    def rejected(path):
        result = assessment(path)
        result.update(quality_status="INVALID", technical_status="TECHNICAL_REJECTED",
                      classifier_status="SKIPPED_TECHNICAL_REJECTION",
                      decision_source="TECHNICAL_RULES", model_version=None,
                      invalid_probability_uncalibrated=None,
                      rejection_reasons=["MULTIPLE_EVENT_CANDIDATES"])
        return result
    monkeypatch.setattr(service, "assess_recording", rejected)
    response = service.create_measurement(b"fixture", AcousticUploadMetadata())
    storage[1].append(response.measurement_id)
    result = service.retrieve_measurement(response.measurement_id)
    assert result.quality_status == "INVALID"
    assert result.model_version is None
    assert result.invalid_probability_uncalibrated is None
    assert result.rejection_reasons == ["MULTIPLE_EVENT_CANDIDATES"]


def test_empty_audio_creates_no_file(storage):
    with pytest.raises(service.AcousticValidationError):
        service.create_measurement(b"", AcousticUploadMetadata())
    assert not list(storage[0].iterdir())


def test_corrupt_wav_is_rejected_and_removed(storage):
    with pytest.raises(service.AcousticValidationError):
        service.create_measurement(b"not a WAV", AcousticUploadMetadata())
    assert not list(storage[0].iterdir())


def test_unavailable_model_removes_unprocessed_recording(storage, assessment, monkeypatch):
    def unavailable(path):
        result = assessment(path)
        result["quality_status"] = "UNAVAILABLE"
        return result
    monkeypatch.setattr(service, "assess_recording", unavailable)
    with pytest.raises(service.AcousticUnavailableError):
        service.create_measurement(b"fixture", AcousticUploadMetadata())
    assert not list(storage[0].iterdir())


def test_missing_measurement(storage):
    with pytest.raises(service.AcousticNotFoundError):
        service.retrieve_measurement(uuid4())


def broken_session():
    raise OperationalError("test failure", {}, RuntimeError("offline"))


def test_database_write_failure_is_not_success_and_retains_wav(storage, assessment, monkeypatch):
    monkeypatch.setattr(service, "SessionLocal", broken_session)
    with pytest.raises(service.AcousticUnavailableError):
        service.create_measurement(b"fixture", AcousticUploadMetadata())
    assert len(list(storage[0].glob("*.wav"))) == 1


def test_database_read_failure_is_unavailable(monkeypatch):
    monkeypatch.setattr(service, "SessionLocal", broken_session)
    with pytest.raises(service.AcousticUnavailableError):
        service.retrieve_measurement(uuid4())


def load_importer():
    project = Path(__file__).resolve().parents[2]
    path = project / "ml-research/component_4_thud_test/src/tools/import_legacy_measurements.py"
    spec = importlib.util.spec_from_file_location("component4_legacy_import_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_import_preserves_result_and_is_idempotent(storage, assessment):
    directory, ids = storage
    response = service.create_measurement(b"legacy fixture", AcousticUploadMetadata())
    ids.append(response.measurement_id)
    json_path = directory / f"{response.measurement_id}.json"
    original_json = response.model_dump_json(indent=2)
    json_path.write_text(original_json, encoding="utf-8")
    with SessionLocal() as db:
        db.execute(delete(AcousticMeasurement).where(AcousticMeasurement.measurement_id == response.measurement_id))
        db.commit()
    importer = load_importer()
    first = importer.import_legacy_measurements(directory, SessionLocal)
    second = importer.import_legacy_measurements(directory, SessionLocal)
    assert first == {"imported": 1, "already_present": 0, "failed": 0}
    assert second == {"imported": 0, "already_present": 1, "failed": 0}
    assert service.retrieve_measurement(response.measurement_id).model_dump(mode="json") == response.model_dump(mode="json")
    assert json_path.read_text(encoding="utf-8") == original_json


def test_legacy_import_refuses_changed_wav(storage, assessment):
    directory, ids = storage
    response = service.create_measurement(b"original", AcousticUploadMetadata())
    ids.append(response.measurement_id)
    (directory / f"{response.measurement_id}.json").write_text(response.model_dump_json(), encoding="utf-8")
    (directory / f"{response.measurement_id}.wav").write_bytes(b"changed")
    result = load_importer().import_legacy_measurements(directory, SessionLocal)
    assert result == {"imported": 0, "already_present": 0, "failed": 1}
