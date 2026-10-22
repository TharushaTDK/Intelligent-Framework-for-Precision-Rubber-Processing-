"""Import previous UUID.json + UUID.wav assessments into PostgreSQL.

Run from Project:
  python ml-research/component_4_thud_test/src/tools/import_legacy_measurements.py

Existing JSON/WAV files are never deleted or overwritten. Import does not
rerun a model or change a measurement ID, timestamp, label or score.
"""
import hashlib
from pathlib import Path
import sys
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.models import AcousticMeasurement
from backend.app.database.session import SessionLocal
from backend.app.schemas.thud_test import AcousticMeasurementResponse
from backend.app.services.thud_test_service import STORAGE_DIR, measurement_to_row


def import_legacy_measurements(storage_dir=STORAGE_DIR, session_factory=SessionLocal):
    counts = {"imported": 0, "already_present": 0, "failed": 0}
    for json_path in sorted(Path(storage_dir).glob("*.json")):
        try:
            measurement_id = UUID(json_path.stem)
            response = AcousticMeasurementResponse.model_validate_json(
                json_path.read_text(encoding="utf-8")
            )
            if response.measurement_id != measurement_id:
                raise ValueError("JSON filename and payload IDs differ.")
            raw_file_name = f"{measurement_id}.wav"
            raw_path = Path(storage_dir) / raw_file_name
            if not raw_path.is_file():
                raise ValueError("Matching original recording is missing.")
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != response.source_sha256:
                raise ValueError("Original WAV SHA-256 does not match the saved assessment.")
            with session_factory() as db:
                existing = db.get(AcousticMeasurement, measurement_id)
                if existing is not None:
                    public = AcousticMeasurementResponse.model_validate(existing, from_attributes=True)
                    if (public.model_dump(mode="json") != response.model_dump(mode="json")
                            or existing.raw_file_name != raw_file_name):
                        raise ValueError("Existing database row differs; refusing to overwrite it.")
                    counts["already_present"] += 1
                    continue
                try:
                    db.add(measurement_to_row(response, raw_file_name))
                    db.commit()
                except Exception:
                    db.rollback()
                    raise
            counts["imported"] += 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"FAILED {json_path.name}: {type(exc).__name__}. Check the file and database configuration.")
    print("Legacy import:", counts)
    return counts


if __name__ == "__main__":
    result = import_legacy_measurements()
    raise SystemExit(1 if result["failed"] else 0)
