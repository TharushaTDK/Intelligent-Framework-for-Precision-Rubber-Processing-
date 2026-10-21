"""Component 4 assessment storage: PostgreSQL results and immutable WAV files."""
from datetime import datetime, timezone
import logging
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError

from backend.app.database.models import AcousticMeasurement
from backend.app.database.session import SessionLocal
from backend.app.ml.component_4_thud_test.model_loader import (
    ModelLoadError, get_quality_model,
)
from backend.app.ml.component_4_thud_test.inference import assess_recording
from backend.app.schemas.thud_test import AcousticMeasurementResponse, AcousticUploadMetadata


logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 10 * 1024 * 1024
STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "component_4_thud_test"


class AcousticValidationError(ValueError):
    pass


class AcousticUnavailableError(RuntimeError):
    pass


class AcousticNotFoundError(LookupError):
    pass


class AcousticServiceError(RuntimeError):
    pass


def model_health():
    """Model availability only; this endpoint is not a database health check."""
    try:
        _, metadata = get_quality_model()
        return {"component": "component_4_thud_test", "model_available": True,
                "model_version": metadata["model_version"], "status": "PRELIMINARY_DEMO"}
    except ModelLoadError:
        return {"component": "component_4_thud_test", "model_available": False,
                "model_version": None, "status": "MODEL_UNAVAILABLE"}


def measurement_to_row(response: AcousticMeasurementResponse, raw_file_name: str):
    """Shared mapping for new assessments and the explicit legacy importer."""
    if Path(raw_file_name).name != raw_file_name:
        raise AcousticValidationError("Recording reference must be a server-generated basename.")
    return AcousticMeasurement(
        **response.model_dump(mode="python"),
        raw_file_name=raw_file_name,
    )


def create_measurement(content: bytes, metadata: AcousticUploadMetadata):
    if not content:
        raise AcousticValidationError("Audio body is empty.")
    if len(content) > MAX_AUDIO_BYTES:
        raise AcousticValidationError("Audio exceeds the prototype size limit.")

    measurement_id = uuid4()
    raw_file_name = f"{measurement_id}.wav"
    raw_path = STORAGE_DIR / raw_file_name
    wrote_raw = False
    database_attempted = False
    completed = False
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            wrote_raw = True
            file.write(content)

        assessment = assess_recording(raw_path)
        if "FILE_OR_FORMAT_ERROR" in assessment["rejection_reasons"]:
            raise AcousticValidationError("Unsupported, unreadable or incomplete mono PCM WAV.")
        if assessment["quality_status"] == "UNAVAILABLE":
            raise AcousticUnavailableError("Quality model unavailable; no prediction was produced.")

        guidance = (
            "Repeat one clean strike using the standard setup."
            if assessment["quality_status"] == "INVALID" else
            "Recording accepted by the preliminary quality pipeline. Moisture prediction is not available yet."
        )
        response = AcousticMeasurementResponse(
            measurement_id=measurement_id,
            created_at=datetime.now(timezone.utc),
            filename=metadata.filename,
            sheet_id=metadata.sheet_id,
            condition_id=metadata.condition_id,
            source_sha256=assessment["source_sha256"],
            technical_status=assessment["technical_status"],
            quality_status=assessment["quality_status"],
            classifier_status=assessment["classifier_status"],
            decision_source=assessment["decision_source"],
            rejection_reasons=assessment["rejection_reasons"],
            warnings=assessment["warnings"],
            rule_version=assessment["rule_version"],
            model_version=assessment["model_version"],
            invalid_probability_uncalibrated=assessment.get("invalid_probability_uncalibrated"),
            candidate_count=assessment.get("candidate_count", 0),
            onset_seconds=assessment.get("onset_seconds"),
            segment_duration_seconds=assessment.get("segment_duration_seconds"),
            peak_percent_full_scale=assessment.get("peak_percent_full_scale"),
            near_full_scale_fraction=assessment.get("near_full_scale_fraction"),
            guidance=guidance,
        )

        # A raw file is written before the row. If commit acknowledgement fails,
        # the transaction outcome may be uncertain. Retain the WAV in that case
        # rather than risk leaving a committed row with a deleted recording.
        database_attempted = True
        with SessionLocal() as db:
            try:
                db.add(measurement_to_row(response, raw_file_name))
                db.commit()
            except SQLAlchemyError:
                db.rollback()
                raise
        completed = True
        return response

    except (AcousticValidationError, AcousticUnavailableError):
        raise
    except SQLAlchemyError as exc:
        logger.exception("Acoustic database write failed for %s.", measurement_id)
        raise AcousticUnavailableError(
            "The measurement could not be confirmed in the database. "
            f"Keep measurement ID {measurement_id} for checking before repeating capture."
        ) from exc
    except Exception as exc:
        logger.exception("Acoustic measurement failed for %s.", measurement_id)
        raise AcousticServiceError("Acoustic processing failed; no confirmed result is available.") from exc
    finally:
        if wrote_raw and not completed and not database_attempted:
            try:
                raw_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove failed recording %s.", measurement_id)


def retrieve_measurement(measurement_id: UUID):
    try:
        with SessionLocal() as db:
            record = db.get(AcousticMeasurement, measurement_id)
            if record is None:
                raise AcousticNotFoundError("Measurement was not found.")
            # Construct the public response while ORM attributes are available.
            return AcousticMeasurementResponse.model_validate(record, from_attributes=True)
    except AcousticNotFoundError:
        raise
    except SQLAlchemyError as exc:
        logger.exception("Acoustic database read failed.")
        raise AcousticUnavailableError("Saved measurements are temporarily unavailable.") from exc
    except Exception as exc:
        logger.exception("Stored acoustic result could not be read.")
        raise AcousticServiceError("Stored measurement could not be retrieved.") from exc
