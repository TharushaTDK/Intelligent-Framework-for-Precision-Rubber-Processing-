"""Raw-WAV upload endpoint; no multipart dependency is needed."""
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.thud_test import AcousticMeasurementResponse, AcousticUploadMetadata
from backend.app.services.thud_test_service import (
    MAX_AUDIO_BYTES, AcousticNotFoundError, AcousticServiceError,
    AcousticUnavailableError, AcousticValidationError,
    create_measurement, model_health, retrieve_measurement,
)


router = APIRouter(prefix="/api/v1/thud-test", tags=["Component 4 Acoustic Quality"])


@router.get("/health")
def health():
    return model_health()


@router.post(
    "/measurements", status_code=201, response_model=AcousticMeasurementResponse,
    openapi_extra={"requestBody": {"required": True, "content": {
        "audio/wav": {"schema": {"type": "string", "format": "binary"}}
    }}},
)
async def upload_measurement(
    request: Request,
    filename: str = Query(default="recording.wav", min_length=1, max_length=160),
    sheet_id: str | None = Query(default=None, min_length=1, max_length=64),
    condition_id: str | None = Query(default=None, min_length=1, max_length=64),
):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in ("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Send the WAV bytes directly with Content-Type: audio/wav.")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio exceeds the 10 MiB prototype limit.")
        content.extend(chunk)
    try:
        metadata = AcousticUploadMetadata(filename=filename, sheet_id=sheet_id, condition_id=condition_id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Filename and supplied identifiers must contain non-whitespace text.") from exc
    try:
        return await run_in_threadpool(create_measurement, bytes(content), metadata)
    except AcousticValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AcousticUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AcousticServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/measurements/{measurement_id}", response_model=AcousticMeasurementResponse)
def get_measurement(measurement_id: UUID):
    try:
        return retrieve_measurement(measurement_id)
    except AcousticNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AcousticUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AcousticServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Live ESP32 capture uses a server-configured address, never a client URL.
from backend.app.services.thud_test_device_service import (
    DeviceBusyError, DeviceCommunicationError, DeviceConfigurationError,
    capture_from_device, device_configuration_status,
)


@router.get("/device")
def device_status():
    return device_configuration_status()


@router.post("/capture", status_code=201, response_model=AcousticMeasurementResponse)
async def capture_measurement(
    sheet_id: str | None = Query(default=None, min_length=1, max_length=64),
    condition_id: str | None = Query(default=None, min_length=1, max_length=64),
):
    try:
        return await run_in_threadpool(capture_from_device, sheet_id, condition_id)
    except DeviceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DeviceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DeviceCommunicationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except AcousticValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AcousticUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AcousticServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
