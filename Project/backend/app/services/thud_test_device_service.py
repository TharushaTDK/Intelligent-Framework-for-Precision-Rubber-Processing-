"""Capture one new WAV from a server-configured ESP32 recorder."""
from datetime import datetime, timezone
import os
import threading
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import ValidationError

from backend.app.schemas.thud_test import AcousticUploadMetadata
from backend.app.services.thud_test_service import (
    MAX_AUDIO_BYTES, AcousticValidationError, create_measurement,
)


_CAPTURE_LOCK = threading.Lock()


class DeviceConfigurationError(RuntimeError):
    pass


class DeviceBusyError(RuntimeError):
    pass


class DeviceCommunicationError(RuntimeError):
    pass


def configured_device_url():
    base_url = os.environ.get("ACOUSTIC_ESP32_BASE_URL", "").strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Invalid recorder base URL")
        _ = parsed.port
    except ValueError as exc:
        raise DeviceConfigurationError("Configure ACOUSTIC_ESP32_BASE_URL with the recorder base address.") from exc
    return base_url


def device_configuration_status():
    try:
        configured_device_url()
        return {"configured": True, "capture_path": "/record"}
    except DeviceConfigurationError:
        return {"configured": False, "capture_path": "/record"}


def capture_from_device(sheet_id=None, condition_id=None):
    base_url = configured_device_url()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        metadata = AcousticUploadMetadata(
            filename=f"LIVE_{stamp}_{uuid4().hex[:8]}.wav",
            sheet_id=sheet_id, condition_id=condition_id,
        )
    except ValidationError as exc:
        raise AcousticValidationError("Supplied identifiers must contain non-whitespace text.") from exc
    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise DeviceBusyError("A capture is already running. Wait for its result.")
    try:
        content = bytearray()
        try:
            # No automatic retry: another POST would trigger another recording.
            with httpx.Client(
                timeout=httpx.Timeout(30.0, connect=5.0),
                trust_env=False, follow_redirects=False,
            ) as client:
                with client.stream("POST", base_url + "/record") as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise DeviceCommunicationError("ESP32 did not return a completed recording.")
                    for chunk in response.iter_bytes():
                        if len(content) + len(chunk) > MAX_AUDIO_BYTES:
                            raise DeviceCommunicationError("ESP32 recording exceeds the size limit.")
                        content.extend(chunk)
        except httpx.HTTPError as exc:
            raise DeviceCommunicationError(
                "Could not complete ESP32 capture. Check device power, address and the shared Wi-Fi network."
            ) from exc
        if not content:
            raise DeviceCommunicationError("ESP32 returned an empty recording.")
        try:
            return create_measurement(bytes(content), metadata)
        except AcousticValidationError as exc:
            raise DeviceCommunicationError("ESP32 returned an unsupported or incomplete WAV.") from exc
    finally:
        _CAPTURE_LOCK.release()
