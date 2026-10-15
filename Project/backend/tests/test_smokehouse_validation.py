from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from backend.app.database.models import SmokehouseBatch
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.schemas.smokehouse import (
    SmokehouseBatchCreate,
    SmokehouseBatchStatusUpdate,
    SmokehouseHistoryReadingInput,
    SmokehousePredictionPreviewRequest,
    SmokehouseReadingCreate,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_BATCH_CODE = "PYTEST_TIMEZONE_BATCH_001"
TEST_SMOKEHOUSE_ID = "PYTEST_TIMEZONE_SH01"

client = TestClient(app)


# ============================================================
# DATABASE CLEANUP
# ============================================================


def cleanup_test_batch() -> None:
    """
    Remove only the dedicated timezone-validation test batch.
    """

    with SessionLocal() as db:

        batch = db.scalar(
            select(
                SmokehouseBatch
            ).where(
                SmokehouseBatch.batch_code
                == TEST_BATCH_CODE
            )
        )

        if batch is not None:
            db.delete(batch)
            db.commit()


@pytest.fixture(autouse=True)
def clean_timezone_test_data():
    """
    Ensure every API validation test starts and ends cleanly.
    """

    cleanup_test_batch()

    yield

    cleanup_test_batch()


# ============================================================
# TEST 1 — AWARE READING TIMESTAMP ACCEPTED
# ============================================================


def test_aware_reading_timestamp_is_accepted():
    """
    Explicit UTC timestamps are valid.
    """

    request = SmokehouseReadingCreate(
        batch_code="TEST",
        layer_code="L1",
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T05:00:00Z",
    )

    assert (
        request.recorded_at
        is not None
    )

    assert (
        request.recorded_at.utcoffset()
        is not None
    )


# ============================================================
# TEST 2 — AWARE OFFSET TIMESTAMP ACCEPTED
# ============================================================


def test_explicit_offset_timestamp_is_accepted():
    """
    A timestamp containing an explicit +05:30 offset is valid.
    """

    request = SmokehouseReadingCreate(
        batch_code="TEST",
        layer_code="L1",
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T10:30:00+05:30",
    )

    assert (
        request.recorded_at
        is not None
    )

    assert (
        request.recorded_at.utcoffset()
        is not None
    )


# ============================================================
# TEST 3 — NAIVE READING TIMESTAMP REJECTED
# ============================================================


def test_naive_reading_timestamp_is_rejected():
    """
    A timestamp without timezone information must never reach
    the runtime feature-engineering pipeline.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "recorded_at must include "
            "timezone information"
        ),
    ):
        SmokehouseReadingCreate(
            batch_code="TEST",
            layer_code="L1",
            temperature_c=40.0,
            humidity_pct=80.0,
            recorded_at="2026-10-15T10:30:00",
        )


# ============================================================
# TEST 4 — NAIVE BATCH START REJECTED
# ============================================================


def test_naive_batch_started_at_is_rejected():
    """
    Batch start timestamps must also be timezone-aware because
    elapsed drying time is calculated from them.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "started_at must include "
            "timezone information"
        ),
    ):
        SmokehouseBatchCreate(
            batch_code="TEST_BATCH",
            smokehouse_id="TEST_SH",
            started_at="2026-10-15T00:00:00",
            layers=[
                {
                    "layer_code": "L1",
                    "number_of_sheets": 40,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L2",
                    "number_of_sheets": 38,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L3",
                    "number_of_sheets": 36,
                    "layer_area_m2": 8.5,
                },
            ],
        )


# ============================================================
# TEST 5 — NAIVE BATCH END REJECTED
# ============================================================


def test_naive_batch_ended_at_is_rejected():
    """
    Explicit batch end timestamps must contain timezone
    information.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "ended_at must include "
            "timezone information"
        ),
    ):
        SmokehouseBatchStatusUpdate(
            status="completed",
            ended_at="2026-10-15T05:00:00",
        )


# ============================================================
# TEST 6 — PREVIEW CURRENT TIMESTAMP REJECTED
# ============================================================


def test_naive_preview_recorded_at_is_rejected():
    """
    The stateless preview endpoint uses the same temporal
    feature logic and therefore needs aware timestamps too.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "recorded_at must include "
            "timezone information"
        ),
    ):
        SmokehousePredictionPreviewRequest(
            smokehouse_id="SH01",
            layer_id="L1",
            temperature_c=40.0,
            humidity_pct=80.0,
            recorded_at="2026-10-15T01:00:00",
            batch_started_at="2026-10-15T00:00:00Z",
            number_of_sheets=40,
            layer_area_m2=8.5,
            history=[],
        )


# ============================================================
# TEST 7 — PREVIEW BATCH START REJECTED
# ============================================================


def test_naive_preview_batch_started_at_is_rejected():
    """
    Preview batch start timestamp must be timezone-aware.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "batch_started_at must include "
            "timezone information"
        ),
    ):
        SmokehousePredictionPreviewRequest(
            smokehouse_id="SH01",
            layer_id="L1",
            temperature_c=40.0,
            humidity_pct=80.0,
            recorded_at="2026-10-15T01:00:00Z",
            batch_started_at="2026-10-15T00:00:00",
            number_of_sheets=40,
            layer_area_m2=8.5,
            history=[],
        )


# ============================================================
# TEST 8 — PREVIEW HISTORY TIMESTAMP REJECTED
# ============================================================


def test_naive_preview_history_timestamp_is_rejected():
    """
    Historical readings supplied to the preview endpoint must
    also use timezone-aware timestamps.
    """

    with pytest.raises(
        ValidationError,
        match=(
            "recorded_at must include "
            "timezone information"
        ),
    ):
        SmokehouseHistoryReadingInput(
            recorded_at="2026-10-15T00:10:00",
            temperature_c=40.0,
            humidity_pct=80.0,
        )


# ============================================================
# TEST 9 — FASTAPI RETURNS 422 FOR NAIVE BATCH START
# ============================================================


def test_api_returns_422_for_naive_batch_timestamp():
    """
    FastAPI must reject an ambiguous batch timestamp before
    calling the database service.
    """

    response = client.post(
        "/api/v1/smokehouse/batches",
        json={
            "batch_code": TEST_BATCH_CODE,
            "smokehouse_id": TEST_SMOKEHOUSE_ID,
            "started_at": "2026-10-15T00:00:00",
            "layers": [
                {
                    "layer_code": "L1",
                    "number_of_sheets": 40,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L2",
                    "number_of_sheets": 38,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L3",
                    "number_of_sheets": 36,
                    "layer_area_m2": 8.5,
                },
            ],
        },
    )

    assert response.status_code == 422

    # Invalid request must not create anything in PostgreSQL.

    with SessionLocal() as db:

        batch = db.scalar(
            select(
                SmokehouseBatch
            ).where(
                SmokehouseBatch.batch_code
                == TEST_BATCH_CODE
            )
        )

        assert batch is None


# ============================================================
# TEST 10 — FASTAPI RETURNS 422 FOR NAIVE READING
# ============================================================


def test_api_returns_422_for_naive_reading_timestamp():
    """
    Verify the persistent readings API rejects a naive
    recorded_at value at request validation.
    """

    # --------------------------------------------------------
    # First create a valid batch.
    # --------------------------------------------------------

    batch_response = client.post(
        "/api/v1/smokehouse/batches",
        json={
            "batch_code": TEST_BATCH_CODE,
            "smokehouse_id": TEST_SMOKEHOUSE_ID,
            "started_at": "2026-10-15T00:00:00Z",
            "layers": [
                {
                    "layer_code": "L1",
                    "number_of_sheets": 40,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L2",
                    "number_of_sheets": 38,
                    "layer_area_m2": 8.5,
                },
                {
                    "layer_code": "L3",
                    "number_of_sheets": 36,
                    "layer_area_m2": 8.5,
                },
            ],
        },
    )

    assert batch_response.status_code == 201

    # --------------------------------------------------------
    # Send an ambiguous timestamp.
    # --------------------------------------------------------

    reading_response = client.post(
        "/api/v1/smokehouse/readings",
        json={
            "batch_code": TEST_BATCH_CODE,
            "layer_code": "L1",
            "temperature_c": 40.0,
            "humidity_pct": 80.0,
            "recorded_at": "2026-10-15T00:10:00",
            "source": "manual",
        },
    )

    assert reading_response.status_code == 422
