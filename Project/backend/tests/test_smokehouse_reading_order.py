from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import SmokehouseBatch
from backend.app.database.session import SessionLocal
from backend.app.main import app


TEST_BATCH_CODE = "PYTEST_ORDER_BATCH_001"
TEST_SMOKEHOUSE_ID = "PYTEST_ORDER_SH01"

client = TestClient(app)


# ============================================================
# CLEANUP
# ============================================================


def cleanup_test_batch() -> None:
    """
    Delete only the dedicated reading-order test batch.
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
def clean_test_data():
    cleanup_test_batch()

    yield

    cleanup_test_batch()


# ============================================================
# HELPERS
# ============================================================


def create_test_batch() -> None:
    response = client.post(
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

    assert response.status_code == 201


def create_reading(
    *,
    layer_code: str,
    recorded_at: str,
    temperature_c: float = 40.0,
    humidity_pct: float = 80.0,
    source: str = "iot",
):
    return client.post(
        "/api/v1/smokehouse/readings",
        json={
            "batch_code": TEST_BATCH_CODE,
            "layer_code": layer_code,
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "recorded_at": recorded_at,
            "source": source,
        },
    )


# ============================================================
# TEST 1 — NORMAL CHRONOLOGICAL READINGS
# ============================================================


def test_chronological_readings_are_accepted():
    """
    Normal append-only sensor readings must continue to work.
    """

    create_test_batch()

    first = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:10:00Z",
        temperature_c=40.0,
        humidity_pct=80.0,
    )

    second = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:20:00Z",
        temperature_c=42.0,
        humidity_pct=78.0,
    )

    third = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:30:00Z",
        temperature_c=44.0,
        humidity_pct=76.0,
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201


# ============================================================
# TEST 2 — STALE / OUT-OF-ORDER READING
# ============================================================


def test_stale_reading_is_rejected():
    """
    Once L1 already contains a later observation, an older
    reading must not be inserted afterwards.

    Example:

        00:10 accepted
        00:30 accepted
        00:20 rejected

    Otherwise the prediction already generated for 00:30
    would have been produced without knowledge of 00:20.
    """

    create_test_batch()

    first = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:10:00Z",
        temperature_c=40.0,
        humidity_pct=80.0,
    )

    later = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:30:00Z",
        temperature_c=44.0,
        humidity_pct=76.0,
    )

    stale = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:20:00Z",
        temperature_c=42.0,
        humidity_pct=78.0,
    )

    assert first.status_code == 201
    assert later.status_code == 201

    assert stale.status_code == 409

    detail = stale.json()["detail"].lower()

    assert (
        "older" in detail
        or "latest" in detail
        or "chronological" in detail
        or "out-of-order" in detail
    )


# ============================================================
# TEST 3 — SAME TIMESTAMP STILL REJECTED
# ============================================================


def test_same_timestamp_is_rejected():
    """
    A timestamp equal to the latest reading is still invalid.

    The existing database uniqueness rule also protects this,
    but the API must continue returning 409.
    """

    create_test_batch()

    first = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:10:00Z",
    )

    duplicate = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:10:00Z",
        temperature_c=41.0,
        humidity_pct=79.0,
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409


# ============================================================
# TEST 4 — ORDERING IS PER LAYER
# ============================================================


def test_reading_order_is_independent_per_layer():
    """
    L1 and L2 maintain independent timelines.

    If L1 is already at 00:30, that must not prevent L2 from
    receiving its own 00:10 and 00:20 observations.
    """

    create_test_batch()

    l1 = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:30:00Z",
        temperature_c=45.0,
        humidity_pct=75.0,
    )

    l2_first = create_reading(
        layer_code="L2",
        recorded_at="2026-10-15T00:10:00Z",
        temperature_c=41.0,
        humidity_pct=79.0,
    )

    l2_second = create_reading(
        layer_code="L2",
        recorded_at="2026-10-15T00:20:00Z",
        temperature_c=43.0,
        humidity_pct=77.0,
    )

    assert l1.status_code == 201
    assert l2_first.status_code == 201
    assert l2_second.status_code == 201

    assert (
        l2_second.json()[
            "historical_reading_count"
        ]
        == 1
    )


# ============================================================
# TEST 5 — REJECTION MUST NOT ALTER HISTORY
# ============================================================


def test_rejected_stale_reading_is_not_persisted():
    """
    Rejecting a late observation must leave the stored layer
    history unchanged.
    """

    create_test_batch()

    first = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:10:00Z",
    )

    latest = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:30:00Z",
    )

    stale = create_reading(
        layer_code="L1",
        recorded_at="2026-10-15T00:20:00Z",
    )

    assert first.status_code == 201
    assert latest.status_code == 201
    assert stale.status_code == 409

    history_response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}/layers/L1/history"
    )

    assert history_response.status_code == 200

    items = history_response.json()["items"]

    assert len(items) == 2

    stored_times = [
        item["reading"]["recorded_at"]
        for item in items
    ]

    assert not any(
        "00:20:00" in timestamp
        for timestamp in stored_times
    )
