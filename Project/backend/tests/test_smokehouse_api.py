from __future__ import annotations
import math
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from backend.app.database.models import (
    SmokehouseBatch,
    SmokehouseLayer,
    SmokehousePrediction,
    SmokehouseReading,
)
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services.smokehouse_service import (
    build_fire_risk_guidance,
)
# ============================================================
# TEST CONFIGURATION
# ============================================================
TEST_BATCH_CODE = "PYTEST_API_BATCH_001"
TEST_SMOKEHOUSE_ID = "PYTEST_SH01"
client = TestClient(app)
# ============================================================
# DATABASE CLEANUP
# ============================================================


def cleanup_test_batch() -> None:
    """
    Remove only the dedicated automated-test batch.
    Deleting the batch cascades to:
        batch
          ↓
        layers
          ↓
        readings
          ↓
        predictions
    Real/manual research batches are not touched.
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
def clean_database_test_data():
    """
    Guarantee that every test starts and ends without the
    dedicated pytest batch in PostgreSQL.
    """
    cleanup_test_batch()
    yield
    cleanup_test_batch()
# ============================================================
# HELPERS
# ============================================================


def create_test_batch():
    """
    Create one deterministic test batch using the real API.
    """
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
    return response.json()


def create_l1_reading(
    *,
    temperature_c: float,
    humidity_pct: float,
    recorded_at: str,
):
    """
    Submit one L1 reading through the real FastAPI API.
    """
    return client.post(
        "/api/v1/smokehouse/readings",
        json={
            "batch_code": TEST_BATCH_CODE,
            "layer_code": "L1",
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "recorded_at": recorded_at,
            "source": "manual",
        },
    )
# ============================================================
# TEST 1 — CREATE + GET BATCH
# ============================================================


def test_create_and_get_smokehouse_batch():
    """
    Verify that FastAPI can create a real PostgreSQL batch
    containing exactly L1, L2 and L3 and retrieve it again.
    """
    created = create_test_batch()
    assert (
        created["batch_code"]
        == TEST_BATCH_CODE
    )
    assert (
        created["smokehouse_id"]
        == TEST_SMOKEHOUSE_ID
    )
    assert (
        created["status"]
        == "active"
    )
    assert (
        created["ended_at"]
        is None
    )
    assert len(
        created["layers"]
    ) == 3
    assert {
        layer["layer_code"]
        for layer in created["layers"]
    } == {
        "L1",
        "L2",
        "L3",
    }
    response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
    )
    assert response.status_code == 200
    loaded = response.json()
    assert (
        loaded["batch_code"]
        == TEST_BATCH_CODE
    )
    assert (
        loaded["smokehouse_id"]
        == TEST_SMOKEHOUSE_ID
    )
    assert (
        loaded["status"]
        == "active"
    )
    assert len(
        loaded["layers"]
    ) == 3
# ============================================================
# TEST 2 — COMPLETE READING / PREDICTION WORKFLOW
# ============================================================


def test_reading_prediction_latest_and_history_workflow():
    """
    Validate the main Component 3 backend workflow:
        create batch
            ↓
        first reading
            ↓
        prediction
            ↓
        second reading
            ↓
        PostgreSQL history
            ↓
        feature construction
            ↓
        prediction
            ↓
        latest API
            ↓
        history API
    """
    create_test_batch()
    # --------------------------------------------------------
    # First L1 reading
    # --------------------------------------------------------
    first_response = create_l1_reading(
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T00:10:00Z",
    )
    assert (
        first_response.status_code
        == 201
    )
    first = first_response.json()
    assert (
        first[
            "historical_reading_count"
        ]
        == 0
    )
    assert set(
        first[
            "missing_numeric_features"
        ]
    ) == {
        "Temperature_Rate_per_10min",
        "Humidity_Rate_per_10min",
        "Temperature_Change_30m",
        "Humidity_Change_30m",
        "Temperature_Change_60m",
        "Humidity_Change_60m",
    }
    assert (
        first["reading"][
            "temperature_c"
        ]
        == 40.0
    )
    assert (
        first["reading"][
            "humidity_pct"
        ]
        == 80.0
    )
    assert (
        first["prediction"][
            "regression_model_version"
        ]
        == "GB_02"
    )
    assert (
        first["prediction"][
            "classification_model_version"
        ]
        == "component3-classification-v2-ordinal"
    )
    assert math.isfinite(
        first["prediction"][
            "remaining_drying_time_hours"
        ]
    )
    first_guidance = first["prediction"]["guidance"]
    assert first_guidance is not None
    assert first_guidance == (
        build_fire_risk_guidance(
            first["prediction"]["fire_risk_label"]
        )
    )
    # --------------------------------------------------------
    # Second L1 reading
    # --------------------------------------------------------
    second_response = create_l1_reading(
        temperature_c=42.0,
        humidity_pct=78.0,
        recorded_at="2026-10-15T00:20:00Z",
    )
    assert (
        second_response.status_code
        == 201
    )
    second = second_response.json()
    assert (
        second[
            "historical_reading_count"
        ]
        == 1
    )
    assert set(
        second[
            "missing_numeric_features"
        ]
    ) == {
        "Temperature_Change_30m",
        "Humidity_Change_30m",
        "Temperature_Change_60m",
        "Humidity_Change_60m",
    }
    assert (
        "Temperature_Rate_per_10min"
        not in second[
            "missing_numeric_features"
        ]
    )
    assert (
        "Humidity_Rate_per_10min"
        not in second[
            "missing_numeric_features"
        ]
    )
    second_guidance = second["prediction"]["guidance"]
    assert second_guidance is not None
    assert second_guidance == (
        build_fire_risk_guidance(
            second["prediction"]["fire_risk_label"]
        )
    )
    # --------------------------------------------------------
    # Fire-risk probability validation
    # --------------------------------------------------------
    probabilities = second[
        "prediction"
    ][
        "probabilities"
    ]
    probability_sum = (
        probabilities["normal"]
        + probabilities["warning"]
        + probabilities["critical"]
    )
    assert probability_sum == pytest.approx(
        1.0,
        abs=1e-6,
    )
    # --------------------------------------------------------
    # Latest endpoint
    # --------------------------------------------------------
    latest_response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
        "/layers/L1/latest"
    )
    assert (
        latest_response.status_code
        == 200
    )
    latest = latest_response.json()
    assert (
        latest["layer_code"]
        == "L1"
    )
    assert (
        latest[
            "latest_reading"
        ][
            "temperature_c"
        ]
        == 42.0
    )
    assert (
        latest[
            "latest_reading"
        ][
            "humidity_pct"
        ]
        == 78.0
    )
    assert (
        latest[
            "latest_prediction"
        ][
            "reading_id"
        ]
        == second[
            "reading"
        ][
            "id"
        ]
    )
    assert (
        latest["latest_prediction"]["guidance"]
        == second_guidance
    )
    # --------------------------------------------------------
    # History endpoint
    # --------------------------------------------------------
    history_response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
        "/layers/L1/history"
    )
    assert (
        history_response.status_code
        == 200
    )
    history = history_response.json()
    assert (
        history["layer_code"]
        == "L1"
    )
    assert len(
        history["items"]
    ) == 2
    assert (
        history["items"][0][
            "reading"
        ][
            "temperature_c"
        ]
        == 40.0
    )
    assert (
        history["items"][1][
            "reading"
        ][
            "temperature_c"
        ]
        == 42.0
    )
    assert (
        history["items"][0][
            "prediction"
        ]
        is not None
    )
    assert (
        history["items"][1][
            "prediction"
        ]
        is not None
    )
    assert (
        history["items"][0]["prediction"]["guidance"]
        == first_guidance
    )
    assert (
        history["items"][1]["prediction"]["guidance"]
        == second_guidance
    )
    # --------------------------------------------------------
    # Verify actual PostgreSQL persistence
    # --------------------------------------------------------
    with SessionLocal() as db:
        batch = db.scalar(
            select(
                SmokehouseBatch
            ).where(
                SmokehouseBatch.batch_code
                == TEST_BATCH_CODE
            )
        )
        assert batch is not None
        layer = db.scalar(
            select(
                SmokehouseLayer
            ).where(
                SmokehouseLayer.batch_id
                == batch.id,
                SmokehouseLayer.layer_code
                == "L1",
            )
        )
        assert layer is not None
        reading_count = db.scalar(
            select(
                func.count()
            )
            .select_from(
                SmokehouseReading
            )
            .where(
                SmokehouseReading.layer_id
                == layer.id
            )
        )
        assert reading_count == 2
        prediction_count = db.scalar(
            select(
                func.count()
            )
            .select_from(
                SmokehousePrediction
            )
            .join(
                SmokehouseReading,
                SmokehousePrediction.reading_id
                == SmokehouseReading.id,
            )
            .where(
                SmokehouseReading.layer_id
                == layer.id
            )
        )
        assert prediction_count == 2
        persisted_predictions = db.scalars(
            select(
                SmokehousePrediction
            )
            .join(
                SmokehouseReading,
                SmokehousePrediction.reading_id
                == SmokehouseReading.id,
            )
            .where(
                SmokehouseReading.layer_id
                == layer.id
            )
            .order_by(
                SmokehouseReading.recorded_at.asc(),
                SmokehousePrediction.id.asc(),
            )
        ).all()
        assert len(persisted_predictions) == 2
        assert (
            persisted_predictions[0].guidance
            == first_guidance
        )
        assert (
            persisted_predictions[1].guidance
            == second_guidance
        )
# ============================================================
# TEST 3 — DUPLICATE READING PROTECTION
# ============================================================


def test_duplicate_layer_timestamp_returns_409():
    """
    One layer cannot store two readings at exactly the same
    timestamp.
    """
    create_test_batch()
    first = create_l1_reading(
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T00:10:00Z",
    )
    assert first.status_code == 201
    duplicate = create_l1_reading(
        temperature_c=41.0,
        humidity_pct=79.0,
        recorded_at="2026-10-15T00:10:00Z",
    )
    assert (
        duplicate.status_code
        == 409
    )
    assert (
        "already exists"
        in duplicate.json()[
            "detail"
        ]
    )
# ============================================================
# TEST 4 — LAYER HISTORY ISOLATION
# ============================================================


def test_layer_history_is_isolated():
    """
    L1 readings must never become historical readings for L2.
    This guards against cross-layer feature leakage.
    """
    create_test_batch()
    # --------------------------------------------------------
    # L1 first reading
    # --------------------------------------------------------
    l1_response = create_l1_reading(
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T00:10:00Z",
    )
    assert (
        l1_response.status_code
        == 201
    )
    # --------------------------------------------------------
    # L2 first reading
    # --------------------------------------------------------
    l2_response = client.post(
        "/api/v1/smokehouse/readings",
        json={
            "batch_code":
                TEST_BATCH_CODE,
            "layer_code":
                "L2",
            "temperature_c":
                41.0,
            "humidity_pct":
                79.0,
            "recorded_at":
                "2026-10-15T00:10:00Z",
            "source":
                "manual",
        },
    )
    assert (
        l2_response.status_code
        == 201
    )
    l2 = l2_response.json()
    # L1 history must not be used by L2.
    assert (
        l2[
            "historical_reading_count"
        ]
        == 0
    )
    assert set(
        l2[
            "missing_numeric_features"
        ]
    ) == {
        "Temperature_Rate_per_10min",
        "Humidity_Rate_per_10min",
        "Temperature_Change_30m",
        "Humidity_Change_30m",
        "Temperature_Change_60m",
        "Humidity_Change_60m",
    }
    # --------------------------------------------------------
    # Confirm independent histories
    # --------------------------------------------------------
    l1_history = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
        "/layers/L1/history"
    )
    l2_history = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
        "/layers/L2/history"
    )
    assert (
        l1_history.status_code
        == 200
    )
    assert (
        l2_history.status_code
        == 200
    )
    assert len(
        l1_history.json()["items"]
    ) == 1
    assert len(
        l2_history.json()["items"]
    ) == 1
# ============================================================
# TEST 5 — UNKNOWN BATCH
# ============================================================


def test_reading_for_unknown_batch_returns_404():
    """
    Readings must not be accepted for a batch that does not
    exist.
    """
    response = client.post(
        "/api/v1/smokehouse/readings",
        json={
            "batch_code":
                TEST_BATCH_CODE,
            "layer_code":
                "L1",
            "temperature_c":
                40.0,
            "humidity_pct":
                80.0,
            "recorded_at":
                "2026-10-15T00:10:00Z",
            "source":
                "manual",
        },
    )
    assert (
        response.status_code
        == 404
    )
# ============================================================
# TEST 6 — COMPLETE BATCH
# ============================================================


def test_complete_batch_and_reject_future_readings():
    """
    Completing a batch must:
    - change status to completed
    - save ended_at
    - persist the new state
    - prevent additional sensor readings
    """
    create_test_batch()
    # --------------------------------------------------------
    # Complete batch
    # --------------------------------------------------------
    response = client.patch(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}/status",
        json={
            "status": "completed",
            "ended_at": "2026-10-15T05:00:00Z",
        },
    )
    assert (
        response.status_code
        == 200
    )
    completed = response.json()
    assert (
        completed["status"]
        == "completed"
    )
    assert (
        completed["ended_at"]
        is not None
    )
    # --------------------------------------------------------
    # Verify GET returns completed state
    # --------------------------------------------------------
    get_response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
    )
    assert (
        get_response.status_code
        == 200
    )
    loaded = get_response.json()
    assert (
        loaded["status"]
        == "completed"
    )
    assert (
        loaded["ended_at"]
        is not None
    )
    # --------------------------------------------------------
    # New readings must now be rejected
    # --------------------------------------------------------
    reading_response = create_l1_reading(
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T05:10:00Z",
    )
    assert (
        reading_response.status_code
        == 409
    )
    assert (
        "not active"
        in reading_response.json()[
            "detail"
        ]
    )
# ============================================================
# TEST 7 — CLOSED BATCH CANNOT CHANGE AGAIN
# ============================================================


def test_closed_batch_cannot_transition_again():
    """
    A terminal batch state is final.
    Example:
        active -> cancelled       allowed
        cancelled -> completed    rejected
    """
    create_test_batch()
    # --------------------------------------------------------
    # Cancel active batch
    # --------------------------------------------------------
    cancelled_response = client.patch(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}/status",
        json={
            "status": "cancelled",
            "ended_at": "2026-10-15T04:00:00Z",
        },
    )
    assert (
        cancelled_response.status_code
        == 200
    )
    cancelled = (
        cancelled_response.json()
    )
    assert (
        cancelled["status"]
        == "cancelled"
    )
    assert (
        cancelled["ended_at"]
        is not None
    )
    # --------------------------------------------------------
    # Attempt second transition
    # --------------------------------------------------------
    second_transition = client.patch(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}/status",
        json={
            "status": "completed",
        },
    )
    assert (
        second_transition.status_code
        == 409
    )
    assert (
        "cannot be changed"
        in second_transition.json()[
            "detail"
        ]
    )
    # --------------------------------------------------------
    # Original cancelled state must remain unchanged
    # --------------------------------------------------------
    get_response = client.get(
        f"/api/v1/smokehouse/batches/"
        f"{TEST_BATCH_CODE}"
    )
    assert (
        get_response.status_code
        == 200
    )
    loaded = get_response.json()
    assert (
        loaded["status"]
        == "cancelled"
    )
