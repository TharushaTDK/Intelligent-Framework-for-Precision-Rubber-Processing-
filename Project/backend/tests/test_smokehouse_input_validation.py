from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.smokehouse import (
    SmokehouseBatchCreate,
    SmokehouseHistoryReadingInput,
    SmokehousePredictionPreviewRequest,
    SmokehouseReadingCreate,
)


# ============================================================
# HELPERS
# ============================================================


def valid_layers():
    return [
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
    ]


# ============================================================
# TEST 1 — WHITESPACE BATCH CODE REJECTED
# ============================================================


def test_whitespace_only_batch_code_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseBatchCreate(
            batch_code="     ",
            smokehouse_id="SH01",
            started_at="2026-10-15T00:00:00Z",
            layers=valid_layers(),
        )


# ============================================================
# TEST 2 — WHITESPACE SMOKEHOUSE ID REJECTED
# ============================================================


def test_whitespace_only_smokehouse_id_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseBatchCreate(
            batch_code="BATCH01",
            smokehouse_id="     ",
            started_at="2026-10-15T00:00:00Z",
            layers=valid_layers(),
        )


# ============================================================
# TEST 3 — IDENTIFIERS ARE TRIMMED
# ============================================================


def test_batch_identifiers_are_trimmed():
    request = SmokehouseBatchCreate(
        batch_code="  BATCH01  ",
        smokehouse_id="  SH01  ",
        started_at="2026-10-15T00:00:00Z",
        layers=valid_layers(),
    )

    assert request.batch_code == "BATCH01"
    assert request.smokehouse_id == "SH01"


# ============================================================
# TEST 4 — READING BATCH CODE IS TRIMMED
# ============================================================


def test_reading_batch_code_is_trimmed():
    request = SmokehouseReadingCreate(
        batch_code="  BATCH01  ",
        layer_code="L1",
        temperature_c=40.0,
        humidity_pct=80.0,
        recorded_at="2026-10-15T00:10:00Z",
    )

    assert request.batch_code == "BATCH01"


# ============================================================
# TEST 5 — WHITESPACE READING BATCH CODE REJECTED
# ============================================================


def test_whitespace_reading_batch_code_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseReadingCreate(
            batch_code="     ",
            layer_code="L1",
            temperature_c=40.0,
            humidity_pct=80.0,
            recorded_at="2026-10-15T00:10:00Z",
        )


# ============================================================
# TEST 6 — NAN TEMPERATURE REJECTED
# ============================================================


def test_nan_temperature_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseReadingCreate(
            batch_code="BATCH01",
            layer_code="L1",
            temperature_c=float("nan"),
            humidity_pct=80.0,
            recorded_at="2026-10-15T00:10:00Z",
        )


# ============================================================
# TEST 7 — INFINITE TEMPERATURE REJECTED
# ============================================================


def test_infinite_temperature_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseReadingCreate(
            batch_code="BATCH01",
            layer_code="L1",
            temperature_c=float("inf"),
            humidity_pct=80.0,
            recorded_at="2026-10-15T00:10:00Z",
        )


# ============================================================
# TEST 8 — NAN HUMIDITY REJECTED
# ============================================================


def test_nan_humidity_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseReadingCreate(
            batch_code="BATCH01",
            layer_code="L1",
            temperature_c=40.0,
            humidity_pct=float("nan"),
            recorded_at="2026-10-15T00:10:00Z",
        )


# ============================================================
# TEST 9 — INFINITE LAYER AREA REJECTED
# ============================================================


def test_infinite_layer_area_is_rejected():
    layers = valid_layers()

    layers[0]["layer_area_m2"] = float("inf")

    with pytest.raises(ValidationError):
        SmokehouseBatchCreate(
            batch_code="BATCH01",
            smokehouse_id="SH01",
            started_at="2026-10-15T00:00:00Z",
            layers=layers,
        )


# ============================================================
# TEST 10 — HISTORY NON-FINITE VALUE REJECTED
# ============================================================


def test_preview_history_non_finite_temperature_is_rejected():
    with pytest.raises(ValidationError):
        SmokehouseHistoryReadingInput(
            recorded_at="2026-10-15T00:10:00Z",
            temperature_c=float("inf"),
            humidity_pct=80.0,
        )


# ============================================================
# TEST 11 — PREVIEW NON-FINITE CURRENT VALUE REJECTED
# ============================================================


def test_preview_non_finite_temperature_is_rejected():
    with pytest.raises(ValidationError):
        SmokehousePredictionPreviewRequest(
            smokehouse_id="SH01",
            layer_id="L1",
            temperature_c=float("nan"),
            humidity_pct=80.0,
            recorded_at="2026-10-15T01:00:00Z",
            batch_started_at="2026-10-15T00:00:00Z",
            number_of_sheets=40,
            layer_area_m2=8.5,
            history=[],
        )
