from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

LayerCode = Literal["L1", "L2", "L3"]
ReadingSource = Literal["manual", "iot"]
BatchStatus = Literal["active", "completed", "cancelled"]
FireRiskLabel = Literal["Normal", "Warning", "Critical"]

# ============================================================
# SHARED VALIDATION HELPERS
# ============================================================


def _require_timezone_aware_datetime(
    value: datetime | None,
    field_name: str,
) -> datetime | None:
    """
    Require all client-supplied timestamps to contain explicit
    timezone information.

    Valid examples:

        2026-10-15T05:00:00Z
        2026-10-15T10:30:00+05:30

    Invalid example:

        2026-10-15T10:30:00
    """

    if value is None:
        return None

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must include timezone information. "
            "Use an explicit UTC offset or 'Z'."
        )

    return value


def _strip_non_empty_identifier(
    value: object,
    field_name: str,
) -> object:
    """
    Trim surrounding whitespace from string identifiers and
    reject identifiers that contain only whitespace.

    Non-string values are returned unchanged so normal Pydantic
    type validation can handle them.
    """

    if not isinstance(value, str):
        return value

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty or whitespace-only."
        )

    return normalized

# ============================================================
# LAYER SCHEMAS
# ============================================================


class SmokehouseLayerCreate(BaseModel):
    """
    Configuration for one smokehouse layer.

    These values are configured once when a drying batch begins.
    """

    layer_code: LayerCode

    number_of_sheets: int = Field(
        gt=0,
        description="Number of rubber sheets placed on this layer.",
    )

    layer_area_m2: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Usable layer area in square metres.",
    )


class SmokehouseLayerResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    layer_code: LayerCode

    number_of_sheets: int

    layer_area_m2: float = Field(
        allow_inf_nan=False,
    )

# ============================================================
# BATCH SCHEMAS
# ============================================================


class SmokehouseBatchCreate(BaseModel):
    """
    Request used to create/start a smokehouse drying batch.
    """

    batch_code: str = Field(
        min_length=1,
        max_length=50,
        examples=["SH01_B01"],
    )

    smokehouse_id: str = Field(
        min_length=1,
        max_length=50,
        examples=["SH01"],
    )

    started_at: datetime | None = Field(
        default=None,
        description=(
            "Optional batch start timestamp. "
            "If omitted, the backend will assign "
            "the current UTC time."
        ),
    )

    layers: list[SmokehouseLayerCreate] = Field(
        min_length=3,
        max_length=3,
    )

    @field_validator(
        "batch_code",
        "smokehouse_id",
        mode="before",
    )
    @classmethod
    def normalize_identifiers(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        return _strip_non_empty_identifier(
            value,
            info.field_name,
        )

    @field_validator("started_at")
    @classmethod
    def validate_started_at_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return _require_timezone_aware_datetime(
            value,
            "started_at",
        )

    @model_validator(mode="after")
    def validate_three_layers(self):
        """
        Ensure the batch contains exactly:

        L1
        L2
        L3
        """

        layer_codes = [
            layer.layer_code
            for layer in self.layers
        ]

        if set(layer_codes) != {
            "L1",
            "L2",
            "L3",
        }:
            raise ValueError(
                "A smokehouse batch must contain exactly "
                "L1, L2 and L3."
            )

        return self


class SmokehouseBatchResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    batch_code: str
    smokehouse_id: str

    started_at: datetime
    ended_at: datetime | None

    status: BatchStatus

    layers: list[
        SmokehouseLayerResponse
    ]


class SmokehouseBatchStatusUpdate(BaseModel):
    """
    Request used to close an active drying batch.

    Allowed transitions:

        active -> completed
        active -> cancelled

    ended_at is optional.

    If ended_at is omitted, the backend will assign
    the current UTC time.
    """

    status: Literal[
        "completed",
        "cancelled",
    ]

    ended_at: datetime | None = Field(
        default=None,
        description=(
            "Optional batch end timestamp. "
            "If omitted, the backend assigns "
            "the current UTC time."
        ),
    )

    @field_validator("ended_at")
    @classmethod
    def validate_ended_at_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return _require_timezone_aware_datetime(
            value,
            "ended_at",
        )


class SmokehouseBatchSummary(BaseModel):
    """
    Lightweight batch summary used when listing every batch
    recorded for one smokehouse_id. Omits layer configuration,
    which callers can fetch via GET /batches/{batch_code} once
    a batch is selected.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    batch_code: str
    smokehouse_id: str

    started_at: datetime
    ended_at: datetime | None

    status: BatchStatus


class SmokehouseBatchListResponse(BaseModel):
    """
    Every drying batch recorded for one smokehouse_id, most
    recently started first.
    """

    smokehouse_id: str

    batches: list[SmokehouseBatchSummary]

# ============================================================
# SENSOR / MANUAL READING SCHEMAS
# ============================================================


class SmokehouseReadingCreate(BaseModel):
    """
    Raw smokehouse observation received from either:

    - temporary manual web interface
    - future ESP32/DHT22 device

    The client does NOT send engineered ML features.

    The backend automatically constructs the finalized
    17-feature model input using:

    - current sensor values
    - batch configuration
    - layer configuration
    - previous readings from PostgreSQL
    """

    batch_code: str = Field(
        min_length=1,
        max_length=50,
        examples=["SH01_B01"],
    )

    layer_code: LayerCode

    temperature_c: float = Field(
        allow_inf_nan=False,
        description=(
            "Current DHT22 temperature reading "
            "in degrees Celsius."
        ),
    )

    humidity_pct: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
        description=(
            "Current relative humidity percentage."
        ),
    )

    recorded_at: datetime | None = Field(
        default=None,
        description=(
            "Observation timestamp. "
            "If omitted, the backend assigns "
            "the current UTC time."
        ),
    )

    source: ReadingSource = "manual"

    @field_validator(
        "batch_code",
        mode="before",
    )
    @classmethod
    def normalize_batch_code(
        cls,
        value: object,
    ) -> object:
        return _strip_non_empty_identifier(
            value,
            "batch_code",
        )

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return _require_timezone_aware_datetime(
            value,
            "recorded_at",
        )


class SmokehouseReadingResponse(BaseModel):
    """
    Stored smokehouse sensor/manual reading.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    id: int
    layer_id: int

    recorded_at: datetime

    temperature_c: float = Field(
        allow_inf_nan=False,
    )

    humidity_pct: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    source: ReadingSource

# ============================================================
# FIRE-RISK PROBABILITY SCHEMA
# ============================================================


class FireRiskProbabilities(BaseModel):
    """
    Final probability distribution produced by the
    ordinal fire-risk classification pipeline.
    """

    normal: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    warning: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    critical: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

# ============================================================
# PREDICTION RESPONSE
# ============================================================


class SmokehousePredictionResponse(BaseModel):
    """
    Prediction returned by the finalized Component 3 ML system.

    remaining_drying_time_hours intentionally has no >= 0
    constraint.

    The backend preserves the raw output produced by the
    finalized regression model.

    Any future UI-only clamping rule must remain separate
    from the research prediction itself.
    """

    prediction_id: int
    reading_id: int

    predicted_at: datetime

    remaining_drying_time_hours: float = Field(
        allow_inf_nan=False,
    )

    fire_risk_label: FireRiskLabel

    probabilities: FireRiskProbabilities

    guidance: str | None = None

    regression_model_version: str
    classification_model_version: str

# ============================================================
# READING + PREDICTION RESPONSE
# ============================================================


class SmokehouseReadingPredictionResponse(BaseModel):
    """
    Response returned after one raw temperature/humidity
    observation has been stored and both finalized models
    have been executed.

    Contains:

    - persisted raw reading
    - persisted ML prediction
    - historical reading count
    - unavailable history-derived numeric features
    """

    reading: SmokehouseReadingResponse

    prediction: SmokehousePredictionResponse

    missing_numeric_features: list[str]

    historical_reading_count: int = Field(
        ge=0,
    )

# ============================================================
# LATEST LAYER STATUS
# ============================================================


class SmokehouseLatestStatusResponse(BaseModel):
    """
    Latest known state for one layer of one drying batch.

    Intended for:

    - temporary validation dashboard
    - future React Native application
    """

    batch_code: str
    smokehouse_id: str
    layer_code: LayerCode

    latest_reading: SmokehouseReadingResponse

    latest_prediction: (
        SmokehousePredictionResponse | None
    ) = None

# ============================================================
# LAYER HISTORY
# ============================================================


class SmokehouseHistoryItemResponse(BaseModel):
    """
    One historical sensor reading together with its persisted
    Component 3 prediction.

    prediction may be None if a reading exists without an
    associated prediction.
    """

    reading: SmokehouseReadingResponse

    prediction: (
        SmokehousePredictionResponse | None
    ) = None


class SmokehouseLayerHistoryResponse(BaseModel):
    """
    Historical readings and predictions for one smokehouse
    layer within one drying batch.
    """

    batch_code: str
    smokehouse_id: str
    layer_code: LayerCode

    items: list[
        SmokehouseHistoryItemResponse
    ]

# ============================================================
# TEMPORARY BACKEND VALIDATION / PREVIEW SCHEMAS
# ============================================================
#
# These schemas support the temporary stateless prediction
# endpoint:
#
# POST /api/v1/smokehouse/predict-preview
#
# Unlike the persistent POST /readings endpoint, the caller
# explicitly supplies historical readings here.
#
# The client still NEVER supplies engineered ML features.
#
# The backend constructs the exact finalized 17-feature input.
# ============================================================


class SmokehouseHistoryReadingInput(BaseModel):
    """
    One historical raw sensor observation supplied to the
    temporary stateless prediction endpoint.
    """

    recorded_at: datetime

    temperature_c: float = Field(
        allow_inf_nan=False,
    )

    humidity_pct: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        validated = _require_timezone_aware_datetime(
            value,
            "recorded_at",
        )

        assert validated is not None

        return validated


class SmokehousePredictionPreviewRequest(BaseModel):
    """
    Temporary stateless prediction request.

    This endpoint is useful for controlled ML/backend
    validation without writing the supplied observation
    to PostgreSQL.
    """

    smokehouse_id: str = Field(
        min_length=1,
        max_length=50,
        examples=["SH01"],
    )

    layer_id: LayerCode

    temperature_c: float = Field(
        allow_inf_nan=False,
    )

    humidity_pct: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
    )

    recorded_at: datetime

    batch_started_at: datetime

    number_of_sheets: int = Field(
        gt=0,
    )

    layer_area_m2: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    history: list[
        SmokehouseHistoryReadingInput
    ] = Field(
        default_factory=list,
    )

    @field_validator(
        "smokehouse_id",
        mode="before",
    )
    @classmethod
    def normalize_smokehouse_id(
        cls,
        value: object,
    ) -> object:
        return _strip_non_empty_identifier(
            value,
            "smokehouse_id",
        )

    @field_validator(
        "recorded_at",
        "batch_started_at",
    )
    @classmethod
    def validate_preview_timestamp_timezone(
        cls,
        value: datetime,
        info: ValidationInfo,
    ) -> datetime:
        validated = _require_timezone_aware_datetime(
            value,
            info.field_name,
        )

        assert validated is not None

        return validated


class SmokehousePredictionPreviewResponse(BaseModel):
    """
    Temporary stateless prediction response.

    Nothing supplied to this endpoint is persisted
    to PostgreSQL.
    """

    remaining_drying_time_hours: float = Field(
        allow_inf_nan=False,
    )

    fire_risk_label: FireRiskLabel

    probabilities: FireRiskProbabilities

    guidance: str

    missing_numeric_features: list[str]

    historical_reading_count: int = Field(
        ge=0,
    )
