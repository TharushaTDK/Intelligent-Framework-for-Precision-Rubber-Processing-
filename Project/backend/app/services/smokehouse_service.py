from __future__ import annotations

from dataclasses import dataclass

from datetime import datetime, timezone

from typing import Sequence

import numpy as np

import pandas as pd

from sqlalchemy import select

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from sqlalchemy.orm import Session, selectinload

from backend.app.database.models import (

    SmokehouseBatch,

    SmokehouseLayer,

    SmokehousePrediction,

    SmokehouseReading,

)

from backend.app.ml.component_3_smokehouse.feature_builder import (

    FeatureBuildError,

    HistoricalReading,

    build_feature_row,

)

from backend.app.ml.component_3_smokehouse.model_loader import (

    ModelLoadError,

    get_models,

)

from backend.app.schemas.smokehouse import (

    SmokehouseBatchCreate,

    SmokehouseBatchStatusUpdate,

    SmokehouseReadingCreate,

)

# ============================================================

# SERVICE EXCEPTIONS

# ============================================================


class SmokehouseServiceError(RuntimeError):

    """

    Base exception for Component 3 backend service failures.

    """


class SmokehouseNotFoundError(SmokehouseServiceError):

    """

    Raised when a requested smokehouse resource does not exist.

    """


class SmokehouseConflictError(SmokehouseServiceError):

    """

    Raised when an operation conflicts with existing data.

    """


class SmokehouseValidationError(SmokehouseServiceError):

    """

    Raised when client-supplied data is structurally valid at

    the API schema level but cannot be used to build the

    finalized runtime feature row.

    Examples include temporal/history inconsistencies detected

    by the feature builder.

    This is different from internal model-loading, inference,

    or database failures.

    """

# ============================================================
# FIRE-RISK STATUS GUIDANCE
# ============================================================


def build_fire_risk_guidance(
    fire_risk_label: str,
) -> str:
    """
    Convert the finalized classifier label into a simple
    operator-facing status message.

    IMPORTANT:

    This is NOT the final Add Fire / Hold Fire / Reduce Fire
    decision engine.

    No temperature, humidity, probability, or drying-time
    thresholds are invented here.

    The message only explains the already-produced ML class.
    """

    guidance_by_label = {
        "Normal": (
            "Normal fire-risk condition. "
            "Continue monitoring smokehouse conditions."
        ),
        "Warning": (
            "Warning fire-risk condition. "
            "Check smokehouse conditions before making any "
            "fire-management adjustment."
        ),
        "Critical": (
            "Critical fire-risk condition. "
            "Immediate operator inspection of smokehouse "
            "conditions is required."
        ),
    }

    try:
        return guidance_by_label[fire_risk_label]
    except KeyError as exc:
        raise SmokehouseServiceError(
            "Cannot generate guidance for an unknown "
            "fire-risk label."
        ) from exc


# ============================================================
# ML INFERENCE RESULT
# ============================================================

@dataclass(frozen=True)
class SmokehouseInferenceResult:

    """

    Final runtime prediction result produced by Component 3.

    """

    remaining_drying_time_hours: float

    fire_risk_label: str

    normal_probability: float

    warning_probability: float

    critical_probability: float

    missing_numeric_features: tuple[str, ...]

    historical_reading_count: int

    regression_model_version: str

    classification_model_version: str

    guidance: str

    feature_row: pd.DataFrame

# ============================================================

# SAVED READING + PREDICTION RESULT

# ============================================================


@dataclass(frozen=True)
class SmokehouseReadingPredictionResult:

    """

    Database result after a sensor reading and its prediction

    are successfully persisted.

    """

    reading: SmokehouseReading

    prediction: SmokehousePrediction

    inference: SmokehouseInferenceResult

# ============================================================

# DATABASE — CREATE BATCH

# ============================================================


def create_smokehouse_batch(

    db: Session,

    request: SmokehouseBatchCreate,

) -> SmokehouseBatch:
    """

    Create one smokehouse drying batch together with its

    three configured layers.

    """

    batch_code = request.batch_code.strip()

    smokehouse_id = request.smokehouse_id.strip()

    # --------------------------------------------------------

    # Check duplicate batch code

    # --------------------------------------------------------

    try:

        existing = db.scalar(

            select(

                SmokehouseBatch

            ).where(

                SmokehouseBatch.batch_code

                == batch_code

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while checking batch code."

        ) from exc

    if existing is not None:

        raise SmokehouseConflictError(

            f"Batch code '{batch_code}' already exists."

        )

    # --------------------------------------------------------

    # Resolve start time

    # --------------------------------------------------------

    started_at = (

        request.started_at

        if request.started_at is not None

        else datetime.now(timezone.utc)

    )

    # --------------------------------------------------------

    # Construct batch and layers

    # --------------------------------------------------------

    batch = SmokehouseBatch(

        batch_code=batch_code,

        smokehouse_id=smokehouse_id,

        started_at=started_at,

        status="active",

    )

    batch.layers = [

        SmokehouseLayer(

            layer_code=layer.layer_code,

            number_of_sheets=(

                layer.number_of_sheets

            ),

            layer_area_m2=(

                layer.layer_area_m2

            ),

        )

        for layer in request.layers

    ]

    # --------------------------------------------------------

    # Save

    # --------------------------------------------------------

    try:

        db.add(batch)

        db.commit()

    except IntegrityError as exc:

        db.rollback()

        raise SmokehouseConflictError(

            "Smokehouse batch could not be created because "

            "the data conflicts with an existing database record."

        ) from exc

    except SQLAlchemyError as exc:

        db.rollback()

        raise SmokehouseServiceError(

            "Database error while creating smokehouse batch."

        ) from exc

    # --------------------------------------------------------

    # Read back with layers

    # --------------------------------------------------------

    try:

        saved = db.scalar(

            select(

                SmokehouseBatch

            )

            .options(

                selectinload(

                    SmokehouseBatch.layers

                )

            )

            .where(

                SmokehouseBatch.id

                == batch.id

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while reading the newly created batch."

        ) from exc

    if saved is None:

        raise SmokehouseServiceError(

            "Batch was committed but could not be read back."

        )

    return saved

# ============================================================

# DATABASE — GET BATCH BY CODE

# ============================================================


def get_smokehouse_batch_by_code(

    db: Session,

    batch_code: str,

) -> SmokehouseBatch:
    """

    Retrieve one batch and its three layers.

    """

    normalized_batch_code = str(

        batch_code

    ).strip()

    if not normalized_batch_code:

        raise SmokehouseServiceError(

            "batch_code must not be empty."

        )

    try:

        batch = db.scalar(

            select(

                SmokehouseBatch

            )

            .options(

                selectinload(

                    SmokehouseBatch.layers

                )

            )

            .where(

                SmokehouseBatch.batch_code

                == normalized_batch_code

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving smokehouse batch."

        ) from exc

    if batch is None:

        raise SmokehouseNotFoundError(

            f"Batch '{normalized_batch_code}' was not found."

        )

    return batch


def list_batches_by_smokehouse(

    db: Session,

    smokehouse_id: str,

) -> list[SmokehouseBatch]:
    """

    Retrieve every drying batch recorded for one smokehouse_id,
    most recently started first.

    This is a listing query, not a lookup by primary identifier:
    an empty result means the smokehouse_id has no recorded
    batches yet, not that something went wrong.

    """

    normalized_smokehouse_id = str(

        smokehouse_id

    ).strip()

    if not normalized_smokehouse_id:

        raise SmokehouseServiceError(

            "smokehouse_id must not be empty."

        )

    try:

        batches = db.scalars(

            select(

                SmokehouseBatch

            )

            .where(

                SmokehouseBatch.smokehouse_id

                == normalized_smokehouse_id

            )

            .order_by(

                SmokehouseBatch.started_at.desc()

            )

        ).all()

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while listing smokehouse batches."

        ) from exc

    return list(batches)

# ============================================================

# ML INFERENCE

# ============================================================


def predict_smokehouse_state(

    *,

    smokehouse_id: str,

    layer_id: str,

    temperature_c: float,

    humidity_pct: float,

    recorded_at: datetime,

    batch_started_at: datetime,

    number_of_sheets: int,

    layer_area_m2: float,

    history: Sequence[HistoricalReading],

) -> SmokehouseInferenceResult:
    """

    Run the complete finalized Component 3 inference workflow.

    Raw/current inputs

            +

    historical readings

            ↓

    exact 17-feature row

            ↓

    GB_02 regression

            +

    ordinal fire-risk classifier

    """

    # ========================================================

    # 1. BUILD EXACT MODEL FEATURES

    # ========================================================

    try:

        feature_result = build_feature_row(

            smokehouse_id=smokehouse_id,

            layer_id=layer_id,

            temperature_c=temperature_c,

            humidity_pct=humidity_pct,

            recorded_at=recorded_at,

            batch_started_at=batch_started_at,

            number_of_sheets=number_of_sheets,

            layer_area_m2=layer_area_m2,

            history=history,

        )

    except FeatureBuildError as exc:

        raise SmokehouseValidationError(

            f"Feature construction failed: {exc}"

        ) from exc

    X = feature_result.features

    # ========================================================

    # 2. LOAD FINALIZED MODELS

    # ========================================================

    try:

        models = get_models()

    except ModelLoadError as exc:

        raise SmokehouseServiceError(

            "Finalized Component 3 models could not be loaded."

        ) from exc

    # ========================================================

    # 3. REGRESSION

    # ========================================================

    try:

        drying_prediction = (

            models.drying_time.predict(X)

        )

    except Exception as exc:

        raise SmokehouseServiceError(

            "Drying-time prediction failed."

        ) from exc

    if len(drying_prediction) != 1:

        raise SmokehouseServiceError(

            "Unexpected drying-time prediction shape."

        )

    remaining_drying_time_hours = float(

        drying_prediction[0]

    )

    if not np.isfinite(

        remaining_drying_time_hours

    ):

        raise SmokehouseServiceError(

            "Drying-time model returned a non-finite value."

        )

    # IMPORTANT:

    #

    # Preserve the finalized model's raw regression output.

    # Do not silently clamp negative values here.

    # ========================================================

    # 4. FIRE-RISK CLASSIFICATION

    # ========================================================

    try:

        fire_labels = (

            models.fire_risk.predict(X)

        )

        fire_probabilities = (

            models.fire_risk.predict_proba(X)

        )

    except Exception as exc:

        raise SmokehouseServiceError(

            "Fire-risk prediction failed."

        ) from exc

    if len(fire_labels) != 1:

        raise SmokehouseServiceError(

            "Unexpected fire-risk prediction shape."

        )

    if fire_probabilities.shape != (

        1,

        3,

    ):

        raise SmokehouseServiceError(

            "Unexpected fire-risk probability shape."

        )

    fire_risk_label = str(

        fire_labels[0]

    )

    allowed_labels = {

        "Normal",

        "Warning",

        "Critical",

    }

    if fire_risk_label not in allowed_labels:

        raise SmokehouseServiceError(

            "Fire-risk model returned an unknown label."

        )

    guidance = build_fire_risk_guidance(
        fire_risk_label
    )

    normal_probability = float(

        fire_probabilities[0][0]

    )

    warning_probability = float(

        fire_probabilities[0][1]

    )

    critical_probability = float(

        fire_probabilities[0][2]

    )

    probabilities = np.array(

        [

            normal_probability,

            warning_probability,

            critical_probability,

        ],

        dtype=float,

    )

    if not np.all(

        np.isfinite(

            probabilities

        )

    ):

        raise SmokehouseServiceError(

            "Fire-risk model returned non-finite probabilities."

        )

    if (

        np.any(probabilities < 0)

        or np.any(probabilities > 1)

    ):

        raise SmokehouseServiceError(

            "Fire-risk probabilities are outside [0, 1]."

        )

    if not np.isclose(

        float(probabilities.sum()),

        1.0,

        atol=1e-6,

    ):

        raise SmokehouseServiceError(

            "Fire-risk probabilities do not sum to 1."

        )

    # ========================================================

    # 5. RETURN RESULT

    # ========================================================

    return SmokehouseInferenceResult(

        remaining_drying_time_hours=(

            remaining_drying_time_hours

        ),

        fire_risk_label=(

            fire_risk_label

        ),

        normal_probability=(

            normal_probability

        ),

        warning_probability=(

            warning_probability

        ),

        critical_probability=(

            critical_probability

        ),

        missing_numeric_features=(

            feature_result.missing_numeric_features

        ),

        historical_reading_count=(

            feature_result.historical_reading_count

        ),

        regression_model_version=(

            models.metadata.regression_model_version

        ),

        classification_model_version=(

            models.metadata.classification_model_version

        ),

        guidance=guidance,

        feature_row=X.copy(),

    )

# ============================================================

# DATABASE — CREATE READING + RUN + SAVE PREDICTION

# ============================================================


def create_reading_and_prediction(

    db: Session,

    request: SmokehouseReadingCreate,

) -> SmokehouseReadingPredictionResult:
    """

    Store one new layer reading and automatically generate the

    finalized Component 3 ML prediction.

    Runtime sequence:

        1. Find batch

        2. Find layer

        3. Resolve current reading timestamp

        4. Check duplicate timestamp

        5. Reject stale/out-of-order reading

        6. Load ONLY previous readings

        7. Build exact finalized ML features

        8. Run both finalized models

        9. Save current reading

        10. Save prediction

    The current reading is NOT included inside its own history.

    No future observations are used.

    """

    batch_code = request.batch_code.strip()

    layer_code = str(request.layer_code).strip()

    # ========================================================

    # 1. FIND BATCH + LAYERS

    # ========================================================

    try:

        batch = db.scalar(

            select(

                SmokehouseBatch

            )

            .options(

                selectinload(

                    SmokehouseBatch.layers

                )

            )

            .where(

                SmokehouseBatch.batch_code

                == batch_code

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving smokehouse batch."

        ) from exc

    if batch is None:

        raise SmokehouseNotFoundError(

            f"Batch '{batch_code}' was not found."

        )

    if batch.status != "active":

        raise SmokehouseConflictError(

            f"Batch '{batch_code}' is not active."

        )

    # ========================================================

    # 2. FIND REQUESTED LAYER

    # ========================================================

    layer = next(

        (

            item

            for item in batch.layers

            if item.layer_code == layer_code

        ),

        None,

    )

    if layer is None:

        raise SmokehouseNotFoundError(

            f"Layer '{layer_code}' was not found "

            f"for batch '{batch_code}'."

        )

    # ========================================================

    # 3. RESOLVE READING TIMESTAMP

    # ========================================================

    recorded_at = (

        request.recorded_at

        if request.recorded_at is not None

        else datetime.now(timezone.utc)

    )

    # ========================================================

    # 4. CHECK DUPLICATE LAYER + TIMESTAMP

    # ========================================================

    try:

        existing_reading = db.scalar(

            select(

                SmokehouseReading

            ).where(

                SmokehouseReading.layer_id

                == layer.id,

                SmokehouseReading.recorded_at

                == recorded_at,

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while checking sensor reading."

        ) from exc

    if existing_reading is not None:

        raise SmokehouseConflictError(

            "A reading already exists for "

            f"{batch_code}/{layer_code} "

            f"at {recorded_at.isoformat()}."

        )

    # ========================================================

    # 5. REJECT STALE / OUT-OF-ORDER READINGS

    # ========================================================

    #

    # Runtime readings are append-only per layer.

    #

    # Example:

    #

    #     existing:

    #         00:10

    #         00:30

    #

    #     incoming:

    #         00:20  -> reject

    #

    # Accepting 00:20 after the 00:30 prediction already

    # exists would make the stored temporal history

    # inconsistent because that 00:30 prediction was created

    # without knowledge of the late 00:20 observation.

    #

    # Ordering is checked independently for each layer.

    try:

        latest_existing_reading = db.scalar(

            select(

                SmokehouseReading

            )

            .where(

                SmokehouseReading.layer_id

                == layer.id

            )

            .order_by(

                SmokehouseReading.recorded_at.desc(),

                SmokehouseReading.id.desc(),

            )

            .limit(1)

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while checking the latest "

            "smokehouse reading."

        ) from exc

    if (

        latest_existing_reading is not None

        and recorded_at

        < latest_existing_reading.recorded_at

    ):

        raise SmokehouseConflictError(

            "Reading timestamp is older than the latest "

            f"stored reading for layer '{layer_code}'. "

            "Readings must be submitted in chronological "

            "order for each layer."

        )

    # ========================================================

    # 6. LOAD STRICTLY EARLIER READINGS

    # ========================================================

    try:

        historical_rows = db.scalars(

            select(

                SmokehouseReading

            )

            .where(

                SmokehouseReading.layer_id

                == layer.id,

                SmokehouseReading.recorded_at

                < recorded_at,

            )

            .order_by(

                SmokehouseReading.recorded_at.asc()

            )

        ).all()

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while loading historical readings."

        ) from exc

    history = [

        HistoricalReading(

            recorded_at=row.recorded_at,

            temperature_c=row.temperature_c,

            humidity_pct=row.humidity_pct,

        )

        for row in historical_rows

    ]

    # ========================================================

    # 7. RUN FINALIZED ML WORKFLOW

    # ========================================================

    inference = predict_smokehouse_state(

        smokehouse_id=batch.smokehouse_id,

        layer_id=layer.layer_code,

        temperature_c=request.temperature_c,

        humidity_pct=request.humidity_pct,

        recorded_at=recorded_at,

        batch_started_at=batch.started_at,

        number_of_sheets=layer.number_of_sheets,

        layer_area_m2=layer.layer_area_m2,

        history=history,

    )

    # ========================================================

    # 8. CREATE READING ORM OBJECT

    # ========================================================

    reading = SmokehouseReading(

        layer_id=layer.id,

        recorded_at=recorded_at,

        temperature_c=request.temperature_c,

        humidity_pct=request.humidity_pct,

        source=request.source,

    )

    # ========================================================

    # 9. SAVE READING + PREDICTION IN ONE TRANSACTION

    # ========================================================

    try:

        db.add(reading)

        # Flush gives us reading.id without committing yet.

        db.flush()

        prediction = SmokehousePrediction(

            reading_id=reading.id,

            predicted_at=datetime.now(

                timezone.utc

            ),

            remaining_drying_time_hours=(

                inference.remaining_drying_time_hours

            ),

            fire_risk_label=(

                inference.fire_risk_label

            ),

            normal_probability=(

                inference.normal_probability

            ),

            warning_probability=(

                inference.warning_probability

            ),

            critical_probability=(

                inference.critical_probability

            ),

            regression_model_version=(

                inference.regression_model_version

            ),

            classification_model_version=(

                inference.classification_model_version

            ),

            guidance=inference.guidance,

        )

        db.add(prediction)

        db.commit()

        db.refresh(reading)

        db.refresh(prediction)

    except IntegrityError as exc:

        db.rollback()

        raise SmokehouseConflictError(

            "The sensor reading or prediction conflicts "

            "with an existing database record."

        ) from exc

    except SQLAlchemyError as exc:

        db.rollback()

        raise SmokehouseServiceError(

            "Database error while saving the reading "

            "and prediction."

        ) from exc

    # ========================================================

    # 10. RETURN COMPLETE SAVED RESULT

    # ========================================================

    return SmokehouseReadingPredictionResult(

        reading=reading,

        prediction=prediction,

        inference=inference,

    )

# ============================================================

# LATEST / HISTORY RESULT OBJECTS

# ============================================================


@dataclass(frozen=True)
class SmokehouseLatestStatusResult:

    """

    Latest persisted reading and prediction for one layer.

    """

    batch_code: str

    smokehouse_id: str

    layer_code: str

    latest_reading: SmokehouseReading

    latest_prediction: SmokehousePrediction | None


@dataclass(frozen=True)
class SmokehouseHistoryItemResult:

    """

    One stored reading together with its latest persisted

    prediction.

    """

    reading: SmokehouseReading

    prediction: SmokehousePrediction | None


@dataclass(frozen=True)
class SmokehouseLayerHistoryResult:

    """

    Complete stored history for one layer in one batch.

    """

    batch_code: str

    smokehouse_id: str

    layer_code: str

    items: tuple[

        SmokehouseHistoryItemResult,

        ...,

    ]

# ============================================================

# INTERNAL — FIND LAYER

# ============================================================


def _find_batch_layer(

    batch: SmokehouseBatch,

    layer_code: str,

) -> SmokehouseLayer:
    """

    Find one configured layer inside a loaded batch.

    """

    normalized_layer_code = str(

        layer_code

    ).strip()

    if normalized_layer_code not in {

        "L1",

        "L2",

        "L3",

    }:

        raise SmokehouseNotFoundError(

            f"Unknown layer '{normalized_layer_code}'."

        )

    layer = next(

        (

            item

            for item in batch.layers

            if item.layer_code

            == normalized_layer_code

        ),

        None,

    )

    if layer is None:

        raise SmokehouseNotFoundError(

            f"Layer '{normalized_layer_code}' "

            f"was not found for batch "

            f"'{batch.batch_code}'."

        )

    return layer

# ============================================================

# DATABASE — GET LATEST LAYER STATUS

# ============================================================


def get_latest_layer_status(

    db: Session,

    batch_code: str,

    layer_code: str,

) -> SmokehouseLatestStatusResult:
    """

    Retrieve the latest stored reading for one layer together

    with its latest persisted prediction.

    No ML inference is performed here.

    This function only reads data that has already been stored

    in PostgreSQL.

    """

    # --------------------------------------------------------

    # 1. Load batch

    # --------------------------------------------------------

    batch = get_smokehouse_batch_by_code(

        db,

        batch_code,

    )

    # --------------------------------------------------------

    # 2. Find requested layer

    # --------------------------------------------------------

    layer = _find_batch_layer(

        batch,

        layer_code,

    )

    # --------------------------------------------------------

    # 3. Find latest reading

    # --------------------------------------------------------

    try:

        latest_reading = db.scalar(

            select(

                SmokehouseReading

            )

            .where(

                SmokehouseReading.layer_id

                == layer.id

            )

            .order_by(

                SmokehouseReading.recorded_at.desc(),

                SmokehouseReading.id.desc(),

            )

            .limit(1)

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving "

            "the latest layer reading."

        ) from exc

    if latest_reading is None:

        raise SmokehouseNotFoundError(

            f"No readings were found for "

            f"{batch.batch_code}/{layer.layer_code}."

        )

    # --------------------------------------------------------

    # 4. Find latest prediction for that reading

    # --------------------------------------------------------

    try:

        latest_prediction = db.scalar(

            select(

                SmokehousePrediction

            )

            .where(

                SmokehousePrediction.reading_id

                == latest_reading.id

            )

            .order_by(

                SmokehousePrediction.predicted_at.desc(),

                SmokehousePrediction.id.desc(),

            )

            .limit(1)

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving "

            "the latest prediction."

        ) from exc

    # --------------------------------------------------------

    # 5. Return result

    # --------------------------------------------------------

    return SmokehouseLatestStatusResult(

        batch_code=batch.batch_code,

        smokehouse_id=batch.smokehouse_id,

        layer_code=layer.layer_code,

        latest_reading=latest_reading,

        latest_prediction=latest_prediction,

    )

# ============================================================

# DATABASE — GET LAYER HISTORY

# ============================================================


def get_layer_history(

    db: Session,

    batch_code: str,

    layer_code: str,

) -> SmokehouseLayerHistoryResult:
    """

    Retrieve all stored readings for one layer in chronological

    order together with each reading's latest persisted

    prediction.

    No feature engineering or ML inference is performed here.

    """

    # --------------------------------------------------------

    # 1. Load batch

    # --------------------------------------------------------

    batch = get_smokehouse_batch_by_code(

        db,

        batch_code,

    )

    # --------------------------------------------------------

    # 2. Find requested layer

    # --------------------------------------------------------

    layer = _find_batch_layer(

        batch,

        layer_code,

    )

    # --------------------------------------------------------

    # 3. Load readings chronologically

    # --------------------------------------------------------

    try:

        readings = db.scalars(

            select(

                SmokehouseReading

            )

            .where(

                SmokehouseReading.layer_id

                == layer.id

            )

            .order_by(

                SmokehouseReading.recorded_at.asc(),

                SmokehouseReading.id.asc(),

            )

        ).all()

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving "

            "layer history."

        ) from exc

    # --------------------------------------------------------

    # 4. Empty history is valid

    # --------------------------------------------------------

    if not readings:

        return SmokehouseLayerHistoryResult(

            batch_code=batch.batch_code,

            smokehouse_id=batch.smokehouse_id,

            layer_code=layer.layer_code,

            items=(),

        )

    # --------------------------------------------------------

    # 5. Load predictions for those readings

    # --------------------------------------------------------

    reading_ids = [

        reading.id

        for reading in readings

    ]

    try:

        predictions = db.scalars(

            select(

                SmokehousePrediction

            )

            .where(

                SmokehousePrediction.reading_id.in_(

                    reading_ids

                )

            )

            .order_by(

                SmokehousePrediction.reading_id.asc(),

                SmokehousePrediction.predicted_at.desc(),

                SmokehousePrediction.id.desc(),

            )

        ).all()

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving "

            "historical predictions."

        ) from exc

    # --------------------------------------------------------

    # 6. Keep latest prediction for each reading

    #

    # Normally each reading currently has one prediction.

    # This also safely supports future model re-runs.

    # --------------------------------------------------------

    latest_prediction_by_reading: dict[

        int,

        SmokehousePrediction,

    ] = {}

    for prediction in predictions:

        latest_prediction_by_reading.setdefault(

            prediction.reading_id,

            prediction,

        )

    # --------------------------------------------------------

    # 7. Combine reading + prediction

    # --------------------------------------------------------

    items = tuple(

        SmokehouseHistoryItemResult(

            reading=reading,

            prediction=(

                latest_prediction_by_reading.get(

                    reading.id

                )

            ),

        )

        for reading in readings

    )

    return SmokehouseLayerHistoryResult(

        batch_code=batch.batch_code,

        smokehouse_id=batch.smokehouse_id,

        layer_code=layer.layer_code,

        items=items,

    )

# ============================================================

# DATABASE — COMPLETE / CANCEL BATCH

# ============================================================


def update_smokehouse_batch_status(

    db: Session,

    batch_code: str,

    request: SmokehouseBatchStatusUpdate,

) -> SmokehouseBatch:
    """

    Close an active drying batch.

    Allowed transitions:

        active -> completed

        active -> cancelled

    Once a batch is completed or cancelled, it cannot be

    reopened or moved to another final state.

    If ended_at is not supplied, the current UTC time is used.

    """

    normalized_batch_code = str(

        batch_code

    ).strip()

    if not normalized_batch_code:

        raise SmokehouseServiceError(

            "batch_code must not be empty."

        )

    # ========================================================

    # 1. LOAD BATCH

    # ========================================================

    try:

        batch = db.scalar(

            select(

                SmokehouseBatch

            )

            .options(

                selectinload(

                    SmokehouseBatch.layers

                )

            )

            .where(

                SmokehouseBatch.batch_code

                == normalized_batch_code

            )

        )

    except SQLAlchemyError as exc:

        raise SmokehouseServiceError(

            "Database error while retrieving smokehouse batch."

        ) from exc

    if batch is None:

        raise SmokehouseNotFoundError(

            f"Batch '{normalized_batch_code}' was not found."

        )

    # ========================================================

    # 2. REQUIRE ACTIVE STATE

    # ========================================================

    if batch.status != "active":

        raise SmokehouseConflictError(

            f"Batch '{normalized_batch_code}' is already "

            f"'{batch.status}' and cannot be changed."

        )

    # ========================================================

    # 3. RESOLVE END TIME

    # ========================================================

    ended_at = (

        request.ended_at

        if request.ended_at is not None

        else datetime.now(timezone.utc)

    )

    # Batch end cannot occur before batch start.

    try:

        ends_before_start = (

            ended_at < batch.started_at

        )

    except TypeError as exc:

        raise SmokehouseServiceError(

            "Batch timestamps must use compatible timezone "

            "information."

        ) from exc

    if ends_before_start:

        raise SmokehouseConflictError(

            "Batch end time cannot be earlier than "

            "the batch start time."

        )

    # ========================================================

    # 4. UPDATE BATCH

    # ========================================================

    batch.status = request.status

    batch.ended_at = ended_at

    # ========================================================

    # 5. COMMIT

    # ========================================================

    try:

        db.commit()

        db.refresh(batch)

    except SQLAlchemyError as exc:

        db.rollback()

        raise SmokehouseServiceError(

            "Database error while updating smokehouse "

            "batch status."

        ) from exc

    # ========================================================

    # 6. RETURN UPDATED BATCH

    # ========================================================

    return batch
