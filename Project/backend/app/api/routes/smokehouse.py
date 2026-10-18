from __future__ import annotations

from typing import Callable

from fastapi import (

    APIRouter,

    Depends,

    HTTPException,

    Request,

    status,

)

from fastapi.exceptions import RequestValidationError

from fastapi.responses import JSONResponse

from fastapi.routing import APIRoute

from starlette.responses import Response

from sqlalchemy.orm import Session

from backend.app.database.models import SmokehousePrediction

from backend.app.database.session import get_db

from backend.app.ml.component_3_smokehouse.feature_builder import (

    HistoricalReading,

)

from backend.app.schemas.smokehouse import (

    FireRiskProbabilities,

    LayerCode,

    SmokehouseBatchCreate,

    SmokehouseBatchResponse,

    SmokehouseBatchStatusUpdate,

    SmokehouseHistoryItemResponse,

    SmokehouseLatestStatusResponse,

    SmokehouseLayerHistoryResponse,

    SmokehousePredictionPreviewRequest,

    SmokehousePredictionPreviewResponse,

    SmokehousePredictionResponse,

    SmokehouseReadingCreate,

    SmokehouseReadingPredictionResponse,

    SmokehouseReadingResponse,

)

from backend.app.services.smokehouse_service import (

    SmokehouseConflictError,

    SmokehouseNotFoundError,

    SmokehouseServiceError,

    SmokehouseValidationError,

    create_reading_and_prediction,

    create_smokehouse_batch,

    get_layer_history,

    get_latest_layer_status,

    get_smokehouse_batch_by_code,

    predict_smokehouse_state,

    update_smokehouse_batch_status,

)

# ============================================================

# ROUTER
#
# NaN/Infinity payload values pass pydantic's allow_inf_nan=False
# check correctly, but FastAPI's default 422 handler echoes the
# rejected input back into the error response, and Starlette's
# JSONResponse serializes with allow_nan=False. That makes
# *building* the 422 body itself raise, which surfaces as a 500.
# This route class strips the unserializable fields before the
# response is built, scoped to Component 3 only.

# ============================================================


class _SmokehouseAPIRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def safe_route_handler(
            request: Request,
        ) -> Response:
            try:
                return await original_route_handler(request)
            except RequestValidationError as exc:
                errors = [
                    {
                        key: value
                        for key, value in error.items()
                        if key not in ("input", "ctx")
                    }
                    for error in exc.errors()
                ]

                return JSONResponse(
                    status_code=(
                        status.HTTP_422_UNPROCESSABLE_CONTENT
                    ),
                    content={"detail": errors},
                )

        return safe_route_handler


router = APIRouter(

    prefix="/api/v1/smokehouse",

    tags=["Component 3 - Smokehouse"],

    route_class=_SmokehouseAPIRoute,

)

# ============================================================

# INTERNAL RESPONSE HELPER

# ============================================================


def _build_prediction_response(

    prediction: SmokehousePrediction | None,

) -> SmokehousePredictionResponse | None:
    """

    Convert a persisted prediction ORM object into the

    public API response format.

    """

    if prediction is None:

        return None

    return SmokehousePredictionResponse(

        prediction_id=prediction.id,

        reading_id=prediction.reading_id,

        predicted_at=prediction.predicted_at,

        remaining_drying_time_hours=(

            prediction.remaining_drying_time_hours

        ),

        fire_risk_label=prediction.fire_risk_label,

        probabilities=FireRiskProbabilities(

            normal=prediction.normal_probability,

            warning=prediction.warning_probability,

            critical=prediction.critical_probability,

        ),

        guidance=prediction.guidance,

        regression_model_version=(

            prediction.regression_model_version

        ),

        classification_model_version=(

            prediction.classification_model_version

        ),

    )

# ============================================================

# CREATE SMOKEHOUSE BATCH

# ============================================================


@router.post(

    "/batches",

    response_model=SmokehouseBatchResponse,

    status_code=status.HTTP_201_CREATED,

)
def create_batch(

    request: SmokehouseBatchCreate,

    db: Session = Depends(get_db),

) -> SmokehouseBatchResponse:
    """

    Create one new smokehouse drying batch.

    The request must contain exactly:

        L1

        L2

        L3

    """

    try:

        batch = create_smokehouse_batch(

            db,

            request,

        )

    except SmokehouseConflictError as exc:

        raise HTTPException(

            status_code=status.HTTP_409_CONFLICT,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    return SmokehouseBatchResponse.model_validate(

        batch

    )

# ============================================================

# GET SMOKEHOUSE BATCH

# ============================================================


@router.get(

    "/batches/{batch_code}",

    response_model=SmokehouseBatchResponse,

    status_code=status.HTTP_200_OK,

)
def get_batch(

    batch_code: str,

    db: Session = Depends(get_db),

) -> SmokehouseBatchResponse:
    """

    Retrieve one drying batch and its configured layers.

    """

    try:

        batch = get_smokehouse_batch_by_code(

            db,

            batch_code,

        )

    except SmokehouseNotFoundError as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    return SmokehouseBatchResponse.model_validate(

        batch

    )

# ============================================================

# UPDATE BATCH STATUS

# ============================================================


@router.patch(

    "/batches/{batch_code}/status",

    response_model=SmokehouseBatchResponse,

    status_code=status.HTTP_200_OK,

)
def update_batch_status(

    batch_code: str,

    request: SmokehouseBatchStatusUpdate,

    db: Session = Depends(get_db),

) -> SmokehouseBatchResponse:
    """

    Close an active drying batch.

    Allowed transitions:

        active -> completed

        active -> cancelled

    Once a batch is completed or cancelled, it cannot be

    reopened or changed to another terminal state.

    """

    try:

        batch = update_smokehouse_batch_status(

            db,

            batch_code,

            request,

        )

    except SmokehouseNotFoundError as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        ) from exc

    except SmokehouseConflictError as exc:

        raise HTTPException(

            status_code=status.HTTP_409_CONFLICT,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    return SmokehouseBatchResponse.model_validate(

        batch

    )

# ============================================================

# GET LATEST LAYER STATUS

# ============================================================


@router.get(

    "/batches/{batch_code}/layers/{layer_code}/latest",

    response_model=SmokehouseLatestStatusResponse,

    status_code=status.HTTP_200_OK,

)
def get_latest_status(

    batch_code: str,

    layer_code: LayerCode,

    db: Session = Depends(get_db),

) -> SmokehouseLatestStatusResponse:
    """

    Retrieve the latest stored reading and persisted prediction

    for one layer.

    This endpoint does NOT rerun the ML models.

    """

    try:

        result = get_latest_layer_status(

            db,

            batch_code,

            layer_code,

        )

    except SmokehouseNotFoundError as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    return SmokehouseLatestStatusResponse(

        batch_code=result.batch_code,

        smokehouse_id=result.smokehouse_id,

        layer_code=result.layer_code,

        latest_reading=(

            SmokehouseReadingResponse.model_validate(

                result.latest_reading

            )

        ),

        latest_prediction=(

            _build_prediction_response(

                result.latest_prediction

            )

        ),

    )

# ============================================================

# GET LAYER HISTORY

# ============================================================


@router.get(

    "/batches/{batch_code}/layers/{layer_code}/history",

    response_model=SmokehouseLayerHistoryResponse,

    status_code=status.HTTP_200_OK,

)
def get_history(

    batch_code: str,

    layer_code: LayerCode,

    db: Session = Depends(get_db),

) -> SmokehouseLayerHistoryResponse:
    """

    Retrieve stored reading/prediction history for one layer.

    Readings are returned in chronological order.

    This endpoint does NOT perform feature engineering

    or rerun ML inference.

    """

    try:

        result = get_layer_history(

            db,

            batch_code,

            layer_code,

        )

    except SmokehouseNotFoundError as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    items = [

        SmokehouseHistoryItemResponse(

            reading=(

                SmokehouseReadingResponse.model_validate(

                    item.reading

                )

            ),

            prediction=(

                _build_prediction_response(

                    item.prediction

                )

            ),

        )

        for item in result.items

    ]

    return SmokehouseLayerHistoryResponse(

        batch_code=result.batch_code,

        smokehouse_id=result.smokehouse_id,

        layer_code=result.layer_code,

        items=items,

    )

# ============================================================

# CREATE READING + AUTOMATIC ML PREDICTION

# ============================================================


@router.post(

    "/readings",

    response_model=SmokehouseReadingPredictionResponse,

    status_code=status.HTTP_201_CREATED,

)
def create_reading(

    request: SmokehouseReadingCreate,

    db: Session = Depends(get_db),

) -> SmokehouseReadingPredictionResponse:
    """

    Save one raw smokehouse temperature/humidity reading and

    automatically run both finalized Component 3 ML models.

    The backend automatically:

        1. finds the batch

        2. finds the requested layer

        3. loads strictly earlier readings

        4. constructs the exact 17 ML features

        5. runs the drying-time model

        6. runs the fire-risk model

        7. stores the raw reading

        8. stores the resulting prediction

    No engineered ML features are supplied by the client.

    """

    try:

        result = create_reading_and_prediction(

            db,

            request,

        )

    except SmokehouseNotFoundError as exc:

        raise HTTPException(

            status_code=status.HTTP_404_NOT_FOUND,

            detail=str(exc),

        ) from exc

    except SmokehouseConflictError as exc:

        raise HTTPException(

            status_code=status.HTTP_409_CONFLICT,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    prediction_response = _build_prediction_response(

        result.prediction

    )

    if prediction_response is None:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail="Persisted prediction could not be returned.",

        )

    return SmokehouseReadingPredictionResponse(

        reading=SmokehouseReadingResponse.model_validate(

            result.reading

        ),

        prediction=prediction_response,

        missing_numeric_features=list(

            result.inference.missing_numeric_features

        ),

        historical_reading_count=(

            result.inference.historical_reading_count

        ),

    )

# ============================================================

# TEMPORARY STATELESS ML PREDICTION ENDPOINT

# ============================================================


@router.post(

    "/predict-preview",

    response_model=SmokehousePredictionPreviewResponse,

    status_code=status.HTTP_200_OK,

)
def predict_preview(

    request: SmokehousePredictionPreviewRequest,

) -> SmokehousePredictionPreviewResponse:
    """

    Temporary stateless Component 3 prediction endpoint.

    Unlike POST /readings:

    - nothing is persisted

    - historical readings are supplied by the client

    This remains useful for controlled backend/model validation.

    """

    history = [

        HistoricalReading(

            recorded_at=reading.recorded_at,

            temperature_c=reading.temperature_c,

            humidity_pct=reading.humidity_pct,

        )

        for reading in request.history

    ]

    try:

        result = predict_smokehouse_state(

            smokehouse_id=request.smokehouse_id,

            layer_id=request.layer_id,

            temperature_c=request.temperature_c,

            humidity_pct=request.humidity_pct,

            recorded_at=request.recorded_at,

            batch_started_at=request.batch_started_at,

            number_of_sheets=request.number_of_sheets,

            layer_area_m2=request.layer_area_m2,

            history=history,

        )

    except SmokehouseValidationError as exc:

        raise HTTPException(

            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=str(exc),

        ) from exc

    except SmokehouseServiceError as exc:

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    return SmokehousePredictionPreviewResponse(

        remaining_drying_time_hours=(

            result.remaining_drying_time_hours

        ),

        fire_risk_label=result.fire_risk_label,

        probabilities=FireRiskProbabilities(

            normal=result.normal_probability,

            warning=result.warning_probability,

            critical=result.critical_probability,

        ),

        guidance=result.guidance,

        missing_numeric_features=list(

            result.missing_numeric_features

        ),

        historical_reading_count=(

            result.historical_reading_count

        ),

    )
