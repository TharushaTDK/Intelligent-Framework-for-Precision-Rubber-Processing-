from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.coagulation import (
    AcidStockAdd,
    AcidStockResponse,
    AcidStockUpdate,
    CoagulationHistoryResponse,
    CoagulationPreviewRequest,
    CoagulationPreviewResponse,
    CoagulationRecommendationCreate,
    CoagulationRecommendationResponse,
    CoagulationStatusUpdate,
    LocationDrcRequest,
    LocationDrcResponse,
    RecommendationStatus,
    WaterAcidResult,
)
from backend.app.services.coagulation_service import (
    acid_stock_status,
    add_acid_stock,
    CoagulationConflictError,
    CoagulationDrcUnavailableError,
    CoagulationNotFoundError,
    CoagulationServiceError,
    CoagulationValidationError,
    add_coagulation_time,
    build_guidance,
    create_recommendation,
    fetch_point_drc,
    get_recommendation,
    list_recommendations,
    notes_for,
    run_coagulation_inference,
    set_acid_stock,
    to_response,
    update_recommendation_status,
)

class _SafeValidationRoute(APIRoute):
    # raw NaN/Infinity in a body can't be echoed back as JSON, so 422 errors leave out the input
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def route_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                errors = [
                    {k: v for k, v in err.items() if k not in ("input", "ctx")}
                    for err in exc.errors()
                ]
                return JSONResponse(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    content={"detail": errors},
                )

        return route_handler


router = APIRouter(
    prefix="/api/v1/coagulation",
    tags=["Component 2 - Coagulation"],
    route_class=_SafeValidationRoute,
)


def _http_error(exc: CoagulationServiceError) -> HTTPException:
    # most specific errors first, they all inherit from CoagulationServiceError
    if isinstance(exc, CoagulationNotFoundError):
        code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, CoagulationConflictError):
        code = status.HTTP_409_CONFLICT
    elif isinstance(exc, CoagulationValidationError):
        code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif isinstance(exc, CoagulationDrcUnavailableError):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return HTTPException(status_code=code, detail=str(exc))


@router.post(
    "/recommendations",
    response_model=CoagulationRecommendationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create(
    request: CoagulationRecommendationCreate,
    db: Session = Depends(get_db),
) -> CoagulationRecommendationResponse:
    """Farmer picks a location on the map and enters latex volume, Component 1 gives the DRC there."""
    try:
        record = create_recommendation(db, request)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return to_response(record)


@router.get(
    "/recommendations",
    response_model=CoagulationHistoryResponse,
)
def history(
    estate_id: str | None = Query(default=None, max_length=50),
    status_filter: RecommendationStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> CoagulationHistoryResponse:
    """Batch history, use status=in_progress to get every running timer."""
    try:
        records = list_recommendations(db, estate_id=estate_id, status=status_filter, limit=limit)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return CoagulationHistoryResponse(items=[to_response(r) for r in records])


@router.get(
    "/recommendations/{recommendation_id}",
    response_model=CoagulationRecommendationResponse,
)
def get_one(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> CoagulationRecommendationResponse:
    try:
        record = get_recommendation(db, recommendation_id)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return to_response(record)


@router.patch(
    "/recommendations/{recommendation_id}/status",
    response_model=CoagulationRecommendationResponse,
)
def update_status(
    recommendation_id: int,
    request: CoagulationStatusUpdate,
    db: Session = Depends(get_db),
) -> CoagulationRecommendationResponse:
    """Start, complete or cancel the coagulation timer for a batch."""
    try:
        record = update_recommendation_status(db, recommendation_id, request)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return to_response(record)


@router.post(
    "/recommendations/{recommendation_id}/coagulation-time",
    response_model=CoagulationRecommendationResponse,
)
def add_time(
    recommendation_id: int,
    db: Session = Depends(get_db),
) -> CoagulationRecommendationResponse:
    """Add the coagulation time to a batch that was first asked only for water and acid."""
    try:
        record = add_coagulation_time(db, recommendation_id)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return to_response(record)


@router.post(
    "/location-drc",
    response_model=LocationDrcResponse,
)
def location_drc(request: LocationDrcRequest) -> LocationDrcResponse:
    """DRC for a map point from Component 1 (weather now, then its DRC model), nothing is saved."""
    try:
        estate = fetch_point_drc(request.latitude, request.longitude)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
    return LocationDrcResponse(
        latitude=request.latitude,
        longitude=request.longitude,
        location=estate.estate_id,
        drc=estate.drc,
        drc_source=estate.source,
        observation_timestamp=estate.observation_timestamp,
    )


@router.post(
    "/predict-preview",
    response_model=CoagulationPreviewResponse,
)
def predict_preview(request: CoagulationPreviewRequest) -> CoagulationPreviewResponse:
    """Stateless check of the models with a given DRC, nothing is saved."""
    show_water_acid = "water_acid" in request.outputs
    try:
        result = run_coagulation_inference(
            drc=request.drc,
            latex_volume_L=request.latex_volume_L,
            include_time="coagulation_time" in request.outputs,
        )
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc

    guidance = build_guidance(
        water_amount_L=result.water_amount_L if show_water_acid else None,
        acid_mL=result.prepared_1pct_formic_acid_solution_mL if show_water_acid else None,
        coagulation_time_hours=result.coagulation_time_hours,
    )

    return CoagulationPreviewResponse(
        drc=result.drc,
        latex_volume_L=result.latex_volume_L,
        water_acid=(
            WaterAcidResult(
                water_amount_L=result.water_amount_L,
                prepared_1pct_formic_acid_solution_mL=result.prepared_1pct_formic_acid_solution_mL,
            )
            if show_water_acid
            else None
        ),
        coagulation_time_hours=result.coagulation_time_hours,
        outside_training_range=result.outside_training_range,
        notes=notes_for(result.outside_training_range),
        guidance=guidance,
    )


@router.get(
    "/acid-stock",
    response_model=AcidStockResponse,
)
def get_acid_stock(db: Session = Depends(get_db)) -> AcidStockResponse:
    """Prepared 1% acid solution on hand and about how many batches it is enough for."""
    try:
        return acid_stock_status(db)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc


@router.put(
    "/acid-stock",
    response_model=AcidStockResponse,
)
def put_acid_stock(
    request: AcidStockUpdate,
    db: Session = Depends(get_db),
) -> AcidStockResponse:
    """Farmer corrects the total 1% acid solution on hand."""
    try:
        return set_acid_stock(db, request.stock_mL)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc


@router.post(
    "/acid-stock/add",
    response_model=AcidStockResponse,
)
def post_acid_stock_add(
    request: AcidStockAdd,
    db: Session = Depends(get_db),
) -> AcidStockResponse:
    """Farmer prepared more 1% acid solution, it is added to the stock."""
    try:
        return add_acid_stock(db, request.added_mL)
    except CoagulationServiceError as exc:
        raise _http_error(exc) from exc
