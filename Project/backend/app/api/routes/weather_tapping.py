from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from pydantic import BaseModel

from sqlalchemy.orm import Session

from backend.app.database.session import get_db

from backend.app.schemas.advisory_record import (
    AdvisoryRecordCreate,
    AdvisoryRecordList,
    AdvisoryRecordResponse,
)

from backend.app.services.advisory_record_service import (
    AdvisoryRecordStorageError,
    create_advisory_record,
    get_advisory_record,
    list_advisory_records,
)

from backend.app.ml.component_1_weather_tapping.model_loader import (
    ModelLoadError,
    get_manifest,
)

from backend.app.schemas.weather_tapping import (
    DrcPredictionRequest,
    DrcPredictionResponse,
    PriceEstimationRequest,
    PriceEstimationResponse,
    TappingSuitabilityRequest,
    TappingSuitabilityResponse,
)

from backend.app.schemas.weather_provider import (
    WeatherSnapshotRequest,
    WeatherSnapshotResponse,
)

from backend.app.services.weather_provider import (
    WeatherProviderDataError,
    WeatherProviderError,
    WeatherTimestampRangeError,
    fetch_weather_snapshot,
)

from backend.app.services.weather_tapping_service import (
    OBSERVATION_TIMEZONE,
    WeatherTappingInputError,
    WeatherTappingModelError,
    predict_drc,
    predict_price_estimation,
    predict_tapping_suitability,
)

router = APIRouter(
    prefix="/api/v1/weather-tapping",
    tags=["Component 1 - Weather Tapping"],
)


# ============================================================
# MODELS METADATA RESPONSE
# ============================================================


class ModelFeatureContract(BaseModel):
    name: str
    target: str
    model_class: str
    runtime_artifact_sha256: str
    original_artifact_sha256: str
    feature_count: int
    features_in_pipeline_order: list[str]
    classes_in_probability_order: list[str] | None = None
    dataset_provenance: str
    target_warning: str | None = None


class ModelCatalogResponse(BaseModel):
    component: str
    observation_timezone: str
    known_limitations: list[str]
    models: list[ModelFeatureContract]


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================


@router.post(
    "/drc-estimation",
    response_model=DrcPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def drc_estimation(
    request: DrcPredictionRequest,
) -> DrcPredictionResponse:
    """
    Stateless latex dry-rubber-content (DRC) prediction.

    Out-of-range and missing sensor values follow the
    research policy: converted to missing and median-
    imputed inside the fitted pipeline.
    """

    try:
        return predict_drc(request)

    except WeatherTappingInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except WeatherTappingModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )


@router.post(
    "/tapping-advisory",
    response_model=TappingSuitabilityResponse,
    status_code=status.HTTP_200_OK,
)
def tapping_advisory(
    request: TappingSuitabilityRequest,
) -> TappingSuitabilityResponse:
    """
    Stateless tapping-suitability proxy prediction.

    This is a rule-derived research proxy trained on
    synthetic noisy labels - not expert-verified tapping
    advice. The response preserves the fitted class order
    and the proxy disclaimer.
    """

    try:
        return predict_tapping_suitability(request)

    except WeatherTappingInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except WeatherTappingModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )


@router.post(
    "/price-estimation",
    response_model=PriceEstimationResponse,
    status_code=status.HTTP_200_OK,
)
def price_estimation(
    request: PriceEstimationRequest,
) -> PriceEstimationResponse:
    """
    Stateless rubber price estimation (LKR per kg).

    latex_drc_pct is supplied explicitly by the caller; the
    models are never chained implicitly.
    """

    try:
        return predict_price_estimation(request)

    except WeatherTappingInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except WeatherTappingModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )


# ============================================================
# WEATHER SNAPSHOT (STANDALONE PROVIDER FETCH)
# ============================================================


@router.post(
    "/weather-snapshot",
    response_model=WeatherSnapshotResponse,
    status_code=status.HTTP_200_OK,
)
def weather_snapshot(
    request: WeatherSnapshotRequest,
) -> WeatherSnapshotResponse:
    """
    Standalone Open-Meteo weather snapshot: hourly
    observations/forecasts at the hour-aligned observation
    time, including the three rainfall windows (all labelled
    unverified_runtime_convention).

    The endpoint is never chained into the DRC, tapping or
    price endpoints. For past observation timestamps,
    forecast_rain_next_6h_mm reflects the provider's current
    best-match series, not an as-issued historical forecast.

    Times up to 7 days ahead are accepted; then every value is
    a forecast and data_quality.observation_in_future is true.
    Times outside the supported range get a 422.

    Provider URLs, parameters and error payloads are never
    exposed to the caller.
    """

    try:
        return fetch_weather_snapshot(
            latitude=request.latitude,
            longitude=request.longitude,
            observation_timestamp=request.observation_timestamp,
        )

    except WeatherTimestampRangeError as exc:
        # The caller asked for an unsupported time - say so plainly
        # instead of blaming the provider.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "observation_timestamp is outside the supported "
                "range: it must be no more than 7 days in the "
                "future and no more than about 89 days in the past."
            ),
        ) from exc

    except WeatherProviderDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Weather provider returned incomplete or "
                "invalid data."
            ),
        ) from exc

    except WeatherProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weather data is currently unavailable.",
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )


# ============================================================
# SAVED ADVISORIES (FOR COMPONENT 2)
# ============================================================
#
# The dashboard saves each advisory it generates, so Component 2
# can read the estimated DRC together with the location and time
# instead of re-running Component 1.


STORAGE_UNAVAILABLE_DETAIL = (
    "Saved advisories are currently unavailable (database error)."
)


@router.post(
    "/advisory-records",
    response_model=AdvisoryRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def save_advisory_record(
    request: AdvisoryRecordCreate,
    db: Session = Depends(get_db),
) -> AdvisoryRecordResponse:
    """
    Save one generated advisory: location, the weather it used,
    the DRC estimate and (when available) tapping and price.
    Values are stored exactly as sent; no model is run here.
    """

    try:
        record = create_advisory_record(db, request)
    except AdvisoryRecordStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STORAGE_UNAVAILABLE_DETAIL,
        ) from exc

    return AdvisoryRecordResponse.model_validate(record)


@router.get(
    "/advisory-records",
    response_model=AdvisoryRecordList,
    status_code=status.HTTP_200_OK,
)
def read_advisory_records(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> AdvisoryRecordList:
    """
    Saved advisories, newest first. This is what Component 2 reads
    to get DRC estimates with their locations.
    """

    try:
        records = list_advisory_records(db, limit)
    except AdvisoryRecordStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STORAGE_UNAVAILABLE_DETAIL,
        ) from exc

    return AdvisoryRecordList(
        count=len(records),
        records=[AdvisoryRecordResponse.model_validate(r) for r in records],
    )


@router.get(
    "/advisory-records/{record_id}",
    response_model=AdvisoryRecordResponse,
    status_code=status.HTTP_200_OK,
)
def read_advisory_record(
    record_id: int,
    db: Session = Depends(get_db),
) -> AdvisoryRecordResponse:
    """One saved advisory by its id."""

    try:
        record = get_advisory_record(db, record_id)
    except AdvisoryRecordStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STORAGE_UNAVAILABLE_DETAIL,
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No saved advisory with id {record_id}.",
        )

    return AdvisoryRecordResponse.model_validate(record)


# ============================================================
# MODEL CATALOG
# ============================================================


@router.get(
    "/models",
    response_model=ModelCatalogResponse,
    status_code=status.HTTP_200_OK,
)
def models_catalog() -> ModelCatalogResponse:
    """
    Verified metadata for all three finalized Component 1
    models: artifact hashes, feature contracts, synthetic
    data provenance, and known limitations.

    No local file paths or internal details are exposed.
    """

    try:
        manifest = get_manifest()

    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The Component 1 model catalog could not be "
                "loaded or failed integrity verification."
            ),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )

    models = [
        ModelFeatureContract(
            name=entry["name"],
            target=entry["target"],
            model_class=entry["model_class"],
            runtime_artifact_sha256=entry[
                "runtime_sha256"
            ],
            original_artifact_sha256=entry[
                "original_artifact"
            ]["sha256"],
            feature_count=len(
                entry["features_in_pipeline_order"]
            ),
            features_in_pipeline_order=entry[
                "features_in_pipeline_order"
            ],
            classes_in_probability_order=entry.get(
                "classes_in_probability_order"
            ),
            dataset_provenance=entry[
                "dataset_provenance"
            ],
            target_warning=entry.get("target_warning"),
        )
        for entry in manifest["models"]
    ]

    return ModelCatalogResponse(
        component=manifest["component"],
        observation_timezone=OBSERVATION_TIMEZONE,
        known_limitations=manifest["known_limitations"],
        models=models,
    )
