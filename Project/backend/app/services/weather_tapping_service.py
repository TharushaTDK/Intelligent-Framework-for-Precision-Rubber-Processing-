from __future__ import annotations

import math

import numpy as np

from backend.app.ml.component_1_weather_tapping.feature_preparation import (
    build_feature_frame,
    derive_calendar_features,
    normalize_estate_id,
    normalize_visible_latex_condition,
    prepare_numeric_feature,
)

from backend.app.ml.component_1_weather_tapping.model_loader import (
    TAPPING_CLASSES_IN_PROBABILITY_ORDER,
    ModelLoadError,
    get_models,
)

from backend.app.schemas.weather_tapping import (
    DrcPredictionRequest,
    DrcPredictionResponse,
    PriceEstimationRequest,
    PriceEstimationResponse,
    TappingClassProbabilities,
    TappingSuitabilityRequest,
    TappingSuitabilityResponse,
)


# ============================================================
# OBSERVATION TIMEZONE
# ============================================================
#
# The configured observation timezone defines the wall
# clock used to derive the month and day_of_year model
# features, reproducing training's observation-local
# timestamp semantics (training parsed the naive
# timestamp_local column and read ts.dt.month /
# ts.dt.dayofyear with no timezone conversion).
#
# This setting is intentionally kept here - separate from
# the pure feature-processing helpers in
# backend.app.ml.component_1_weather_tapping.feature_preparation
# - and is never silently replaced by UTC.
#
# The default reflects the Sri Lanka estate context of the
# research dataset (LKR prices, estate-local operations).
# It is a documented, configurable domain default, not a
# model assumption.
# ============================================================


OBSERVATION_TIMEZONE = "Asia/Colombo"


# ============================================================
# SERVICE EXCEPTIONS
# ============================================================


class WeatherTappingServiceError(RuntimeError):
    """
    Base exception for Component 1 backend service failures.
    """


class WeatherTappingInputError(WeatherTappingServiceError):
    """
    Raised when client-supplied data is structurally valid at
    the API schema level but cannot be used to build the
    finalized runtime feature row.
    """


class WeatherTappingModelError(WeatherTappingServiceError):
    """
    Raised when the finalized Component 1 models cannot be
    verified/loaded, or when inference fails or produces
    unusable output.

    Messages never contain file paths, tracebacks, secrets,
    or serialized artifact contents.
    """


# ============================================================
# DERIVED-FEATURE AND MODEL CONSTANTS
# ============================================================


MONTH_FEATURE = "month"
DAY_OF_YEAR_FEATURE = "day_of_year"

ESTATE_ID_FEATURE = "estate_id"
LATEX_CONDITION_FEATURE = "visible_latex_condition"

CALENDAR_FEATURES = (MONTH_FEATURE, DAY_OF_YEAR_FEATURE)

# Only the DRC model has categorical inputs since the October
# 2026 retrain; the tapping and price models are all numeric.
CATEGORICAL_FEATURES = (
    ESTATE_ID_FEATURE,
    LATEX_CONDITION_FEATURE,
)


# ============================================================
# INTERNAL HELPERS
# ============================================================


def _load_verified_models():
    """
    Load the verified Component 1 runtime models.

    Integrity or loading failures are re-raised as a narrow
    model error without exposing file paths or underlying
    exception details.
    """

    try:
        return get_models()

    except ModelLoadError as exc:
        raise WeatherTappingModelError(
            "The Component 1 prediction models could not be "
            "loaded or failed integrity verification."
        ) from exc

    except Exception as exc:
        raise WeatherTappingModelError(
            "The Component 1 prediction models could not be "
            "loaded."
        ) from exc


def _ordered_features(manifest_entry: dict) -> list[str]:
    """
    Read the exact ordered feature list recorded in the
    verified runtime manifest for one model.
    """

    features = manifest_entry.get(
        "features_in_pipeline_order"
    )

    if (
        not isinstance(features, list)
        or not features
        or not all(
            isinstance(name, str) for name in features
        )
    ):
        raise WeatherTappingModelError(
            "The verified Component 1 artifact manifest does "
            "not contain a valid feature contract."
        )

    return list(features)


def _manifest_entry(manifest: dict, model_name: str) -> dict:
    """
    Look up one model entry in the verified runtime
    manifest.
    """

    models = manifest.get("models")

    if isinstance(models, list):
        for entry in models:
            if (
                isinstance(entry, dict)
                and entry.get("name") == model_name
            ):
                return entry

    raise WeatherTappingModelError(
        "The verified Component 1 artifact manifest does "
        f"not contain the '{model_name}' model entry."
    )


def _model_identifier_fields(manifest_entry: dict) -> dict:
    """
    Extract the model identifier fields for a response from
    the verified runtime manifest.
    """

    return {
        "model_name": manifest_entry.get("name", ""),
        "model_class": manifest_entry.get(
            "model_class", ""
        ),
        "runtime_artifact_sha256": manifest_entry.get(
            "runtime_sha256", ""
        ),
        "original_artifact_sha256": (
            manifest_entry.get("original_artifact", {}) or {}
        ).get("sha256", ""),
        "dataset_provenance": manifest_entry.get(
            "dataset_provenance", ""
        ),
    }


def _build_numeric_features(
    ordered_features: list[str],
    request,
) -> dict[str, float]:
    """
    Build the numeric part of the feature row by applying
    the documented research missing-value policy to each
    numeric feature supplied by the request.
    """

    numeric_values: dict[str, float] = {}

    for feature_name in ordered_features:
        if feature_name in CATEGORICAL_FEATURES:
            continue

        if feature_name in CALENDAR_FEATURES:
            continue

        numeric_values[feature_name] = (
            prepare_numeric_feature(
                feature_name,
                getattr(request, feature_name, None),
            )
        )

    return numeric_values


def _calendar_feature_values(request) -> tuple[int, int]:
    """
    Derive month and day_of_year from the explicit
    observation timestamp, using the configured observation
    timezone (never silently UTC).
    """

    try:
        return derive_calendar_features(
            request.observation_timestamp,
            OBSERVATION_TIMEZONE,
        )

    except ValueError as exc:
        raise WeatherTappingInputError(str(exc)) from exc


def _run_regression_pipeline(
    pipeline,
    feature_frame,
    model_label: str,
) -> float:
    """
    Execute one regression pipeline and validate its output.

    The raw prediction is preserved exactly (no clamping or
    rounding); non-finite or unexpected-shaped outputs are
    rejected.
    """

    try:
        raw_output = pipeline.predict(feature_frame)

    except ValueError as exc:
        raise WeatherTappingModelError(
            f"The finalized {model_label} pipeline rejected "
            "the feature row during inference."
        ) from exc

    predictions = np.asarray(raw_output)

    if predictions.shape != (1,):
        raise WeatherTappingModelError(
            f"The finalized {model_label} pipeline produced "
            "an unexpected prediction shape."
        )

    predicted_value = float(predictions.reshape(-1)[0])

    if not math.isfinite(predicted_value):
        raise WeatherTappingModelError(
            f"The finalized {model_label} pipeline produced "
            "a non-finite prediction."
        )

    return predicted_value


# ============================================================
# OPERATION 1 - DRC PREDICTION
# ============================================================


def predict_drc(
    request: DrcPredictionRequest,
) -> DrcPredictionResponse:
    """
    Stateless latex dry-rubber-content (DRC) prediction.

    Constructs the exact feature row defined by the verified
    runtime manifest (estate_name is never used), applies the
    documented research missing-value policy, and runs the
    finalized pipeline with its embedded preprocessing.
    """

    models = _load_verified_models()

    manifest_entry = _manifest_entry(
        models.manifest,
        "drc_prediction",
    )

    ordered_features = _ordered_features(
        manifest_entry
    )

    numeric_values = _build_numeric_features(
        ordered_features,
        request,
    )

    month, day_of_year = _calendar_feature_values(request)

    numeric_values[MONTH_FEATURE] = float(month)
    numeric_values[DAY_OF_YEAR_FEATURE] = float(
        day_of_year
    )

    feature_frame = build_feature_frame(
        ordered_features=ordered_features,
        numeric_values=numeric_values,
        categorical_values={
            ESTATE_ID_FEATURE: normalize_estate_id(
                request.estate_id
            ),
            LATEX_CONDITION_FEATURE: (
                normalize_visible_latex_condition(
                    request.visible_latex_condition
                )
            ),
        },
    )

    predicted_drc_pct = _run_regression_pipeline(
        models.drc,
        feature_frame,
        "DRC",
    )

    return DrcPredictionResponse(
        **_model_identifier_fields(manifest_entry),
        observation_month=month,
        observation_day_of_year=day_of_year,
        predicted_latex_drc_pct=predicted_drc_pct,
    )


# ============================================================
# OPERATION 2 - TAPPING SUITABILITY PROXY
# ============================================================


def predict_tapping_suitability(
    request: TappingSuitabilityRequest,
) -> TappingSuitabilityResponse:
    """
    Stateless tapping-suitability PROXY prediction.

    The response preserves the fitted model's exact class
    labels and probability-column order, and is explicitly
    marked as a rule-derived research proxy (rainfall
    thresholds) - not expert-verified tapping advice.
    """

    models = _load_verified_models()

    manifest_entry = _manifest_entry(
        models.manifest,
        "tapping_advisory",
    )

    ordered_features = _ordered_features(
        manifest_entry
    )

    numeric_values = _build_numeric_features(
        ordered_features,
        request,
    )

    month, day_of_year = _calendar_feature_values(request)

    numeric_values[MONTH_FEATURE] = float(month)
    numeric_values[DAY_OF_YEAR_FEATURE] = float(
        day_of_year
    )

    feature_frame = build_feature_frame(
        ordered_features=ordered_features,
        numeric_values=numeric_values,
        categorical_values={},
    )

    bundle = models.tapping

    try:
        raw_encoded = bundle.pipeline.predict(
            feature_frame
        )

        raw_probabilities = (
            bundle.pipeline.predict_proba(
                feature_frame
            )
        )

    except ValueError as exc:
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "rejected the feature row during inference."
        ) from exc

    encoded_labels = np.asarray(raw_encoded)
    probabilities = np.asarray(
        raw_probabilities,
        dtype=float,
    )

    expected_classes = tuple(
        TAPPING_CLASSES_IN_PROBABILITY_ORDER
    )

    if tuple(bundle.classes) != expected_classes:
        raise WeatherTappingModelError(
            "The tapping-advisory bundle class order does "
            "not match the trusted fitted class order."
        )

    if encoded_labels.shape != (1,):
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "produced an unexpected prediction shape."
        )

    if probabilities.shape != (
        1,
        len(expected_classes),
    ):
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "produced an unexpected probability shape."
        )

    if not np.all(np.isfinite(probabilities)):
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "produced non-finite probabilities."
        )

    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "produced probabilities outside [0, 1]."
        )

    try:
        decoded_label = str(
            bundle.label_encoder.inverse_transform(
                encoded_labels
            )[0]
        )

    except Exception as exc:
        raise WeatherTappingModelError(
            "The finalized tapping-advisory label encoder "
            "could not decode the predicted label."
        ) from exc

    if decoded_label not in expected_classes:
        raise WeatherTappingModelError(
            "The finalized tapping-advisory pipeline "
            "produced an unknown class label."
        )

    # predict_proba() column order is exactly the fitted
    # classes order: High=0, Low=1, Moderate=2.
    return TappingSuitabilityResponse(
        **_model_identifier_fields(manifest_entry),
        observation_month=month,
        observation_day_of_year=day_of_year,
        predicted_label=decoded_label,  # type: ignore[arg-type]
        class_order=list(expected_classes),  # type: ignore[arg-type]
        probabilities=TappingClassProbabilities(
            high=float(probabilities[0, 0]),
            low=float(probabilities[0, 1]),
            moderate=float(probabilities[0, 2]),
        ),
        is_rule_derived_proxy=True,
        research_warning=bundle.warning,
    )


# ============================================================
# OPERATION 3 - PRICE ESTIMATION
# ============================================================


def predict_price_estimation(
    request: PriceEstimationRequest,
) -> PriceEstimationResponse:
    """
    Stateless field latex price estimation (LKR per kg of
    field latex).

    latex_drc_pct is supplied explicitly by the caller (a
    measured value or the output of the DRC prediction
    operation); the service never chains models implicitly
    and never uses collection-time features.
    """

    models = _load_verified_models()

    manifest_entry = _manifest_entry(
        models.manifest,
        "price_estimation",
    )

    ordered_features = _ordered_features(
        manifest_entry
    )

    numeric_values = _build_numeric_features(
        ordered_features,
        request,
    )

    month, day_of_year = _calendar_feature_values(request)

    numeric_values[MONTH_FEATURE] = float(month)
    numeric_values[DAY_OF_YEAR_FEATURE] = float(
        day_of_year
    )

    feature_frame = build_feature_frame(
        ordered_features=ordered_features,
        numeric_values=numeric_values,
        categorical_values={},
    )

    predicted_price = _run_regression_pipeline(
        models.price,
        feature_frame,
        "price-estimation",
    )

    return PriceEstimationResponse(
        **_model_identifier_fields(manifest_entry),
        observation_month=month,
        observation_day_of_year=day_of_year,
        predicted_price_lkr_per_kg=predicted_price,
    )
