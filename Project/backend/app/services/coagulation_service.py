from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import PROJECT_ROOT
from backend.app.database.models import CoagulationRecommendation
from backend.app.ml.component_2_coagulation.model_loader import ModelLoadError, get_models
from backend.app.schemas.coagulation import (
    CoagulationAlert,
    CoagulationGuidance,
    CoagulationRecommendationCreate,
    CoagulationRecommendationResponse,
    CoagulationStatusUpdate,
    WaterAcidResult,
)
from backend.app.services import coagulation_drc_provider


class CoagulationServiceError(RuntimeError):
    pass


class CoagulationNotFoundError(CoagulationServiceError):
    pass


class CoagulationConflictError(CoagulationServiceError):
    pass


class CoagulationValidationError(CoagulationServiceError):
    pass


class CoagulationDrcUnavailableError(CoagulationServiceError):
    pass


OUT_OF_RANGE_NOTE = (
    "DRC or latex volume is outside the range the models were trained on, "
    "so this result is less reliable."
)

class _AlertSettings(BaseSettings):
    # set COAGULATION_ALERT_INTERVAL_MINUTES in .env to a small value only for quick local testing
    coagulation_alert_interval_minutes: int = Field(default=30, ge=1)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


ALERT_INTERVAL_MINUTES = _AlertSettings().coagulation_alert_interval_minutes

# small allowance for phone and server clocks being slightly apart
CLOCK_TOLERANCE = timedelta(minutes=5)

# allowed status changes for the coagulation timer
ALLOWED_TRANSITIONS = {
    "recommended": {"in_progress", "cancelled"},
    "in_progress": {"completed", "cancelled"},
}


@dataclass(frozen=True)
class CoagulationInferenceResult:
    drc: float
    latex_volume_L: float
    water_amount_L: float
    prepared_1pct_formic_acid_solution_mL: float
    coagulation_time_hours: float | None
    outside_training_range: bool
    water_acid_model_version: str
    coagulation_time_model_version: str | None


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise CoagulationServiceError(f"Model returned a non-finite {name}.")
    return value


def _predict_time(models, drc: float, latex_volume_L: float, water: float, acid: float) -> float:
    row = {
        "drc": drc,
        "latex_volume_L": latex_volume_L,
        "water_amount_L": water,
        "prepared_1pct_formic_acid_solution_mL": acid,
    }
    try:
        x_time = pd.DataFrame([row], columns=list(models.metadata.coagulation_time_features))
        t = np.asarray(models.coagulation_time.predict(x_time)).reshape(-1)
    except Exception as exc:
        raise CoagulationServiceError("Coagulation time prediction failed.") from exc
    if t.shape != (1,):
        raise CoagulationServiceError("Unexpected coagulation time prediction shape.")
    return _finite(t[0], "coagulation time")


def predict_time_for_batch(record: CoagulationRecommendation) -> tuple[float, str]:
    """Time model on the batch's own stored DRC, volume and recommended water/acid."""
    try:
        models = get_models()
    except ModelLoadError as exc:
        raise CoagulationServiceError("Component 2 models could not be loaded.") from exc
    hours = _predict_time(
        models,
        record.drc,
        record.latex_volume_l,
        record.water_amount_l,
        record.prepared_1pct_formic_acid_solution_ml,
    )
    return hours, models.metadata.coagulation_time_version


def run_coagulation_inference(
    *,
    drc: float,
    latex_volume_L: float,
    include_time: bool,
) -> CoagulationInferenceResult:
    """Water/acid model first, then the time model on those amounts when time is asked for."""
    if not (math.isfinite(drc) and 0 < drc <= 100):
        raise CoagulationValidationError("DRC must be a finite value between 0 and 100.")
    if not (math.isfinite(latex_volume_L) and latex_volume_L > 0):
        raise CoagulationValidationError("Latex volume must be a finite value above 0.")

    try:
        models = get_models()
    except ModelLoadError as exc:
        raise CoagulationServiceError("Component 2 models could not be loaded.") from exc

    meta = models.metadata
    inputs = {"drc": drc, "latex_volume_L": latex_volume_L}

    try:
        x_wa = pd.DataFrame([inputs], columns=list(meta.water_acid_features))
        wa = np.asarray(models.water_acid.predict(x_wa))
    except Exception as exc:
        raise CoagulationServiceError("Water and acid prediction failed.") from exc

    if wa.shape != (1, 2):
        raise CoagulationServiceError("Unexpected water and acid prediction shape.")

    water = _finite(wa[0, 0], "water amount")
    acid = _finite(wa[0, 1], "acid amount")

    time_hours = None
    if include_time:
        time_hours = _predict_time(models, drc, latex_volume_L, water, acid)

    outside = any(
        not (low <= inputs[name] <= high)
        for name, (low, high) in meta.training_ranges.items()
    )

    return CoagulationInferenceResult(
        drc=drc,
        latex_volume_L=latex_volume_L,
        water_amount_L=water,
        prepared_1pct_formic_acid_solution_mL=acid,
        coagulation_time_hours=time_hours,
        outside_training_range=outside,
        water_acid_model_version=meta.water_acid_version,
        coagulation_time_model_version=meta.coagulation_time_version if include_time else None,
    )


def notes_for(outside_training_range: bool) -> list[str]:
    return [OUT_OF_RANGE_NOTE] if outside_training_range else []


def hours_and_minutes(hours: float) -> str:
    # farmers read "7 hours 48 minutes" more easily than 7.79 hours
    h, m = divmod(round(hours * 60), 60)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m or not h:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    return " ".join(parts)


def build_guidance(
    *,
    water_amount_L: float | None,
    acid_mL: float | None,
    coagulation_time_hours: float | None,
) -> list[CoagulationGuidance]:
    """Farmer guidance for each shown output, filled with this batch's predicted values."""
    items = []

    if water_amount_L is not None:
        water = f"{water_amount_L:.2f}"
        items.append(
            CoagulationGuidance(
                output="water",
                section="1. Water recommendation",
                title="Predicted water requirement",
                value=water_amount_L,
                unit="L",
                display_value=f"{water} L",
                guidance=[
                    f"For this latex batch, the recommended water quantity is {water} litres.",
                    "Measure this quantity using a suitable measuring container and add it to the "
                    "batch according to the validated latex-processing procedure.",
                    "Ensure that the recommended quantity is measured accurately before proceeding "
                    "to the next preparation step.",
                ],
            )
        )

    if acid_mL is not None:
        acid = f"{acid_mL:.1f}"
        items.append(
            CoagulationGuidance(
                output="acid",
                section="2. Prepared 1% formic acid solution recommendation",
                title="Predicted acid solution requirement",
                value=acid_mL,
                unit="mL",
                display_value=f"{acid} mL",
                guidance=[
                    "For this latex batch, the recommended quantity of prepared 1% formic acid "
                    f"solution is {acid} millilitres.",
                    "Measure this quantity separately using a suitable measuring device.",
                    "Handle the solution carefully and follow the approved procedure for adding it "
                    "to the latex batch.",
                    "Do not confuse the prepared 1% solution with concentrated formic acid.",
                ],
            )
        )

    if coagulation_time_hours is not None:
        duration = hours_and_minutes(coagulation_time_hours)
        items.append(
            CoagulationGuidance(
                output="coagulation_time",
                section="3. Coagulation time prediction",
                title="Predicted coagulation time",
                value=coagulation_time_hours,
                unit="hours",
                display_value=duration,
                guidance=[
                    f"The estimated coagulation time for this batch is {duration}.",
                    "Start the in-app timer when the defined coagulation process begins.",
                    "When the timer reaches zero, inspect the batch and check whether coagulation "
                    "is complete using the established practical criteria.",
                    "If the batch is not ready, follow the approved processing procedure rather than "
                    "relying on the predicted time alone.",
                ],
            )
        )

    return items


def _fetch_estate_drc(estate_id: str) -> coagulation_drc_provider.EstateDrc:
    try:
        estate = coagulation_drc_provider.get_estate_drc(estate_id)
    except coagulation_drc_provider.EstateNotFoundError as exc:
        raise CoagulationNotFoundError(f"Estate '{estate_id}' was not found.") from exc
    except coagulation_drc_provider.EstateDrcUnavailableError as exc:
        raise CoagulationDrcUnavailableError(str(exc)) from exc

    # never recommend from a broken upstream DRC value
    if not (isinstance(estate.drc, (int, float)) and math.isfinite(estate.drc) and 0 < estate.drc <= 100):
        raise CoagulationDrcUnavailableError(
            f"Estate DRC received for '{estate_id}' is not a valid percentage."
        )
    return estate


def create_recommendation(
    db: Session,
    request: CoagulationRecommendationCreate,
) -> CoagulationRecommendation:
    if request.drc is not None:
        estate = coagulation_drc_provider.EstateDrc(request.estate_id, request.drc, source="manual")
    else:
        estate = _fetch_estate_drc(request.estate_id)
    include_time = "coagulation_time" in request.outputs

    result = run_coagulation_inference(
        drc=float(estate.drc),
        latex_volume_L=request.latex_volume_L,
        include_time=include_time,
    )

    record = CoagulationRecommendation(
        estate_id=request.estate_id,
        latex_volume_l=request.latex_volume_L,
        drc=result.drc,
        drc_source=estate.source,
        requested_water_acid="water_acid" in request.outputs,
        requested_coagulation_time=include_time,
        water_amount_l=result.water_amount_L,
        prepared_1pct_formic_acid_solution_ml=result.prepared_1pct_formic_acid_solution_mL,
        coagulation_time_hours=result.coagulation_time_hours,
        outside_training_range=result.outside_training_range,
        water_acid_model_version=result.water_acid_model_version,
        coagulation_time_model_version=result.coagulation_time_model_version,
        status="recommended",
        created_at=datetime.now(timezone.utc),
    )

    try:
        db.add(record)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise CoagulationServiceError("Recommendation could not be saved.") from exc

    db.refresh(record)
    return record


def get_recommendation(
    db: Session,
    recommendation_id: int,
    *,
    lock: bool = False,
) -> CoagulationRecommendation:
    try:
        record = db.get(CoagulationRecommendation, recommendation_id, with_for_update=lock)
    except SQLAlchemyError as exc:
        raise CoagulationServiceError("Recommendation could not be loaded.") from exc
    if record is None:
        raise CoagulationNotFoundError(f"Recommendation {recommendation_id} was not found.")
    return record


def list_recommendations(
    db: Session,
    *,
    estate_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[CoagulationRecommendation]:
    query = select(CoagulationRecommendation)
    if estate_id:
        query = query.where(CoagulationRecommendation.estate_id == estate_id)
    if status:
        query = query.where(CoagulationRecommendation.status == status)
    query = query.order_by(
        CoagulationRecommendation.created_at.desc(),
        CoagulationRecommendation.id.desc(),
    ).limit(limit)

    try:
        return list(db.scalars(query))
    except SQLAlchemyError as exc:
        raise CoagulationServiceError("Recommendation history could not be loaded.") from exc


def update_recommendation_status(
    db: Session,
    recommendation_id: int,
    request: CoagulationStatusUpdate,
) -> CoagulationRecommendation:
    # row lock so two quick taps can't both change the same batch
    record = get_recommendation(db, recommendation_id, lock=True)

    try:
        if request.status not in ALLOWED_TRANSITIONS.get(record.status, set()):
            raise CoagulationConflictError(
                f"Cannot change status from '{record.status}' to '{request.status}'."
            )

        now = datetime.now(timezone.utc)
        changed_at = request.changed_at or now
        if changed_at > now + CLOCK_TOLERANCE:
            raise CoagulationValidationError("Status time cannot be in the future.")

        if request.status == "in_progress":
            if changed_at < record.created_at - CLOCK_TOLERANCE:
                raise CoagulationValidationError("Start time cannot be before the recommendation was made.")
            # water/acid only batches still need a time for their alerts
            if record.coagulation_time_hours is None:
                hours, version = predict_time_for_batch(record)
                record.coagulation_time_hours = hours
                record.coagulation_time_model_version = version
            record.started_at = changed_at
            record.expected_completion_at = changed_at + timedelta(hours=record.coagulation_time_hours)

        elif request.status == "completed":
            if changed_at < record.started_at:
                raise CoagulationValidationError("Completion time cannot be before the start time.")
            record.completed_at = changed_at
            record.actual_coagulation_time_hours = (changed_at - record.started_at).total_seconds() / 3600

        record.status = request.status
        db.commit()
    except CoagulationServiceError:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise CoagulationServiceError("Status could not be updated.") from exc

    db.refresh(record)
    return record


def add_coagulation_time(db: Session, recommendation_id: int) -> CoagulationRecommendation:
    """Farmer asks for the time later on a water/acid batch, so the same batch gets it."""
    record = get_recommendation(db, recommendation_id, lock=True)

    try:
        if record.status not in ("recommended", "in_progress"):
            raise CoagulationConflictError(
                f"Cannot add coagulation time to a '{record.status}' batch."
            )
        if record.coagulation_time_hours is None:
            hours, version = predict_time_for_batch(record)
            record.coagulation_time_hours = hours
            record.coagulation_time_model_version = version
        record.requested_coagulation_time = True
        db.commit()
    except CoagulationServiceError:
        db.rollback()
        raise
    except SQLAlchemyError as exc:
        db.rollback()
        raise CoagulationServiceError("Coagulation time could not be saved.") from exc

    db.refresh(record)
    return record


def _format_duration(td: timedelta) -> str:
    # round up so a short wait never shows as 0 min
    minutes = max(math.ceil(td.total_seconds() / 60), 0)
    hours, minutes = divmod(minutes, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"
    return f"{hours} h" if hours else f"{minutes} min"


def _batch_label(record: CoagulationRecommendation) -> str:
    return f"Estate {record.estate_id} batch #{record.id} ({record.latex_volume_l:g} L)"


def build_alert_schedule(
    record: CoagulationRecommendation,
    now: datetime,
) -> list[CoagulationAlert]:
    """Progress alert every 30 min after start, then a due alert at the expected time."""
    if record.started_at is None or record.expected_completion_at is None:
        return []

    step = timedelta(minutes=ALERT_INTERVAL_MINUTES)
    due = record.expected_completion_at
    label = _batch_label(record)

    alerts = []
    at = record.started_at + step
    while at < due:
        alerts.append(
            CoagulationAlert(
                alert_at=at,
                type="progress",
                message=f"{label}: about {_format_duration(due - at)} left for coagulation.",
                is_past=at <= now,
            )
        )
        at += step

    alerts.append(
        CoagulationAlert(
            alert_at=due,
            type="due",
            message=f"{label}: expected coagulation time reached, check the latex.",
            is_past=due <= now,
        )
    )
    return alerts


def next_alert(
    record: CoagulationRecommendation,
    schedule: list[CoagulationAlert],
    now: datetime,
) -> datetime | None:
    if not schedule:
        return None
    for alert in schedule:
        if alert.alert_at > now:
            return alert.alert_at
    # overdue and not completed yet, keep reminding every 30 min
    step = timedelta(minutes=ALERT_INTERVAL_MINUTES)
    passed = (now - record.expected_completion_at) // step + 1
    return record.expected_completion_at + passed * step


def to_response(
    record: CoagulationRecommendation,
    now: datetime | None = None,
) -> CoagulationRecommendationResponse:
    """Only return the outputs the farmer asked for, plus live timer and alert values."""
    outputs = []
    if record.requested_water_acid:
        outputs.append("water_acid")
    if record.requested_coagulation_time:
        outputs.append("coagulation_time")

    # once the timer is started the time is shown too, the alerts depend on it
    shown_time = (
        record.coagulation_time_hours
        if record.requested_coagulation_time or record.started_at is not None
        else None
    )

    remaining = None
    is_due = None
    alerts: list[CoagulationAlert] = []
    next_alert_at = None
    if record.status == "in_progress" and record.expected_completion_at is not None:
        now = now or datetime.now(timezone.utc)
        remaining = max((record.expected_completion_at - now).total_seconds() / 3600, 0.0)
        is_due = now >= record.expected_completion_at
        alerts = build_alert_schedule(record, now)
        next_alert_at = next_alert(record, alerts, now)
        if is_due:
            alerts.append(
                CoagulationAlert(
                    alert_at=next_alert_at,
                    type="overdue",
                    message=(
                        f"{_batch_label(record)}: overdue by "
                        f"{_format_duration(next_alert_at - record.expected_completion_at)}, "
                        "check the latex and mark the batch completed."
                    ),
                    is_past=False,
                )
            )

    return CoagulationRecommendationResponse(
        id=record.id,
        estate_id=record.estate_id,
        latex_volume_L=record.latex_volume_l,
        drc=record.drc,
        drc_source=record.drc_source,
        outputs=outputs,
        water_acid=(
            WaterAcidResult(
                water_amount_L=record.water_amount_l,
                prepared_1pct_formic_acid_solution_mL=record.prepared_1pct_formic_acid_solution_ml,
            )
            if record.requested_water_acid
            else None
        ),
        coagulation_time_hours=shown_time,
        outside_training_range=record.outside_training_range,
        notes=notes_for(record.outside_training_range),
        guidance=build_guidance(
            water_amount_L=record.water_amount_l if record.requested_water_acid else None,
            acid_mL=record.prepared_1pct_formic_acid_solution_ml if record.requested_water_acid else None,
            coagulation_time_hours=shown_time,
        ),
        water_acid_model_version=record.water_acid_model_version,
        coagulation_time_model_version=record.coagulation_time_model_version,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        expected_completion_at=record.expected_completion_at,
        completed_at=record.completed_at,
        actual_coagulation_time_hours=record.actual_coagulation_time_hours,
        time_remaining_hours=remaining,
        is_due=is_due,
        alert_interval_minutes=ALERT_INTERVAL_MINUTES if alerts else None,
        next_alert_at=next_alert_at,
        alerts=alerts,
    )
