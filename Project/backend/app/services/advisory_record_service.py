from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.database.models import TappingAdvisory
from backend.app.schemas.advisory_record import AdvisoryRecordCreate


# Saving and reading Component 1 advisories. Plain storage: nothing
# here runs a model or changes a value - what was sent is what is kept.


class AdvisoryRecordStorageError(RuntimeError):
    """The database could not be reached or refused the write."""


def create_advisory_record(
    db: Session,
    request: AdvisoryRecordCreate,
) -> TappingAdvisory:
    record = TappingAdvisory(**request.model_dump())

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except SQLAlchemyError as exc:
        db.rollback()
        raise AdvisoryRecordStorageError(
            "The advisory could not be saved."
        ) from exc

    return record


def list_advisory_records(
    db: Session,
    limit: int,
) -> list[TappingAdvisory]:
    try:
        return list(
            db.scalars(
                select(TappingAdvisory)
                .order_by(TappingAdvisory.created_at.desc(), TappingAdvisory.id.desc())
                .limit(limit)
            )
        )
    except SQLAlchemyError as exc:
        raise AdvisoryRecordStorageError(
            "Saved advisories could not be read."
        ) from exc


def get_advisory_record(
    db: Session,
    record_id: int,
) -> TappingAdvisory | None:
    try:
        return db.get(TappingAdvisory, record_id)
    except SQLAlchemyError as exc:
        raise AdvisoryRecordStorageError(
            "The saved advisory could not be read."
        ) from exc
