from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.session import Base


class CoagulationRecommendation(Base):
    """One latex batch recommendation from Component 2 and its coagulation timer."""

    __tablename__ = "coagulation_recommendations"

    __table_args__ = (
        CheckConstraint("latex_volume_l > 0", name="ck_coagulation_volume_positive"),
        CheckConstraint("drc > 0 AND drc <= 100", name="ck_coagulation_drc_range"),
        CheckConstraint(
            "requested_water_acid OR requested_coagulation_time",
            name="ck_coagulation_output_requested",
        ),
        CheckConstraint(
            "status IN ('recommended', 'in_progress', 'completed', 'cancelled')",
            name="ck_coagulation_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    estate_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    latex_volume_l: Mapped[float] = mapped_column(Float, nullable=False)

    # DRC actually used, with where it came from
    drc: Mapped[float] = mapped_column(Float, nullable=False)
    drc_source: Mapped[str] = mapped_column(String(30), nullable=False)

    requested_water_acid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    requested_coagulation_time: Mapped[bool] = mapped_column(Boolean, nullable=False)

    water_amount_l: Mapped[float] = mapped_column(Float, nullable=False)
    prepared_1pct_formic_acid_solution_ml: Mapped[float] = mapped_column(Float, nullable=False)
    coagulation_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)

    outside_training_range: Mapped[bool] = mapped_column(Boolean, nullable=False)

    water_acid_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    coagulation_time_model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="recommended")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expected_completion_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actual_coagulation_time_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
