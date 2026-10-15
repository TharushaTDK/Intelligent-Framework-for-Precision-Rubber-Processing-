from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.session import Base


# ============================================================
# SMOKEHOUSE BATCH
# ============================================================


class SmokehouseBatch(Base):
    """
    One rubber-sheet drying batch inside a smokehouse.

    A batch contains exactly the three logical smokehouse layers:

        L1
        L2
        L3

    The application/Pydantic layer validates that all three layers
    are created for a batch.
    """

    __tablename__ = "smokehouse_batches"

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'completed', 'cancelled')",
            name="ck_smokehouse_batches_status",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    batch_code: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
        index=True,
    )

    smokehouse_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
    )

    layers: Mapped[list["SmokehouseLayer"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
    )


# ============================================================
# SMOKEHOUSE LAYER
# ============================================================


class SmokehouseLayer(Base):
    """
    One physical layer belonging to a drying batch.

    Component 3 uses exactly:

        L1 = bottom layer
        L2 = middle layer
        L3 = top layer
    """

    __tablename__ = "smokehouse_layers"

    __table_args__ = (
        UniqueConstraint(
            "batch_id",
            "layer_code",
            name="uq_smokehouse_layer_per_batch",
        ),
        CheckConstraint(
            "number_of_sheets > 0",
            name="ck_smokehouse_layers_sheet_count_positive",
        ),
        CheckConstraint(
            "layer_area_m2 > 0",
            name="ck_smokehouse_layers_area_positive",
        ),
        CheckConstraint(
            "layer_code IN ('L1', 'L2', 'L3')",
            name="ck_smokehouse_layers_layer_code",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    batch_id: Mapped[int] = mapped_column(
        ForeignKey(
            "smokehouse_batches.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    layer_code: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    number_of_sheets: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    layer_area_m2: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    batch: Mapped["SmokehouseBatch"] = relationship(
        back_populates="layers",
    )

    readings: Mapped[list["SmokehouseReading"]] = relationship(
        back_populates="layer",
        cascade="all, delete-orphan",
    )


# ============================================================
# SMOKEHOUSE SENSOR READING
# ============================================================


class SmokehouseReading(Base):
    """
    One timestamped temperature/humidity observation from one layer.

    source:
        manual = temporary validation/testing interface
        iot    = future ESP32/DHT22 sensor input
    """

    __tablename__ = "smokehouse_readings"

    __table_args__ = (
        UniqueConstraint(
            "layer_id",
            "recorded_at",
            name="uq_smokehouse_reading_layer_time",
        ),
        CheckConstraint(
            "humidity_pct >= 0 AND humidity_pct <= 100",
            name="ck_smokehouse_readings_humidity_range",
        ),
        CheckConstraint(
            "source IN ('manual', 'iot')",
            name="ck_smokehouse_readings_source",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    layer_id: Mapped[int] = mapped_column(
        ForeignKey(
            "smokehouse_layers.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    temperature_c: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    humidity_pct: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="manual",
    )

    layer: Mapped["SmokehouseLayer"] = relationship(
        back_populates="readings",
    )

    predictions: Mapped[list["SmokehousePrediction"]] = relationship(
        back_populates="reading",
        cascade="all, delete-orphan",
    )


# ============================================================
# SMOKEHOUSE ML PREDICTION
# ============================================================


class SmokehousePrediction(Base):
    """
    Stores the output produced by both finalized Component 3
    machine-learning pipelines for a sensor reading.

    IMPORTANT:

    remaining_drying_time_hours is intentionally NOT constrained
    to >= 0 at the database layer.

    The finalized regression model output is preserved exactly.
    If UI-level clamping is later introduced, it must remain
    separate and explicitly documented.
    """

    __tablename__ = "smokehouse_predictions"

    __table_args__ = (
        CheckConstraint(
            "fire_risk_label IN ('Normal', 'Warning', 'Critical')",
            name="ck_smokehouse_predictions_fire_risk",
        ),
        CheckConstraint(
            "normal_probability >= 0 "
            "AND normal_probability <= 1",
            name="ck_smokehouse_predictions_normal_probability",
        ),
        CheckConstraint(
            "warning_probability >= 0 "
            "AND warning_probability <= 1",
            name="ck_smokehouse_predictions_warning_probability",
        ),
        CheckConstraint(
            "critical_probability >= 0 "
            "AND critical_probability <= 1",
            name="ck_smokehouse_predictions_critical_probability",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    reading_id: Mapped[int] = mapped_column(
        ForeignKey(
            "smokehouse_readings.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    remaining_drying_time_hours: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    fire_risk_label: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    normal_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    warning_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    critical_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    regression_model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    classification_model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    guidance: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reading: Mapped["SmokehouseReading"] = relationship(
        back_populates="predictions",
    )
