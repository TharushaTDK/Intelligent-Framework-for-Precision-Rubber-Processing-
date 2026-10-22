"""Create Component 4 acoustic measurements.

Revision ID: c4d2a8f60101
Revises: b7c41e2a9d03
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c4d2a8f60101"
down_revision = "b7c41e2a9d03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "acoustic_measurements",
        sa.Column(
            "measurement_id",
            sa.Uuid(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(160), nullable=False),
        sa.Column("raw_file_name", sa.String(80), nullable=False),
        sa.Column("sheet_id", sa.String(64), nullable=True),
        sa.Column("condition_id", sa.String(64), nullable=True),
        sa.Column("source_sha256", sa.String(64), nullable=False),

        sa.Column("processing_status", sa.String(32), nullable=False),
        sa.Column("technical_status", sa.String(32), nullable=False),
        sa.Column("quality_status", sa.String(16), nullable=False),
        sa.Column("classifier_status", sa.String(64), nullable=False),
        sa.Column("decision_source", sa.String(32), nullable=False),

        sa.Column(
            "rejection_reasons",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),

        sa.Column("rule_version", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(100), nullable=True),
        sa.Column("model_status", sa.String(32), nullable=False),

        sa.Column(
            "invalid_probability_uncalibrated",
            sa.Float(),
            nullable=True,
        ),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("onset_seconds", sa.Float(), nullable=True),
        sa.Column("segment_duration_seconds", sa.Float(), nullable=True),
        sa.Column("peak_percent_full_scale", sa.Float(), nullable=True),
        sa.Column("near_full_scale_fraction", sa.Float(), nullable=True),
        sa.Column("moisture_percent", sa.Float(), nullable=True),
        sa.Column("guidance", sa.Text(), nullable=False),

        sa.PrimaryKeyConstraint("measurement_id"),
        sa.UniqueConstraint("raw_file_name"),

        sa.CheckConstraint(
            "quality_status IN ('VALID', 'INVALID')",
            name="ck_acoustic_measurements_quality",
        ),
        sa.CheckConstraint(
            "technical_status IN "
            "('TECHNICAL_PASSED', 'TECHNICAL_REJECTED')",
            name="ck_acoustic_measurements_technical",
        ),
        sa.CheckConstraint(
            "decision_source IN ('TECHNICAL_RULES', 'ML_CLASSIFIER')",
            name="ck_acoustic_measurements_decision",
        ),
        sa.CheckConstraint(
            "invalid_probability_uncalibrated IS NULL OR "
            "invalid_probability_uncalibrated BETWEEN 0 AND 1",
            name="ck_acoustic_measurements_invalid_score",
        ),
        sa.CheckConstraint(
            "candidate_count >= 0",
            name="ck_acoustic_measurements_candidates",
        ),
        sa.CheckConstraint(
            "moisture_percent IS NULL OR "
            "moisture_percent BETWEEN 0 AND 100",
            name="ck_acoustic_measurements_moisture",
        ),
    )

    op.create_index(
        "ix_acoustic_measurements_created_at",
        "acoustic_measurements",
        ["created_at"],
    )
    op.create_index(
        "ix_acoustic_measurements_sheet_id",
        "acoustic_measurements",
        ["sheet_id"],
    )
    op.create_index(
        "ix_acoustic_measurements_source_sha256",
        "acoustic_measurements",
        ["source_sha256"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_acoustic_measurements_source_sha256",
        table_name="acoustic_measurements",
    )
    op.drop_index(
        "ix_acoustic_measurements_sheet_id",
        table_name="acoustic_measurements",
    )
    op.drop_index(
        "ix_acoustic_measurements_created_at",
        table_name="acoustic_measurements",
    )
    op.drop_table("acoustic_measurements")