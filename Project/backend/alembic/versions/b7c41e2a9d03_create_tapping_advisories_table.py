"""create tapping advisories table

Saves Component 1 advisories (location, weather used, estimated
DRC, tapping and price) so Component 2 can use the DRC and location.
Only adds a new table; the smokehouse and coagulation tables are not
touched.

Runs after the Component 2 coagulation migration so the history
stays a single line (both used to start from the smokehouse one).

Revision ID: b7c41e2a9d03
Revises: 18ffbd7abbde
Create Date: 2026-10-19 22:30:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7c41e2a9d03'
down_revision: Union[str, Sequence[str], None] = '18ffbd7abbde'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('tapping_advisories',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('latitude', sa.Float(), nullable=False),
    sa.Column('longitude', sa.Float(), nullable=False),
    sa.Column('location_source', sa.String(length=10), nullable=True),
    sa.Column('observation_timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.Column('forecast_based', sa.Boolean(), nullable=False),
    sa.Column('temperature_c', sa.Float(), nullable=False),
    sa.Column('relative_humidity_pct', sa.Float(), nullable=False),
    sa.Column('wind_speed_kmh', sa.Float(), nullable=False),
    sa.Column('rainfall_last_6h_mm', sa.Float(), nullable=False),
    sa.Column('rainfall_last_3days_mm', sa.Float(), nullable=False),
    sa.Column('forecast_rain_next_6h_mm', sa.Float(), nullable=False),
    sa.Column('predicted_latex_drc_pct', sa.Float(), nullable=False),
    sa.Column('drc_model_name', sa.String(length=100), nullable=False),
    sa.Column('drc_runtime_artifact_sha256', sa.String(length=64), nullable=False),
    sa.Column('tapping_label', sa.String(length=10), nullable=True),
    sa.Column('tapping_probability_high', sa.Float(), nullable=True),
    sa.Column('tapping_probability_low', sa.Float(), nullable=True),
    sa.Column('tapping_probability_moderate', sa.Float(), nullable=True),
    sa.Column('predicted_price_lkr_per_kg', sa.Float(), nullable=True),
    sa.CheckConstraint('latitude BETWEEN -90 AND 90', name='ck_tapping_advisories_latitude'),
    sa.CheckConstraint('longitude BETWEEN -180 AND 180', name='ck_tapping_advisories_longitude'),
    sa.CheckConstraint("location_source IS NULL OR location_source IN ('map', 'gps', 'typed')", name='ck_tapping_advisories_location_source'),
    sa.CheckConstraint("tapping_label IS NULL OR tapping_label IN ('High', 'Low', 'Moderate')", name='ck_tapping_advisories_tapping_label'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tapping_advisories_created_at'), 'tapping_advisories', ['created_at'], unique=False)
    op.create_index(op.f('ix_tapping_advisories_observation_timestamp'), 'tapping_advisories', ['observation_timestamp'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_tapping_advisories_observation_timestamp'), table_name='tapping_advisories')
    op.drop_index(op.f('ix_tapping_advisories_created_at'), table_name='tapping_advisories')
    op.drop_table('tapping_advisories')
