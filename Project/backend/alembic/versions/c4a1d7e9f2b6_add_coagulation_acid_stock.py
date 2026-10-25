"""add coagulation acid stock

Revision ID: c4a1d7e9f2b6
Revises: c4d2a8f60101
Create Date: 2026-10-25 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4a1d7e9f2b6'
down_revision: Union[str, Sequence[str], None] = 'c4d2a8f60101'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('coagulation_acid_stock',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('stock_ml', sa.Float(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint('stock_ml >= 0', name='ck_coagulation_acid_stock_positive'),
    sa.PrimaryKeyConstraint('id')
    )
    op.add_column('coagulation_recommendations', sa.Column('acid_used_ml', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('coagulation_recommendations', 'acid_used_ml')
    op.drop_table('coagulation_acid_stock')
