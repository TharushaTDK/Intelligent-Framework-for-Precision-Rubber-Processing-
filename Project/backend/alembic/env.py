from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.app.database.session import Base, DATABASE_URL

# Import ORM models so SQLAlchemy registers all tables
# inside Base.metadata before Alembic autogeneration runs.
from backend.app.database.models import (  # noqa: F401
    SmokehouseBatch,
    SmokehouseLayer,
    SmokehousePrediction,
    SmokehouseReading,
    TappingAdvisory,
)


# ============================================================
# ALEMBIC CONFIGURATION
# ============================================================

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# ============================================================
# DATABASE URL
# ============================================================
#
# The database URL is constructed by the existing application
# configuration:
#
# .env
#   ↓
# backend/app/core/config.py
#   ↓
# backend/app/database/session.py
#
# No password is stored in alembic.ini.
# ============================================================

config.set_main_option(
    "sqlalchemy.url",
    DATABASE_URL.render_as_string(
        hide_password=False
    ).replace("%", "%%"),
)


# ============================================================
# SQLALCHEMY METADATA
# ============================================================

target_metadata = Base.metadata


# ============================================================
# OFFLINE MIGRATIONS
# ============================================================


def run_migrations_offline() -> None:
    """
    Run migrations without creating a live DB connection.
    """

    url = config.get_main_option(
        "sqlalchemy.url"
    )

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named"
        },
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ============================================================
# ONLINE MIGRATIONS
# ============================================================


def run_migrations_online() -> None:
    """
    Run migrations using a real PostgreSQL connection.
    """

    connectable = engine_from_config(
        config.get_section(
            config.config_ini_section,
            {}
        ),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# ============================================================
# ENTRY POINT
# ============================================================

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
