from collections.abc import Generator

from sqlalchemy import URL, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import settings


# Build the PostgreSQL connection URL safely.

# Example:
# postgresql+psycopg://rubber_user:password@localhost:5432/rubber_research
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=settings.postgres_user,
    password=settings.postgres_password,
    host=settings.postgres_host,
    port=settings.postgres_port,
    database=settings.postgres_db,
)


# SQLAlchemy engine

# pool_pre_ping=True checks whether a pooled database connection
# is still alive before SQLAlchemy gives it to the application.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# Database session factory.

# Each API request will later receive its own Session.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# All future SQLAlchemy database models will inherit from this class.
class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """
    Provide one SQLAlchemy database session.

    The session is always closed after the request finishes,
    even if an error occurs.
    """
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
