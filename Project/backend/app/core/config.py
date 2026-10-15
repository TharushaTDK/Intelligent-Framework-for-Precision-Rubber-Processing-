from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Application
    app_name: str = "Rubber Processing Research API"
    app_env: str = "development"
    app_version: str = "0.1.0"

    # PostgreSQL
    postgres_db: str = "rubber_research"
    postgres_user: str = "rubber_user"
    postgres_password: str = "change_me_local"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
