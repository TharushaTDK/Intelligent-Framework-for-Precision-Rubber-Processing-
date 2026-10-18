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

    # Weather provider (Open-Meteo Forecast API, keyless)
    open_meteo_base_url: str = (
        "https://api.open-meteo.com/v1/forecast"
    )
    weather_provider_connect_timeout_seconds: float = 5.0
    weather_provider_read_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
