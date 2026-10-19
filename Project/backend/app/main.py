import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.app.api.routes.coagulation import (
    router as coagulation_router,
)
from backend.app.api.routes.smokehouse import (
    router as smokehouse_router,
)

from backend.app.api.routes.weather_tapping import (
    router as weather_tapping_router,
)

from backend.app.core.config import settings

from backend.app.ml.component_1_weather_tapping.model_loader import (
    get_models as get_component1_models,
)


logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

STATIC_DIR = APP_DIR / "static"

SMOKEHOUSE_DASHBOARD_FILE = (
    STATIC_DIR
    / "smokehouse_dashboard.html"
)

COAGULATION_DASHBOARD_FILE = STATIC_DIR / "coagulation_dashboard.html"
WEATHER_TAPPING_DASHBOARD_FILE = (
    STATIC_DIR
    / "weather_tapping_dashboard.html"
)


# ============================================================
# STARTUP: PRELOAD COMPONENT 1 MODELS
# ============================================================


def _preload_component1_models() -> None:
    """
    Load the Component 1 models in the background so the first
    advisory doesn't wait for scikit-learn/XGBoost to import
    (about 20 s on a cold start, which made the dashboard's first
    requests time out). If it fails, the models simply load on the
    first request as before.
    """

    try:
        get_component1_models()
    except Exception:
        logger.exception(
            "Component 1 models could not be preloaded; "
            "they will load on the first request."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A background thread, so startup isn't delayed and Component 3
    # is available straight away.
    threading.Thread(
        target=_preload_component1_models,
        name="component1-model-preload",
        daemon=True,
    ).start()
    yield


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(
    smokehouse_router
)

app.include_router(
    coagulation_router
)

app.include_router(
    weather_tapping_router
)


# ============================================================
# SYSTEM HEALTH
# ============================================================


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, str]:
    """
    Basic backend health endpoint.
    """

    return {
        "status": "ok",
        "environment": settings.app_env,
    }


# ============================================================
# COMPONENT 3 TEMPORARY VALIDATION DASHBOARD
# ============================================================


@app.get(
    "/smokehouse-dashboard",
    include_in_schema=False,
)
def smokehouse_dashboard() -> FileResponse:
    """
    Serve the temporary Component 3 validation dashboard.

    The dashboard uses the same FastAPI backend for:

    - batch creation
    - raw sensor reading submission
    - ML prediction
    - latest layer status
    - layer history

    This is a development/research validation interface,
    not the final React Native application.
    """

    return FileResponse(
        path=SMOKEHOUSE_DASHBOARD_FILE,
        media_type="text/html",
    )


@app.get(
    "/coagulation-dashboard",
    include_in_schema=False,
)
def coagulation_dashboard() -> FileResponse:
    """Temporary Component 2 validation dashboard, not the final mobile app."""

    return FileResponse(
        path=COAGULATION_DASHBOARD_FILE,
        media_type="text/html",
        # always send the latest page so old copies are not shown while testing
        headers={"Cache-Control": "no-store"},
    )


# ============================================================
# COMPONENT 1 TEMPORARY TESTING DASHBOARD
# ============================================================


@app.get(
    "/weather-tapping-dashboard",
    include_in_schema=False,
)
def weather_tapping_dashboard() -> FileResponse:
    """
    Serve the temporary Component 1 testing dashboard.

    The farmer picks an estate, fetches its weather through
    /weather-snapshot, then calls the DRC, tapping and price
    endpoints separately. Served from this app so the page
    can use same-origin requests and no CORS setup is needed.

    This is only a testing page, not the Flutter application.
    """

    return FileResponse(
        path=WEATHER_TAPPING_DASHBOARD_FILE,
        media_type="text/html",
    )
