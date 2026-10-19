from __future__ import annotations

import re
import threading

import pytest

from fastapi.testclient import TestClient

import backend.app.main as main_module
from backend.app.main import WEATHER_TAPPING_DASHBOARD_FILE, app
from backend.app.schemas.weather_tapping import (
    DrcPredictionRequest,
    PriceEstimationRequest,
    TappingSuitabilityRequest,
)

client = TestClient(app)


def dashboard_source() -> str:
    return WEATHER_TAPPING_DASHBOARD_FILE.read_text(encoding="utf-8")


def endpoint_fields_from_page(kind: str) -> set[str]:
    """
    Pull one ENDPOINT_FIELDS array out of the page's script so we
    can compare it with the real request schema.
    """

    source = dashboard_source()
    block = re.search(
        r"const ENDPOINT_FIELDS = \{(.*?)\n        \};",
        source,
        re.S,
    )
    assert block, "ENDPOINT_FIELDS block not found in dashboard"

    entry = re.search(
        rf"{kind}: \[(.*?)\]",
        block.group(1),
        re.S,
    )
    assert entry, f"{kind} entry not found in ENDPOINT_FIELDS"

    return set(re.findall(r'"([a-z0-9_]+)"', entry.group(1)))


def test_dashboard_route_serves_html():
    response = client.get("/weather-tapping-dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Generate Advisory" in response.text


def test_smokehouse_dashboard_still_served():
    # Component 3's page must keep working alongside the new one.
    response = client.get("/smokehouse-dashboard")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_dashboard_is_hidden_from_openapi():
    paths = client.get("/openapi.json").json()["paths"]

    assert "/weather-tapping-dashboard" not in paths


@pytest.mark.parametrize(
    ("kind", "schema"),
    [
        ("drc", DrcPredictionRequest),
        ("tapping", TappingSuitabilityRequest),
        ("price", PriceEstimationRequest),
    ],
)
def test_page_field_lists_match_request_schemas(kind, schema):
    # The schemas forbid extra fields, so the page must send exactly
    # these names (observation_timestamp is added separately).
    expected = set(schema.model_fields) - {"observation_timestamp"}

    assert endpoint_fields_from_page(kind) == expected


def test_startup_preloads_component1_models(monkeypatch):
    # Without the preload, the first advisory after a restart waited
    # ~20 s for scikit-learn to import and the page timed out.
    loaded = threading.Event()
    monkeypatch.setattr(main_module, "get_component1_models", loaded.set)

    with TestClient(app) as started:
        assert loaded.wait(timeout=5)
        assert started.get("/health").status_code == 200


def test_failed_preload_does_not_stop_the_app(monkeypatch):
    attempted = threading.Event()

    def broken_loader():
        attempted.set()
        raise RuntimeError("artifact missing")

    monkeypatch.setattr(main_module, "get_component1_models", broken_loader)

    with TestClient(app) as started:
        assert attempted.wait(timeout=5)
        assert started.get("/health").status_code == 200
        assert started.get("/smokehouse-dashboard").status_code == 200


def test_page_never_sends_estate_name():
    assert "estate_name" not in endpoint_fields_from_page("drc")


def test_location_is_chosen_on_a_map_not_an_estate_dropdown():
    source = dashboard_source()

    assert 'id="map"' in source
    assert "Use My Current Location" in source
    assert 'id="estateSelect"' not in source


def test_leaflet_is_pinned_with_integrity_hashes():
    source = dashboard_source()

    assert "unpkg.com/leaflet@1.9.4/dist/leaflet.js" in source
    assert 'integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="' in source
    assert 'integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="' in source


def test_gps_is_only_requested_from_the_button():
    # Static check: the browser's GPS API is touched in exactly one helper,
    # and that helper is only called from the button's click handler.
    source = dashboard_source()

    assert source.count("getCurrentPosition(") == 1
    assert source.count("requestCurrentPosition(navigator.geolocation)") == 1
    assert '$("gpsBtn").addEventListener("click", onUseMyLocation);' in source
    assert "watchPosition" not in source


def test_page_uses_same_origin_api_paths():
    source = dashboard_source()

    assert 'const API_BASE = "/api/v1/weather-tapping";' in source
    for path in (
        "/weather-snapshot",
        "/drc-estimation",
        "/tapping-advisory",
        "/price-estimation",
    ):
        assert path in source
