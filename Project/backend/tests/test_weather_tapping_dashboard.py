from __future__ import annotations

import re

import pytest

from fastapi.testclient import TestClient

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


def test_page_never_sends_estate_name_or_geolocation():
    source = dashboard_source()

    assert "estate_name" not in endpoint_fields_from_page("drc")
    assert "navigator.geolocation" not in source


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
