from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app.database.models import CoagulationRecommendation
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services import coagulation_drc_provider, coagulation_service
from backend.app.services.coagulation_drc_provider import EstateDrc

TEST_ESTATE_ID = "PYTEST_C2_EST"
TEST_DRC = 35.26
BASE = "/api/v1/coagulation"

client = TestClient(app)


def cleanup() -> None:
    # only removes rows created by these tests
    with SessionLocal() as db:
        db.execute(
            delete(CoagulationRecommendation).where(
                CoagulationRecommendation.estate_id.like("PYTEST_C2_%")
            )
        )
        db.commit()


@pytest.fixture(autouse=True)
def clean_database():
    cleanup()
    yield
    cleanup()


@pytest.fixture(autouse=True)
def real_alert_interval(monkeypatch):
    # alert tests use the real 30 minute interval even if .env sets a short one
    monkeypatch.setattr(coagulation_service, "ALERT_INTERVAL_MINUTES", 30)


@pytest.fixture
def estate_drc(monkeypatch):
    # stands in for Component 1 until its DRC service is merged
    def fake(estate_id: str) -> EstateDrc:
        if estate_id == "PYTEST_C2_UNKNOWN":
            raise coagulation_drc_provider.EstateNotFoundError(estate_id)
        if estate_id == "PYTEST_C2_BADDRC":
            return EstateDrc(estate_id=estate_id, drc=float("nan"), source="component_1")
        return EstateDrc(estate_id=estate_id, drc=TEST_DRC, source="component_1")

    monkeypatch.setattr(coagulation_drc_provider, "get_estate_drc", fake)


def create(outputs, estate_id=TEST_ESTATE_ID, volume=162.58):
    res = client.post(
        f"{BASE}/recommendations",
        json={"estate_id": estate_id, "latex_volume_L": volume, "outputs": outputs},
    )
    if res.status_code == 201:
        # backdate so tests can start timers in the past
        with SessionLocal() as db:
            row = db.get(CoagulationRecommendation, res.json()["id"])
            row.created_at = datetime.now(timezone.utc) - timedelta(hours=48)
            db.commit()
    return res


def test_water_acid_only(estate_drc):
    res = create(["water_acid"])
    assert res.status_code == 201
    body = res.json()
    assert body["outputs"] == ["water_acid"]
    assert body["drc"] == TEST_DRC
    assert body["drc_source"] == "component_1"
    assert body["water_acid"]["water_amount_L"] > 0
    assert body["water_acid"]["prepared_1pct_formic_acid_solution_mL"] > 0
    assert body["coagulation_time_hours"] is None
    assert body["coagulation_time_model_version"] is None
    assert body["water_acid_model_version"] == "component2-water-acid-v1"
    assert body["status"] == "recommended"


def test_coagulation_time_only(estate_drc):
    res = create(["coagulation_time"])
    assert res.status_code == 201
    body = res.json()
    assert body["outputs"] == ["coagulation_time"]
    assert body["water_acid"] is None
    assert body["coagulation_time_hours"] > 0
    assert body["coagulation_time_model_version"] == "component2-coagulation-time-v1"


def test_both_outputs_match_preview(estate_drc):
    res = create(["water_acid", "coagulation_time"])
    preview = client.post(
        f"{BASE}/predict-preview",
        json={"drc": TEST_DRC, "latex_volume_L": 162.58, "outputs": ["water_acid", "coagulation_time"]},
    )
    assert res.status_code == 201 and preview.status_code == 200
    body, prev = res.json(), preview.json()
    for key, value in prev["water_acid"].items():
        assert body["water_acid"][key] == pytest.approx(value, rel=1e-12)
    assert body["coagulation_time_hours"] == pytest.approx(prev["coagulation_time_hours"], rel=1e-12)


def test_recommendation_is_persisted_with_provenance(estate_drc):
    rec_id = create(["coagulation_time"]).json()["id"]
    with SessionLocal() as db:
        row = db.scalar(select(CoagulationRecommendation).where(CoagulationRecommendation.id == rec_id))
    assert row is not None
    assert row.estate_id == TEST_ESTATE_ID
    assert row.created_at.tzinfo is not None
    assert row.requested_water_acid is False
    assert row.requested_coagulation_time is True
    # water and acid are stored because the time model used them
    assert row.water_amount_l > 0


def test_get_and_history(estate_drc):
    first = create(["water_acid"]).json()["id"]
    second = create(["coagulation_time"]).json()["id"]

    one = client.get(f"{BASE}/recommendations/{first}")
    assert one.status_code == 200
    assert one.json()["id"] == first

    history = client.get(f"{BASE}/recommendations", params={"estate_id": TEST_ESTATE_ID})
    assert history.status_code == 200
    ids = [item["id"] for item in history.json()["items"]]
    assert ids == [second, first]


def test_out_of_range_input_is_flagged(estate_drc):
    body = create(["water_acid"], volume=500.0).json()
    assert body["outside_training_range"] is True
    assert body["notes"]


def test_timer_flow(estate_drc):
    rec = create(["coagulation_time"]).json()
    started = datetime.now(timezone.utc) - timedelta(hours=24)

    res = client.patch(
        f"{BASE}/recommendations/{rec['id']}/status",
        json={"status": "in_progress", "changed_at": started.isoformat()},
    )
    assert res.status_code == 200
    body = res.json()
    expected = started + timedelta(hours=rec["coagulation_time_hours"])
    assert datetime.fromisoformat(body["expected_completion_at"]) == expected
    assert body["is_due"] is True
    assert body["time_remaining_hours"] == 0

    done = client.patch(
        f"{BASE}/recommendations/{rec['id']}/status",
        json={"status": "completed", "changed_at": (started + timedelta(hours=7.5)).isoformat()},
    )
    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert done.json()["actual_coagulation_time_hours"] == pytest.approx(7.5)


def test_timer_not_due_yet(estate_drc):
    rec = create(["coagulation_time"]).json()
    res = client.patch(f"{BASE}/recommendations/{rec['id']}/status", json={"status": "in_progress"})
    body = res.json()
    assert body["is_due"] is False
    assert 0 < body["time_remaining_hours"] <= rec["coagulation_time_hours"]


def start_batch(rec_id: int, started: datetime):
    return client.patch(
        f"{BASE}/recommendations/{rec_id}/status",
        json={"status": "in_progress", "changed_at": started.isoformat()},
    ).json()


def test_alerts_every_30_minutes(estate_drc):
    rec = create(["coagulation_time"]).json()
    started = (datetime.now(timezone.utc) - timedelta(minutes=70)).replace(microsecond=0)
    body = start_batch(rec["id"], started)

    alerts = body["alerts"]
    times = [datetime.fromisoformat(a["alert_at"]) for a in alerts]
    expected_done = datetime.fromisoformat(body["expected_completion_at"])

    assert body["alert_interval_minutes"] == 30
    assert times[0] == started + timedelta(minutes=30)
    assert all(a["type"] == "progress" for a in alerts[:-1])
    assert all(b - a == timedelta(minutes=30) for a, b in zip(times[:-1], times[1:-1]))
    assert alerts[-1]["type"] == "due" and times[-1] == expected_done
    assert times[-2] < expected_done <= times[-2] + timedelta(minutes=30)
    assert datetime.fromisoformat(body["next_alert_at"]) == started + timedelta(minutes=90)
    assert rec["id"] and f"batch #{rec['id']}" in alerts[0]["message"]


def test_overdue_alert_repeats_every_30_minutes(estate_drc):
    rec = create(["coagulation_time"]).json()
    body = start_batch(rec["id"], datetime.now(timezone.utc) - timedelta(hours=24))

    assert body["is_due"] is True
    assert body["alerts"][-1]["type"] == "overdue"
    next_at = datetime.fromisoformat(body["next_alert_at"])
    expected_done = datetime.fromisoformat(body["expected_completion_at"])
    now = datetime.now(timezone.utc)
    assert now < next_at <= now + timedelta(minutes=30)
    assert (next_at - expected_done) % timedelta(minutes=30) == timedelta(0)


def test_no_alerts_before_start(estate_drc):
    rec = create(["coagulation_time"]).json()
    assert rec["alerts"] == [] and rec["next_alert_at"] is None


def test_water_acid_only_batch_gets_alerts_when_started(estate_drc):
    rec = create(["water_acid"]).json()
    assert rec["coagulation_time_hours"] is None

    body = start_batch(rec["id"], datetime.now(timezone.utc))
    preview = client.post(
        f"{BASE}/predict-preview",
        json={"drc": TEST_DRC, "latex_volume_L": 162.58, "outputs": ["coagulation_time"]},
    ).json()

    # same time as the normal water/acid -> time chain for this batch
    assert body["coagulation_time_hours"] == pytest.approx(preview["coagulation_time_hours"], rel=1e-9)
    assert body["coagulation_time_model_version"] == "component2-coagulation-time-v1"
    assert body["alerts"] and body["next_alert_at"] is not None
    assert body["outputs"] == ["water_acid"]


def test_alerts_stop_after_complete_or_cancel(estate_drc):
    done = create(["coagulation_time"]).json()["id"]
    start_batch(done, datetime.now(timezone.utc) - timedelta(hours=1))
    body = client.patch(f"{BASE}/recommendations/{done}/status", json={"status": "completed"}).json()
    assert body["alerts"] == [] and body["next_alert_at"] is None and body["is_due"] is None

    cancelled = create(["coagulation_time"]).json()["id"]
    start_batch(cancelled, datetime.now(timezone.utc))
    body = client.patch(f"{BASE}/recommendations/{cancelled}/status", json={"status": "cancelled"}).json()
    assert body["alerts"] == [] and body["next_alert_at"] is None


def test_past_and_upcoming_alerts_are_marked(estate_drc):
    rec = create(["coagulation_time"]).json()
    body = start_batch(rec["id"], datetime.now(timezone.utc) - timedelta(minutes=65))
    flags = [a["is_past"] for a in body["alerts"]]
    assert flags[:2] == [True, True]
    assert not any(flags[2:])


def test_two_estates_have_independent_alerts(monkeypatch):
    drc_by_estate = {"PYTEST_C2_EST_A": 35.26, "PYTEST_C2_EST_B": 28.72}
    monkeypatch.setattr(
        coagulation_drc_provider,
        "get_estate_drc",
        lambda estate_id: EstateDrc(estate_id, drc_by_estate[estate_id], "component_1"),
    )
    now = datetime.now(timezone.utc)
    a = create(["coagulation_time"], estate_id="PYTEST_C2_EST_A", volume=50).json()
    b = create(["coagulation_time"], estate_id="PYTEST_C2_EST_B", volume=30).json()
    assert a["drc"] != b["drc"]
    assert a["coagulation_time_hours"] != b["coagulation_time_hours"]

    body_a = start_batch(a["id"], now - timedelta(minutes=40))
    body_b = start_batch(b["id"], now - timedelta(minutes=20))

    def expected_done(body):
        return datetime.fromisoformat(body["started_at"]) + timedelta(hours=body["coagulation_time_hours"])

    assert datetime.fromisoformat(body_a["expected_completion_at"]) == expected_done(body_a)
    assert datetime.fromisoformat(body_b["expected_completion_at"]) == expected_done(body_b)
    assert body_a["next_alert_at"] != body_b["next_alert_at"]
    assert all("PYTEST_C2_EST_A" in x["message"] and "(50 L)" in x["message"] for x in body_a["alerts"])
    assert all("PYTEST_C2_EST_B" in x["message"] and "(30 L)" in x["message"] for x in body_b["alerts"])

    running = client.get(f"{BASE}/recommendations", params={"status": "in_progress"}).json()["items"]
    ours = {item["id"] for item in running if item["estate_id"].startswith("PYTEST_C2_")}
    assert ours == {a["id"], b["id"]}


def test_start_time_in_future_is_rejected(estate_drc):
    rec_id = create(["coagulation_time"]).json()["id"]
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    res = client.patch(
        f"{BASE}/recommendations/{rec_id}/status",
        json={"status": "in_progress", "changed_at": future.isoformat()},
    )
    assert res.status_code == 422
    # batch is left unchanged after a rejected update
    assert client.get(f"{BASE}/recommendations/{rec_id}").json()["status"] == "recommended"


def test_start_before_recommendation_is_rejected(estate_drc):
    rec_id = create(["coagulation_time"]).json()["id"]
    res = client.patch(
        f"{BASE}/recommendations/{rec_id}/status",
        json={"status": "in_progress", "changed_at": "2020-01-01T00:00:00Z"},
    )
    assert res.status_code == 422


def test_status_filter_returns_running_batches(estate_drc):
    running = create(["coagulation_time"]).json()["id"]
    create(["coagulation_time"], estate_id="PYTEST_C2_EST_2")
    start_batch(running, datetime.now(timezone.utc))

    res = client.get(f"{BASE}/recommendations", params={"status": "in_progress"})
    ids = [item["id"] for item in res.json()["items"] if item["estate_id"].startswith("PYTEST_C2_")]
    assert ids == [running]
    assert client.get(f"{BASE}/recommendations", params={"status": "bad"}).status_code == 422


@pytest.mark.parametrize(
    "outputs, expected",
    [
        (["water_acid"], ["water", "acid"]),
        (["coagulation_time"], ["coagulation_time"]),
        (["water_acid", "coagulation_time"], ["water", "acid", "coagulation_time"]),
    ],
)
def test_guidance_matches_chosen_outputs(estate_drc, outputs, expected):
    body = create(outputs).json()
    assert [g["output"] for g in body["guidance"]] == expected


def test_guidance_uses_this_batch_values(estate_drc):
    small = create(["water_acid", "coagulation_time"], volume=20).json()
    large = create(["water_acid", "coagulation_time"], volume=180).json()

    for body in (small, large):
        water, acid, time = body["guidance"]
        assert water["value"] == body["water_acid"]["water_amount_L"]
        assert acid["value"] == body["water_acid"]["prepared_1pct_formic_acid_solution_mL"]
        assert time["value"] == body["coagulation_time_hours"]
        # first point carries this batch's own value
        assert f"{water['value']:.2f} litres" in water["guidance"][0]
        assert f"{acid['value']:.1f} millilitres" in acid["guidance"][0]
        assert time["display_value"] in time["guidance"][0]
        assert "minute" in time["display_value"] or "hour" in time["display_value"]
        assert water["display_value"] == f"{water['value']:.2f} L"
        assert acid["guidance"][-1] == "Do not confuse the prepared 1% solution with concentrated formic acid."
        assert [len(water["guidance"]), len(acid["guidance"]), len(time["guidance"])] == [3, 4, 4]
        assert [g["section"] for g in body["guidance"]] == [
            "1. Water recommendation",
            "2. Prepared 1% formic acid solution recommendation",
            "3. Coagulation time prediction",
        ]
        assert all(g["guidance_heading"] == "Guidance for the farmer" for g in body["guidance"])
        assert all(isinstance(p, str) and p for g in body["guidance"] for p in g["guidance"])

    assert small["guidance"][0]["guidance"][0] != large["guidance"][0]["guidance"][0]


def test_time_guidance_added_when_water_acid_batch_starts(estate_drc):
    rec = create(["water_acid"]).json()
    assert [g["output"] for g in rec["guidance"]] == ["water", "acid"]
    body = start_batch(rec["id"], datetime.now(timezone.utc))
    assert [g["output"] for g in body["guidance"]] == ["water", "acid", "coagulation_time"]


def test_preview_includes_guidance():
    res = client.post(
        f"{BASE}/predict-preview",
        json={"drc": 30.0, "latex_volume_L": 40.0, "outputs": ["coagulation_time"]},
    ).json()
    assert [g["output"] for g in res["guidance"]] == ["coagulation_time"]
    assert res["guidance"][0]["unit"] == "hours"


def test_preview_returns_only_requested_output():
    res = client.post(
        f"{BASE}/predict-preview",
        json={"drc": 30.0, "latex_volume_L": 40.0, "outputs": ["water_acid"]},
    )
    assert res.status_code == 200
    assert res.json()["coagulation_time_hours"] is None
    assert res.json()["water_acid"] is not None


# error handling


def test_drc_unavailable_returns_503():
    # real provider, Component 1 is not connected yet
    res = create(["water_acid"])
    assert res.status_code == 503
    assert "not available" in res.json()["detail"]


def test_manual_drc_is_used_and_saved():
    res = client.post(
        f"{BASE}/recommendations",
        json={"estate_id": TEST_ESTATE_ID, "drc": 30.0, "latex_volume_L": 40.0, "outputs": ["water_acid", "coagulation_time"]},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["drc"] == 30.0
    assert body["drc_source"] == "manual"

    preview = client.post(
        f"{BASE}/predict-preview",
        json={"drc": 30.0, "latex_volume_L": 40.0, "outputs": ["water_acid", "coagulation_time"]},
    ).json()
    assert body["water_acid"] == pytest.approx(preview["water_acid"])
    assert body["coagulation_time_hours"] == pytest.approx(preview["coagulation_time_hours"])


def test_manual_drc_overrides_component_1(estate_drc):
    res = client.post(
        f"{BASE}/recommendations",
        json={"estate_id": TEST_ESTATE_ID, "drc": 30.0, "latex_volume_L": 40.0, "outputs": ["water_acid"]},
    )
    assert res.json()["drc"] == 30.0
    assert res.json()["drc_source"] == "manual"


@pytest.mark.parametrize("drc", [0, -5, 101])
def test_invalid_manual_drc_returns_422(drc):
    res = client.post(
        f"{BASE}/recommendations",
        json={"estate_id": TEST_ESTATE_ID, "drc": drc, "latex_volume_L": 40.0, "outputs": ["water_acid"]},
    )
    assert res.status_code == 422


def test_add_time_to_water_acid_batch(estate_drc):
    rec = create(["water_acid"]).json()
    assert rec["coagulation_time_hours"] is None

    res = client.post(f"{BASE}/recommendations/{rec['id']}/coagulation-time")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == rec["id"]
    assert body["outputs"] == ["water_acid", "coagulation_time"]

    preview = client.post(
        f"{BASE}/predict-preview",
        json={"drc": TEST_DRC, "latex_volume_L": 162.58, "outputs": ["coagulation_time"]},
    ).json()
    assert body["coagulation_time_hours"] == pytest.approx(preview["coagulation_time_hours"])
    assert [g["output"] for g in body["guidance"]] == ["water", "acid", "coagulation_time"]


def test_add_time_to_closed_batch_returns_409(estate_drc):
    rec_id = create(["water_acid"]).json()["id"]
    client.patch(f"{BASE}/recommendations/{rec_id}/status", json={"status": "cancelled"})
    assert client.post(f"{BASE}/recommendations/{rec_id}/coagulation-time").status_code == 409
    assert client.post(f"{BASE}/recommendations/999999999/coagulation-time").status_code == 404


def test_unknown_estate_returns_404(estate_drc):
    assert create(["water_acid"], estate_id="PYTEST_C2_UNKNOWN").status_code == 404


def test_invalid_upstream_drc_returns_503(estate_drc):
    assert create(["water_acid"], estate_id="PYTEST_C2_BADDRC").status_code == 503


def test_missing_recommendation_returns_404():
    assert client.get(f"{BASE}/recommendations/999999999").status_code == 404
    assert (
        client.patch(f"{BASE}/recommendations/999999999/status", json={"status": "cancelled"}).status_code
        == 404
    )


def test_invalid_payload_returns_422():
    assert create([]).status_code == 422
    assert create(["water_acid"], volume=-1).status_code == 422
    assert client.post(
        f"{BASE}/recommendations",
        content='{"estate_id": "EST-001", "latex_volume_L": NaN, "outputs": ["water_acid"]}',
        headers={"content-type": "application/json"},
    ).status_code == 422


def test_invalid_status_transition_returns_409(estate_drc):
    rec_id = create(["coagulation_time"]).json()["id"]
    assert client.patch(f"{BASE}/recommendations/{rec_id}/status", json={"status": "completed"}).status_code == 409
    assert client.patch(f"{BASE}/recommendations/{rec_id}/status", json={"status": "cancelled"}).status_code == 200
    assert client.patch(f"{BASE}/recommendations/{rec_id}/status", json={"status": "in_progress"}).status_code == 409


def test_completion_before_start_returns_422(estate_drc):
    rec_id = create(["coagulation_time"]).json()["id"]
    start = datetime.now(timezone.utc)
    client.patch(
        f"{BASE}/recommendations/{rec_id}/status",
        json={"status": "in_progress", "changed_at": start.isoformat()},
    )
    res = client.patch(
        f"{BASE}/recommendations/{rec_id}/status",
        json={"status": "completed", "changed_at": (start - timedelta(hours=1)).isoformat()},
    )
    assert res.status_code == 422


def test_model_failure_returns_500(estate_drc, monkeypatch):
    from backend.app.ml.component_2_coagulation.model_loader import ModelLoadError

    def broken():
        raise ModelLoadError("missing")

    monkeypatch.setattr(coagulation_service, "get_models", broken)
    assert create(["water_acid"]).status_code == 500


def test_dashboard_is_served():
    res = client.get("/coagulation-dashboard")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "innerHTML" not in res.text
