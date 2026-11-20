from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.app.database.models import CoagulationRecommendation
from backend.app.database.models.coagulation import CoagulationAcidStock
from backend.app.database.session import SessionLocal
from backend.app.main import app
from backend.app.services import coagulation_drc_provider, coagulation_service
from backend.app.services.coagulation_drc_provider import EstateDrc

TEST_ESTATE_ID = "PYTEST_C2_FEAT"
TEST_DRC = 43.8
BASE = "/api/v1/coagulation"

client = TestClient(app)


def cleanup() -> None:
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
def saved_stock():
    # keep the farmer's real stock row and put it back after each test
    with SessionLocal() as db:
        row = db.get(CoagulationAcidStock, coagulation_service.ACID_STOCK_ID)
        saved = (row.stock_ml, row.updated_at) if row else None
        if row:
            db.delete(row)
        db.commit()
    yield
    with SessionLocal() as db:
        row = db.get(CoagulationAcidStock, coagulation_service.ACID_STOCK_ID)
        if row:
            db.delete(row)
            db.flush()
        if saved:
            db.add(CoagulationAcidStock(id=coagulation_service.ACID_STOCK_ID, stock_ml=saved[0], updated_at=saved[1]))
        db.commit()


@pytest.fixture(autouse=True)
def estate_drc(monkeypatch):
    def fake(db, location_id: int) -> EstateDrc:
        return EstateDrc(estate_id=TEST_ESTATE_ID, drc=TEST_DRC, source="component_1")

    monkeypatch.setattr(coagulation_drc_provider, "get_location_drc", fake)


def create(outputs=("water_acid", "coagulation_time"), volume=1.6):
    res = client.post(
        f"{BASE}/recommendations",
        json={"location_id": 1, "latex_volume_L": volume, "outputs": list(outputs)},
    )
    assert res.status_code == 201
    with SessionLocal() as db:
        row = db.get(CoagulationRecommendation, res.json()["id"])
        row.created_at = datetime.now(timezone.utc) - timedelta(hours=48)
        db.commit()
    return res.json()


def start(rec_id: int, started: datetime | None = None):
    body = {"status": "in_progress"}
    if started:
        body["changed_at"] = started.isoformat()
    return client.patch(f"{BASE}/recommendations/{rec_id}/status", json=body)


def set_stock(ml: float):
    return client.put(f"{BASE}/acid-stock", json={"stock_mL": ml})


@pytest.mark.parametrize(
    "share, key, number",
    [(0.0, "liquid", 1), (0.24, "liquid", 1), (0.25, "thickening", 2), (0.59, "thickening", 2),
     (0.6, "setting", 3), (0.99, "setting", 3), (1.0, "firm", 4), (1.8, "firm", 4)],
)
def test_stage_from_share_of_time(share, key, number):
    stage = coagulation_service.coagulation_stage(share)
    assert stage.key == key and stage.number == number


def test_no_stage_before_start():
    rec = create()
    assert rec["stage"] is None and rec["progress_pct"] is None


def test_stage_while_running():
    rec = create()
    hours = rec["coagulation_time_hours"]
    body = start(rec["id"], datetime.now(timezone.utc) - timedelta(hours=hours * 0.4)).json()
    assert body["stage"]["key"] == "thickening"
    assert body["progress_pct"] == pytest.approx(40, abs=1)


def test_stage_firm_when_due_and_completed():
    rec = create()
    started = datetime.now(timezone.utc) - timedelta(hours=24)
    body = start(rec["id"], started).json()
    assert body["stage"]["key"] == "firm" and body["progress_pct"] == 100
    done = client.patch(f"{BASE}/recommendations/{rec['id']}/status", json={"status": "completed"}).json()
    assert done["stage"]["key"] == "firm" and done["progress_pct"] == 100


def test_steps_use_this_batch_amounts():
    rec = create()
    steps = rec["steps"]
    assert [s["number"] for s in steps] == list(range(1, 8))
    water = f"{rec['water_acid']['water_amount_L']:.1f} L"
    acid = f"{rec['water_acid']['prepared_1pct_formic_acid_solution_mL']:.0f} mL"
    text = " ".join(s["detail"] for s in steps)
    assert "1.6 L of latex" in text and water in text and acid in text
    assert coagulation_service.hours_and_minutes(rec["coagulation_time_hours"]) in text


def test_steps_hide_amounts_not_asked_for():
    rec = create(outputs=["coagulation_time"])
    text = " ".join(s["detail"] for s in rec["steps"])
    assert "recommended amount of clean water" in text


def test_stock_not_entered_yet():
    body = client.get(f"{BASE}/acid-stock").json()
    assert body["stock_mL"] is None and body["low_stock"] is False
    assert "Enter" in body["message"]


def test_start_without_stock_records_acid_used():
    rec = create()
    body = start(rec["id"]).json()
    assert body["acid_used_mL"] == round(rec["water_acid"]["prepared_1pct_formic_acid_solution_mL"])


def test_start_lowers_stock():
    set_stock(1000)
    rec = create()
    needed = round(rec["water_acid"]["prepared_1pct_formic_acid_solution_mL"])
    assert start(rec["id"]).status_code == 200
    stock = client.get(f"{BASE}/acid-stock").json()
    assert stock["stock_mL"] == pytest.approx(1000 - needed)
    assert stock["average_batch_acid_mL"] > 0


def test_time_only_batch_takes_no_acid():
    set_stock(10)
    rec = create(outputs=["coagulation_time"])
    body = start(rec["id"]).json()
    assert body["status"] == "in_progress" and body["acid_used_mL"] is None
    assert client.get(f"{BASE}/acid-stock").json()["stock_mL"] == 10


def test_not_enough_acid_blocks_start():
    rec = create()
    set_stock(10)
    res = start(rec["id"])
    assert res.status_code == 409
    assert "Not enough 1% acid solution" in res.json()["detail"]
    # nothing changed
    assert client.get(f"{BASE}/acid-stock").json()["stock_mL"] == 10
    assert client.get(f"{BASE}/recommendations/{rec['id']}").json()["status"] == "recommended"


def test_low_stock_message(monkeypatch):
    # the real average depends on batches already in the database
    needed = 200
    monkeypatch.setattr(coagulation_service, "_average_batch_acid", lambda db: needed)
    body = set_stock(needed * 1.5).json()
    assert body["batches_left"] == 1 and body["low_stock"] is True
    assert "Only enough for 1 more batch" in body["message"]
    body = set_stock(needed * 5).json()
    assert body["batches_left"] == 5 and body["low_stock"] is False


def test_empty_stock_is_low():
    body = set_stock(0).json()
    assert body["low_stock"] is True and "empty" in body["message"]


@pytest.mark.parametrize("body", [{"stock_mL": -1}, {"stock_mL": "abc"}, {}, {"stock_mL": 5, "x": 1}])
def test_bad_stock_returns_422(body):
    assert client.put(f"{BASE}/acid-stock", json=body).status_code == 422


def test_add_new_solution_on_top_of_stock():
    set_stock(150)
    body = client.post(f"{BASE}/acid-stock/add", json={"added_mL": 1000}).json()
    assert body["stock_mL"] == 1150


def test_add_before_any_stock_starts_the_stock():
    body = client.post(f"{BASE}/acid-stock/add", json={"added_mL": 500}).json()
    assert body["stock_mL"] == 500


def test_low_stock_message_says_add_here(monkeypatch):
    monkeypatch.setattr(coagulation_service, "_average_batch_acid", lambda db: 200)
    assert "then add it here" in set_stock(250).json()["message"]


@pytest.mark.parametrize("body", [{"added_mL": 0}, {"added_mL": -5}, {"added_mL": 200000}, {}])
def test_bad_add_returns_422(body):
    assert client.post(f"{BASE}/acid-stock/add", json=body).status_code == 422
