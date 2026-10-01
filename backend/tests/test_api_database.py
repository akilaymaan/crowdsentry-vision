"""Database-backed API contract tests.

These tests use the configured MongoDB database: fixture docs are created with a
unique TEST- marker and deleted at teardown. Set ``MONGO_URI`` to a dedicated test
database when running them against a shared environment. They skip cleanly when
MongoDB is not available, while the database-free API contract tests in
``test_api.py`` remain runnable everywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.core.database import ALERTS, CAMERAS, OBSERVATIONS, RISK_SCORES, db, get_db, next_id, point
from app.main import app


@pytest.fixture
def database_client(monkeypatch):
    """TestClient plus one camera and its related docs, cleaned up at teardown."""
    monkeypatch.setattr(settings, "realtime_enabled", False)
    # These tests exercise the endpoints, not auth: force it off regardless of the
    # ambient API_KEY.
    monkeypatch.setattr(settings, "api_key", "")

    try:
        db.command("ping")
    except PyMongoError as exc:
        pytest.skip(f"database-backed API tests require MongoDB: {exc}")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    camera_id = next_id(db, CAMERAS)
    tag = uuid4().hex[:12]
    camera_doc = {
        "id": camera_id,
        "name": f"TEST-API-{tag}",
        "location_name": "API Test Gate",
        "latitude": 12.97,
        "longitude": 77.59,
        "location": point(77.59, 12.97),
        "area_sq_meters": 100.0,
        "pixels_per_meter": 40.0,
        "stream_url": None,
        "is_active": True,
        "created_at": now,
    }
    db[CAMERAS].insert_one(camera_doc)

    observation_doc = {
        "id": next_id(db, OBSERVATIONS),
        "camera_id": camera_id,
        "timestamp": now - timedelta(minutes=2),
        "person_count": 42,
        "density": 0.42,
        "mean_flow_speed": 0.8,
        "flow_direction_variance": 0.2,
        "optical_flow_entropy": 1.0,
        "stop_ratio": 0.1,
        "density_rate_of_change": 0.01,
        "historical_deviation": 0.5,
    }
    score_doc = {
        "id": next_id(db, RISK_SCORES),
        "camera_id": camera_id,
        "timestamp": now - timedelta(minutes=2),
        "risk_score": 71.0,
        "risk_level": "HIGH",
        "contributing_features": {"top_features": []},
    }
    alert_doc = {
        "id": next_id(db, ALERTS),
        "camera_id": camera_id,
        "timestamp": now - timedelta(minutes=1),
        "risk_score": 71.0,
        "risk_level": "HIGH",
        "message": "High crowd risk detected near API Test Gate",
        "acknowledged": False,
    }
    db[OBSERVATIONS].insert_one(observation_doc)
    db[RISK_SCORES].insert_one(score_doc)
    db[ALERTS].insert_one(alert_doc)

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client, camera_id, now
    finally:
        app.dependency_overrides.pop(get_db, None)
        db[CAMERAS].delete_one({"id": camera_id})
        for collection in (OBSERVATIONS, RISK_SCORES, ALERTS):
            db[collection].delete_many({"camera_id": camera_id})


def test_camera_list_detail_and_history_use_the_database(database_client):
    client, camera_id, now = database_client

    cameras = client.get("/api/cameras").raise_for_status().json()
    camera = next(item for item in cameras if item["id"] == camera_id)
    assert camera["latest_observation"]["person_count"] == 42
    assert camera["latest_risk"]["risk_level"] == "HIGH"

    detail = client.get(f"/api/cameras/{camera_id}").raise_for_status().json()
    assert detail["unacknowledged_alerts"] == 1
    assert detail["latest_observation"]["density"] == pytest.approx(0.42)

    start = (now - timedelta(hours=1)).isoformat()
    end = (now + timedelta(minutes=1)).isoformat()
    history = client.get(
        f"/api/cameras/{camera_id}/history",
        params={"from": start, "to": end},
    ).raise_for_status().json()
    assert history["camera_id"] == camera_id
    assert len(history["observations"]) == 1
    assert len(history["risk_scores"]) == 1


def test_dashboard_and_alert_endpoints_use_the_database(database_client):
    client, camera_id, _ = database_client

    summary = client.get(
        "/api/dashboard/summary", params={"freshness_seconds": 3600}
    ).raise_for_status().json()
    assert summary["reporting_cameras"] >= 1
    assert summary["total_people"] >= 42
    assert summary["unacknowledged_alerts"] >= 1

    alerts = client.get(
        "/api/alerts", params={"camera_id": camera_id, "acknowledged": "false"}
    ).raise_for_status().json()
    assert alerts["total"] == 1
    alert_id = alerts["alerts"][0]["id"]
    assert alerts["alerts"][0]["camera_name"].startswith("TEST-API-")

    acknowledged = client.post(f"/api/alerts/{alert_id}/acknowledge")
    acknowledged.raise_for_status()
    assert acknowledged.json()["acknowledged"] is True

    remaining = client.get(
        "/api/alerts", params={"camera_id": camera_id, "acknowledged": "false"}
    ).raise_for_status().json()
    assert remaining["total"] == 0


def test_missing_camera_and_alert_return_not_found(database_client):
    client, _, _ = database_client

    assert client.get("/api/cameras/999999999").status_code == 404
    assert client.get("/api/cameras/999999999/history").status_code == 404
    assert client.post("/api/alerts/999999999/acknowledge").status_code == 404
