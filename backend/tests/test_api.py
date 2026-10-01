"""Tests for the REST and WebSocket API.

The database is stubbed with a dependency override, so these run with no MongoDB and
no camera workers. They cover the response contracts and the routing/serialisation
wiring; the queries themselves are exercised in test_api_database.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import get_db
from app.models import RiskLevel
from app.schemas import (
    AlertWithCamera,
    CameraSummary,
    DashboardSummary,
    LiveRiskEvent,
    RiskLevelCounts,
)
from app.services.broadcaster import LiveBroadcaster

NOW = datetime(2026, 3, 4, 14, 30, tzinfo=timezone.utc)


@pytest.fixture
def client(monkeypatch):
    """A TestClient with realtime disabled, auth off, and the DB stubbed out."""
    monkeypatch.setattr(settings, "realtime_enabled", False)
    # Deterministic regardless of the developer's ambient API_KEY.
    monkeypatch.setattr(settings, "api_key", "")
    from app.main import app

    def _no_db():
        # A clean 503 rather than an AssertionError: TestClient re-raises server
        # exceptions, which would mask the status code the test is checking.
        raise HTTPException(status_code=503, detail="no database in these tests")

    app.dependency_overrides[get_db] = _no_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------------------
# schema contracts
# --------------------------------------------------------------------------------------


def test_camera_summary_allows_a_never_scored_camera():
    """A camera that has never produced a score must serialise, not fail validation."""
    summary = CameraSummary(
        id=1,
        name="CAM-01",
        location_name="Gate",
        latitude=1.0,
        longitude=2.0,
        area_sq_meters=50.0,
        pixels_per_meter=None,
        stream_configured=False,
        is_active=True,
        created_at=NOW,
    )
    assert summary.latest_risk is None
    assert summary.model_dump()["pixels_per_meter"] is None


def test_risk_level_serialises_to_its_string_value():
    event = LiveRiskEvent(
        camera_id=1,
        camera_name="CAM-01",
        location_name="Gate",
        timestamp=NOW,
        risk_score=71.5,
        risk_level=RiskLevel.CRITICAL,
        person_count=12.4,
        density=1.8,
    )
    dumped = event.model_dump(mode="json")
    assert dumped["risk_level"] == "CRITICAL"
    assert dumped["type"] == "risk_score"


def test_dashboard_summary_risk_counts_default_to_zero():
    summary = DashboardSummary(
        generated_at=NOW,
        total_cameras=0,
        active_cameras=0,
        reporting_cameras=0,
        stale_cameras=0,
        total_people=0,
        cameras_by_risk_level=RiskLevelCounts(),
        freshness_seconds=120.0,
    )
    assert summary.cameras_by_risk_level.CRITICAL == 0
    assert summary.highest_risk_level is None


def test_alert_with_camera_carries_the_camera_name():
    alert = AlertWithCamera(
        id=1,
        camera_id=2,
        timestamp=NOW,
        risk_score=80.0,
        message="High crowd risk detected near Gate",
        acknowledged=False,
        camera_name="CAM-02",
        camera_location="Gate",
    )
    assert alert.camera_name == "CAM-02"


# --------------------------------------------------------------------------------------
# routing / OpenAPI wiring
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/cameras",
        "/api/cameras/{camera_id}",
        "/api/cameras/{camera_id}/history",
        "/api/cameras/{camera_id}/start",
        "/api/cameras/{camera_id}/stop",
        "/api/alerts",
        "/api/alerts/{alert_id}/acknowledge",
        "/api/dashboard/summary",
    ],
)
def test_every_endpoint_is_registered(client, path):
    assert path in client.app.openapi()["paths"]


def test_history_uses_from_and_to_as_query_names(client):
    """`from` is a Python keyword, so it must be exposed via an alias."""
    spec = client.app.openapi()["paths"]["/api/cameras/{camera_id}/history"]["get"]
    names = {p["name"] for p in spec["parameters"]}
    assert {"from", "to"} <= names


def test_health_does_not_require_the_database(client):
    assert client.get("/health").json()["status"] == "ok"


# --------------------------------------------------------------------------------------
# authentication
# --------------------------------------------------------------------------------------


@pytest.fixture
def secured_client(monkeypatch):
    """A TestClient with API-key auth switched on and the DB stubbed out."""
    monkeypatch.setattr(settings, "realtime_enabled", False)
    monkeypatch.setattr(settings, "api_key", "test-secret")
    from app.main import app

    def _no_db():
        raise HTTPException(status_code=503, detail="no database in these tests")

    app.dependency_overrides[get_db] = _no_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_requests_are_open_when_no_api_key_is_configured(client):
    """Local dev default: an empty API_KEY means no auth challenge at all."""
    response = client.get("/api/cameras")
    assert response.status_code == 503  # past auth; only the DB stub failed


def test_api_requires_a_key_when_one_is_configured(secured_client):
    assert secured_client.get("/api/cameras").status_code == 401
    assert secured_client.post("/api/alerts/1/acknowledge").status_code == 401
    assert secured_client.post("/api/cameras/1/start").status_code == 401
    assert secured_client.post("/api/cameras/1/stop").status_code == 401
    assert secured_client.get("/api/dashboard/summary").status_code == 401
    assert secured_client.get("/api/processor/status").status_code == 401


def test_api_accepts_the_key_as_header_or_bearer(secured_client):
    # 503 means auth passed and the request reached the stubbed DB dependency.
    assert secured_client.get(
        "/api/cameras", headers={"X-API-Key": "test-secret"}
    ).status_code == 503
    assert secured_client.get(
        "/api/cameras", headers={"Authorization": "Bearer test-secret"}
    ).status_code == 503
    assert (
        secured_client.get("/api/cameras", headers={"X-API-Key": "wrong"}).status_code
        == 401
    )


def test_probes_stay_open_when_auth_is_on(secured_client):
    """Health checks come from load balancers that carry no credentials."""
    assert secured_client.get("/health").status_code == 200


def test_docs_are_disabled_unless_explicitly_enabled():
    """A fresh Settings (no .env, no ambient var) must not advertise the API surface."""
    from app.core.config import Settings

    assert Settings(_env_file=None).api_docs_enabled is False


def test_websocket_rejects_a_missing_key(secured_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with secured_client.websocket_connect("/ws/live"):
            pass


def test_websocket_accepts_the_key_as_a_query_param(secured_client):
    """Browsers cannot set WS headers, so ?api_key= is the dashboard's credential."""
    with secured_client.websocket_connect("/ws/live?api_key=test-secret"):
        pass


def test_websocket_is_open_when_no_key_is_configured(client):
    with client.websocket_connect("/ws/live"):
        pass


def test_stream_urls_never_appear_in_api_responses():
    """stream_url can embed feed credentials; only the boolean flag may leak out."""
    fields = set(CameraSummary.model_fields)
    assert "stream_url" not in fields
    assert "stream_configured" in fields


def test_stream_credentials_are_redacted_from_status():
    from app.core.security import redact_url

    assert redact_url("rtsp://admin:s3cret@cam-01.local:554/stream") == (
        "rtsp://cam-01.local:554/stream"
    )
    assert redact_url("data/samples/crowd.mp4") == "data/samples/crowd.mp4"
    assert redact_url(None) is None


def test_security_headers_are_set_on_api_responses(client):
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_cors_allows_the_vite_dev_server(client):
    response = client.options(
        "/api/cameras",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_an_unknown_origin(client):
    response = client.options(
        "/api/cameras",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_both_localhost_and_loopback_are_allowed():
    """They are distinct origins to a browser; missing one breaks the dev server."""
    assert "http://localhost:5173" in settings.cors_origin_list
    assert "http://127.0.0.1:5173" in settings.cors_origin_list


# --------------------------------------------------------------------------------------
# broadcaster
# --------------------------------------------------------------------------------------


def test_publish_without_a_bound_loop_is_a_no_op():
    """Processing must not fail when the API is not running."""
    broadcaster = LiveBroadcaster()
    broadcaster.publish_threadsafe({"type": "risk_score"})   # must not raise


def test_publish_with_no_subscribers_is_a_no_op():
    async def exercise():
        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        broadcaster.publish_threadsafe({"type": "risk_score"})
        assert broadcaster.subscriber_count == 0

    asyncio.run(exercise())


def test_subscribers_receive_published_events():
    async def exercise():
        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        first = broadcaster.subscribe("a")
        second = broadcaster.subscribe("b")

        broadcaster.publish({"type": "risk_score", "camera_id": 1})

        assert first.queue.get_nowait()["camera_id"] == 1
        assert second.queue.get_nowait()["camera_id"] == 1

    asyncio.run(exercise())


def test_publish_threadsafe_delivers_from_another_thread():
    """The real path: worker thread publishes, event loop delivers."""

    async def exercise():
        import threading

        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        subscriber = broadcaster.subscribe("a")

        threading.Thread(
            target=broadcaster.publish_threadsafe,
            args=({"type": "risk_score", "camera_id": 7},),
        ).start()

        event = await asyncio.wait_for(subscriber.queue.get(), timeout=2.0)
        assert event["camera_id"] == 7

    asyncio.run(exercise())


def test_a_lagging_subscriber_drops_the_oldest_event():
    """A stale reading is worthless; the newest one is the point."""

    async def exercise():
        from app.services import broadcaster as module

        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        subscriber = broadcaster.subscribe("slow")

        for index in range(module.QUEUE_MAXSIZE + 5):
            broadcaster.publish({"seq": index})

        assert subscriber.queue.qsize() == module.QUEUE_MAXSIZE
        assert subscriber.dropped == 5
        # The oldest five are gone; the newest is still queued.
        assert subscriber.queue.get_nowait()["seq"] == 5

    asyncio.run(exercise())


def test_unsubscribe_stops_delivery():
    async def exercise():
        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        subscriber = broadcaster.subscribe("a")
        broadcaster.unsubscribe(subscriber)

        broadcaster.publish({"type": "risk_score"})

        assert broadcaster.subscriber_count == 0
        assert subscriber.queue.empty()

    asyncio.run(exercise())


def test_unbind_loop_stops_threadsafe_publishing():
    async def exercise():
        broadcaster = LiveBroadcaster()
        broadcaster.bind_loop()
        subscriber = broadcaster.subscribe("a")
        broadcaster.unbind_loop()

        broadcaster.publish_threadsafe({"type": "risk_score"})
        await asyncio.sleep(0)

        assert subscriber.queue.empty()

    asyncio.run(exercise())


# --------------------------------------------------------------------------------------
# history bounds
# --------------------------------------------------------------------------------------


def test_history_row_cap_is_enforced_in_the_schema(client):
    spec = client.app.openapi()["paths"]["/api/cameras/{camera_id}/history"]["get"]
    limit = next(p for p in spec["parameters"] if p["name"] == "limit")
    from app.routers.cameras import MAX_HISTORY_ROWS

    assert limit["schema"]["maximum"] == MAX_HISTORY_ROWS


def test_default_history_window_is_an_hour():
    from app.routers.cameras import DEFAULT_HISTORY_HOURS

    assert timedelta(hours=DEFAULT_HISTORY_HOURS) == timedelta(hours=1)
