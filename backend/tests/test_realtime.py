"""Tests for the realtime processor's alerting, status, and lifespan wiring.

No video, no model, and no database: the alert logic is pure, and the camera worker is
driven directly with synthetic feature vectors and predictions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.ml.predict import FeatureContribution, RiskScore
from app.services.realtime_processor import (
    CameraWorker,
    RealtimeProcessor,
    _CameraConfig,
)
from app.vision.features import FeatureVector

START = datetime(2026, 3, 4, 14, 30, 0, tzinfo=timezone.utc)


def make_config() -> _CameraConfig:
    return _CameraConfig(
        id=1,
        name="CAM-TEST",
        location_name="North Gate Concourse",
        stream_url="data/samples/crowd.mp4",
        area_sq_meters=50.0,
        pixels_per_meter=40.0,
    )


def make_worker() -> CameraWorker:
    # db_factory is never used: these tests only exercise pure alert logic.
    return CameraWorker(make_config(), db_factory=None)


def make_features(density: float = 1.5, offset: float = 0.0) -> FeatureVector:
    return FeatureVector(
        camera_id=1,
        window_start=START + timedelta(seconds=offset),
        window_end=START + timedelta(seconds=offset + 5),
        window_seconds=5.0,
        person_count=density * 50.0,
        unique_track_count=int(density * 50),
        density=density,
        density_rate_of_change=0.01,
        density_delta=0.05,
        mean_flow_speed_m_s=0.4,
        mean_flow_speed_px_s=16.0,
        flow_direction_variance=0.5,
        optical_flow_entropy=2.0,
        stop_ratio=0.3,
        newly_stopped_ratio=0.1,
        historical_deviation=2.0,
        frame_count=25,
    )


def make_risk(level: str, score: float = 70.0, driver: str = "density") -> RiskScore:
    return RiskScore(
        risk_score=score,
        risk_level=level,
        probabilities={"LOW": 0.05, "MODERATE": 0.05, "HIGH": 0.5, "CRITICAL": 0.4},
        top_features=[FeatureContribution(feature=driver, value=1.5, contribution=1.9)],
        feature_values={"density": 1.5},
    )


# --------------------------------------------------------------------------------------
# which levels alert
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["LOW", "MODERATE"])
def test_low_and_moderate_do_not_alert(level):
    worker = make_worker()
    assert worker._build_alert(make_features(), make_risk(level)) is None


@pytest.mark.parametrize("level", ["HIGH", "CRITICAL"])
def test_high_and_critical_alert(level):
    worker = make_worker()
    alert = worker._build_alert(make_features(), make_risk(level))

    assert alert is not None
    assert alert.camera_id == 1
    assert alert.risk_score == 70.0
    assert alert.acknowledged is not True   # defaults to False on insert


# --------------------------------------------------------------------------------------
# message content
# --------------------------------------------------------------------------------------


def test_alert_message_names_the_location():
    worker = make_worker()
    alert = worker._build_alert(make_features(), make_risk("HIGH"))

    assert alert.message.startswith("High crowd risk detected near North Gate Concourse")


def test_critical_message_is_distinguishable_from_high():
    worker = make_worker()
    high = worker._build_alert(make_features(), make_risk("HIGH")).message

    worker = make_worker()
    critical = worker._build_alert(make_features(), make_risk("CRITICAL")).message

    assert "CRITICAL crowd risk" in critical
    assert "High crowd risk" in high
    assert high != critical


def test_alert_message_carries_the_numbers_an_operator_needs():
    worker = make_worker()
    alert = worker._build_alert(make_features(density=1.42), make_risk("HIGH", score=68.4))

    assert "68/100" in alert.message
    assert "1.42 people/m2" in alert.message
    assert "driven by density" in alert.message


# --------------------------------------------------------------------------------------
# cooldown / de-duplication
# --------------------------------------------------------------------------------------


def test_repeated_high_windows_are_rate_limited():
    """A camera parked at HIGH must not emit an alert every window."""
    worker = make_worker()

    first = worker._build_alert(make_features(), make_risk("HIGH"))
    assert first is not None
    # _build_alert does not update the cooldown; _persist does. Simulate that.
    worker._last_alert_at = __import__("time").monotonic()
    worker._last_alert_level = "HIGH"

    for _ in range(5):
        assert worker._build_alert(make_features(), make_risk("HIGH")) is None


def test_escalation_alerts_even_inside_the_cooldown():
    """HIGH -> CRITICAL is new information and must never be suppressed."""
    import time

    worker = make_worker()
    worker._last_alert_at = time.monotonic()
    worker._last_alert_level = "HIGH"

    assert worker._build_alert(make_features(), make_risk("HIGH")) is None
    assert worker._build_alert(make_features(), make_risk("CRITICAL")) is not None


def test_a_sustained_calm_period_rearms_alerting():
    """A crowd that calms down and later surges again is a new event."""
    import time

    worker = make_worker()
    worker._last_alert_at = time.monotonic()
    worker._last_alert_level = "HIGH"

    for _ in range(settings.realtime_alert_rearm_windows):
        assert worker._build_alert(make_features(), make_risk("LOW")) is None

    assert worker._last_alert_at is None
    assert worker._build_alert(make_features(), make_risk("HIGH")) is not None


def test_a_single_calm_window_does_not_rearm_alerting():
    """Flapping across the HIGH boundary must not alert on every other window."""
    import time

    worker = make_worker()
    worker._last_alert_at = time.monotonic()
    worker._last_alert_level = "HIGH"

    assert worker._build_alert(make_features(), make_risk("MODERATE")) is None
    assert worker._build_alert(make_features(), make_risk("HIGH")) is None


def test_calm_streak_resets_on_an_alerting_window():
    """Calm windows must be consecutive to count toward re-arming."""
    import time

    worker = make_worker()
    worker._last_alert_at = time.monotonic()
    worker._last_alert_level = "HIGH"

    worker._build_alert(make_features(), make_risk("LOW"))     # calm 1
    worker._build_alert(make_features(), make_risk("HIGH"))    # breaks the streak
    worker._build_alert(make_features(), make_risk("LOW"))     # calm 1 again

    assert worker._last_alert_at is not None


def test_cooldown_expires(monkeypatch):
    import time

    worker = make_worker()
    worker._last_alert_level = "HIGH"
    worker._last_alert_at = time.monotonic() - (
        settings.realtime_alert_cooldown_seconds + 1
    )

    assert worker._build_alert(make_features(), make_risk("HIGH")) is not None


def test_first_alert_is_never_suppressed():
    worker = make_worker()
    assert worker._last_alert_at is None
    assert worker._build_alert(make_features(), make_risk("CRITICAL")) is not None


# --------------------------------------------------------------------------------------
# status reporting
# --------------------------------------------------------------------------------------


def test_status_snapshot_is_json_serialisable():
    import json

    worker = make_worker()
    worker.status.last_window_at = START
    blob = json.loads(json.dumps(worker.status.snapshot()))

    assert blob["name"] == "CAM-TEST"
    assert blob["state"] == "stopped"
    assert blob["last_window_at"].startswith("2026-03-04")


def test_processor_status_reports_every_worker():
    processor = RealtimeProcessor()
    processor.workers = {1: make_worker()}

    status = processor.status()
    assert status["camera_count"] == 1
    assert status["cameras"][0]["name"] == "CAM-TEST"


def test_processor_status_is_safe_before_start():
    status = RealtimeProcessor().status()
    assert status["running"] is False
    assert status["cameras"] == []


# --------------------------------------------------------------------------------------
# camera selection
# --------------------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=1):
        return sorted(self._docs, key=lambda doc: doc[key])


class _FakeCollection:
    def __init__(self, docs):
        self._docs = docs

    def find(self, _query=None):
        return _FakeCursor(self._docs)

    def find_one(self, query=None):
        for doc in self._docs:
            if all(doc.get(key) == value for key, value in (query or {}).items()):
                return doc
        return None


class _FakeDb:
    """Minimal stand-in for a pymongo Database returning fixed camera docs."""

    def __init__(self, camera_docs):
        self._camera_docs = camera_docs

    def __getitem__(self, name):
        assert name == "cameras"
        return _FakeCollection(self._camera_docs)


def _fake_camera_doc(id, name, stream_url):
    return {
        "id": id,
        "name": name,
        "location_name": f"loc-{name}",
        "latitude": 0.0,
        "longitude": 0.0,
        "area_sq_meters": 50.0,
        "pixels_per_meter": 40.0,
        "stream_url": stream_url,
        "is_active": True,
        "created_at": START,
    }


def test_cameras_without_a_stream_url_are_skipped():
    """A registered camera with no feed configured must not spawn a worker."""
    cameras = [
        _fake_camera_doc(1, "CAM-A", "rtsp://host/a"),
        _fake_camera_doc(2, "CAM-B", None),
        _fake_camera_doc(3, "CAM-C", ""),
    ]
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb(cameras))

    configs = processor.load_active_cameras()

    assert [c.name for c in configs] == ["CAM-A"]


def test_camera_config_copies_fields_off_the_document():
    """Workers hold plain config values, not database documents."""
    cameras = [_fake_camera_doc(1, "CAM-A", "rtsp://host/a")]
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb(cameras))

    config = processor.load_active_cameras()[0]

    assert isinstance(config, _CameraConfig)
    assert config.area_sq_meters == 50.0
    assert not hasattr(config, "_sa_instance_state")


# --------------------------------------------------------------------------------------
# lifespan wiring
# --------------------------------------------------------------------------------------


def test_lifespan_runs_startup_and_shutdown_with_realtime_disabled(monkeypatch):
    """The API must come up and go down cleanly when workers are switched off."""
    monkeypatch.setattr(settings, "realtime_enabled", False)

    from app.main import app

    async def exercise():
        async with app.router.lifespan_context(app):
            return True
        return False

    assert asyncio.run(exercise()) is True


def test_lifespan_starts_and_stops_the_processor(monkeypatch):
    """With realtime on, startup starts the processor and shutdown stops it."""
    monkeypatch.setattr(settings, "realtime_enabled", True)

    import app.services.realtime_processor as module
    from app.main import app

    events: list[str] = []

    class _StubProcessor:
        def __init__(self):
            self._running = False

        async def start(self):
            events.append("start")
            self._running = True

        async def stop(self, timeout=None):
            events.append("stop")
            self._running = False

        def status(self):
            return {"running": self._running, "camera_count": 0, "cameras": []}

    monkeypatch.setattr(module, "_processor", _StubProcessor())

    async def exercise():
        async with app.router.lifespan_context(app):
            events.append("serving")

    asyncio.run(exercise())

    assert events == ["start", "serving", "stop"]


def test_processor_start_is_idempotent():
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb([]))

    asyncio.run(processor.start())
    assert processor.is_running

    # A second start must not spawn a duplicate set of workers.
    asyncio.run(processor.start())
    assert processor.is_running
    assert len(processor.workers) == 0

    asyncio.run(processor.stop())
    assert not processor.is_running


def test_stop_before_start_is_a_no_op():
    processor = RealtimeProcessor()
    asyncio.run(processor.stop())
    assert not processor.is_running


# --------------------------------------------------------------------------------------
# per-camera start/stop control
# --------------------------------------------------------------------------------------


def test_load_camera_config_filters_inactive_and_unconfigured():
    cameras = [
        _fake_camera_doc(1, "CAM-A", "rtsp://host/a"),
        {**_fake_camera_doc(2, "CAM-B", "rtsp://host/b"), "is_active": False},
        _fake_camera_doc(3, "CAM-C", None),
    ]
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb(cameras))

    assert processor.load_camera_config(1).name == "CAM-A"
    assert processor.load_camera_config(2) is None  # inactive
    assert processor.load_camera_config(3) is None  # no stream_url
    assert processor.load_camera_config(99) is None  # missing


def test_start_camera_spawns_a_worker_and_stop_removes_it(monkeypatch):
    """The control path must manage one worker without touching the others."""
    monkeypatch.setattr(settings, "realtime_restart_delay_seconds", 0.01)
    # A path that fails fast, so the worker exercises the reconnect loop rather
    # than blocking on a real source.
    cameras = [_fake_camera_doc(1, "CAM-A", "does-not-exist.mp4")]
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb(cameras))

    async def exercise():
        worker = await processor.start_camera(1)
        assert worker is not None
        assert processor.is_running
        assert 1 in processor.workers

        # Idempotent: starting a running camera returns the same worker.
        assert await processor.start_camera(1) is worker

        stopped = await processor.stop_camera(1, timeout=5)
        assert stopped is worker
        assert 1 not in processor.workers
        # Stopping a camera with no worker is a no-op, not an error.
        assert await processor.stop_camera(99) is None

    asyncio.run(exercise())


def test_start_camera_returns_none_for_inactive_or_missing(monkeypatch):
    monkeypatch.setattr(settings, "realtime_restart_delay_seconds", 0.01)
    cameras = [{**_fake_camera_doc(1, "CAM-A", "rtsp://host/a"), "is_active": False}]
    processor = RealtimeProcessor(db_factory=lambda: _FakeDb(cameras))

    async def exercise():
        assert await processor.start_camera(1) is None
        assert await processor.start_camera(99) is None
        assert processor.workers == {}

    asyncio.run(exercise())
