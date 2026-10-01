"""Realtime processing service: video in, risk scores and alerts out.

One worker per active camera, each running the full pipeline end to end::

    VideoSource -> PersonTracker -> FeatureExtractor -> predict_risk
                                          |                  |
                                   crowd_observations    risk_scores
                                                             |
                                                    alerts (HIGH/CRITICAL)

**Threading.** The pipeline is entirely blocking, CPU-bound work: YOLO inference, OpenCV
decoding, and synchronous PyMongo writes. None of it can run on the event loop without
stalling every HTTP request. Each camera therefore runs its whole pipeline in its own
thread via ``asyncio.to_thread``, and the async layer only supervises -- start, stop,
restart-on-failure, and report status. This is not a workaround: torch, OpenCV and
PyMongo all release the GIL during their heavy work, so the threads genuinely run in
parallel.

The alternative -- awaiting each frame -- would mean thousands of thread hops per second
to do the same blocking work, with a more tangled shutdown path and no benefit.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.database import CAMERAS, RISK_SCORES, ALERTS, insert_document
from app.core.logging import get_logger
from app.core.security import redact_url
from app.ml.predict import ModelNotTrainedError, RiskScore, predict_risk
from app.models import Alert, Camera, RiskLevel
from app.schemas import LiveRiskEvent
from app.services.broadcaster import get_broadcaster
from app.vision.features import FeatureExtractor, FeatureVector
from app.vision.source import VideoSourceError, open_source
from app.vision.tracking import PersonTracker, TrackedFrame

logger = get_logger("realtime")

# Levels that raise an operator alert.
ALERTING_LEVELS = ("HIGH", "CRITICAL")
# Ordering used to decide whether a level is an escalation over the last alert.
LEVEL_ORDER = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


@dataclass
class CameraStatus:
    """Live counters for one camera worker, safe to read from the event loop."""

    camera_id: int
    name: str
    location_name: str
    stream_url: str | None
    state: str = "stopped"          # stopped | starting | running | reconnecting | failed
    frames_processed: int = 0
    windows_processed: int = 0
    alerts_raised: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    last_window_at: datetime | None = None
    last_risk_level: str | None = None
    last_risk_score: float | None = None
    last_density: float | None = None
    started_at: datetime | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location_name": self.location_name,
            # Stream specs can carry embedded credentials (rtsp://user:pass@...), so
            # the status endpoint reports the redacted form only.
            "stream_url": redact_url(self.stream_url),
            "state": self.state,
            "frames_processed": self.frames_processed,
            "windows_processed": self.windows_processed,
            "alerts_raised": self.alerts_raised,
            "last_error": self.last_error,
            "last_window_at": (
                self.last_window_at.isoformat() if self.last_window_at else None
            ),
            "last_risk_level": self.last_risk_level,
            "last_risk_score": self.last_risk_score,
            "last_density": self.last_density,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }


@dataclass
class _CameraConfig:
    """Everything a worker needs, read once so the thread never touches a DB doc.

    Copying the handful of fields keeps the worker independent of the document shape
    and of whatever the reader thread was doing.
    """

    id: int
    name: str
    location_name: str
    stream_url: str
    area_sq_meters: float
    pixels_per_meter: float | None


class CameraWorker:
    """Runs the full pipeline for one camera in a dedicated thread."""

    def __init__(
        self,
        config: _CameraConfig,
        db_factory=None,
        loop_video: bool = False,
        window_seconds: float | None = None,
    ) -> None:
        self.config = config
        self.loop_video = loop_video
        self.window_seconds = window_seconds
        self._db_factory = db_factory

        self.status = CameraStatus(
            camera_id=config.id,
            name=config.name,
            location_name=config.location_name,
            stream_url=config.stream_url,
        )

        self._stop = threading.Event()
        self._task: asyncio.Task | None = None
        self.log = logger.bind(camera=config.name)

        # Alert de-duplication state.
        self._last_alert_at: float | None = None
        self._last_alert_level: str | None = None
        self._calm_windows = 0

    # -- database handle ---------------------------------------------------------

    def _db(self):
        if self._db_factory is not None:
            return self._db_factory()
        from app.core.database import db

        return db

    # -- async supervision -------------------------------------------------------

    async def start(self) -> None:
        self._stop.clear()
        self.status.state = "starting"
        self.status.started_at = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._supervise(), name=f"camera-{self.config.name}")

    async def stop(self, timeout: float = 20.0) -> None:
        """Signal the worker to stop and wait for its thread to unwind."""
        self._stop.set()
        if self._task is None:
            return

        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
        except asyncio.TimeoutError:
            # The thread is inside a blocking call (a frame read, an inference) and
            # cannot be cancelled; it will exit at the next stop check. Say so plainly
            # rather than pretending shutdown was clean.
            self.log.warning("worker did not stop in time", timeout=timeout)
        except asyncio.CancelledError:
            pass
        finally:
            self.status.state = "stopped"

    async def _supervise(self) -> None:
        """Run the pipeline, restarting it if the source fails."""
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self._run_pipeline)

                if self._stop.is_set():
                    break

                # The pipeline returned without an error: the source ran out (a
                # non-looping file). That is completion, not failure.
                self.log.info("source exhausted", windows=self.status.windows_processed)
                self.status.state = "stopped"
                return

            except VideoSourceError as exc:
                self._note_failure(str(exc), "source unavailable")
            except ModelNotTrainedError as exc:
                # Retrying will not help: no model on disk.
                self.log.error("risk model unavailable, stopping camera", error=str(exc))
                self.status.state = "failed"
                self.status.last_error = str(exc)
                return
            except Exception as exc:  # noqa: BLE001 - one camera must not kill the rest
                self.log.exception("pipeline error", error=type(exc).__name__)
                self._note_failure(f"{type(exc).__name__}: {exc}", "pipeline error")

            if self._stop.is_set():
                break

            if self.status.consecutive_failures >= settings.realtime_max_consecutive_failures:
                self.log.error(
                    "giving up after repeated failures",
                    failures=self.status.consecutive_failures,
                )
                self.status.state = "failed"
                return

            self.status.state = "reconnecting"
            await asyncio.sleep(settings.realtime_restart_delay_seconds)

        self.status.state = "stopped"

    def _note_failure(self, error: str, message: str) -> None:
        self.status.consecutive_failures += 1
        self.status.last_error = error
        self.log.warning(
            message, error=error, attempt=self.status.consecutive_failures
        )

    # -- the blocking pipeline ---------------------------------------------------

    def _stoppable(self, frames: Iterator[TrackedFrame]) -> Iterator[TrackedFrame]:
        """Pass frames through, aborting promptly when a stop is requested."""
        for frame in frames:
            if self._stop.is_set():
                return
            self.status.frames_processed += 1
            yield frame

    def _run_pipeline(self) -> None:
        """The whole per-camera pipeline. Runs in a worker thread."""
        config = self.config

        source_kwargs = {}
        is_file = "://" not in config.stream_url and not config.stream_url.isdigit()
        if self.loop_video and is_file:
            source_kwargs["loop"] = True

        source = open_source(config.stream_url, **source_kwargs)

        tracker = PersonTracker(pixels_per_meter=config.pixels_per_meter)
        extractor = FeatureExtractor(
            camera_id=config.id,
            area_sq_meters=config.area_sq_meters,
            pixels_per_meter=config.pixels_per_meter,
            window_seconds=self.window_seconds,
            persist=True,
            db_factory=self._db_factory,
        )

        self.log.info(
            "starting",
            source=redact_url(config.stream_url),
            area=config.area_sq_meters,
            ppm=config.pixels_per_meter,
            calibrated=extractor.calibrated,
            window=extractor.window_seconds,
            stride=settings.realtime_frame_stride,
        )

        self.status.state = "running"
        # A successful open resets the failure count, so intermittent dropouts do not
        # accumulate toward the give-up threshold over hours of healthy running.
        self.status.consecutive_failures = 0

        tracked = tracker.track(source, stride=settings.realtime_frame_stride)

        for features in extractor.extract(self._stoppable(tracked)):
            if self._stop.is_set():
                break
            self._handle_window(features)

        self.log.info("stopped", windows=self.status.windows_processed)

    # -- per-window handling -----------------------------------------------------

    def _handle_window(self, features: FeatureVector) -> None:
        """Score one window, persist the result, and alert if warranted."""
        started = time.perf_counter()
        risk = predict_risk(features)

        self.status.windows_processed += 1
        self.status.last_window_at = features.window_end
        self.status.last_risk_level = risk.risk_level
        self.status.last_risk_score = risk.risk_score
        self.status.last_density = round(features.density, 4)

        self._persist(features, risk)
        self._broadcast(features, risk)

        self.log.info(
            "window scored",
            level=risk.risk_level,
            score=risk.risk_score,
            confidence=round(risk.confidence, 2),
            density=round(features.density, 3),
            people=round(features.person_count, 1),
            stop_ratio=round(features.stop_ratio, 2),
            driver=risk.top_features[0].feature if risk.top_features else None,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _broadcast(self, features: FeatureVector, risk: RiskScore) -> None:
        """Push this score to any connected dashboards.

        Fire-and-forget from a worker thread: publish_threadsafe hands the event to the
        event loop and returns immediately, so the pipeline never waits on a client and
        a broadcast failure can never stall processing.
        """
        try:
            event = LiveRiskEvent(
                camera_id=self.config.id,
                camera_name=self.config.name,
                location_name=self.config.location_name,
                timestamp=features.window_end,
                risk_score=risk.risk_score,
                risk_level=RiskLevel(risk.risk_level),
                person_count=round(features.person_count, 2),
                density=round(features.density, 4),
                top_feature=(
                    risk.top_features[0].feature if risk.top_features else None
                ),
            )
            get_broadcaster().publish_threadsafe(event.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - broadcasting is never worth failing a window
            self.log.exception("could not broadcast live event")

    def _persist(self, features: FeatureVector, risk: RiskScore) -> None:
        """Write the risk score and, when warranted, an alert.

        Two independent inserts rather than one transaction: MongoDB document writes are
        each atomic, and multi-document transactions need a replica set -- not worth
        requiring here, since a crash between the two writes is rare and benign (an
        alert without its score still names the camera, level, and time).
        """
        alert = self._build_alert(features, risk)
        alert_message = alert.message if alert is not None else None

        database = self._db()
        insert_document(
            database,
            RISK_SCORES,
            risk.to_doc(camera_id=self.config.id, timestamp=features.window_end),
        )
        if alert is not None:
            insert_document(database, ALERTS, alert.to_doc())

        if alert is not None:
            self.status.alerts_raised += 1
            self._last_alert_at = time.monotonic()
            self._last_alert_level = risk.risk_level
            self.log.warning(
                "ALERT raised",
                level=risk.risk_level,
                score=risk.risk_score,
                message=alert_message,
            )

    def _build_alert(self, features: FeatureVector, risk: RiskScore) -> Alert | None:
        """An Alert document when this window warrants one, else None."""
        if risk.risk_level not in ALERTING_LEVELS:
            self._calm_windows += 1
            if self._calm_windows >= settings.realtime_alert_rearm_windows:
                # The condition genuinely ended. Clear the cooldown as well as the
                # remembered level: a crowd that calms down and later surges again is a
                # new event, and should not be silenced by a stale cooldown from the
                # previous one.
                self._last_alert_level = None
                self._last_alert_at = None
            return None

        self._calm_windows = 0

        if not self._should_alert(risk.risk_level):
            return None

        return Alert(
            camera_id=self.config.id,
            timestamp=features.window_end,
            risk_score=risk.risk_score,
            risk_level=RiskLevel(risk.risk_level),
            message=self._alert_message(features, risk),
        )

    def _should_alert(self, level: str) -> bool:
        """Rate-limit alerts, but never suppress an escalation.

        A camera parked at HIGH would otherwise emit an alert every window. Escalating
        HIGH -> CRITICAL is new information and always gets through.
        """
        if self._last_alert_level is not None and LEVEL_ORDER.index(level) > LEVEL_ORDER.index(
            self._last_alert_level
        ):
            return True

        if self._last_alert_at is None:
            return True

        elapsed = time.monotonic() - self._last_alert_at
        if elapsed >= settings.realtime_alert_cooldown_seconds:
            return True

        self.log.debug(
            "alert suppressed by cooldown", level=level, elapsed=round(elapsed, 1)
        )
        return False

    def _alert_message(self, features: FeatureVector, risk: RiskScore) -> str:
        headline = (
            "CRITICAL crowd risk"
            if risk.risk_level == "CRITICAL"
            else "High crowd risk"
        )
        message = (
            f"{headline} detected near {self.config.location_name} "
            f"- risk {risk.risk_score:.0f}/100, "
            f"density {features.density:.2f} people/m2"
        )
        if risk.top_features:
            message += f", driven by {risk.top_features[0].feature}"
        return message


class RealtimeProcessor:
    """Owns one :class:`CameraWorker` per active camera."""

    def __init__(
        self,
        db_factory=None,
        loop_video: bool | None = None,
        window_seconds: float | None = None,
    ) -> None:
        self._db_factory = db_factory
        self.loop_video = (
            loop_video if loop_video is not None else settings.realtime_loop_video
        )
        self.window_seconds = window_seconds
        self.workers: dict[int, CameraWorker] = {}
        self._running = False
        # Serialises start_camera/stop_camera so an API call cannot race the
        # lifecycle of the worker it is starting or stopping.
        self._control_lock = asyncio.Lock()

    def _db(self):
        if self._db_factory is not None:
            return self._db_factory()
        from app.core.database import db

        return db

    @staticmethod
    def _config_for(camera: Camera) -> _CameraConfig | None:
        """A worker config, or None when the camera has no feed configured."""
        if not camera.stream_url:
            return None
        return _CameraConfig(
            id=camera.id,
            name=camera.name,
            location_name=camera.location_name,
            stream_url=camera.stream_url,
            area_sq_meters=camera.area_sq_meters,
            pixels_per_meter=camera.pixels_per_meter,
        )

    def load_active_cameras(self) -> list[_CameraConfig]:
        """Cameras that are active and actually have a feed configured."""
        camera_docs = self._db()[CAMERAS].find({"is_active": True}).sort("id", 1)

        configs = []
        skipped = []
        for camera in (Camera.from_doc(doc) for doc in camera_docs):
            config = self._config_for(camera)
            if config is None:
                skipped.append(camera.name)
                continue
            configs.append(config)

        if skipped:
            logger.info("skipping cameras with no stream_url", cameras=",".join(skipped))

        return configs

    def load_camera_config(self, camera_id: int) -> _CameraConfig | None:
        """Config for one active camera, or None when missing/inactive/unconfigured."""
        doc = self._db()[CAMERAS].find_one({"id": camera_id})
        if doc is None:
            return None
        camera = Camera.from_doc(doc)
        if not camera.is_active:
            return None
        return self._config_for(camera)

    @staticmethod
    def _limit_torch_threads() -> None:
        """Stop inference from monopolising every core.

        Torch's default intra-op thread count is half the machine's cores, which starves
        the event loop and the other camera workers. Process-global, so it is set once.
        """
        if settings.realtime_torch_threads <= 0:
            return
        try:
            import torch

            torch.set_num_threads(settings.realtime_torch_threads)
            logger.info("capped torch threads", threads=settings.realtime_torch_threads)
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            logger.warning("could not cap torch threads", error=str(exc))

    async def start(self) -> None:
        if self._running:
            return

        await asyncio.to_thread(self._limit_torch_threads)

        try:
            configs = await asyncio.to_thread(self.load_active_cameras)
        except Exception as exc:  # noqa: BLE001 - the API must still come up
            logger.exception("could not load cameras; processor idle", error=str(exc))
            return

        if not configs:
            logger.info(
                "no active cameras with a stream_url; processor idle "
                "(set cameras.stream_url and cameras.is_active)"
            )
            self._running = True
            return

        logger.info("starting workers", cameras=len(configs))
        for config in configs:
            worker = CameraWorker(
                config,
                db_factory=self._db_factory,
                loop_video=self.loop_video,
                window_seconds=self.window_seconds,
            )
            self.workers[config.id] = worker
            await worker.start()

        self._running = True

    async def stop(self, timeout: float | None = None) -> None:
        if not self._running:
            return

        timeout = timeout or settings.realtime_shutdown_timeout_seconds
        if self.workers:
            logger.info("stopping workers", cameras=len(self.workers))
            await asyncio.gather(
                *(worker.stop(timeout=timeout) for worker in self.workers.values()),
                return_exceptions=True,
            )

        self.workers.clear()
        self._running = False
        logger.info("processor stopped")

    async def start_camera(self, camera_id: int) -> CameraWorker | None:
        """Start (or return the running) worker for one active camera.

        Returns None when the camera does not exist, is inactive, or has no
        stream_url -- the caller translates that into an HTTP error. Idempotent:
        starting an already-running camera just returns its worker.
        """
        async with self._control_lock:
            worker = self.workers.get(camera_id)
            if worker is not None:
                return worker

            config = await asyncio.to_thread(self.load_camera_config, camera_id)
            if config is None:
                return None

            worker = CameraWorker(
                config,
                db_factory=self._db_factory,
                loop_video=self.loop_video,
                window_seconds=self.window_seconds,
            )
            self.workers[camera_id] = worker
            await worker.start()
            self._running = True
            logger.info("camera worker started via control endpoint", camera=config.name)
            return worker

    async def stop_camera(
        self, camera_id: int, timeout: float | None = None
    ) -> CameraWorker | None:
        """Stop one camera's worker without disturbing the others.

        Returns the stopped worker, or None when the camera had no worker. The
        stop itself happens outside the control lock so a slow unwind does not
        block start/stop calls for other cameras.
        """
        async with self._control_lock:
            worker = self.workers.pop(camera_id, None)

        if worker is None:
            return None
        await worker.stop(timeout=timeout or settings.realtime_shutdown_timeout_seconds)
        logger.info("camera worker stopped via control endpoint", camera=worker.config.name)
        return worker

    async def wait(self) -> None:
        """Block until every worker has finished. Used by tests."""
        tasks = [w._task for w in self.workers.values() if w._task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "camera_count": len(self.workers),
            "cameras": [w.status.snapshot() for w in self.workers.values()],
        }


# --------------------------------------------------------------------------------------
# Process-wide instance, wired to the FastAPI lifespan
# --------------------------------------------------------------------------------------

_processor: RealtimeProcessor | None = None


def get_processor() -> RealtimeProcessor:
    global _processor
    if _processor is None:
        _processor = RealtimeProcessor()
    return _processor


async def start_processor() -> RealtimeProcessor:
    processor = get_processor()
    await processor.start()
    return processor


async def stop_processor() -> None:
    global _processor
    if _processor is not None:
        await _processor.stop()
        _processor = None


__all__ = [
    "CameraStatus",
    "CameraWorker",
    "RealtimeProcessor",
    "get_processor",
    "start_processor",
    "stop_processor",
]
