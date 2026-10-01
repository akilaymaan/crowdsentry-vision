"""Multi-object person tracking on top of the YOLOv8 detector.

Uses Ultralytics' built-in ByteTrack. That is the cleanest integration available here:
``model.track(persist=True)`` runs detection and association in one call, so there is no
second copy of the boxes to keep in sync, and ByteTrack's two-stage association (it
matches high-confidence boxes first, then rescues low-confidence ones against the
leftover tracks) is exactly right for crowds, where people are constantly half-occluded
and drop to low confidence for a few frames without actually disappearing.

Usage::

    from app.vision.tracking import track_people

    for tracked in track_people("data/samples/crowd.mp4"):
        for person in tracked.people:
            print(person.track_id, person.centroid, person.speed_m_per_s)

One :class:`PersonTracker` follows exactly one video stream. ByteTrack keeps its state
inside the Ultralytics model object, so two concurrent streams need two trackers -- share
one and their identities will contaminate each other.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.vision.detection import PERSON_CLASS_ID, BoundingBox, _resize_to_max_dimension
from app.vision.source import VideoSource, open_source

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    """One person in one frame, with an identity that persists across frames."""

    track_id: int
    bbox: BoundingBox

    #: Box centre, in frame pixel coordinates.
    centroid: tuple[float, float]

    #: Bottom-centre of the box -- where the person meets the ground. Velocity is
    #: measured from this point, not the centroid; see PersonTracker for why.
    foot_point: tuple[float, float]

    #: Velocity of the foot point as (vx, vy) in pixels/second. Screen axes, so vy is
    #: positive downwards.
    velocity_px_per_s: tuple[float, float]

    #: Magnitude of the above, pixels/second.
    speed_px_per_s: float

    #: Real-world speed in metres/second, or None when the camera has no
    #: pixels-per-metre calibration.
    speed_m_per_s: float | None

    #: How many frames this track has been alive, including this one.
    age: int

    @property
    def confidence(self) -> float:
        return self.bbox.confidence

    @property
    def direction_radians(self) -> float | None:
        """Heading of travel, or None when effectively stationary.

        Measured with the y-axis flipped, so 0 is right and pi/2 is *up* the screen --
        conventional maths orientation rather than screen orientation.
        """
        vx, vy = self.velocity_px_per_s
        if self.speed_px_per_s < 1e-6:
            return None
        return float(np.arctan2(-vy, vx))


@dataclass(slots=True)
class TrackedFrame:
    """Every tracked person in a single frame, plus when that frame happened."""

    frame_index: int

    #: Wall-clock timestamp for this frame. For a file this is the run's start time plus
    #: the frame's media offset, so it stays consistent regardless of how fast the file
    #: is processed.
    timestamp: datetime

    #: Seconds from the start of the stream. This -- not wall clock -- is what velocity
    #: is computed against.
    source_time: float

    people: list[TrackedPerson]

    #: The BGR frame these detections came from, for rendering and dense optical flow.
    #: Optional so a TrackedFrame can be built synthetically in tests without a video.
    frame: np.ndarray | None = field(default=None, repr=False)

    @property
    def person_count(self) -> int:
        return len(self.people)

    @property
    def width(self) -> int | None:
        return None if self.frame is None else self.frame.shape[1]

    @property
    def height(self) -> int | None:
        return None if self.frame is None else self.frame.shape[0]

    def mean_speed_m_per_s(self) -> float | None:
        """Mean real-world speed across tracks, or None without calibration."""
        speeds = [p.speed_m_per_s for p in self.people if p.speed_m_per_s is not None]
        return float(np.mean(speeds)) if speeds else None

    def __repr__(self) -> str:
        return (
            f"<TrackedFrame index={self.frame_index} t={self.source_time:.2f}s "
            f"people={self.person_count}>"
        )


@dataclass(slots=True)
class _TrackHistory:
    """Rolling position history for one track, used to estimate velocity."""

    # (source_time, x, y) samples of the foot point, oldest first.
    samples: deque[tuple[float, float, float]]
    age: int = 0
    last_seen: float = 0.0


class PersonTracker:
    """ByteTrack person tracker over a single video stream.

    Args:
        model_path: Ultralytics weights (defaults to the detection setting).
        confidence_threshold: Minimum detection score fed to the tracker.
        max_dimension: Inference resolution, as in :class:`PersonDetector`.
        iou_threshold: NMS IoU threshold.
        device: ``"cpu"``, ``"cuda"``, ... or None for auto.
        pixels_per_meter: Ground-plane scale for this camera. ``None`` or ``0`` leaves
            ``speed_m_per_s`` as None rather than reporting a fabricated number.
        velocity_window_seconds: Velocity is measured across this much history rather
            than between adjacent frames. Frame-to-frame differences are dominated by
            box jitter -- a box that wobbles two pixels at 25 fps implies a phantom
            50 px/s -- and a window of a few tenths of a second averages that out while
            still responding to genuine changes in pace.
        tracker_config: Ultralytics tracker YAML (``bytetrack.yaml`` or ``botsort.yaml``).
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        max_dimension: int | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
        pixels_per_meter: float | None = None,
        velocity_window_seconds: float | None = None,
        tracker_config: str | None = None,
    ) -> None:
        self.model_path = model_path or settings.detection_model
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.detection_confidence
        )
        self.max_dimension = (
            max_dimension if max_dimension is not None else settings.detection_max_dimension
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else settings.detection_iou
        )
        self.device = device if device is not None else settings.detection_device
        self.tracker_config = tracker_config or settings.tracking_tracker

        resolved_ppm = (
            pixels_per_meter
            if pixels_per_meter is not None
            else settings.tracking_pixels_per_meter
        )
        # Treat 0 and negative as "uncalibrated" rather than dividing by them.
        self.pixels_per_meter: float | None = (
            float(resolved_ppm) if resolved_ppm and resolved_ppm > 0 else None
        )

        self.velocity_window_seconds = (
            velocity_window_seconds
            if velocity_window_seconds is not None
            else settings.tracking_velocity_window_seconds
        )

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
        if self.velocity_window_seconds <= 0:
            raise ValueError(
                f"velocity_window_seconds must be > 0, got {self.velocity_window_seconds}"
            )

        self.imgsz = ((self.max_dimension + 31) // 32) * 32
        self._histories: dict[int, _TrackHistory] = {}

    @cached_property
    def model(self):
        """The Ultralytics model, loaded on first access.

        This object also holds ByteTrack's state, which is why a tracker instance is
        bound to one stream.
        """
        from ultralytics import YOLO

        logger.info("Loading YOLO weights for tracking: %s", self.model_path)
        model = YOLO(self.model_path)
        if self.device:
            model.to(self.device)
        return model

    def reset(self) -> None:
        """Forget all track state, so the next frame starts fresh identities."""
        self._histories.clear()
        # Ultralytics attaches its trackers to the predictor; dropping them makes the
        # next track() call rebuild from scratch.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()

    # -- velocity ----------------------------------------------------------------

    def _update_velocity(
        self, track_id: int, source_time: float, point: tuple[float, float]
    ) -> tuple[tuple[float, float], int]:
        """Record a position and return ``((vx, vy), age)`` in pixels/second."""
        history = self._histories.get(track_id)
        if history is None:
            history = _TrackHistory(samples=deque(maxlen=120))
            self._histories[track_id] = history

        history.samples.append((source_time, point[0], point[1]))
        history.age += 1
        history.last_seen = source_time

        # Drop samples that have fallen out of the window, but always keep at least two
        # so a track that reappears after a gap still produces an estimate.
        cutoff = source_time - self.velocity_window_seconds
        while len(history.samples) > 2 and history.samples[0][0] < cutoff:
            history.samples.popleft()

        if len(history.samples) < 2:
            return (0.0, 0.0), history.age

        t_old, x_old, y_old = history.samples[0]
        t_new, x_new, y_new = history.samples[-1]
        dt = t_new - t_old

        if dt <= 1e-6:
            return (0.0, 0.0), history.age

        return ((x_new - x_old) / dt, (y_new - y_old) / dt), history.age

    def _prune_histories(self, source_time: float, max_age_seconds: float = 5.0) -> None:
        """Drop histories for tracks not seen recently.

        Without this, a long run over a busy scene accumulates one entry per person who
        has ever appeared.
        """
        stale = [
            track_id
            for track_id, history in self._histories.items()
            if source_time - history.last_seen > max_age_seconds
        ]
        for track_id in stale:
            del self._histories[track_id]

    def _to_speed_m_per_s(self, speed_px_per_s: float) -> float | None:
        if self.pixels_per_meter is None:
            return None
        return speed_px_per_s / self.pixels_per_meter

    # -- tracking ----------------------------------------------------------------

    def track_frame(self, frame: np.ndarray, source_time: float) -> list[TrackedPerson]:
        """Track people in one frame.

        ``source_time`` is seconds since the start of the stream and drives the velocity
        estimate, so it must come from media time for files and a real clock for live
        feeds -- never from a frame counter alone unless the frame rate is exact.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")

        resized, scale = _resize_to_max_dimension(frame, self.max_dimension)
        inverse = 1.0 / scale

        results = self.model.track(
            resized,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID],
            device=self.device or None,
            tracker=self.tracker_config,
            # Carry tracker state across calls; without this every frame starts new IDs.
            persist=True,
            verbose=False,
        )

        self._prune_histories(source_time)

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            # boxes.id is None on frames where nothing was successfully associated.
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        people: list[TrackedPerson] = []
        for (x1, y1, x2, y2), confidence, track_id in zip(xyxy, confidences, track_ids):
            bbox = BoundingBox(
                x1=float(x1) * inverse,
                y1=float(y1) * inverse,
                x2=float(x2) * inverse,
                y2=float(y2) * inverse,
                confidence=float(confidence),
            )
            foot_point = bbox.foot_point
            velocity, age = self._update_velocity(int(track_id), source_time, foot_point)
            speed_px = float(np.hypot(*velocity))

            people.append(
                TrackedPerson(
                    track_id=int(track_id),
                    bbox=bbox,
                    centroid=bbox.center,
                    foot_point=foot_point,
                    velocity_px_per_s=velocity,
                    speed_px_per_s=speed_px,
                    speed_m_per_s=self._to_speed_m_per_s(speed_px),
                    age=age,
                )
            )

        people.sort(key=lambda person: person.track_id)
        return people

    def track(
        self,
        video_source: VideoSource | str | int | Path,
        max_frames: int | None = None,
        stride: int = 1,
    ) -> Iterator[TrackedFrame]:
        """Track people across a video source, yielding one :class:`TrackedFrame` each.

        Accepts an already-built :class:`VideoSource` or anything
        :func:`app.vision.source.open_source` understands.
        """
        source = (
            video_source
            if isinstance(video_source, VideoSource)
            else open_source(video_source)
        )

        self.reset()
        started_wall_clock = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()

        with source:
            fps = source.fps
            is_live = source.is_live

            for frame_index, frame in source.frames(max_frames=max_frames, stride=stride):
                if is_live:
                    # A live feed has no reliable media clock; use the real one.
                    source_time = time.monotonic() - started_monotonic
                else:
                    # Media time, so velocity does not change with how fast we happen to
                    # process the file. A frame counter alone would be wrong whenever
                    # stride > 1 or the file's fps is not what we assumed.
                    source_time = frame_index / fps if fps > 0 else frame_index

                people = self.track_frame(frame, source_time)

                yield TrackedFrame(
                    frame_index=frame_index,
                    timestamp=started_wall_clock + timedelta(seconds=source_time),
                    source_time=source_time,
                    people=people,
                    frame=frame,
                )


def track_people(
    video_source: VideoSource | str | int | Path,
    max_frames: int | None = None,
    stride: int = 1,
    **tracker_kwargs,
) -> Iterator[TrackedFrame]:
    """Track people in a video source, yielding :class:`TrackedFrame` objects.

    Convenience wrapper that builds a fresh :class:`PersonTracker` -- correct by default,
    since tracker state must not be shared between streams. Extra keyword arguments go to
    the tracker, e.g. ``track_people(path, pixels_per_meter=42.0)``.
    """
    tracker = PersonTracker(**tracker_kwargs)
    yield from tracker.track(video_source, max_frames=max_frames, stride=stride)


__all__ = ["PersonTracker", "TrackedFrame", "TrackedPerson", "track_people"]
