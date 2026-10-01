"""Feature extraction: tracked frames in, model-ready feature vectors out.

This is the join between the vision pipeline and the risk model. It consumes the
:class:`~app.vision.tracking.TrackedFrame` stream, aggregates it into fixed time windows
(5 seconds by default), and emits one :class:`FeatureVector` per window, optionally
persisting each to ``crowd_observations``.

    tracked frames  ->  [ 5s window ]  ->  FeatureVector  ->  crowd_observations
                                                          ->  risk model (later)

Usage::

    from app.vision.features import FeatureExtractor
    from app.vision.tracking import track_people

    extractor = FeatureExtractor(camera_id=2, persist=True)
    for features in extractor.extract(track_people("data/samples/crowd.mp4")):
        print(features.density, features.historical_deviation)

Aggregation is **per track, not per frame**. A person visible in 100 frames would
otherwise count 100 times toward the averages and drown out someone who walked through in
10. Each track is reduced to one value for the window first, then those are combined, so
every person counts once regardless of how long they were on screen.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from app.core.config import settings
from app.vision.tracking import TrackedFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Aggregated crowd features for one camera over one time window."""

    camera_id: int | None
    window_start: datetime
    window_end: datetime
    window_seconds: float

    #: Mean number of people visible per frame across the window. This -- not the count
    #: of distinct track IDs -- is the instantaneous crowd size: over 5 seconds people
    #: walk through, so distinct IDs measures throughput, not occupancy.
    person_count: float

    #: Distinct track IDs seen in the window. Kept alongside person_count because the
    #: gap between the two is itself meaningful: equal values mean a static crowd,
    #: unique_track_count >> person_count means people are flowing through.
    unique_track_count: int

    #: people per square metre
    density: float

    #: Change in density since the previous window, people/m^2 per second.
    density_rate_of_change: float | None

    #: Raw difference in density vs the previous window (not divided by time).
    density_delta: float | None

    #: Mean speed across tracks, metres/second. None when the camera is uncalibrated.
    mean_flow_speed_m_s: float | None

    #: Mean speed across tracks in pixels/second. Always available.
    mean_flow_speed_px_s: float | None

    #: Circular variance of movement headings, 0-1. 0 = everyone moving the same way,
    #: 1 = headings uniformly scattered. None when nobody is moving.
    flow_direction_variance: float | None

    #: Shannon entropy (bits) of the dense optical flow direction histogram. None when
    #: optical flow could not be computed (no frames, or disabled).
    optical_flow_entropy: float | None

    #: Fraction of tracks in this window that were near-stationary, 0-1.
    stop_ratio: float

    #: Fraction of tracks present in *both* this and the previous window that were
    #: moving before and are stopped now. None when no tracks carried over. This is the
    #: transition signal -- a crowd seizing up -- as opposed to stop_ratio's level.
    newly_stopped_ratio: float | None

    #: Standard deviations from this camera's baseline for the window's hour and
    #: weekday. Falls back to 0.0 -- "no evidence of deviation" -- when no baseline
    #: exists for the slot yet, which is the normal state until the baseline job has
    #: run over real history.
    historical_deviation: float

    #: Frames that went into this window.
    frame_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def model_features(self) -> dict[str, float | None]:
        """Just the seven model inputs, in a stable order."""
        return {
            "density": self.density,
            "density_rate_of_change": self.density_rate_of_change,
            # m/s or None -- the model is trained on m/s, so px/s would mix units.
            "mean_flow_speed": self.mean_flow_speed_m_s,
            "flow_direction_variance": self.flow_direction_variance,
            "optical_flow_entropy": self.optical_flow_entropy,
            "stop_ratio": self.stop_ratio,
            "historical_deviation": self.historical_deviation,
        }

    def __repr__(self) -> str:
        return (
            f"<FeatureVector camera={self.camera_id} "
            f"{self.window_start:%H:%M:%S}+{self.window_seconds:g}s "
            f"density={self.density:.3f} stop={self.stop_ratio:.2f}>"
        )


@dataclass(slots=True)
class _TrackWindowState:
    """Per-track accumulation within one window."""

    speeds_px: list[float] = field(default_factory=list)
    speeds_m: list[float] = field(default_factory=list)
    # Sum of unit heading vectors, for a per-track mean direction.
    heading_x: float = 0.0
    heading_y: float = 0.0
    heading_samples: int = 0

    def mean_speed_px(self) -> float:
        return float(np.mean(self.speeds_px)) if self.speeds_px else 0.0

    def mean_speed_m(self) -> float | None:
        return float(np.mean(self.speeds_m)) if self.speeds_m else None

    def mean_heading(self) -> tuple[float, float] | None:
        """Mean unit heading vector for this track, or None if never moving."""
        if self.heading_samples == 0:
            return None
        magnitude = math.hypot(self.heading_x, self.heading_y)
        if magnitude < 1e-9:
            return None
        return self.heading_x / magnitude, self.heading_y / magnitude


def circular_variance(headings: Iterable[tuple[float, float]]) -> float | None:
    """Circular variance of a set of unit direction vectors, in ``[0, 1]``.

    ``1 - R`` where ``R`` is the length of the mean resultant vector. 0 means every
    heading is identical; 1 means they cancel out completely. Returns None for an empty
    input, where the quantity is undefined rather than zero.

    Direction vectors are used rather than angles because averaging angles directly is
    wrong at the wrap-around: the mean of 359 degrees and 1 degree is 0, not 180.
    """
    vectors = list(headings)
    if not vectors:
        return None

    mean_x = float(np.mean([v[0] for v in vectors]))
    mean_y = float(np.mean([v[1] for v in vectors]))
    resultant = math.hypot(mean_x, mean_y)
    # Clamp: floating point can nudge a single unit vector marginally above 1.
    return float(max(0.0, min(1.0, 1.0 - resultant)))


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a histogram, in bits.

    Zero-weight bins are dropped rather than special-cased, since ``0 * log2(0)`` is
    defined as 0 in this context but evaluates to NaN.
    """
    total = float(counts.sum())
    if total <= 0:
        return 0.0

    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def optical_flow_direction_entropy(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    bins: int = 16,
    min_magnitude: float = 0.5,
) -> float:
    """Entropy (bits) of the direction histogram of dense Farneback optical flow.

    A crowd moving as one body concentrates flow into a few bins and scores low. Milling,
    turbulent, or counter-flowing movement spreads it out and scores high -- which is the
    regime that precedes crowd-crush incidents, and is why this is worth its cost.

    Bins are weighted by flow magnitude, so a fast-moving region counts for more than a
    slow one, and near-zero vectors are excluded as noise rather than contributing a
    meaningless direction.
    """
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    moving = magnitude > min_magnitude

    if not np.any(moving):
        # A completely still scene has no directional uncertainty.
        return 0.0

    histogram, _ = np.histogram(
        angle[moving], bins=bins, range=(0.0, 2.0 * math.pi), weights=magnitude[moving]
    )
    return shannon_entropy(histogram)


class FeatureExtractor:
    """Aggregates a :class:`TrackedFrame` stream into per-window feature vectors.

    Args:
        camera_id: Camera these frames belong to. Required for persistence and for the
            historical-baseline lookup; may be None for offline experiments.
        area_sq_meters: Ground area covered, the denominator for density. Loaded from
            the camera document when not given.
        pixels_per_meter: Calibration for speed. Loaded from the camera document when
            not given. Without it, ``mean_flow_speed_m_s`` stays None.
        window_seconds: Aggregation window length.
        persist: Write each vector to ``crowd_observations``.
        db_factory: Overridable for tests; defaults to the app's Mongo database handle.
        compute_optical_flow: Farneback is by far the most expensive feature here;
            disable it when frames are unavailable or the cost is not worth it.
    """

    def __init__(
        self,
        camera_id: int | None = None,
        area_sq_meters: float | None = None,
        pixels_per_meter: float | None = None,
        window_seconds: float | None = None,
        persist: bool = False,
        db_factory=None,
        compute_optical_flow: bool | None = None,
        stationary_speed: float | None = None,
        moving_speed: float | None = None,
        no_baseline_deviation: float | None = None,
    ) -> None:
        self.camera_id = camera_id
        # `or` would treat an explicit 0 as unset and silently substitute the
        # default, so the validation below could never fire.
        self.window_seconds = (
            window_seconds if window_seconds is not None else settings.features_window_seconds
        )
        self.persist = persist
        self._db_factory = db_factory
        self.compute_optical_flow = (
            compute_optical_flow
            if compute_optical_flow is not None
            else settings.features_optical_flow_enabled
        )

        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds}")

        self.area_sq_meters = area_sq_meters
        self.pixels_per_meter = (
            float(pixels_per_meter) if pixels_per_meter and pixels_per_meter > 0 else None
        )

        # Fill anything missing from the camera row.
        if camera_id is not None and (area_sq_meters is None or pixels_per_meter is None):
            self._load_camera_settings(camera_id)

        if self.area_sq_meters is None or self.area_sq_meters <= 0:
            raise ValueError(
                "area_sq_meters is required and must be > 0 (pass it directly or give a "
                "camera_id whose row has it set) -- density is meaningless without it"
            )

        # Thresholds switch units with calibration: comparing a px/s speed against an
        # m/s threshold would silently classify everyone as stopped, or nobody.
        self.calibrated = self.pixels_per_meter is not None
        if stationary_speed is not None:
            self.stationary_speed = stationary_speed
        else:
            self.stationary_speed = (
                settings.features_stationary_speed_m_s
                if self.calibrated
                else settings.features_stationary_speed_px_s
            )
        if moving_speed is not None:
            self.moving_speed = moving_speed
        else:
            self.moving_speed = (
                settings.features_moving_speed_m_s
                if self.calibrated
                else settings.features_moving_speed_px_s
            )

        if not self.calibrated:
            logger.warning(
                "Camera %s has no pixels_per_meter: speeds will be in px/s and "
                "crowd_observations.mean_flow_speed will be left NULL, since mixing "
                "px/s and m/s in one column across cameras would corrupt the model.",
                camera_id,
            )

        # Cold-start fallback for historical_deviation.
        self.no_baseline_deviation = (
            no_baseline_deviation
            if no_baseline_deviation is not None
            else settings.features_historical_deviation_default
        )
        self._warned_no_baseline = False

        # Cross-window state.
        self._previous_density: float | None = None
        self._previous_window_end: datetime | None = None
        self._previous_stopped: dict[int, bool] = {}
        # (hour, weekday) -> (baseline_or_None, monotonic time it was read). Entries
        # expire so a long-running worker picks up baselines written after it started.
        self._baseline_cache: dict[
            tuple[int, int], tuple[tuple[float, float] | None, float]
        ] = {}

    # -- database ----------------------------------------------------------------

    def _db(self):
        if self._db_factory is not None:
            return self._db_factory()
        from app.core.database import db

        return db

    def _load_camera_settings(self, camera_id: int) -> None:
        from app.core.database import CAMERAS

        doc = self._db()[CAMERAS].find_one({"id": camera_id})

        if doc is None:
            raise ValueError(f"No camera with id {camera_id}")

        if self.area_sq_meters is None:
            self.area_sq_meters = doc["area_sq_meters"]
        if self.pixels_per_meter is None and doc.get("pixels_per_meter"):
            self.pixels_per_meter = float(doc["pixels_per_meter"])

    def _historical_deviation(self, density: float, moment: datetime) -> float:
        """Z-score of ``density`` against this camera's baseline for that time slot.

        Falls back to :attr:`no_baseline_deviation` (0.0 by default) whenever a z-score
        cannot be computed -- no camera, no baseline for this slot yet, or a baseline
        with zero variance. This is the cold-start path: a fresh deployment has no
        history at all, and every window would otherwise be missing this feature until
        the baseline job has run over real data.

        Zero is the right neutral here rather than NaN. The model was trained with this
        feature always present, so it never learned a default branch for a missing
        value; feeding NaN would route the prediction down whichever side XGBoost
        happened to pick, while zero says exactly what is true -- no evidence that this
        window deviates from normal.

        The slot key is (UTC hour, UTC weekday) and must stay in step with how the
        baseline job groups observations; see app/services/baseline_job.py.
        """
        if self.camera_id is None:
            return self.no_baseline_deviation

        key = (moment.hour, moment.weekday())
        cached = self._baseline_cache.get(key)
        # >= rather than >: a TTL of 0 must mean "never trust the cache". With >, a
        # lookup landing inside the same clock tick as the write (the monotonic clock
        # has ~15 ms granularity on Windows) is treated as fresh and a zero TTL does
        # nothing.
        expired = (
            cached is None
            or (time.monotonic() - cached[1]) >= settings.features_baseline_cache_seconds
        )
        if expired:
            baseline = self._load_baseline(*key)
            self._baseline_cache[key] = (baseline, time.monotonic())
            # A slot that was empty and now is not means the baseline job has run; say
            # so once so the transition is visible in the log.
            if baseline is not None and cached is not None and cached[0] is None:
                logger.info(
                    "Baseline now available for camera %s at hour=%d weekday=%d",
                    self.camera_id,
                    key[0],
                    key[1],
                )
                self._warned_no_baseline = False
        else:
            baseline = cached[0]
        if baseline is None:
            if not self._warned_no_baseline:
                # Once per extractor: a camera with no baselines at all would otherwise
                # log this for every window forever.
                logger.info(
                    "No baseline for camera %s at hour=%d weekday=%d; "
                    "historical_deviation defaults to %.1f until "
                    "scripts/compute_baselines.py has run over real history.",
                    self.camera_id,
                    key[0],
                    key[1],
                    self.no_baseline_deviation,
                )
                self._warned_no_baseline = True
            return self.no_baseline_deviation

        avg, stddev = baseline
        if stddev <= 0:
            # A zero-variance baseline would make every deviation infinite.
            return self.no_baseline_deviation

        return (density - avg) / stddev

    def _load_baseline(self, hour: int, weekday: int) -> tuple[float, float] | None:
        from app.core.database import BASELINES

        try:
            doc = self._db()[BASELINES].find_one(
                {
                    "camera_id": self.camera_id,
                    "hour_of_day": hour,
                    "day_of_week": weekday,
                }
            )
        except Exception:
            logger.exception(
                "Could not load baseline for camera %s at hour=%d weekday=%d",
                self.camera_id,
                hour,
                weekday,
            )
            return None

        if doc is None:
            return None
        return float(doc["avg_density"]), float(doc["stddev_density"])

    def _persist(self, features: FeatureVector) -> None:
        if self.camera_id is None:
            raise ValueError("persist=True requires a camera_id")

        from app.core.database import OBSERVATIONS, insert_document
        from app.models import CrowdObservation

        insert_document(
            self._db(),
            OBSERVATIONS,
            CrowdObservation(
                camera_id=self.camera_id,
                timestamp=features.window_end,
                # The collection stores a whole-person count; the window mean is
                # fractional, so round for storage and keep the exact value on the
                # feature vector.
                person_count=int(round(features.person_count)),
                density=features.density,
                # null rather than px/s: see the uncalibrated warning above.
                mean_flow_speed=features.mean_flow_speed_m_s,
                flow_direction_variance=features.flow_direction_variance,
                optical_flow_entropy=features.optical_flow_entropy,
                stop_ratio=features.stop_ratio,
                density_rate_of_change=features.density_rate_of_change,
                historical_deviation=features.historical_deviation,
            ).to_doc(),
        )

    # -- windowing ---------------------------------------------------------------

    def extract(self, frames: Iterable[TrackedFrame]) -> Iterator[FeatureVector]:
        """Consume tracked frames and yield one feature vector per completed window.

        Windows are cut on ``source_time``, so they are stable regardless of how fast
        frames arrive. A trailing partial window is emitted when the stream ends, since
        for a short clip it may be the only data there is.
        """
        buffer: list[TrackedFrame] = []
        current_window: int | None = None

        for tracked in frames:
            window_index = int(tracked.source_time // self.window_seconds)

            if current_window is None:
                current_window = window_index

            if window_index != current_window:
                if buffer:
                    yield self._build(buffer, current_window)
                buffer = []
                current_window = window_index

            buffer.append(tracked)

        if buffer and current_window is not None:
            yield self._build(buffer, current_window)

    def _build(self, frames: list[TrackedFrame], window_index: int) -> FeatureVector:
        """Compute one feature vector from the frames of a single window."""
        window_start = frames[0].timestamp
        window_end = frames[-1].timestamp
        elapsed = max(frames[-1].source_time - frames[0].source_time, 0.0)

        # -- per-track aggregation, so each person counts once ---------------------
        tracks: dict[int, _TrackWindowState] = {}
        for tracked in frames:
            for person in tracked.people:
                state = tracks.setdefault(person.track_id, _TrackWindowState())
                state.speeds_px.append(person.speed_px_per_s)
                if person.speed_m_per_s is not None:
                    state.speeds_m.append(person.speed_m_per_s)

                speed = (
                    person.speed_m_per_s
                    if self.calibrated and person.speed_m_per_s is not None
                    else person.speed_px_per_s
                )
                if speed >= self.moving_speed:
                    vx, vy = person.velocity_px_per_s
                    norm = math.hypot(vx, vy)
                    if norm > 1e-9:
                        # Flip y so headings are in maths orientation, matching
                        # TrackedPerson.direction_radians.
                        state.heading_x += vx / norm
                        state.heading_y += -vy / norm
                        state.heading_samples += 1

        # -- density ---------------------------------------------------------------
        person_count = float(np.mean([f.person_count for f in frames]))
        density = person_count / self.area_sq_meters

        density_delta: float | None = None
        density_rate: float | None = None
        if self._previous_density is not None:
            density_delta = density - self._previous_density
            # Divide by the gap between window starts rather than the nominal window
            # length, so a dropped or short window does not inflate the rate.
            gap = self.window_seconds
            if self._previous_window_end is not None:
                measured = (window_end - self._previous_window_end).total_seconds()
                if measured > 0:
                    gap = measured
            density_rate = density_delta / gap

        # -- speed -----------------------------------------------------------------
        track_speeds_px = [s.mean_speed_px() for s in tracks.values()]
        track_speeds_m = [
            s.mean_speed_m() for s in tracks.values() if s.mean_speed_m() is not None
        ]
        mean_speed_px = float(np.mean(track_speeds_px)) if track_speeds_px else None
        mean_speed_m = float(np.mean(track_speeds_m)) if track_speeds_m else None

        # -- direction variance ----------------------------------------------------
        headings = [h for h in (s.mean_heading() for s in tracks.values()) if h is not None]
        direction_variance = circular_variance(headings)

        # -- stop ratio ------------------------------------------------------------
        stopped_now: dict[int, bool] = {}
        for track_id, state in tracks.items():
            speed = (
                state.mean_speed_m()
                if self.calibrated and state.mean_speed_m() is not None
                else state.mean_speed_px()
            )
            stopped_now[track_id] = speed < self.stationary_speed

        stop_ratio = (
            sum(1 for is_stopped in stopped_now.values() if is_stopped) / len(stopped_now)
            if stopped_now
            else 0.0
        )

        carried_over = set(stopped_now) & set(self._previous_stopped)
        newly_stopped_ratio: float | None = None
        if carried_over:
            newly_stopped = sum(
                1
                for track_id in carried_over
                if stopped_now[track_id] and not self._previous_stopped[track_id]
            )
            newly_stopped_ratio = newly_stopped / len(carried_over)

        # -- optical flow entropy --------------------------------------------------
        entropy = self._window_entropy(frames) if self.compute_optical_flow else None

        # -- historical deviation --------------------------------------------------
        # window_end is the timestamp the observation is stored under and the slot
        # the baseline job groups by, so a window straddling an hour boundary is
        # compared against the same slot it will be counted in.
        deviation = self._historical_deviation(density, window_end)

        features = FeatureVector(
            camera_id=self.camera_id,
            window_start=window_start,
            window_end=window_end,
            window_seconds=self.window_seconds,
            person_count=person_count,
            unique_track_count=len(tracks),
            density=density,
            density_rate_of_change=density_rate,
            density_delta=density_delta,
            mean_flow_speed_m_s=mean_speed_m,
            mean_flow_speed_px_s=mean_speed_px,
            flow_direction_variance=direction_variance,
            optical_flow_entropy=entropy,
            stop_ratio=stop_ratio,
            newly_stopped_ratio=newly_stopped_ratio,
            historical_deviation=deviation,
            frame_count=len(frames),
        )

        # Roll state forward before returning.
        self._previous_density = density
        self._previous_window_end = window_end
        self._previous_stopped = stopped_now

        if self.persist:
            self._persist(features)

        logger.debug("Window %d (%.1fs elapsed): %r", window_index, elapsed, features)
        return features

    def _window_entropy(self, frames: list[TrackedFrame]) -> float | None:
        """Mean flow-direction entropy over a few frame pairs in the window."""
        usable = [f.frame for f in frames if f.frame is not None]
        if len(usable) < 2:
            return None

        import cv2

        from app.vision.detection import _resize_to_max_dimension

        max_pairs = max(1, settings.features_optical_flow_max_pairs)
        # Evenly spaced pairs rather than the first N, so the sample spans the window.
        step = max(1, (len(usable) - 1) // max_pairs)
        indices = list(range(0, len(usable) - 1, step))[:max_pairs]

        entropies: list[float] = []
        for index in indices:
            first, _ = _resize_to_max_dimension(
                usable[index], settings.features_optical_flow_max_dimension
            )
            second, _ = _resize_to_max_dimension(
                usable[index + 1], settings.features_optical_flow_max_dimension
            )
            entropies.append(
                optical_flow_direction_entropy(
                    cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(second, cv2.COLOR_BGR2GRAY),
                    bins=settings.features_optical_flow_bins,
                    min_magnitude=settings.features_optical_flow_min_magnitude,
                )
            )

        return float(np.mean(entropies)) if entropies else None


__all__ = [
    "FeatureExtractor",
    "FeatureVector",
    "circular_variance",
    "optical_flow_direction_entropy",
    "shannon_entropy",
]
