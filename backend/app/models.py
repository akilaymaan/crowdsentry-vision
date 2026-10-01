"""Document models for the CrowdSentry spatio-temporal risk schema, backed by MongoDB.

Data flow across these collections:

    cameras --+--> crowd_observations   raw per-window measurements from the vision pipeline
              +--> historical_baselines what "normal" looks like per (hour, weekday)
              +--> risk_scores          model output, one doc per scored window
              +--> alerts               operator-facing events raised from a score

These are plain dataclasses -- the Mongo collections hold plain dicts, so each model
knows how to serialise itself (``to_doc``) and rebuild itself (``from_doc``). The pydantic
schemas keep ``from_attributes`` working unchanged, and the public ``id`` stays an
integer allocated from the ``counters`` collection (``database.next_id``).

There is no schema migration layer: indexes are created idempotently by
``database.ensure_indexes`` at startup, and optional fields simply omit themselves or
read back as ``None``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.database import point, utcnow


class RiskLevel(str, enum.Enum):
    """Discrete risk bands derived from the continuous 0-100 risk score."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Camera:
    """A fixed monitoring camera covering a known ground area.

    ``area_sq_meters`` is the real-world footprint of the camera's field of view and is
    the denominator for density: ``density = person_count / area_sq_meters``.
    """

    id: int
    name: str
    location_name: str
    latitude: float
    longitude: float
    area_sq_meters: float
    # Ground-plane scale: image pixels per real metre, used to convert tracked pixel
    # velocity into m/s. None means uncalibrated -- speeds stay in pixels/second.
    pixels_per_meter: float | None
    # What the processor opens: an RTSP/HTTP URL, a webcam index ("0"), or a file path.
    # None means registered-but-no-feed, and the processor skips it.
    stream_url: str | None
    # Whether the processor runs a worker for this camera.
    is_active: bool = True
    created_at: datetime = field(default_factory=utcnow)

    @property
    def stream_configured(self) -> bool:
        """Whether a feed is wired up -- the API-facing view of stream_url.

        The URL itself is deliberately never serialised: RTSP specs routinely embed
        credentials (rtsp://user:pass@host/...), and clients only need to know that a
        feed exists.
        """
        return bool(self.stream_url)

    def to_doc(self) -> dict:
        # GeoJSON alongside the flat fields: the flat pair is the read path, the point
        # carries the 2dsphere index for "cameras within N metres" style queries. Both
        # are written here, so the two representations can never drift apart.
        return {
            "id": self.id,
            "name": self.name,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location": point(self.longitude, self.latitude),
            "area_sq_meters": self.area_sq_meters,
            "pixels_per_meter": self.pixels_per_meter,
            "stream_url": self.stream_url,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> Camera:
        return cls(
            id=doc["id"],
            name=doc["name"],
            location_name=doc["location_name"],
            latitude=doc["latitude"],
            longitude=doc["longitude"],
            area_sq_meters=doc["area_sq_meters"],
            pixels_per_meter=doc.get("pixels_per_meter"),
            stream_url=doc.get("stream_url"),
            is_active=doc.get("is_active", True),
            created_at=doc.get("created_at") or utcnow(),
        )


@dataclass
class CrowdObservation:
    """One spatio-temporal measurement window for a camera.

    Written by the vision pipeline; these feature fields are the inputs the risk model
    consumes alongside the camera's baseline for the same hour and weekday.
    """

    camera_id: int
    timestamp: datetime = field(default_factory=utcnow)
    person_count: int = 0
    # People per square metre. Around 4+ is widely treated as the onset of crush risk.
    density: float = 0.0

    # Mean crowd movement speed, metres/second. None when the camera is uncalibrated.
    mean_flow_speed: float | None = None
    # Circular variance of movement headings: low = orderly flow, high = disordered.
    flow_direction_variance: float | None = None
    # Shannon entropy of the optical-flow field; a turbulence proxy.
    optical_flow_entropy: float | None = None
    # Fraction of tracked people who are stationary (0-1); rises as a crowd jams.
    stop_ratio: float | None = None
    # Signed change in density since the previous window, people/m^2 per second.
    density_rate_of_change: float | None = None
    # Standard deviations from this camera's baseline for the current slot. Stored
    # rather than recomputed on read: the baseline drifts as history accumulates, and
    # we want the value that was actually true when the window was scored.
    historical_deviation: float | None = None

    id: int | None = None

    def to_doc(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "person_count": self.person_count,
            "density": self.density,
            "mean_flow_speed": self.mean_flow_speed,
            "flow_direction_variance": self.flow_direction_variance,
            "optical_flow_entropy": self.optical_flow_entropy,
            "stop_ratio": self.stop_ratio,
            "density_rate_of_change": self.density_rate_of_change,
            "historical_deviation": self.historical_deviation,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> CrowdObservation:
        return cls(
            id=doc["id"],
            camera_id=doc["camera_id"],
            timestamp=doc["timestamp"],
            person_count=doc["person_count"],
            density=doc["density"],
            mean_flow_speed=doc.get("mean_flow_speed"),
            flow_direction_variance=doc.get("flow_direction_variance"),
            optical_flow_entropy=doc.get("optical_flow_entropy"),
            stop_ratio=doc.get("stop_ratio"),
            density_rate_of_change=doc.get("density_rate_of_change"),
            historical_deviation=doc.get("historical_deviation"),
        )


@dataclass
class HistoricalBaseline:
    """Rolling "normal" density for a camera at a given hour and weekday.

    Lets the system express risk as a deviation from what is typical for *this place at
    this time*: a packed platform at 08:00 Monday is routine, the same density at 03:00
    Sunday is not::

        z = (observed_density - avg_density) / stddev_density

    Keyed by (camera_id, hour_of_day, day_of_week) with exactly one doc per slot, so it
    is upserted rather than appended.
    """

    camera_id: int
    # 0-23 in UTC. FeatureExtractor reads `window_end.hour` off a UTC-aware datetime --
    # the same timestamp the observation doc is stored under -- so a baseline job that
    # grouped by local hour would pair every observation with the wrong slot, silently.
    hour_of_day: int
    # 0 = Monday ... 6 = Sunday, matching Python's datetime.weekday().
    day_of_week: int

    avg_density: float
    stddev_density: float = 0.0
    # Slots built from a handful of samples are statistically weak; callers can require
    # a minimum sample_count before trusting the z-score.
    sample_count: int = 0
    updated_at: datetime = field(default_factory=utcnow)

    def to_doc(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "avg_density": self.avg_density,
            "stddev_density": self.stddev_density,
            "sample_count": self.sample_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> HistoricalBaseline:
        return cls(
            camera_id=doc["camera_id"],
            hour_of_day=doc["hour_of_day"],
            day_of_week=doc["day_of_week"],
            avg_density=doc["avg_density"],
            stddev_density=doc.get("stddev_density", 0.0),
            sample_count=doc.get("sample_count", 0),
            updated_at=doc.get("updated_at") or utcnow(),
        )


@dataclass
class RiskScore:
    """Model output for one scoring window.

    ``contributing_features`` stores the per-feature attribution behind the score --
    ``{"top_features": [...], "probabilities": {...}, "feature_values": {...}}`` -- so
    the dashboard can explain *why* a camera is flagged. It stays a subdocument, which
    Mongo queries natively.
    """

    camera_id: int
    risk_score: float
    risk_level: RiskLevel | str
    timestamp: datetime = field(default_factory=utcnow)
    contributing_features: dict[str, Any] | None = None
    id: int | None = None

    def to_doc(self) -> dict:
        level = self.risk_level
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "risk_level": level.value if isinstance(level, RiskLevel) else level,
            "contributing_features": self.contributing_features,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> RiskScore:
        return cls(
            id=doc["id"],
            camera_id=doc["camera_id"],
            timestamp=doc["timestamp"],
            risk_score=doc["risk_score"],
            risk_level=RiskLevel(doc["risk_level"]),
            contributing_features=doc.get("contributing_features"),
        )


@dataclass
class Alert:
    """An operator-facing event raised when a risk score breaches its threshold."""

    camera_id: int
    timestamp: datetime
    risk_score: float
    message: str
    # Absent on docs written before the field existed; new alerts always set it.
    risk_level: RiskLevel | str | None = None
    acknowledged: bool = False
    id: int | None = None

    def to_doc(self) -> dict:
        level = self.risk_level
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "risk_score": self.risk_score,
            "risk_level": level.value if isinstance(level, RiskLevel) else level,
            "message": self.message,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> Alert:
        level = doc.get("risk_level")
        return cls(
            id=doc["id"],
            camera_id=doc["camera_id"],
            timestamp=doc["timestamp"],
            risk_score=doc["risk_score"],
            risk_level=RiskLevel(level) if level is not None else None,
            message=doc["message"],
            acknowledged=doc.get("acknowledged", False),
        )


__all__ = [
    "Alert",
    "Camera",
    "CrowdObservation",
    "HistoricalBaseline",
    "RiskLevel",
    "RiskScore",
]
