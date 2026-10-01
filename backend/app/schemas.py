"""Pydantic response schemas for the REST and WebSocket API.

These are the API's contract, kept deliberately separate from the ORM models: the
database schema should be free to change shape without silently changing what clients
receive. Everything here is read-only output except the handful of request bodies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import RiskLevel

ORM = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------------------
# risk / observations
# --------------------------------------------------------------------------------------


class RiskScoreOut(BaseModel):
    """One scored window."""

    model_config = ORM

    id: int
    camera_id: int
    timestamp: datetime
    risk_score: float = Field(description="0-100", ge=0, le=100)
    risk_level: RiskLevel
    contributing_features: dict[str, Any] | None = Field(
        default=None, description="Per-prediction feature attributions and probabilities."
    )


class LatestRisk(BaseModel):
    """The compact form embedded in camera listings."""

    model_config = ORM

    timestamp: datetime
    risk_score: float
    risk_level: RiskLevel


class LatestObservation(BaseModel):
    """Compact measurement embedded in camera listings.

    The camera list needs a person count per card; without this the UI would have to
    fetch every camera's detail separately.
    """

    model_config = ORM

    timestamp: datetime
    person_count: int
    density: float


class CrowdObservationOut(BaseModel):
    """One feature window as measured by the vision pipeline."""

    model_config = ORM

    id: int
    camera_id: int
    timestamp: datetime
    person_count: int
    density: float = Field(description="people per square metre")
    mean_flow_speed: float | None = Field(
        default=None, description="metres/second; null when the camera is uncalibrated"
    )
    flow_direction_variance: float | None = None
    optical_flow_entropy: float | None = None
    stop_ratio: float | None = None
    density_rate_of_change: float | None = None
    historical_deviation: float | None = Field(
        default=None, description="standard deviations from this camera's baseline"
    )


# --------------------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------------------


class CameraBase(BaseModel):
    model_config = ORM

    id: int
    name: str
    location_name: str
    latitude: float
    longitude: float
    area_sq_meters: float
    pixels_per_meter: float | None = Field(
        default=None,
        description="Ground-plane calibration. Null means speeds are unavailable in m/s.",
    )
    # The raw stream_url is never exposed: it can embed feed credentials. A boolean
    # carries everything a client needs -- "is there a feed configured".
    stream_configured: bool = Field(
        default=False, description="True when the camera has a stream_url configured."
    )
    is_active: bool
    created_at: datetime


class CameraSummary(CameraBase):
    """A camera plus its most recent risk score, for the camera list."""

    latest_risk: LatestRisk | None = Field(
        default=None, description="Null when this camera has never been scored."
    )
    latest_observation: LatestObservation | None = Field(
        default=None, description="Null when this camera has never reported."
    )


class CameraDetail(CameraBase):
    """A camera with its latest risk score and latest measurement."""

    latest_risk: LatestRisk | None = None
    latest_observation: CrowdObservationOut | None = None
    unacknowledged_alerts: int = 0


class CameraControlOut(BaseModel):
    """Result of a start/stop control call on one camera."""

    camera_id: int
    is_active: bool
    worker_state: str | None = Field(
        default=None,
        description="Worker state after the call (starting/running/stopped); "
        "null when the camera has no worker.",
    )


class CameraHistory(BaseModel):
    """Time-series for one camera, for charting.

    Observations and risk scores are returned as parallel series rather than joined:
    they are written by different stages and a window can produce one without the other,
    so joining them here would silently drop rows.
    """

    camera_id: int
    camera_name: str
    start: datetime = Field(description="inclusive lower bound actually applied")
    end: datetime = Field(description="inclusive upper bound actually applied")
    truncated: bool = Field(
        default=False,
        description="True when the row limit was hit and older rows were omitted.",
    )
    observations: list[CrowdObservationOut] = []
    risk_scores: list[RiskScoreOut] = []


# --------------------------------------------------------------------------------------
# alerts
# --------------------------------------------------------------------------------------


class AlertOut(BaseModel):
    model_config = ORM

    id: int
    camera_id: int
    timestamp: datetime
    risk_score: float
    # Null on rows written before the column existed; the UI falls back to inferring
    # the level from the message text for those.
    risk_level: RiskLevel | None = None
    message: str
    acknowledged: bool


class AlertWithCamera(AlertOut):
    """An alert carrying the camera's name, so the UI needs no second request."""

    camera_name: str
    camera_location: str


class AlertListOut(BaseModel):
    total: int = Field(description="Matching alerts, ignoring limit/offset.")
    count: int = Field(description="Alerts in this response.")
    alerts: list[AlertWithCamera] = []


# --------------------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------------------


class RiskLevelCounts(BaseModel):
    """Cameras currently at each risk level."""

    LOW: int = 0
    MODERATE: int = 0
    HIGH: int = 0
    CRITICAL: int = 0


class DashboardSummary(BaseModel):
    """Header figures for the dashboard."""

    generated_at: datetime
    total_cameras: int
    active_cameras: int
    reporting_cameras: int = Field(
        description="Cameras with a measurement inside the freshness window."
    )
    stale_cameras: int = Field(
        description="Active cameras whose most recent measurement is older than that."
    )
    total_people: int = Field(
        description="People summed across cameras reporting recently. Excludes stale "
        "cameras, so a camera that stopped hours ago does not inflate the live count."
    )
    mean_density: float | None = Field(
        default=None, description="Mean people/m^2 across reporting cameras."
    )
    peak_density: float | None = None
    cameras_by_risk_level: RiskLevelCounts
    highest_risk_level: RiskLevel | None = None
    unacknowledged_alerts: int = 0
    freshness_seconds: float = Field(
        description="How recent a measurement must be to count as reporting."
    )


# --------------------------------------------------------------------------------------
# websocket
# --------------------------------------------------------------------------------------


class LiveRiskEvent(BaseModel):
    """Broadcast whenever a new risk score is computed for any camera."""

    type: str = "risk_score"
    camera_id: int
    camera_name: str
    location_name: str
    timestamp: datetime
    risk_score: float
    risk_level: RiskLevel
    person_count: float = Field(description="Mean people per frame over the window.")
    density: float
    top_feature: str | None = Field(
        default=None, description="Feature that most moved this score."
    )


class LiveHello(BaseModel):
    """First frame sent on connect, so a client can render before any window completes."""

    type: str = "hello"
    server_time: datetime
    subscribers: int
    cameras: list[CameraSummary] = []


__all__ = [
    "AlertListOut",
    "AlertOut",
    "AlertWithCamera",
    "CameraControlOut",
    "CameraDetail",
    "CameraHistory",
    "CameraSummary",
    "CrowdObservationOut",
    "DashboardSummary",
    "LatestObservation",
    "LatestRisk",
    "LiveHello",
    "LiveRiskEvent",
    "RiskLevelCounts",
    "RiskScoreOut",
]
