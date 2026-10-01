"""Dashboard aggregates: the numbers along the top of the operator view."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pymongo.database import Database

from app.core.database import ALERTS, CAMERAS, OBSERVATIONS, RISK_SCORES, get_db
from app.models import RiskLevel
from app.routers.cameras import latest_per_camera
from app.schemas import DashboardSummary, RiskLevelCounts

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# A camera that stopped reporting an hour ago should not still be contributing people to
# a "right now" headline figure. Two minutes is generous against the 5 s window.
DEFAULT_FRESHNESS_SECONDS = 120.0

_LEVEL_SEVERITY = {
    RiskLevel.LOW: 0,
    RiskLevel.MODERATE: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@router.get("/summary", response_model=DashboardSummary, summary="Dashboard header")
def dashboard_summary(
    db: Database = Depends(get_db),
    freshness_seconds: float = Query(
        DEFAULT_FRESHNESS_SECONDS,
        ge=1,
        le=86_400,
        description="How recent a measurement must be for a camera to count as reporting.",
    ),
) -> DashboardSummary:
    """Aggregate counts across all cameras.

    Everything here is restricted to *recent* data. A camera whose feed died an hour ago
    still has a most-recent observation stored, and counting it would quietly inflate
    the live headcount and hold the risk level at whatever it was when the feed stopped.
    Those cameras are reported separately as ``stale_cameras``.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=freshness_seconds)

    total_cameras = db[CAMERAS].count_documents({})
    active_cameras = db[CAMERAS].count_documents({"is_active": True})

    # Latest observation per camera, in one aggregation pass.
    observations = latest_per_camera(db, OBSERVATIONS)
    fresh = [doc for doc in observations.values() if doc["timestamp"] >= cutoff]
    total_people = sum(doc["person_count"] for doc in fresh)
    densities = [doc["density"] for doc in fresh]

    # Latest risk score per camera, same trick.
    counts = RiskLevelCounts()
    highest: RiskLevel | None = None
    for doc in latest_per_camera(db, RISK_SCORES).values():
        if doc["timestamp"] < cutoff:
            continue
        level = RiskLevel(doc["risk_level"])
        setattr(counts, level.value, getattr(counts, level.value) + 1)
        if highest is None or _LEVEL_SEVERITY[level] > _LEVEL_SEVERITY[highest]:
            highest = level

    reporting_ids = {doc["camera_id"] for doc in fresh}
    stale_query: dict = {"is_active": True}
    if reporting_ids:
        stale_query["id"] = {"$nin": list(reporting_ids)}
    stale_active = db[CAMERAS].count_documents(stale_query)

    unacknowledged = db[ALERTS].count_documents({"acknowledged": False})

    return DashboardSummary(
        generated_at=now,
        total_cameras=total_cameras,
        active_cameras=active_cameras,
        reporting_cameras=len(fresh),
        stale_cameras=stale_active,
        total_people=int(total_people),
        mean_density=round(sum(densities) / len(densities), 4) if densities else None,
        peak_density=round(max(densities), 4) if densities else None,
        cameras_by_risk_level=counts,
        highest_risk_level=highest,
        unacknowledged_alerts=unacknowledged,
        freshness_seconds=freshness_seconds,
    )
