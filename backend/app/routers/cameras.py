"""Camera endpoints: listing, detail, and history for charting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo.database import Database

from app.core.database import CAMERAS, OBSERVATIONS, RISK_SCORES, ALERTS, get_db
from app.models import Camera, CrowdObservation, RiskScore
from app.schemas import (
    CameraDetail,
    CameraHistory,
    CameraSummary,
    CrowdObservationOut,
    LatestObservation,
    LatestRisk,
    RiskScoreOut,
)

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

# History is for charting, not bulk export. At one window every 5 s this is ~14 hours
# per series, and it stops a wide date range from trying to serialise a million rows.
MAX_HISTORY_ROWS = 10_000
DEFAULT_HISTORY_HOURS = 1


def latest_per_camera(db: Database, collection: str) -> dict[int, dict]:
    """camera_id -> newest document in ``collection``, in one aggregation pass.

    The (camera_id, timestamp) index feeds the $sort/$group without an in-memory
    table scan, the same job DISTINCT ON did in Postgres. Used by the camera list,
    the dashboard summary, and the WebSocket hello so all three agree on "latest".
    """
    pipeline = [
        {"$sort": {"camera_id": 1, "timestamp": -1}},
        {"$group": {"_id": "$camera_id", "doc": {"$first": "$$ROOT"}}},
    ]
    return {row["_id"]: row["doc"] for row in db[collection].aggregate(pipeline)}


def _summaries(
    camera_docs: list[dict],
    latest_risks: dict[int, dict],
    latest_observations: dict[int, dict],
) -> list[CameraSummary]:
    summaries = []
    for doc in camera_docs:
        summary = CameraSummary.model_validate(Camera.from_doc(doc))
        risk = latest_risks.get(summary.id)
        if risk is not None:
            summary.latest_risk = LatestRisk(
                timestamp=risk["timestamp"],
                risk_score=risk["risk_score"],
                risk_level=risk["risk_level"],
            )
        observation = latest_observations.get(summary.id)
        if observation is not None:
            summary.latest_observation = LatestObservation(
                timestamp=observation["timestamp"],
                person_count=observation["person_count"],
                density=observation["density"],
            )
        summaries.append(summary)
    return summaries


@router.get("", response_model=list[CameraSummary], summary="List cameras")
def list_cameras(
    db: Database = Depends(get_db),
    active_only: bool = Query(False, description="Only cameras marked active."),
) -> list[CameraSummary]:
    """Every camera, each with its most recent risk score (null if never scored)."""
    query = {"is_active": True} if active_only else {}
    camera_docs = list(db[CAMERAS].find(query).sort("name", 1))
    return _summaries(
        camera_docs,
        latest_per_camera(db, RISK_SCORES),
        latest_per_camera(db, OBSERVATIONS),
    )


@router.get("/{camera_id}", response_model=CameraDetail, summary="Camera detail")
def get_camera(camera_id: int, db: Database = Depends(get_db)) -> CameraDetail:
    doc = db[CAMERAS].find_one({"id": camera_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No camera with id {camera_id}"
        )
    camera = Camera.from_doc(doc)

    latest_risk = db[RISK_SCORES].find_one(
        {"camera_id": camera_id}, sort=[("timestamp", -1)]
    )
    latest_observation = db[OBSERVATIONS].find_one(
        {"camera_id": camera_id}, sort=[("timestamp", -1)]
    )
    unacknowledged = db[ALERTS].count_documents(
        {"camera_id": camera_id, "acknowledged": False}
    )

    detail = CameraDetail.model_validate(camera)
    detail.latest_risk = (
        LatestRisk.model_validate(RiskScore.from_doc(latest_risk)) if latest_risk else None
    )
    detail.latest_observation = (
        CrowdObservationOut.model_validate(CrowdObservation.from_doc(latest_observation))
        if latest_observation
        else None
    )
    detail.unacknowledged_alerts = unacknowledged
    return detail


@router.get(
    "/{camera_id}/history",
    response_model=CameraHistory,
    summary="Time-series history for charting",
)
def get_camera_history(
    camera_id: int,
    db: Database = Depends(get_db),
    start: datetime | None = Query(
        None, alias="from", description="ISO-8601. Defaults to one hour ago."
    ),
    end: datetime | None = Query(
        None, alias="to", description="ISO-8601. Defaults to now."
    ),
    limit: int = Query(
        MAX_HISTORY_ROWS, ge=1, le=MAX_HISTORY_ROWS, description="Max rows per series."
    ),
) -> CameraHistory:
    """Observations and risk scores for a camera over a time range, oldest first.

    The two series are returned separately rather than joined: a window can produce an
    observation without a score (or vice versa) and joining would silently drop rows.
    """
    doc = db[CAMERAS].find_one({"id": camera_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No camera with id {camera_id}"
        )
    camera = Camera.from_doc(doc)

    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(hours=DEFAULT_HISTORY_HOURS)

    # Naive datetimes are ambiguous against stored BSON dates; treat them as UTC.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if start >= end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'from' must be earlier than 'to'",
        )

    # Take the newest docs when the range holds more than the limit, then flip to
    # chronological order for charting -- truncating the recent end would be useless.
    # Fetching limit+1 makes "truncated" exact: len == limit alone cannot tell a range
    # holding precisely limit rows apart from one holding more.
    window = {
        "camera_id": camera_id,
        "timestamp": {"$gte": start, "$lte": end},
    }
    observations = list(
        db[OBSERVATIONS].find(window).sort("timestamp", -1).limit(limit + 1)
    )
    risk_scores = list(
        db[RISK_SCORES].find(window).sort("timestamp", -1).limit(limit + 1)
    )

    truncated = len(observations) > limit or len(risk_scores) > limit
    observations = observations[:limit][::-1]
    risk_scores = risk_scores[:limit][::-1]

    return CameraHistory(
        camera_id=camera.id,
        camera_name=camera.name,
        start=start,
        end=end,
        truncated=truncated,
        observations=[
            CrowdObservationOut.model_validate(CrowdObservation.from_doc(o))
            for o in observations
        ],
        risk_scores=[
            RiskScoreOut.model_validate(RiskScore.from_doc(r)) for r in risk_scores
        ],
    )
