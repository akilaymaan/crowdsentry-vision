"""Alert endpoints: the operator queue, and acknowledging entries in it."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo import ReturnDocument
from pymongo.database import Database

from app.core.database import ALERTS, CAMERAS, get_db
from app.core.logging import get_logger
from app.schemas import AlertListOut, AlertOut, AlertWithCamera

router = APIRouter(prefix="/api/alerts", tags=["alerts"])
logger = get_logger("api.alerts")


def _camera_names(db: Database, camera_ids: set[int]) -> dict[int, dict]:
    """id -> {name, location_name} for the cameras behind a page of alerts, in one
    query rather than a lookup per row."""
    return {
        doc["id"]: doc
        for doc in db[CAMERAS].find(
            {"id": {"$in": list(camera_ids)}},
            projection={"id": 1, "name": 1, "location_name": 1},
        )
    }


def _with_camera(alert: dict, camera: dict | None) -> AlertWithCamera:
    """Combine an alert doc with its camera's name for the response."""
    return AlertWithCamera(
        **AlertOut.model_validate(alert).model_dump(),
        camera_name=(camera or {}).get("name", f"camera-{alert['camera_id']}"),
        camera_location=(camera or {}).get("location_name", ""),
    )


@router.get("", response_model=AlertListOut, summary="List alerts")
def list_alerts(
    db: Database = Depends(get_db),
    acknowledged: bool | None = Query(
        None,
        description="Filter by acknowledgement. Omit for all; false for the open queue.",
    ),
    camera_id: int | None = Query(None, description="Restrict to one camera."),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> AlertListOut:
    """Alerts, most recent first.

    ``total`` is the full count matching the filter, independent of paging, so the UI can
    show "12 open alerts" without fetching all of them.
    """
    query: dict = {}
    if acknowledged is not None:
        query["acknowledged"] = acknowledged
    if camera_id is not None:
        query["camera_id"] = camera_id

    total = db[ALERTS].count_documents(query)
    rows = list(
        db[ALERTS]
        .find(query)
        .sort([("timestamp", -1), ("id", -1)])
        .skip(offset)
        .limit(limit)
    )

    cameras = _camera_names(db, {row["camera_id"] for row in rows})
    alerts = [_with_camera(row, cameras.get(row["camera_id"])) for row in rows]

    return AlertListOut(total=total, count=len(alerts), alerts=alerts)


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AlertWithCamera,
    summary="Acknowledge an alert",
)
def acknowledge_alert(alert_id: int, db: Database = Depends(get_db)) -> AlertWithCamera:
    """Mark an alert acknowledged.

    Idempotent: acknowledging an already-acknowledged alert succeeds and returns it
    unchanged, so a double-click or a retried request is not an error.
    """
    alert = db[ALERTS].find_one_and_update(
        {"id": alert_id},
        {"$set": {"acknowledged": True}},
        return_document=ReturnDocument.AFTER,
    )
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No alert with id {alert_id}"
        )

    camera = db[CAMERAS].find_one(
        {"id": alert["camera_id"]}, projection={"name": 1, "location_name": 1}
    )
    logger.info("alert acknowledged", alert_id=alert_id, camera=(camera or {}).get("name"))

    return _with_camera(alert, camera)
