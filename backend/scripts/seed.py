"""Seed the database with the development camera (the laptop webcam).

Usage (from the backend/ directory, with the venv active and MongoDB reachable):

    python -m scripts.seed
    python -m scripts.seed --reset   # delete existing cameras first

Idempotent: cameras are matched by their unique ``name``, so re-running updates the
existing documents instead of creating duplicates.
"""

from __future__ import annotations

import argparse
import sys

from pymongo.errors import PyMongoError

from app.core.database import (
    ALERTS,
    BASELINES,
    CAMERAS,
    OBSERVATIONS,
    RISK_SCORES,
    db,
    ensure_indexes,
    next_id,
    point,
    utcnow,
)

# The development setup has one camera: the laptop webcam (device index 0).
# pixels_per_meter stays None -- without calibration, speeds report in px/s rather
# than a fabricated m/s figure. Coordinates are the dev machine's approximate area.
SAMPLE_CAMERAS: list[dict] = [
    {
        "name": "CAM-1",
        "location_name": "Laptop webcam",
        "latitude": 12.978900,
        "longitude": 77.599800,
        "area_sq_meters": 10.0,
        "pixels_per_meter": None,
        "stream_url": "0",
    },
]

# Child collections that reference camera_id, emptied alongside cameras on --reset
# (there are no foreign-key cascades in a document store).
CHILD_COLLECTIONS = [OBSERVATIONS, RISK_SCORES, ALERTS, BASELINES]


def seed(reset: bool = False) -> int:
    """Insert or update the development camera. Returns the number of cameras written."""
    ensure_indexes()
    written = 0

    if reset:
        camera_ids = [doc["id"] for doc in db[CAMERAS].find({}, projection={"id": 1})]
        deleted = db[CAMERAS].delete_many({}).deleted_count
        for collection in CHILD_COLLECTIONS:
            db[collection].delete_many({"camera_id": {"$in": camera_ids}})
        print(f"  reset: deleted {deleted} existing camera(s) and their child documents")

    for spec in SAMPLE_CAMERAS:
        sets = dict(spec)
        sets["location"] = point(spec["longitude"], spec["latitude"])
        existing = db[CAMERAS].find_one({"name": spec["name"]})

        db[CAMERAS].update_one(
            {"name": spec["name"]},
            {
                "$set": sets,
                "$setOnInsert": {
                    "id": next_id(db, CAMERAS),
                    "is_active": True,
                    "created_at": utcnow(),
                },
            },
            upsert=True,
        )
        action = "updated" if existing else "created"
        written += 1
        print(f"  {action}: {spec['name']:<20} {spec['location_name']}")

    # Read back the GeoJSON point so it is obvious the location actually landed.
    for doc in db[CAMERAS].find(
        {"name": {"$in": [s["name"] for s in SAMPLE_CAMERAS]}},
        projection={"name": 1, "location": 1},
    ).sort("name", 1):
        coords = doc.get("location", {}).get("coordinates", [])
        print(f"  geom:    {doc['name']:<20} POINT({coords[0]} {coords[1]})")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed CrowdSentry sample data.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete all existing cameras (and their observations, scores and alerts) first",
    )
    args = parser.parse_args()

    print("Seeding CrowdSentry development camera...")
    try:
        count = seed(reset=args.reset)
    except PyMongoError as exc:
        print(f"\nSeeding failed: {exc}", file=sys.stderr)
        print(
            "Is MongoDB reachable? Check MONGO_URI in backend/.env "
            "('docker compose up -d' for the local one).",
            file=sys.stderr,
        )
        return 1

    print(f"Done. {count} camera(s) present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
