"""Seed the database with sample cameras for local development.

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

# Three venues with genuinely different crowd dynamics, so the risk model has
# contrasting baselines to work against. Coordinates are around Bengaluru.
SAMPLE_CAMERAS: list[dict] = [
    {
        "name": "CAM-01-NORTH-GATE",
        "location_name": "Stadium North Gate - Main Entry",
        "latitude": 12.978900,
        "longitude": 77.599800,
        # Wide funnel approach; bursty inflow around gate-opening time.
        "area_sq_meters": 450.0,
        # Placeholder calibration -- measure a known distance in each camera's own
        # footage and set this properly before trusting any m/s figure.
        "pixels_per_meter": 38.0,
    },
    {
        "name": "CAM-02-CONCOURSE",
        "location_name": "Stadium Upper Concourse - Section B",
        "latitude": 12.979350,
        "longitude": 77.600450,
        # Circulation corridor; sustained bidirectional flow at half-time.
        "area_sq_meters": 280.0,
        "pixels_per_meter": 52.0,
    },
    {
        "name": "CAM-03-METRO-EXIT",
        "location_name": "Metro Station Exit C - Pedestrian Plaza",
        "latitude": 12.976100,
        "longitude": 77.603200,
        # Open plaza, but the narrow stair mouth makes it a pinch point on egress.
        "area_sq_meters": 620.0,
        "pixels_per_meter": 26.0,
    },
]

# Child collections that reference camera_id, emptied alongside cameras on --reset
# (there are no foreign-key cascades in a document store).
CHILD_COLLECTIONS = [OBSERVATIONS, RISK_SCORES, ALERTS, BASELINES]


def seed(reset: bool = False) -> int:
    """Insert or update the sample cameras. Returns the number of cameras written."""
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
                    "stream_url": None,
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

    print("Seeding CrowdSentry sample cameras...")
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
