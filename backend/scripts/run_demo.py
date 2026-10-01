"""Run the realtime processor against sample video files, on loop.

Everything the live service does -- detection, tracking, feature extraction, risk
scoring, persistence, alerting -- but fed from video files instead of cameras, so the
whole pipeline can be demonstrated without hardware.

Usage (from backend/, with MongoDB reachable and the model trained):

    python -m scripts.run_demo
    python -m scripts.run_demo --window 2 --duration 60
    python -m scripts.run_demo --videos data/samples/crowd.mp4 data/samples/moving.mp4

It creates (or updates) one demo camera per video file, points it at that file, marks it
active, and runs the processor until interrupted with Ctrl+C. Demo cameras are named
``DEMO-*`` and are deactivated on exit so they do not start up with the API afterwards.

Note the window length: the default 5 s window means a 4 s sample clip produces roughly
one scored window per loop. ``--window 2`` gives more frequent output for a demo.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from pathlib import Path

from pymongo.errors import PyMongoError

from app.core.config import settings
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
from app.core.logging import configure_logging, get_logger
from app.ml.predict import ModelNotTrainedError, get_model
from app.services.realtime_processor import RealtimeProcessor

logger = get_logger("demo")

DEFAULT_VIDEOS = [
    Path("data/samples/crowd.mp4"),
    Path("data/samples/moving.mp4"),
]

# Plausible stand-ins so density and calibration are not nonsense. Real deployments
# measure these; see the README on calibration.
DEMO_CAMERAS = [
    {
        "name": "DEMO-01-GATE",
        "location_name": "Demo Stadium North Gate",
        "latitude": 12.9789,
        "longitude": 77.5998,
        "area_sq_meters": 45.0,
        "pixels_per_meter": 38.0,
    },
    {
        "name": "DEMO-02-PLAZA",
        "location_name": "Demo Metro Exit Plaza",
        "latitude": 12.9761,
        "longitude": 77.6032,
        "area_sq_meters": 60.0,
        "pixels_per_meter": 26.0,
    },
]


def check_prerequisites(videos: list[Path]) -> list[Path]:
    """Fail early and legibly, rather than deep inside a worker thread."""
    missing = [video for video in videos if not video.exists()]
    if missing:
        raise SystemExit(
            "Sample video(s) not found: "
            + ", ".join(str(m) for m in missing)
            + "\nPut a video with people in it at backend/data/samples/ and pass it with"
            " --videos."
        )

    try:
        get_model().metadata
    except ModelNotTrainedError as exc:
        raise SystemExit(f"{exc}") from exc

    try:
        db.command("ping")
    except PyMongoError as exc:
        raise SystemExit(
            f"Cannot reach MongoDB: {exc}\n"
            "Check MONGO_URI in backend/.env ('docker compose up -d' for the local one)."
        ) from exc

    return videos


def _upsert_camera(spec: dict) -> int:
    """Create or refresh a demo camera document; returns its public id."""
    sets = dict(spec)
    sets["location"] = point(spec["longitude"], spec["latitude"])
    existing = db[CAMERAS].find_one({"name": spec["name"]}, projection={"id": 1})
    db[CAMERAS].update_one(
        {"name": spec["name"]},
        {
            "$set": sets,
            "$setOnInsert": {"id": next_id(db, CAMERAS), "created_at": utcnow()},
        },
        upsert=True,
    )
    if existing is not None:
        return existing["id"]
    return db[CAMERAS].find_one({"name": spec["name"]}, projection={"id": 1})["id"]


def configure_demo_cameras(
    videos: list[Path], with_baselines: bool, area_override: float | None = None
) -> list[str]:
    """Create/refresh one demo camera per video. Returns their names."""
    ensure_indexes()
    names: list[str] = []
    now = utcnow()

    # Any other active camera would also be picked up by the processor and compete for
    # CPU with the demo, so park them for the duration.
    others = db[CAMERAS].update_many(
        {"is_active": True, "name": {"$not": {"$regex": "^DEMO-"}}},
        {"$set": {"is_active": False}},
    )
    if others.modified_count:
        logger.info(
            "deactivated non-demo cameras for the demo", cameras=others.modified_count
        )

    for index, video in enumerate(videos):
        spec = DEMO_CAMERAS[index % len(DEMO_CAMERAS)].copy()
        if index >= len(DEMO_CAMERAS):
            spec["name"] = f"{spec['name']}-{index}"
        if area_override is not None:
            spec["area_sq_meters"] = area_override

        spec["stream_url"] = str(video)
        spec["is_active"] = True
        camera_id = _upsert_camera(spec)

        if with_baselines:
            # Without a baseline, historical_deviation falls back to 0.0 for every
            # window. A synthetic one makes the demo show the feature actually working;
            # it is not a real measurement.
            db[BASELINES].update_one(
                {
                    "camera_id": camera_id,
                    "hour_of_day": now.hour,
                    "day_of_week": now.weekday(),
                },
                {
                    "$set": {
                        "avg_density": 0.15,
                        "stddev_density": 0.06,
                        "sample_count": 200,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )

        names.append(spec["name"])
        logger.info(
            "demo camera ready",
            camera=spec["name"],
            video=str(video),
            area=spec["area_sq_meters"],
            ppm=spec["pixels_per_meter"],
        )

    return names


def deactivate_demo_cameras() -> None:
    db[CAMERAS].update_many(
        {"name": {"$regex": "^DEMO-"}}, {"$set": {"is_active": False}}
    )
    logger.info("demo cameras deactivated")


def print_summary(names: list[str]) -> None:
    """What actually landed in the database."""
    cameras = list(db[CAMERAS].find({"name": {"$in": names}}))

    print()
    print("=" * 72)
    print("  Demo run summary")
    print("=" * 72)

    for camera in cameras:
        camera_id = camera["id"]
        observations = db[OBSERVATIONS].count_documents({"camera_id": camera_id})
        scores = list(
            db[RISK_SCORES]
            .find({"camera_id": camera_id}, projection={"risk_level": 1, "risk_score": 1})
            .sort("timestamp", 1)
        )
        alerts = list(
            db[ALERTS]
            .find({"camera_id": camera_id}, projection={"message": 1})
            .sort("timestamp", 1)
        )

        print(f"  {camera['name']}  ({camera['location_name']})")
        print(f"    observations   {observations}")
        print(f"    risk scores    {len(scores)}")
        if scores:
            levels: dict[str, int] = {}
            for score in scores:
                levels[score["risk_level"]] = levels.get(score["risk_level"], 0) + 1
            spread = "  ".join(f"{k}={v}" for k, v in levels.items())
            print(f"      levels       {spread}")
            print(
                f"      score range  {min(s['risk_score'] for s in scores):.1f}"
                f" - {max(s['risk_score'] for s in scores):.1f}"
            )
        print(f"    alerts         {len(alerts)}")
        for alert in alerts[:3]:
            print(f"      - {alert['message']}")
        if len(alerts) > 3:
            print(f"      ... and {len(alerts) - 3} more")
        print()

    print("  Inspect the data with:")
    print('    docker compose exec db mongosh crowdsentry --eval \\')
    print('      "db.risk_scores.find().sort({timestamp:-1}).limit(10)"')
    print("=" * 72)


async def run(args: argparse.Namespace, names: list[str]) -> None:
    processor = RealtimeProcessor(loop_video=True, window_seconds=args.window)
    await processor.start()

    if not processor.workers:
        logger.error("no workers started; nothing to demo")
        return

    stop_event = asyncio.Event()

    def request_stop() -> None:
        logger.info("shutdown requested")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            # Windows event loops do not implement add_signal_handler; the
            # KeyboardInterrupt path below covers Ctrl+C there.
            loop.add_signal_handler(sig, request_stop)

    print()
    logger.info(
        "processing; press Ctrl+C to stop",
        cameras=len(processor.workers),
        window=args.window or settings.features_window_seconds,
        duration=args.duration or "until interrupted",
    )
    print()

    try:
        if args.duration:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
        else:
            await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        await processor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the realtime processor against sample videos on loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos", type=Path, nargs="+", default=DEFAULT_VIDEOS,
        help="sample video files, one demo camera per file",
    )
    parser.add_argument(
        "--window", type=float, default=2.0,
        help="feature window in seconds (shorter than production, for livelier output)",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="stop after this many seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--area", type=float, default=None,
        help="override each demo camera's area_sq_meters. The stock sample clips hold "
             "only 3-4 people, so at a realistic 45-60 m2 footprint density never leaves "
             "LOW and the alerting path is never exercised. Shrinking the assumed "
             "footprint (e.g. --area 3) models a camera watching a tight doorway and "
             "pushes the same footage into HIGH/CRITICAL. The density is computed "
             "honestly from this number; it is the number itself that is a stand-in.",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="skip seeding synthetic historical baselines",
    )
    parser.add_argument(
        "--keep-active", action="store_true",
        help="leave demo cameras active on exit, so the API picks them up too",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)

    videos = check_prerequisites(list(args.videos))
    names = configure_demo_cameras(
        videos, with_baselines=not args.no_baselines, area_override=args.area
    )

    try:
        asyncio.run(run(args, names))
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        if not args.keep_active:
            deactivate_demo_cameras()
        print_summary(names)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
