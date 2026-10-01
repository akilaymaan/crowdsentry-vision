"""Recompute historical_baselines from accumulated observations.

Run this nightly. Until it has run over real history, every camera's
``historical_deviation`` is 0.0 -- the cold-start default -- and the risk model is
effectively working from six features instead of seven.

Usage (from backend/, with the venv active):

    python -m scripts.compute_baselines
    python -m scripts.compute_baselines --dry-run
    python -m scripts.compute_baselines --days 14 --min-samples 3
    python -m scripts.compute_baselines --camera CAM-02-CONCOURSE
    python -m scripts.compute_baselines --exclude-alerting

Scheduling
----------
No scheduler library needed -- this is one command, it is idempotent (every slot is an
upsert), and it takes seconds. Point the operating system's scheduler at it.

**Linux / macOS (cron).** ``crontab -e``, then, to run at 03:15 UTC nightly:

    15 3 * * *  cd /path/to/CrowdSentry/backend && .venv/bin/python -m scripts.compute_baselines >> /var/log/crowdsentry-baselines.log 2>&1

**Windows (Task Scheduler).** As one command:

    schtasks /create /tn "CrowdSentry baselines" /sc daily /st 03:15 ^
      /tr "cmd /c cd /d D:\\code\\CrowdSentry\\backend && .venv\\Scripts\\python.exe -m scripts.compute_baselines"

**Docker/systemd.** A systemd timer or a Kubernetes CronJob invoking the same command
works identically; nothing here holds state between runs.

**In-process alternative.** Set ``BASELINE_AUTO_ENABLED=true`` and the API will run the
job itself once a day (see app/services/baseline_job.py and the lifespan in app/main.py).
That is convenient for a single-node deployment, but an external scheduler is better
behaved: it still runs when the API is down, it logs where your other jobs log, and it
does not run N times when you run N API replicas.

Pick a time when the venue is quiet. The job reads a month of observations in one
aggregate query, which is cheap, but there is no reason to compete with peak-hour
processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both ``python -m scripts.compute_baselines`` from backend/ and direct
# invocation such as ``python backend/scripts/compute_baselines.py`` from the repo
# root. The latter otherwise puts only backend/scripts on sys.path, so ``app`` cannot
# be imported.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.database import BASELINES, CAMERAS, OBSERVATIONS, db
from app.core.logging import configure_logging
from app.services.baseline_job import compute_baselines

TOTAL_SLOTS = 24 * 7
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute historical_baselines from crowd_observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Schedule this nightly; see the module docstring for cron and Task "
        "Scheduler recipes.",
    )
    parser.add_argument(
        "--days", type=int, default=settings.baseline_lookback_days,
        help="how much history to build baselines from",
    )
    parser.add_argument(
        "--min-samples", type=int, default=settings.baseline_min_samples,
        help="skip slots with fewer observations than this",
    )
    parser.add_argument(
        "--camera", action="append", default=None, metavar="NAME",
        help="restrict to this camera (repeatable); default is all cameras",
    )
    parser.add_argument(
        "--exclude-alerting", action="store_true",
        help="ignore observations that were scored HIGH or CRITICAL. Baselines built "
             "from data that includes incidents teach the system that dangerous is "
             "normal, which is self-defeating -- but on a young deployment it can "
             "remove most of the data, so it is opt-in.",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="delete each processed camera's existing baselines first, so slots that no "
             "longer have data disappear instead of going stale",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument(
        "--show-coverage", action="store_true",
        help="print a per-camera grid of which (weekday, hour) slots now have a baseline",
    )
    return parser


def resolve_cameras(names: list[str] | None) -> list[int] | None:
    if not names:
        return None

    rows = list(
        db[CAMERAS].find(
            {"name": {"$in": names}}, projection={"id": 1, "name": 1}
        )
    )

    found = {row["name"] for row in rows}
    missing = set(names) - found
    if missing:
        raise SystemExit(f"No camera named: {', '.join(sorted(missing))}")

    return [row["id"] for row in rows]


def print_coverage() -> None:
    """A 7x24 grid per camera: which slots have a baseline, and how warm they are."""
    for camera in db[CAMERAS].find({}).sort("name", 1):
        rows = list(
            db[BASELINES].find(
                {"camera_id": camera["id"]},
                projection={"day_of_week": 1, "hour_of_day": 1, "sample_count": 1},
            )
        )
        if not rows:
            continue

        grid = {(row["day_of_week"], row["hour_of_day"]): row["sample_count"] for row in rows}
        print(f"\n  {camera['name']}  ({len(rows)}/{TOTAL_SLOTS} slots)")
        print("        " + "".join(f"{hour:>3}" for hour in range(24)))
        for day in range(7):
            cells = []
            for hour in range(24):
                samples = grid.get((day, hour))
                # Density of the mark stands in for how many samples back the slot:
                # a sparse slot is a weak baseline even though it exists.
                cells.append(
                    "  ." if samples is None
                    else "  o" if samples < 20
                    else "  #"
                )
            print(f"    {DAYS[day]} " + "".join(cells))
        print("        . none   o thin (<20 samples)   # solid")


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(settings.log_level)

    camera_ids = resolve_cameras(args.camera)

    try:
        observation_count = db[OBSERVATIONS].estimated_document_count()
        result = compute_baselines(
            db,
            lookback_days=args.days,
            min_samples=args.min_samples,
            camera_ids=camera_ids,
            exclude_alerting=args.exclude_alerting,
            replace=args.replace,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Baseline job failed: {exc}", file=sys.stderr)
        print(
            "Is MongoDB reachable? Check MONGO_URI in backend/.env "
            "('docker compose up -d' for the local one).",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 68)
    print("  Historical baselines" + ("  (DRY RUN - nothing written)" if args.dry_run else ""))
    print("=" * 68)
    print(f"  observations in table   {observation_count}")
    print(f"  lookback window         {args.days} days "
          f"({result.window_start:%Y-%m-%d %H:%M} -> {result.window_end:%Y-%m-%d %H:%M} UTC)")
    print(f"  observations used       {result.observations_considered}")
    print(f"  min samples per slot    {args.min_samples}")
    if args.exclude_alerting:
        print("  excluding               windows scored HIGH/CRITICAL")
    print("-" * 68)
    print(f"  cameras                 {result.cameras_processed}")
    print(f"  slots written           {result.slots_written}")
    print(f"  slots skipped (thin)    {result.slots_skipped}")
    if result.per_camera:
        print("-" * 68)
        for name, count in sorted(result.per_camera.items()):
            bar = "#" * min(40, count)
            print(f"  {name:<22}{count:>4}/{TOTAL_SLOTS}  {bar}")
    print("=" * 68)

    if result.slots_written == 0:
        print()
        if observation_count == 0:
            print("  No observations yet. Baselines are built from crowd_observations,")
            print("  so run the processor first (uvicorn app.main:app, or")
            print("  python -m scripts.run_demo) and let it collect some history.")
        else:
            print(f"  No slot reached {args.min_samples} samples. Either the history is")
            print("  still too short, or it is concentrated in a few hours. Lower")
            print("  --min-samples to see partial baselines, but treat them as weak.")
        print()
        print("  historical_deviation stays at 0.0 until baselines exist - the model")
        print("  is running on six of its seven features.")

    if args.show_coverage:
        print_coverage()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
