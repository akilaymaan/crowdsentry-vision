"""Recompute ``historical_baselines`` from accumulated observations.

``FeatureExtractor.historical_deviation`` asks "how unusual is this density *for this
camera at this hour on this weekday*", which only means anything once something has
written down what normal looks like. This job is what writes it down.

For every camera it groups ``crowd_observations`` into (hour_of_day, day_of_week) slots,
computes the mean and standard deviation of density in each, and upserts one doc per
slot. There are 168 slots per camera (24 x 7).

Run it nightly -- see ``scripts/compute_baselines.py`` for the CLI and the cron/Task
Scheduler recipes.

**Timezone.** Slots are keyed in **UTC**, because that is how the lookup key is built:
``FeatureExtractor`` reads ``window_end.hour`` off a UTC-aware datetime -- the same
timestamp the observation doc is stored under. Mongo's ``$hour`` / ``$isoDayOfWeek``
date operators default to UTC, which keeps the two sides in step. Grouping by local
hour would pair observations with the wrong slot, and would do it silently -- the
z-scores would still look plausible, just computed against the wrong "normal".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pymongo import UpdateOne

from app.core.config import settings
from app.core.database import BASELINES, CAMERAS, OBSERVATIONS, RISK_SCORES
from app.core.logging import get_logger
from app.models import RiskLevel

logger = get_logger("baselines")

_ALERTING_LEVELS = [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]


@dataclass
class BaselineResult:
    """What one run of the job did."""

    cameras_processed: int = 0
    slots_written: int = 0
    slots_skipped: int = 0
    observations_considered: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    per_camera: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False

    def summary(self) -> str:
        mode = "would write" if self.dry_run else "wrote"
        return (
            f"{mode} {self.slots_written} slots across {self.cameras_processed} cameras "
            f"from {self.observations_considered} observations "
            f"({self.slots_skipped} slots skipped)"
        )


def compute_baselines(
    database,
    *,
    lookback_days: int | None = None,
    min_samples: int | None = None,
    camera_ids: list[int] | None = None,
    exclude_alerting: bool = False,
    replace: bool = False,
    dry_run: bool = False,
) -> BaselineResult:
    """Recompute and upsert baselines.

    Args:
        database: a PyMongo ``Database`` handle.
        lookback_days: only use observations this recent. Crowd patterns drift, so
            "normal" should reflect recent weeks rather than everything ever recorded.
        min_samples: skip slots with fewer observations than this. A slot built from one
            sample has a standard deviation of zero, which would make every later
            reading infinitely deviant.
        camera_ids: restrict to these cameras; ``None`` means all of them.
        exclude_alerting: drop observations that were scored HIGH or CRITICAL. See the
            note in the CLI: including incidents teaches the baseline that dangerous is
            normal.
        replace: delete existing baselines for the processed cameras first, so slots
            that no longer have data disappear instead of lingering.
        dry_run: compute and report without writing.
    """
    lookback_days = (
        settings.baseline_lookback_days if lookback_days is None else lookback_days
    )
    min_samples = min_samples if min_samples is not None else settings.baseline_min_samples

    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than zero")
    if min_samples < 1:
        raise ValueError("min_samples must be at least one")

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=lookback_days)

    match: dict = {"timestamp": {"$gte": since}}
    if camera_ids is not None:
        match["camera_id"] = {"$in": camera_ids}

    pipeline: list[dict] = [{"$match": match}]

    if exclude_alerting:
        # An observation is dropped when a risk score at the same camera and instant was
        # HIGH or CRITICAL. Scores are written with the window's own end timestamp, so
        # this matches exactly rather than by proximity.
        pipeline += [
            {
                "$lookup": {
                    "from": RISK_SCORES,
                    "let": {"cid": "$camera_id", "ts": "$timestamp"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$camera_id", "$$cid"]},
                                        {"$eq": ["$timestamp", "$$ts"]},
                                        {"$in": ["$risk_level", _ALERTING_LEVELS]},
                                    ]
                                }
                            }
                        },
                        {"$limit": 1},
                    ],
                    "as": "_alerting",
                }
            },
            {"$match": {"_alerting": []}},
        ]

    # $hour / $isoDayOfWeek are UTC on BSON dates by default. isoDayOfWeek is
    # 1=Monday..7=Sunday; subtracting one matches Python's weekday() -- the same pact
    # the SQL version documented.
    pipeline.append(
        {
            "$group": {
                "_id": {
                    "camera_id": "$camera_id",
                    "hour_of_day": {"$hour": "$timestamp"},
                    "day_of_week": {"$isoDayOfWeek": "$timestamp"},
                },
                "avg_density": {"$avg": "$density"},
                # $stdDevSamp is null for a single doc; a zero-variance slot is the
                # honest reading there, and min_samples normally filters it anyway.
                "stddev_density": {"$stdDevSamp": "$density"},
                "sample_count": {"$sum": 1},
            }
        }
    )

    rows = [
        {
            "camera_id": row["_id"]["camera_id"],
            "hour_of_day": int(row["_id"]["hour_of_day"]),
            "day_of_week": int(row["_id"]["day_of_week"]) - 1,
            "avg_density": float(row["avg_density"]),
            "stddev_density": float(row["stddev_density"] or 0.0),
            "sample_count": int(row["sample_count"]),
        }
        for row in database[OBSERVATIONS].aggregate(pipeline)
    ]

    result = BaselineResult(window_start=since, window_end=now, dry_run=dry_run)

    keep = [row for row in rows if row["sample_count"] >= min_samples]
    result.slots_skipped = len(rows) - len(keep)
    result.observations_considered = sum(row["sample_count"] for row in rows)

    camera_names = {
        doc["id"]: doc["name"]
        for doc in database[CAMERAS].find({}, projection={"id": 1, "name": 1})
    }
    touched = {row["camera_id"] for row in keep}
    result.cameras_processed = len(touched)
    for row in keep:
        name = camera_names.get(row["camera_id"], str(row["camera_id"]))
        result.per_camera[name] = result.per_camera.get(name, 0) + 1

    if dry_run:
        result.slots_written = len(keep)
        return result

    # When replacing, include cameras with no qualifying rows as well. Otherwise a
    # camera whose recent history has gone quiet would retain stale baseline slots
    # forever. ``camera_ids=None`` means all cameras, so resolve that scope explicitly.
    if replace:
        replace_camera_ids = (
            set(camera_ids)
            if camera_ids is not None
            else {doc["id"] for doc in database[CAMERAS].find({}, projection={"id": 1})}
        )
        if replace_camera_ids:
            database[BASELINES].delete_many(
                {"camera_id": {"$in": list(replace_camera_ids)}}
            )

    if keep:
        # One upsert per slot on the unique (camera, hour, weekday) index, so re-running
        # the job refreshes docs in place rather than failing or duplicating -- which is
        # what makes it safe to run on a schedule.
        database[BASELINES].bulk_write(
            [
                UpdateOne(
                    {
                        "camera_id": row["camera_id"],
                        "hour_of_day": row["hour_of_day"],
                        "day_of_week": row["day_of_week"],
                    },
                    {"$set": {**row, "updated_at": now}},
                    upsert=True,
                )
                for row in keep
            ]
        )

    result.slots_written = len(keep)
    return result


def run_baseline_job(db_factory=None, **kwargs) -> BaselineResult:
    """Resolve the database handle and recompute baselines. The scheduled entry point."""
    if db_factory is None:
        from app.core.database import db

        database = db
    else:
        database = db_factory()

    result = compute_baselines(database, **kwargs)

    logger.info(
        "baseline job complete",
        cameras=result.cameras_processed,
        slots=result.slots_written,
        skipped=result.slots_skipped,
        observations=result.observations_considered,
        dry_run=result.dry_run or None,
    )
    return result


def baseline_coverage(database, camera_id: int) -> tuple[int, int]:
    """``(slots_with_a_baseline, 168)`` for one camera.

    Useful for answering "is this camera's history warm enough to trust its deviation
    scores yet", which during the first week is usually "no".
    """
    filled = database[BASELINES].count_documents({"camera_id": camera_id})
    return int(filled), 24 * 7


async def nightly_baseline_loop() -> None:
    """Run the baseline job once a day, inside the API process.

    Deliberately trivial: sleep until the next occurrence of the configured UTC hour,
    run, repeat. A scheduler library would buy cron expressions and persistence, neither
    of which a single daily job needs.

    Failures are logged and the loop continues -- a bad night must not silently stop
    every subsequent night.
    """
    import asyncio

    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(
            hour=settings.baseline_auto_hour_utc, minute=0, second=0, microsecond=0
        )
        if target <= now:
            target += timedelta(days=1)

        delay = (target - now).total_seconds()
        logger.info(
            "baseline job scheduled", next_run=target.isoformat(), in_hours=round(delay / 3600, 1)
        )
        await asyncio.sleep(delay)

        try:
            # Blocking DB work: keep it off the event loop.
            result = await asyncio.to_thread(run_baseline_job)
            logger.info("scheduled baseline job done", slots=result.slots_written)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("scheduled baseline job failed", error=str(exc))


__all__ = [
    "BaselineResult",
    "baseline_coverage",
    "compute_baselines",
    "nightly_baseline_loop",
    "run_baseline_job",
]
