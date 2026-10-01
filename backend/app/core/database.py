"""MongoDB client, collection access, and index management.

One ``MongoClient`` is shared process-wide -- it is internally a connection pool and
is safe to use from worker threads and request handlers alike. The previous SQLAlchemy
"session per request" pattern has no equivalent here: a ``Database`` handle is cheap,
immutable, and needs no lifecycle management, so ``get_db`` just yields the shared one.

Document conventions:

* Every collection carries an ``id`` integer field assigned from the ``counters``
  collection (``next_id``), keeping the API's numeric ids stable -- the BSON ``_id``
  stays an internal ObjectId.
* ``cameras`` stores GeoJSON ``location`` (2dsphere) alongside flat ``latitude`` /
  ``longitude`` fields; the flat fields are the read path, the point exists for
  geospatial queries.
* Timestamps are stored as BSON dates (UTC); the client is ``tz_aware`` so reads come
  back timezone-aware.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("db")

client: MongoClient = MongoClient(
    settings.mongo_uri,
    tz_aware=True,
    # Startup must not hang for minutes when Atlas is unreachable: fail fast and let
    # the readiness probe report it.
    serverSelectionTimeoutMS=5000,
    appname="crowdsentry-api",
)
db: Database = client[settings.mongo_db]

# Collection names, kept in one place so a typo fails loudly here rather than
# silently creating a misspelt collection at write time.
CAMERAS = "cameras"
OBSERVATIONS = "crowd_observations"
RISK_SCORES = "risk_scores"
ALERTS = "alerts"
BASELINES = "historical_baselines"
COUNTERS = "counters"


def get_db() -> Generator[Database, None, None]:
    """FastAPI dependency yielding the shared database handle."""
    yield db


def next_id(database: Database, collection: str) -> int:
    """Monotonically increasing integer id for a collection's public ``id`` field.

    Keeps ``/api/cameras/{id}`` and friends numeric across a document store that would
    otherwise hand us ObjectIds. One atomic ``findOneAndUpdate`` on ``counters`` --
    safe under concurrent workers.
    """
    row = database[COUNTERS].find_one_and_update(
        {"_id": collection},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(row["seq"])


def insert_document(database: Database, collection: str, doc: dict) -> int:
    """Assign the public ``id`` and insert; returns the assigned id."""
    doc["id"] = next_id(database, collection)
    database[collection].insert_one(doc)
    return doc["id"]


def point(longitude: float, latitude: float) -> dict:
    """GeoJSON Point. Note the order: GeoJSON coordinates are [longitude, latitude]."""
    return {"type": "Point", "coordinates": [longitude, latitude]}


def ensure_indexes(database: Database | None = None) -> None:
    """Create the collection indexes the queries rely on. Idempotent and cheap:
    ``createIndex`` is a no-op when the index already exists, so this runs on every
    startup instead of a separate migration step.

    Failures are logged, not raised -- an API that cannot create an index should still
    boot and let the readiness probe report the database's real state.
    """
    database = db if database is None else database
    try:
        cameras: Collection = database[CAMERAS]
        cameras.create_index([("id", ASCENDING)], unique=True)
        cameras.create_index([("name", ASCENDING)], unique=True)
        cameras.create_index([("location", "2dsphere")])

        database[OBSERVATIONS].create_index(
            [("camera_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        database[OBSERVATIONS].create_index([("timestamp", DESCENDING)])

        database[RISK_SCORES].create_index(
            [("camera_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        database[RISK_SCORES].create_index([("timestamp", DESCENDING)])
        # "Everything that went HIGH/CRITICAL recently", across cameras.
        database[RISK_SCORES].create_index(
            [("risk_level", ASCENDING), ("timestamp", DESCENDING)]
        )

        database[ALERTS].create_index(
            [("camera_id", ASCENDING), ("timestamp", DESCENDING)]
        )
        # The dashboard's hot query is the unacknowledged queue.
        database[ALERTS].create_index(
            [("acknowledged", ASCENDING), ("timestamp", DESCENDING)]
        )

        # One baseline per (camera, hour, weekday) slot -- enforced unique so the job's
        # upserts can never produce duplicates.
        database[BASELINES].create_index(
            [
                ("camera_id", ASCENDING),
                ("hour_of_day", ASCENDING),
                ("day_of_week", ASCENDING),
            ],
            unique=True,
        )
    except PyMongoError as exc:
        logger.warning("could not ensure indexes", error=type(exc).__name__)


def ping(database: Database | None = None) -> bool:
    """Liveness check used by the readiness probe."""
    try:
        (db if database is None else database).command("ping")
        return True
    except PyMongoError:
        return False


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "ALERTS",
    "BASELINES",
    "CAMERAS",
    "COUNTERS",
    "OBSERVATIONS",
    "RISK_SCORES",
    "client",
    "db",
    "ensure_indexes",
    "get_db",
    "insert_document",
    "next_id",
    "ping",
    "point",
    "utcnow",
]
