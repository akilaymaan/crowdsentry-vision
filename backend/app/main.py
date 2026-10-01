"""CrowdSentry - Spatio-Temporal Crowd Behavior Analysis and Predictive Risk Management.

The API process also hosts the realtime processor: on startup it spawns one background
worker per active camera (see :mod:`app.services.realtime_processor`), and on shutdown it
stops them. Set ``REALTIME_ENABLED=false`` to run a pure API process with no workers.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import settings
from app.core.database import db, ensure_indexes
from app.core.logging import configure_logging, get_logger
from app.core.security import require_api_key
from app.routers import alerts, cameras, dashboard, live
from app.services.baseline_job import nightly_baseline_loop
from app.services.broadcaster import get_broadcaster
from app.services.realtime_processor import get_processor, start_processor, stop_processor

configure_logging(settings.log_level)
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start camera workers with the server, and stop them with it."""
    # Camera workers publish live events from their own threads, so the broadcaster
    # needs a handle on this loop to schedule the sends onto. Bind it before any worker
    # starts, or the first few events have nowhere to go.
    get_broadcaster().bind_loop()

    # Collection indexes double as the schema "migration": createIndex is a no-op when
    # the index exists, so this replaces Alembic for the document store. Failure is
    # logged inside ensure_indexes -- an unreachable Mongo must not block the API boot.
    await asyncio.to_thread(ensure_indexes)

    if settings.realtime_enabled:
        # Deliberately not awaited to completion beyond spawning: start_processor
        # returns once workers are launched, so a slow camera cannot delay the port
        # opening and make the server look dead to a health checker.
        await start_processor()
    else:
        logger.info("realtime processing disabled (REALTIME_ENABLED=false)")

    baseline_task = None
    if settings.baseline_auto_enabled:
        baseline_task = asyncio.create_task(nightly_baseline_loop(), name="baselines")
        logger.info("nightly baseline job enabled", hour_utc=settings.baseline_auto_hour_utc)

    if not settings.api_key:
        logger.warning(
            "API key auth is DISABLED (API_KEY unset). Anyone who can reach this "
            "service can read camera data and acknowledge alerts -- set API_KEY "
            "before exposing it to a network."
        )

    logger.info("API ready", version=settings.version)

    yield

    if baseline_task is not None:
        baseline_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await baseline_task

    if settings.realtime_enabled:
        await stop_processor()
    get_broadcaster().unbind_loop()
    logger.info("API shut down")


_docs_url = "/docs" if settings.api_docs_enabled else None
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Backend for spatio-temporal crowd behavior analysis and predictive "
        "risk management."
    ),
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)

# CORS for the Vite dev server / deployed dashboard origin. Note this governs HTTP
# only -- the WebSocket handshake is exempt from CORS, so /ws/live checks the API key
# itself rather than relying on this middleware (see app.routers.live).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# History endpoints return up to 10k-row series; JSON compresses well.
app.add_middleware(GZipMiddleware, minimum_size=1000)

if settings.trusted_host_list:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)


@app.middleware("http")
async def security_headers(request, call_next):
    """Baseline headers on every response; API payloads are never cacheable."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


# Every /api router requires the API key (when configured); /health, /health/ready
# and / stay open so load-balancer and container probes work without credentials.
app.include_router(cameras.router, dependencies=[Depends(require_api_key)])
app.include_router(alerts.router, dependencies=[Depends(require_api_key)])
app.include_router(dashboard.router, dependencies=[Depends(require_api_key)])
app.include_router(live.router)   # /ws/live authenticates inside the handshake


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Liveness probe used by the frontend and by docker-compose healthchecks.

    Async so it is served on the event loop directly rather than dispatched to the
    threadpool: it does no blocking work, and the probe should stay cheap under load.
    """
    return {"status": "ok", "service": settings.app_name, "version": settings.version}


@app.get("/health/ready", tags=["system"])
async def health_ready() -> dict[str, str]:
    """Readiness probe: the API plus a live database round-trip.

    Unlike /health this does block on the database, so it runs in a thread -- a wedged
    connection pool should mark the pod unready, not stall the event loop.
    """
    def _check() -> None:
        db.command("ping")

    try:
        await asyncio.wait_for(asyncio.to_thread(_check), timeout=5.0)
    except Exception as exc:  # noqa: BLE001 - report failure, never leak internals
        from fastapi import HTTPException

        logger.warning("readiness check failed", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="database unreachable") from exc
    return {"status": "ready"}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running."}


@app.get("/api/processor/status", tags=["system"], dependencies=[Depends(require_api_key)])
async def processor_status() -> dict:
    """Per-camera processing state: counters, last risk level, and any error.

    The quickest way to tell whether the workers are actually consuming video, and the
    endpoint the dashboard will poll.
    """
    return get_processor().status()
