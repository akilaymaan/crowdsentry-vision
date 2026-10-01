"""WebSocket endpoint for live risk updates.

Clients connect to ``/ws/live`` and receive a JSON frame every time any camera produces
a new risk score, so the dashboard never has to poll.

Protocol:

    ->  {"type": "hello",      "server_time": ..., "subscribers": 2, "cameras": [...]}
    ->  {"type": "risk_score", "camera_id": 2, "risk_level": "HIGH", ...}
    ->  {"type": "pong"}                        in reply to a client "ping"

The ``hello`` frame carries the current camera list so a freshly-opened dashboard can
render immediately, rather than showing nothing until the first window completes -- which
with a 5 s window is a visibly empty screen.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.database import CAMERAS, OBSERVATIONS, RISK_SCORES, db
from app.core.logging import get_logger
from app.core.security import websocket_authorized
from app.routers.cameras import _summaries, latest_per_camera
from app.schemas import CameraSummary, LiveHello
from app.services.broadcaster import get_broadcaster

router = APIRouter(tags=["live"])
logger = get_logger("api.live")

# Sent when no event has occurred for this long, so idle connections stay open through
# proxies and clients can distinguish "quiet" from "dead".
KEEPALIVE_SECONDS = 25.0


def _camera_snapshot() -> list[CameraSummary]:
    """Current cameras with their latest risk and observation, for the hello frame.

    Built from the same helpers as ``GET /api/cameras`` so the two can never disagree
    about what "latest" means.
    """
    camera_docs = list(db[CAMERAS].find({}).sort("name", 1))
    return _summaries(
        camera_docs,
        latest_per_camera(db, RISK_SCORES),
        latest_per_camera(db, OBSERVATIONS),
    )


@router.websocket("/ws/live")
async def live_updates(websocket: WebSocket) -> None:
    """Stream risk-score events to a connected dashboard."""
    # CORS does not cover the WS handshake, so the key is checked explicitly here.
    # Closing before accept() makes the handshake itself fail with HTTP 403 rather
    # than opening a socket that dies immediately.
    if not websocket_authorized(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()

    client = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    broadcaster = get_broadcaster()
    subscriber = broadcaster.subscribe(client)

    try:
        # The snapshot query is blocking; keep it off the event loop.
        cameras = await asyncio.to_thread(_camera_snapshot)
        hello = LiveHello(
            server_time=datetime.now(timezone.utc),
            subscribers=broadcaster.subscriber_count,
            cameras=cameras,
        )
        await websocket.send_text(hello.model_dump_json())
    except Exception:  # noqa: BLE001 - a failed snapshot must not kill the stream
        logger.exception("could not send hello frame", client=client)

    # Two concurrent jobs: pushing events out, and reading from the socket so a client
    # disconnect is noticed promptly rather than on the next send attempt.
    sender = asyncio.create_task(_send_events(websocket, subscriber), name="ws-send")
    receiver = asyncio.create_task(_read_client(websocket), name="ws-recv")

    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        broadcaster.unsubscribe(subscriber)
        try:
            await websocket.close()
        except RuntimeError:
            # Already closed by the peer.
            pass


async def _send_events(websocket: WebSocket, subscriber) -> None:
    """Drain this subscriber's queue to the socket."""
    while True:
        try:
            event = await asyncio.wait_for(
                subscriber.queue.get(), timeout=KEEPALIVE_SECONDS
            )
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "keepalive"})
            continue

        await websocket.send_json(event)


async def _read_client(websocket: WebSocket) -> None:
    """Consume client frames; answers ping, otherwise just detects disconnect."""
    while True:
        try:
            message = await websocket.receive_text()
        except WebSocketDisconnect:
            return

        if message.strip().lower() in {"ping", '{"type":"ping"}', '{"type": "ping"}'}:
            await websocket.send_json({"type": "pong"})
