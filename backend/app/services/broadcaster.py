"""Live event fan-out to WebSocket subscribers.

The producer and the consumers live on opposite sides of a thread boundary: risk scores
are computed inside camera worker *threads*, while WebSocket sends must happen on the
event loop. :meth:`LiveBroadcaster.publish_threadsafe` is the bridge -- it is safe to call
from any thread and never blocks the caller, so a slow or stalled subscriber can never
slow down the processing pipeline.

Each subscriber gets its own bounded queue. When a client is too slow to keep up its
queue fills and the *oldest* event is dropped rather than the newest: for a live
dashboard, a stale reading is worthless and the current one is the whole point.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.logging import get_logger

logger = get_logger("broadcast")

# Per-subscriber buffer. A dashboard receives roughly one event per camera per window
# (~0.2/s per camera), so 64 is many seconds of slack before anything is dropped.
QUEUE_MAXSIZE = 64


class Subscriber:
    """One connected client, with its own buffer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.dropped = 0

    def offer(self, event: dict[str, Any]) -> None:
        """Queue an event, discarding the oldest if the client has fallen behind."""
        try:
            self.queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        try:
            self.queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
            pass

        self.dropped += 1
        if self.dropped in (1, 10, 100) or self.dropped % 500 == 0:
            logger.warning("subscriber lagging, dropping events", client=self.name,
                           dropped=self.dropped)

        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover
            pass


class LiveBroadcaster:
    """Fan-out hub. One instance per process."""

    def __init__(self) -> None:
        self._subscribers: set[Subscriber] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Remember the event loop that publishes must be scheduled onto.

        Called from the FastAPI lifespan. Without it, publishing from a worker thread
        has no loop to hand the event to and is silently dropped.
        """
        self._loop = loop or asyncio.get_running_loop()

    def unbind_loop(self) -> None:
        self._loop = None

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self, name: str) -> Subscriber:
        subscriber = Subscriber(name)
        self._subscribers.add(subscriber)
        logger.info("subscriber connected", client=name, total=len(self._subscribers))
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.discard(subscriber)
        logger.info(
            "subscriber disconnected",
            client=subscriber.name,
            total=len(self._subscribers),
            dropped=subscriber.dropped or None,
        )

    # -- publishing --------------------------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Hand an event to every subscriber. Must run on the event loop."""
        for subscriber in list(self._subscribers):
            subscriber.offer(event)

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Publish from any thread.

        Returns immediately. If no loop is bound, or nobody is listening, this is a
        cheap no-op -- the processing pipeline must never depend on a dashboard being
        connected.
        """
        loop = self._loop
        if loop is None or not self._subscribers:
            return

        try:
            loop.call_soon_threadsafe(self.publish, event)
        except RuntimeError:
            # The loop closed between the check and the call, i.e. we are shutting down.
            logger.debug("event loop closed; dropping live event")


_broadcaster = LiveBroadcaster()


def get_broadcaster() -> LiveBroadcaster:
    return _broadcaster


__all__ = ["LiveBroadcaster", "Subscriber", "get_broadcaster"]
