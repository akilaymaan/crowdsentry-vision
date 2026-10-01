"""Video input abstraction.

One interface over the three things a camera feed can be in this project:

    FileVideoSource     an mp4/avi on disk (development and evaluation)
    WebcamVideoSource   a locally attached camera (quick manual testing)
    StreamVideoSource   an RTSP/HTTP network camera (deployment)

Use :func:`open_source` to get the right one from a string, so calling code never has to
care which it got::

    with open_source("rtsp://cam-01.local/stream") as src:
        for frame_index, frame in src:
            ...

The distinction that matters downstream is :attr:`VideoSource.is_live`. A file is finite
and seekable and can be processed slower than real time; a live feed is unbounded, and
frames that arrive while you are busy are stale, so the reader drops them rather than
queueing them.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Treated as a network stream by open_source().
_STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class VideoSourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed unrecoverably."""


class VideoSource(ABC):
    """A frame producer backed by an OpenCV ``VideoCapture``.

    Subclasses only supply :meth:`_create_capture` and the :attr:`is_live` flag; opening,
    iteration, and teardown are handled here.
    """

    #: Live feeds are unbounded and drop stale frames; files are finite and never drop.
    is_live: bool = False

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._frame_index = -1

    # -- lifecycle ---------------------------------------------------------------

    @abstractmethod
    def _create_capture(self) -> cv2.VideoCapture:
        """Build the underlying ``VideoCapture``. Called by :meth:`open`."""

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable description, used in logs and error messages."""

    def open(self) -> VideoSource:
        """Open the source. Idempotent; returns ``self`` so it can be chained."""
        if self._capture is not None and self._capture.isOpened():
            return self

        capture = self._create_capture()
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Could not open video source: {self.describe()}")

        self._capture = capture
        logger.info(
            "Opened %s (%dx%d @ %.2f fps)", self.describe(), self.width, self.height, self.fps
        )
        return self

    def release(self) -> None:
        """Release the underlying capture. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # -- properties --------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _prop(self, prop: int, default: float = 0.0) -> float:
        if self._capture is None:
            return default
        value = self._capture.get(prop)
        return value if value and value > 0 else default

    @property
    def width(self) -> int:
        return int(self._prop(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._prop(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Frames per second, falling back to 25.0 when the source does not report it.

        Webcams and many RTSP streams report 0; the fallback keeps output videos from
        being written with a nonsensical frame rate.
        """
        return self._prop(cv2.CAP_PROP_FPS, default=25.0)

    @property
    def frame_count(self) -> int | None:
        """Total frames for a finite source, or ``None`` for a live feed."""
        if self.is_live:
            return None
        count = int(self._prop(cv2.CAP_PROP_FRAME_COUNT))
        return count or None

    # -- reading -----------------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """Return the next frame as BGR, or ``None`` when the source is exhausted."""
        if self._capture is None:
            raise VideoSourceError(f"Source is not open: {self.describe()}")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None

        self._frame_index += 1
        return frame

    def frames(self, max_frames: int | None = None, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs.

        ``stride`` keeps every Nth frame, which is the cheapest way to cut inference
        cost: crowd density does not change meaningfully between adjacent frames at
        25 fps. Skipped frames are still decoded (OpenCV gives no way around that for
        streams) but never reach the model.
        """
        if not self.is_open:
            self.open()

        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        yielded = 0
        while True:
            frame = self.read()
            if frame is None:
                return

            if self._frame_index % stride != 0:
                continue

            yield self._frame_index, frame

            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        return self.frames()


class FileVideoSource(VideoSource):
    """A video file on disk. Finite, seekable, and safe to process slower than real time."""

    is_live = False

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        #: Restart from the first frame when the file ends, so a short clip can stand
        #: in for a continuous feed in demos.
        self.loop = loop
        self._loops = 0

    def _create_capture(self) -> cv2.VideoCapture:
        if not self.path.exists():
            raise VideoSourceError(f"Video file not found: {self.path}")
        return cv2.VideoCapture(str(self.path))

    def read(self) -> np.ndarray | None:
        frame = super().read()
        if frame is not None or not self.loop or self._capture is None:
            return frame

        # Rewind, but deliberately do NOT reset _frame_index: source_time stays
        # monotonic across the seam, so velocity and window boundaries do not see time
        # jump backwards every time the clip restarts.
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._loops += 1
        logger.debug("Looping %s (loop %d)", self.describe(), self._loops)

        frame = super().read()
        if frame is None:
            logger.warning("Loop restart produced no frame for %s", self.describe())
        return frame

    def describe(self) -> str:
        return f"file:{self.path}"


class WebcamVideoSource(VideoSource):
    """A locally attached camera, addressed by device index."""

    is_live = True

    def __init__(self, device_index: int = 0) -> None:
        super().__init__()
        self.device_index = device_index

    def _create_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.device_index)
        # Keep only the newest frame: a stale webcam frame is worthless for live risk.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def describe(self) -> str:
        return f"webcam:{self.device_index}"


class StreamVideoSource(VideoSource):
    """A network camera (RTSP/HTTP).

    Network feeds drop out routinely, so a short read failure triggers a reconnect
    rather than ending the stream. ``max_reconnects`` bounds that so a permanently dead
    camera does not spin forever.
    """

    is_live = True

    def __init__(
        self,
        url: str,
        max_reconnects: int = 3,
        reconnect_delay_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.max_reconnects = max_reconnects
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._reconnects = 0

    def _create_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # A large buffer means we would process frames from several seconds ago.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def read(self) -> np.ndarray | None:
        frame = super().read()
        if frame is not None:
            self._reconnects = 0
            return frame

        if self._reconnects >= self.max_reconnects:
            logger.error(
                "Stream %s failed after %d reconnect attempts", self.url, self._reconnects
            )
            return None

        self._reconnects += 1
        logger.warning(
            "Stream %s dropped; reconnecting (%d/%d)",
            self.url,
            self._reconnects,
            self.max_reconnects,
        )
        self.release()
        time.sleep(self.reconnect_delay_seconds)

        try:
            self.open()
        except VideoSourceError:
            logger.exception("Reconnect to %s failed", self.url)
            return None

        return super().read()

    def describe(self) -> str:
        return f"stream:{self.url}"


def open_source(spec: str | int | Path, **kwargs) -> VideoSource:
    """Build the right :class:`VideoSource` for ``spec`` (not yet opened).

    ``0`` or ``"0"``          -> webcam device 0
    ``"rtsp://..."``          -> network stream
    anything else             -> file path

    Extra keyword arguments are forwarded to the chosen source, so stream tuning such as
    ``open_source(url, max_reconnects=10)`` works without special-casing at the call site.
    """
    if isinstance(spec, int):
        return WebcamVideoSource(spec, **kwargs)

    text = str(spec).strip()

    if text.isdigit():
        return WebcamVideoSource(int(text), **kwargs)

    if text.lower().startswith(_STREAM_PREFIXES):
        return StreamVideoSource(text, **kwargs)

    return FileVideoSource(text, **kwargs)


__all__ = [
    "FileVideoSource",
    "StreamVideoSource",
    "VideoSource",
    "VideoSourceError",
    "WebcamVideoSource",
    "open_source",
]
