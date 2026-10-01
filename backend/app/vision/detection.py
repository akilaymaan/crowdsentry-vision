"""Per-frame person detection with YOLOv8.

Detection only -- no tracking, no identity across frames. Each call is independent, so a
person in frame N has no relationship to a person in frame N+1. Tracking comes later and
will consume these boxes.

Typical use::

    from app.vision.detection import detect_people

    boxes = detect_people(frame)
    print(len(boxes), "people")

``detect_people`` uses a lazily-created shared detector, which is what you want in a
long-running process: the model is loaded once, on first call. Construct
:class:`PersonDetector` directly when you need per-camera settings or several models at
once.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# "person" is class 0 in the COCO dataset that the pretrained YOLOv8 weights use.
PERSON_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """One detected person in a single frame, in that frame's pixel coordinates.

    Corners are ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right, as floats -- the
    model's output is subpixel and rounding is left to whoever draws it.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box centre. Useful as a coarse position for density work."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the person meets the ground.

        This is the point to homography-project onto a floor plan -- the box centre
        floats around chest height and would place people behind where they stand.
        """
        return (self.x1 + self.x2) / 2.0, self.y2

    def scaled(self, factor: float) -> BoundingBox:
        """Return this box scaled about the origin, for mapping between frame sizes."""
        return BoundingBox(
            x1=self.x1 * factor,
            y1=self.y1 * factor,
            x2=self.x2 * factor,
            y2=self.y2 * factor,
            confidence=self.confidence,
        )

    def to_xywh(self) -> tuple[float, float, float, float]:
        """As ``(x, y, width, height)``, the format most trackers expect."""
        return self.x1, self.y1, self.width, self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
        }


def _resize_to_max_dimension(
    frame: np.ndarray, max_dimension: int
) -> tuple[np.ndarray, float]:
    """Downscale ``frame`` so its longer side is ``max_dimension``.

    Returns the resized frame and the scale factor applied, so detections can be mapped
    back to the original resolution by dividing by it. Never upscales: feeding a model
    more pixels than the source has invents no detail and costs real time.
    """
    height, width = frame.shape[:2]
    longest = max(height, width)

    if longest <= max_dimension:
        return frame, 1.0

    scale = max_dimension / longest
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        # INTER_AREA is the correct filter for shrinking; it averages the pixels being
        # discarded instead of point-sampling them.
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


class PersonDetector:
    """YOLOv8 person detector.

    The model is loaded lazily on first use, so importing this module (or constructing a
    detector at import time) does not pay the weight-loading cost.

    Args:
        model_path: Ultralytics weights. A bare name like ``"yolov8n.pt"`` is downloaded
            on first use and cached. ``n`` is the smallest and fastest variant; step up
            to ``s``/``m`` for better recall on small, distant people in a crowd.
        confidence_threshold: Minimum score to report a detection (0-1). Crowd scenes
            justify a lower value than the usual 0.5 -- distant and heavily occluded
            people score low, and missing them undercounts density exactly where the
            crowd is thickest.
        max_dimension: Frames are downscaled so their longer side is at most this many
            pixels before inference. The dominant cost knob.
        iou_threshold: IoU above which two boxes are treated as the same person by
            non-maximum suppression. Raised from the default for crowds, where people
            genuinely overlap and an aggressive NMS deletes real detections.
        device: ``"cpu"``, ``"cuda"``, ``"0"``, ... or ``None`` to let Ultralytics pick.
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        max_dimension: int | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path or settings.detection_model
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.detection_confidence
        )
        self.max_dimension = (
            max_dimension if max_dimension is not None else settings.detection_max_dimension
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else settings.detection_iou
        )
        self.device = device if device is not None else settings.detection_device

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
        if self.max_dimension < 32:
            raise ValueError(f"max_dimension must be >= 32, got {self.max_dimension}")

        # YOLO's stride-32 backbone requires an input size that is a multiple of 32;
        # Ultralytics silently rounds up otherwise, so do it here where it is visible.
        self.imgsz = ((self.max_dimension + 31) // 32) * 32

    @cached_property
    def model(self):
        """The Ultralytics model, loaded on first access."""
        # Imported here rather than at module scope: ultralytics pulls in torch, which
        # costs seconds of import time that the API process should not pay to serve
        # /health.
        from ultralytics import YOLO

        logger.info("Loading YOLO weights: %s", self.model_path)
        model = YOLO(self.model_path)
        if self.device:
            model.to(self.device)
        return model

    def warmup(self) -> None:
        """Load the weights and run one dummy inference.

        The first real inference is several times slower than steady state (weight load,
        lazy kernel init). Call this at startup so the first actual frame is not the one
        that pays for it.
        """
        dummy = np.zeros((self.max_dimension, self.max_dimension, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("Detector warmed up (%s)", self.model_path)

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Detect people in one BGR frame.

        Returns boxes in the coordinate space of the frame that was passed in, highest
        confidence first, regardless of any internal downscaling.
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame is empty")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR frame, got shape {frame.shape}")

        resized, scale = _resize_to_max_dimension(frame, self.max_dimension)

        results = self.model.predict(
            resized,
            # Without this Ultralytics letterboxes to its own default (640) and the
            # pre-resize above buys nothing: max_dimension would silently stop being a
            # cost knob. Passing it means the network genuinely runs at this size.
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            # Filter to people inside the model, so NMS and postprocessing never touch
            # the other 79 COCO classes.
            classes=[PERSON_CLASS_ID],
            device=self.device or None,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        # Undo our downscale so callers get coordinates in their own frame's space.
        inverse = 1.0 / scale

        detections = [
            BoundingBox(
                x1=float(x1) * inverse,
                y1=float(y1) * inverse,
                x2=float(x2) * inverse,
                y2=float(y2) * inverse,
                confidence=float(confidence),
            )
            for (x1, y1, x2, y2), confidence in zip(xyxy, confidences)
        ]
        detections.sort(key=lambda box: box.confidence, reverse=True)
        return detections


# --------------------------------------------------------------------------------------
# Module-level convenience API
# --------------------------------------------------------------------------------------

_default_detector: PersonDetector | None = None
_detector_lock = threading.Lock()


def get_detector() -> PersonDetector:
    """Return the process-wide detector, creating it on first call.

    Double-checked locking: FastAPI serves requests from a thread pool, and two
    concurrent first-requests would otherwise each load their own copy of the weights.
    """
    global _default_detector

    if _default_detector is None:
        with _detector_lock:
            if _default_detector is None:
                _default_detector = PersonDetector()

    return _default_detector


def reset_detector() -> None:
    """Discard the shared detector so the next call rebuilds it from current settings."""
    global _default_detector

    with _detector_lock:
        _default_detector = None


def detect_people(frame: np.ndarray) -> list[BoundingBox]:
    """Detect people in one BGR frame using the shared detector.

    Convenience wrapper over :meth:`PersonDetector.detect`; construct a
    :class:`PersonDetector` directly when you need non-default settings.
    """
    return get_detector().detect(frame)


__all__ = [
    "PERSON_CLASS_ID",
    "BoundingBox",
    "PersonDetector",
    "detect_people",
    "get_detector",
    "reset_detector",
]
