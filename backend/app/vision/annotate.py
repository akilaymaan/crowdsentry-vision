"""Drawing helpers for visual verification of the detector.

Debug/verification only -- the analytics pipeline never renders anything.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.vision.detection import BoundingBox

# BGR, since that is what OpenCV uses.
_BOX_COLOR = (80, 220, 80)
_TEXT_COLOR = (20, 20, 20)
_PANEL_COLOR = (30, 30, 30)
_PANEL_TEXT_COLOR = (240, 240, 240)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    frame: np.ndarray,
    detections: list[BoundingBox],
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with each detection outlined.

    Box thickness and label size scale with frame size so output is legible at both
    640p and 4K.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    thickness = max(1, round(min(width, height) / 400))
    font_scale = max(0.35, min(width, height) / 1400)

    for box in detections:
        x1, y1 = int(round(box.x1)), int(round(box.y1))
        x2, y2 = int(round(box.x2)), int(round(box.y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, thickness)

        if not show_confidence:
            continue

        label = f"{box.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)

        # Put the label inside the box when it would otherwise run off the top edge.
        label_bottom = y1 if y1 - text_h - baseline >= 0 else y2 + text_h + baseline
        label_top = label_bottom - text_h - baseline

        cv2.rectangle(
            canvas, (x1, label_top), (x1 + text_w + 4, label_bottom), _BOX_COLOR, -1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_bottom - baseline),
            _FONT,
            font_scale,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return canvas


def track_color(track_id: int) -> tuple[int, int, int]:
    """A stable, well-separated BGR colour for a track ID.

    Derived from the ID itself rather than a lookup table, so the same person keeps the
    same colour for the whole run without any state. The golden-angle step around the
    hue circle keeps consecutive IDs far apart in colour, which is what makes an ID
    swap visible at a glance.
    """
    hue = int((track_id * 137.508) % 180)  # OpenCV hue is 0-179
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    people: list,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    show_velocity: bool = True,
    velocity_scale: float = 0.5,
) -> np.ndarray:
    """Draw tracked people with per-ID colours, velocity vectors, and motion trails.

    Args:
        people: :class:`~app.vision.tracking.TrackedPerson` objects.
        trails: Recent foot-point history per track ID, drawn as a fading polyline.
        velocity_scale: Seconds of predicted travel the arrow represents. At 0.5 the
            arrow shows where the person would be in half a second, which reads as
            speed without cluttering the frame.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    thickness = max(1, round(min(width, height) / 400))
    font_scale = max(0.35, min(width, height) / 1400)

    for person in people:
        color = track_color(person.track_id)
        box = person.bbox
        x1, y1 = int(round(box.x1)), int(round(box.y1))
        x2, y2 = int(round(box.x2)), int(round(box.y2))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Trail first, so the box and arrow sit on top of it.
        if trails:
            points = trails.get(person.track_id)
            if points and len(points) > 1:
                path = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
                cv2.polylines(canvas, [path], False, color, max(1, thickness - 1), cv2.LINE_AA)

        foot = (int(round(person.foot_point[0])), int(round(person.foot_point[1])))
        cv2.circle(canvas, foot, max(2, thickness + 1), color, -1)

        if show_velocity and person.speed_px_per_s > 1.0:
            vx, vy = person.velocity_px_per_s
            tip = (
                int(round(foot[0] + vx * velocity_scale)),
                int(round(foot[1] + vy * velocity_scale)),
            )
            cv2.arrowedLine(canvas, foot, tip, color, thickness + 1, cv2.LINE_AA, tipLength=0.3)

        # Label: ID always, speed when the camera is calibrated.
        if person.speed_m_per_s is not None:
            label = f"#{person.track_id}  {person.speed_m_per_s:.1f}m/s"
        else:
            label = f"#{person.track_id}  {person.speed_px_per_s:.0f}px/s"

        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)
        label_bottom = y1 if y1 - text_h - baseline >= 0 else y2 + text_h + baseline
        label_top = label_bottom - text_h - baseline

        cv2.rectangle(canvas, (x1, label_top), (x1 + text_w + 4, label_bottom), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_bottom - baseline),
            _FONT,
            font_scale,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_stats_panel(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a translucent panel of text in the top-left corner. Modifies in place."""
    if not lines:
        return frame

    height, width = frame.shape[:2]
    font_scale = max(0.4, min(width, height) / 1200)
    line_height = int(28 * font_scale / 0.5)
    padding = 10

    panel_w = max(
        cv2.getTextSize(line, _FONT, font_scale, 1)[0][0] for line in lines
    ) + padding * 2
    panel_h = line_height * len(lines) + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (padding, padding + line_height * (index + 1) - 8),
            _FONT,
            font_scale,
            _PANEL_TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    return frame


__all__ = ["draw_detections", "draw_stats_panel", "draw_tracks", "track_color"]
