"""Run person tracking over a video source and write an annotated copy.

Visual verification of track stability -- each person gets a consistently coloured box
and ID, a motion trail, and a velocity arrow. Nothing here touches the database.

Usage (from backend/, with the venv active):

    python -m scripts.test_tracking --source data/samples/crowd.mp4
    python -m scripts.test_tracking --source data/samples/crowd.mp4 --pixels-per-meter 42
    python -m scripts.test_tracking --source data/samples/crowd.mp4 --camera CAM-1
    python -m scripts.test_tracking --source 0 --max-frames 200        # webcam

What to look for in the output:

  * A person keeps one colour and one ID for as long as they are visible. A colour
    change mid-walk is an identity switch -- the thing this script exists to expose.
  * ID numbers stay low. Rapidly climbing IDs mean tracks are being dropped and
    recreated rather than maintained.
  * Trails are smooth. Jagged trails mean the detector is jittering.
  * Arrows point where people are actually walking, and lengthen with pace.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from app.vision.annotate import draw_stats_panel, draw_tracks
from app.vision.source import VideoSourceError, open_source
from app.vision.tracking import PersonTracker

DEFAULT_OUTPUT_DIR = Path("data/output")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("test_tracking")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ByteTrack person tracking over a video source and save an "
        "annotated output video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="video file path, webcam index (e.g. 0), or rtsp:// URL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output video path (default: data/output/<source name>_tracked.mp4)",
    )
    parser.add_argument("--model", default=None, help="Ultralytics weights to use")
    parser.add_argument(
        "--tracker",
        default=None,
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Ultralytics tracker config",
    )
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold")
    parser.add_argument(
        "--max-dimension", type=int, default=None, help="inference resolution"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--device", default=None, help="'cpu', 'cuda', ...")
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="ground-plane calibration; without it speeds are reported in px/s",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="camera NAME in the database to read pixels_per_meter from "
        "(overridden by --pixels-per-meter)",
    )
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=2.0,
        help="how much motion history to draw behind each person",
    )
    return parser


def resolve_pixels_per_meter(args: argparse.Namespace) -> float | None:
    """Explicit flag wins; otherwise look the camera up in the database."""
    if args.pixels_per_meter is not None:
        return args.pixels_per_meter

    if not args.camera:
        return None

    # Imported lazily so the script still runs with no database available.
    from app.core.database import CAMERAS, db

    camera = db[CAMERAS].find_one({"name": args.camera})

    if camera is None:
        raise SystemExit(f"No camera named {args.camera!r} in the database.")
    if camera.get("pixels_per_meter") is None:
        logger.warning(
            "Camera %s has no pixels_per_meter set; speeds will be in px/s.", args.camera
        )
        return None

    logger.info(
        "Using calibration from camera %s: %.1f px/m", args.camera, camera["pixels_per_meter"]
    )
    return camera["pixels_per_meter"]


def resolve_output_path(source: str, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        stem = Path(source).stem if not source.isdigit() else f"webcam{source}"
        path = DEFAULT_OUTPUT_DIR / f"{stem or 'stream'}_tracked.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = build_parser().parse_args()

    if args.max_frames is None and (args.source.isdigit() or "://" in args.source):
        logger.warning(
            "Live source with no --max-frames: this will run until interrupted (Ctrl+C)."
        )

    pixels_per_meter = resolve_pixels_per_meter(args)

    tracker = PersonTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_dimension=args.max_dimension,
        device=args.device,
        pixels_per_meter=pixels_per_meter,
        tracker_config=args.tracker,
    )

    output_path = resolve_output_path(args.source, args.output)

    try:
        source = open_source(args.source)
        source.open()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return 1

    fps = source.fps / max(1, args.stride)
    trail_length = max(2, int(fps * args.trail_seconds))
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))

    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    total_people = 0
    peak_people = 0
    track_lifetimes: dict[int, int] = defaultdict(int)
    speeds: list[float] = []
    started = time.perf_counter()

    logger.info("Loading model (tracker: %s)...", tracker.tracker_config)

    try:
        for tracked in tracker.track(
            source, max_frames=args.max_frames, stride=args.stride
        ):
            frames_processed += 1
            total_people += tracked.person_count
            peak_people = max(peak_people, tracked.person_count)

            for person in tracked.people:
                trails[person.track_id].append(person.foot_point)
                track_lifetimes[person.track_id] += 1
                if person.speed_m_per_s is not None:
                    speeds.append(person.speed_m_per_s)
                else:
                    speeds.append(person.speed_px_per_s)

            annotated = draw_tracks(
                tracked.frame,
                tracked.people,
                trails={tid: list(pts) for tid, pts in trails.items()},
            )

            unit = "m/s" if pixels_per_meter else "px/s"
            mean_speed = (
                tracked.mean_speed_m_per_s()
                if pixels_per_meter
                else (
                    sum(p.speed_px_per_s for p in tracked.people) / tracked.person_count
                    if tracked.person_count
                    else None
                )
            )
            draw_stats_panel(
                annotated,
                [
                    f"frame {tracked.frame_index}  t={tracked.source_time:.1f}s",
                    f"tracked: {tracked.person_count}   ids seen: {len(track_lifetimes)}",
                    f"mean speed: {mean_speed:.2f} {unit}" if mean_speed is not None
                    else f"mean speed: -- {unit}",
                ],
            )

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    logger.error("Could not open video writer for %s", output_path)
                    return 1

            writer.write(annotated)

            if frames_processed % 25 == 0:
                logger.info(
                    "  %d frames | %d tracked | %d ids so far",
                    frames_processed,
                    tracked.person_count,
                    len(track_lifetimes),
                )

    except KeyboardInterrupt:
        logger.info("Interrupted; writing out what was processed so far.")
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started

    if frames_processed == 0:
        logger.error("No frames were read from %s", args.source)
        return 1

    unit = "m/s" if pixels_per_meter else "px/s"
    lifetimes = sorted(track_lifetimes.values(), reverse=True)
    # A track alive for only a frame or two is almost always a false start rather than a
    # person: a high share of them is the clearest signal of an unstable tracker.
    fleeting = sum(1 for length in lifetimes if length <= 2)

    print()
    print("=" * 64)
    print(f"  source            {args.source}")
    print(f"  model / tracker   {tracker.model_path} / {tracker.tracker_config}")
    print(f"  calibration       "
          f"{f'{pixels_per_meter:.1f} px/m' if pixels_per_meter else 'none (px/s only)'}")
    print("-" * 64)
    print(f"  frames processed  {frames_processed}")
    print(f"  people per frame  {total_people / frames_processed:.1f} avg, {peak_people} peak")
    print(f"  unique track ids  {len(track_lifetimes)}")
    if lifetimes:
        print(f"  track lifetime    {max(lifetimes)} frames longest, "
              f"{sum(lifetimes)/len(lifetimes):.1f} avg")
        print(f"  fleeting tracks   {fleeting} of {len(lifetimes)} lasted <= 2 frames")
    if speeds:
        print(f"  speed             {sum(speeds)/len(speeds):.2f} {unit} mean, "
              f"{max(speeds):.2f} {unit} peak")
    print(f"  throughput        {frames_processed / elapsed:.1f} fps ({elapsed:.1f}s total)")
    print("-" * 64)
    print(f"  annotated video   {output_path}")
    print("=" * 64)

    if lifetimes and len(track_lifetimes) > peak_people * 3:
        print()
        print(f"  {len(track_lifetimes)} IDs for a peak of {peak_people} people suggests")
        print("  identity switching. Try --tracker botsort.yaml (appearance re-ID),")
        print("  a larger --model, or a lower --conf so people are not lost mid-track.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
