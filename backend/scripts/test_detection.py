"""Run person detection over a video source and write an annotated copy.

Purely a visual sanity check for the detector -- nothing here touches the database.

Usage (from backend/, with the venv active):

    python -m scripts.test_detection --source data/samples/crowd.mp4
    python -m scripts.test_detection --source data/samples/crowd.mp4 --conf 0.25 --max-frames 300
    python -m scripts.test_detection --source 0 --max-frames 200        # webcam
    python -m scripts.test_detection --source rtsp://user:pass@host/stream --max-frames 200

The annotated video defaults to ``data/output/<source name>_detected.mp4``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from app.vision.annotate import draw_detections, draw_stats_panel
from app.vision.detection import PersonDetector
from app.vision.source import VideoSourceError, open_source

DEFAULT_OUTPUT_DIR = Path("data/output")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("test_detection")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 person detection over a video source and save an "
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
        help="output video path (default: data/output/<source name>_detected.mp4)",
    )
    parser.add_argument("--model", default=None, help="Ultralytics weights to use")
    parser.add_argument(
        "--conf", type=float, default=None, help="confidence threshold (0-1)"
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="downscale frames to this longer-side size before inference",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many processed frames (required to end a live source)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="process every Nth frame",
    )
    parser.add_argument(
        "--device", default=None, help="'cpu', 'cuda', '0', ... (default: auto)"
    )
    return parser


def resolve_output_path(source: str, override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        stem = Path(source).stem if not source.isdigit() else f"webcam{source}"
        # A source like an rtsp:// URL has no usable stem.
        stem = stem or "stream"
        path = DEFAULT_OUTPUT_DIR / f"{stem}_detected.mp4"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = build_parser().parse_args()

    if args.max_frames is None and (args.source.isdigit() or "://" in args.source):
        logger.warning(
            "Live source with no --max-frames: this will run until interrupted (Ctrl+C)."
        )

    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_dimension=args.max_dimension,
        device=args.device,
    )

    output_path = resolve_output_path(args.source, args.output)

    try:
        source = open_source(args.source)
        source.open()
    except VideoSourceError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Loading model (first run downloads the weights)...")
    detector.warmup()

    writer: cv2.VideoWriter | None = None
    frames_processed = 0
    total_detections = 0
    peak_detections = 0
    inference_seconds = 0.0
    started = time.perf_counter()

    try:
        with source:
            fps = source.fps / max(1, args.stride)
            total = source.frame_count
            logger.info(
                "Source %s: %dx%d @ %.2f fps%s",
                source.describe(),
                source.width,
                source.height,
                source.fps,
                f", {total} frames" if total else "",
            )

            for frame_index, frame in source.frames(
                max_frames=args.max_frames, stride=args.stride
            ):
                inference_started = time.perf_counter()
                detections = detector.detect(frame)
                inference_seconds += time.perf_counter() - inference_started

                frames_processed += 1
                total_detections += len(detections)
                peak_detections = max(peak_detections, len(detections))

                annotated = draw_detections(frame, detections)
                draw_stats_panel(
                    annotated,
                    [
                        f"frame {frame_index}",
                        f"people: {len(detections)}",
                        f"conf >= {detector.confidence_threshold:.2f}",
                    ],
                )

                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        logger.error("Could not open video writer for %s", output_path)
                        return 1

                writer.write(annotated)

                if frames_processed % 25 == 0:
                    logger.info(
                        "  %d frames | %d people in this frame", frames_processed, len(detections)
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

    print()
    print("=" * 62)
    print(f"  source            {args.source}")
    print(f"  model             {detector.model_path}  (device: {detector.device or 'auto'})")
    print(f"  confidence        {detector.confidence_threshold}")
    print(f"  max dimension     {detector.max_dimension}px")
    print("-" * 62)
    print(f"  frames processed  {frames_processed}")
    print(f"  detections        {total_detections} total, "
          f"{total_detections / frames_processed:.1f} avg, {peak_detections} peak")
    print(f"  inference         {inference_seconds / frames_processed * 1000:.1f} ms/frame "
          f"({frames_processed / inference_seconds:.1f} fps)")
    print(f"  wall clock        {elapsed:.1f}s end to end")
    print("-" * 62)
    print(f"  annotated video   {output_path}")
    print("=" * 62)

    if total_detections == 0:
        print()
        print("  No people detected. If the video definitely contains people, try a")
        print("  lower --conf (e.g. 0.15) or a larger model (--model yolov8s.pt).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
