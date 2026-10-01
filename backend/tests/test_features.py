"""Unit tests for the feature extraction pipeline.

Entirely synthetic: TrackedFrame objects are constructed directly, so no video, no model
weights, and no database are involved. Every expected value below is derived by hand in
the test rather than by re-running the implementation, so a wrong implementation fails
instead of agreeing with itself.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.vision.detection import BoundingBox
from app.vision.features import (
    FeatureExtractor,
    circular_variance,
    optical_flow_direction_entropy,
    shannon_entropy,
)
from app.vision.tracking import TrackedFrame, TrackedPerson

START = datetime(2026, 3, 4, 14, 30, 0, tzinfo=timezone.utc)  # a Wednesday, 14:xx
AREA = 100.0  # m^2, so density == person_count / 100
FPS = 10.0


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------


def make_person(
    track_id: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    position: tuple[float, float] = (100.0, 200.0),
    pixels_per_meter: float | None = None,
) -> TrackedPerson:
    """A TrackedPerson with an exact, known velocity."""
    speed_px = math.hypot(*velocity)
    x, y = position
    bbox = BoundingBox(x1=x - 10, y1=y - 40, x2=x + 10, y2=y, confidence=0.9)
    return TrackedPerson(
        track_id=track_id,
        bbox=bbox,
        centroid=bbox.center,
        foot_point=bbox.foot_point,
        velocity_px_per_s=velocity,
        speed_px_per_s=speed_px,
        speed_m_per_s=(speed_px / pixels_per_meter) if pixels_per_meter else None,
        age=10,
    )


def make_frames(
    per_frame_people: list[list[TrackedPerson]], start_time: float = 0.0
) -> list[TrackedFrame]:
    """Turn a list-of-lists into a TrackedFrame stream at FPS."""
    frames = []
    for index, people in enumerate(per_frame_people):
        source_time = start_time + index / FPS
        frames.append(
            TrackedFrame(
                frame_index=index,
                timestamp=START + timedelta(seconds=source_time),
                source_time=source_time,
                people=people,
            )
        )
    return frames


def extractor(**kwargs) -> FeatureExtractor:
    kwargs.setdefault("area_sq_meters", AREA)
    kwargs.setdefault("window_seconds", 5.0)
    kwargs.setdefault("compute_optical_flow", False)
    return FeatureExtractor(**kwargs)


# --------------------------------------------------------------------------------------
# density
# --------------------------------------------------------------------------------------


def test_density_is_person_count_over_area():
    # 20 people in every frame of a 5s window, 100 m^2 -> 0.2 people/m^2
    frames = make_frames([[make_person(i) for i in range(20)] for _ in range(50)])
    (features,) = list(extractor().extract(frames))

    assert features.person_count == pytest.approx(20.0)
    assert features.density == pytest.approx(0.2)
    assert features.unique_track_count == 20
    assert features.frame_count == 50


def test_person_count_is_the_per_frame_mean_not_distinct_ids():
    """10 people per frame, but everyone is replaced halfway: density must stay 10/area."""
    first_half = [[make_person(i) for i in range(10)] for _ in range(25)]
    second_half = [[make_person(100 + i) for i in range(10)] for _ in range(25)]
    frames = make_frames(first_half + second_half)

    (features,) = list(extractor().extract(frames))

    assert features.person_count == pytest.approx(10.0)
    assert features.density == pytest.approx(0.1)
    # 20 distinct people passed through, but only 10 were ever present at once.
    assert features.unique_track_count == 20


def test_density_rate_of_change_across_windows():
    # window 0: 10 people (density 0.10); window 1: 30 people (density 0.30)
    window0 = [[make_person(i) for i in range(10)] for _ in range(50)]
    window1 = [[make_person(i) for i in range(30)] for _ in range(50)]
    frames = make_frames(window0 + window1)

    first, second = list(extractor().extract(frames))

    # No previous window to compare against.
    assert first.density_rate_of_change is None
    assert first.density_delta is None

    assert second.density_delta == pytest.approx(0.2)
    # Windows are 5s apart (measured end-to-end), so 0.2 / 5 = 0.04 per second.
    assert second.density_rate_of_change == pytest.approx(second.density_delta / 5.0, rel=0.05)
    assert second.density_rate_of_change > 0


def test_density_rate_of_change_is_negative_when_crowd_disperses():
    window0 = [[make_person(i) for i in range(40)] for _ in range(50)]
    window1 = [[make_person(i) for i in range(10)] for _ in range(50)]
    frames = make_frames(window0 + window1)

    _, second = list(extractor().extract(frames))

    assert second.density_delta == pytest.approx(-0.3)
    assert second.density_rate_of_change < 0


# --------------------------------------------------------------------------------------
# mean flow speed
# --------------------------------------------------------------------------------------


def test_mean_flow_speed_px_is_mean_over_tracks():
    # Three people at 10, 20, 30 px/s -> mean 20 px/s
    people = [
        make_person(1, velocity=(10.0, 0.0)),
        make_person(2, velocity=(20.0, 0.0)),
        make_person(3, velocity=(30.0, 0.0)),
    ]
    frames = make_frames([people for _ in range(50)])

    (features,) = list(extractor().extract(frames))

    assert features.mean_flow_speed_px_s == pytest.approx(20.0)
    # Uncalibrated: m/s is left as None rather than guessed.
    assert features.mean_flow_speed_m_s is None


def test_mean_flow_speed_converts_to_m_s_when_calibrated():
    ppm = 50.0  # 50 px per metre
    people = [
        make_person(1, velocity=(100.0, 0.0), pixels_per_meter=ppm),  # 2.0 m/s
        make_person(2, velocity=(200.0, 0.0), pixels_per_meter=ppm),  # 4.0 m/s
    ]
    frames = make_frames([people for _ in range(50)])

    (features,) = list(extractor(pixels_per_meter=ppm).extract(frames))

    assert features.mean_flow_speed_px_s == pytest.approx(150.0)
    assert features.mean_flow_speed_m_s == pytest.approx(3.0)


def test_each_track_counts_once_regardless_of_how_long_it_is_visible():
    """A person on screen for the whole window must not outweigh a passer-by."""
    long_stayer = make_person(1, velocity=(10.0, 0.0))
    brief = make_person(2, velocity=(30.0, 0.0))

    # Track 1 in all 50 frames, track 2 in only 5.
    per_frame = [[long_stayer] + ([brief] if i < 5 else []) for i in range(50)]
    (features,) = list(extractor().extract(make_frames(per_frame)))

    # Per-track mean of (10, 30) = 20. A per-detection mean would be ~11.8.
    assert features.mean_flow_speed_px_s == pytest.approx(20.0)


# --------------------------------------------------------------------------------------
# flow direction variance
# --------------------------------------------------------------------------------------


def test_circular_variance_of_identical_headings_is_zero():
    assert circular_variance([(1.0, 0.0)] * 5) == pytest.approx(0.0)


def test_circular_variance_of_opposing_headings_is_one():
    # Two exactly opposite directions cancel: resultant length 0 -> variance 1.
    assert circular_variance([(1.0, 0.0), (-1.0, 0.0)]) == pytest.approx(1.0)


def test_circular_variance_handles_the_angle_wraparound():
    """Headings at +1 and -1 degrees are nearly identical, so variance must be ~0.

    Averaging the angles as plain numbers (1 and 359) would give 180 degrees and a large
    spurious variance; this is the case that catches that bug.
    """
    a = math.radians(1.0)
    b = math.radians(-1.0)
    variance = circular_variance([(math.cos(a), math.sin(a)), (math.cos(b), math.sin(b))])
    assert variance == pytest.approx(0.0, abs=1e-3)


def test_circular_variance_of_evenly_scattered_headings_is_one():
    # Four headings at 90-degree spacing cancel exactly.
    headings = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    assert circular_variance(headings) == pytest.approx(1.0)


def test_circular_variance_of_nothing_is_none():
    assert circular_variance([]) is None


def test_flow_direction_variance_low_when_crowd_moves_together():
    people = [make_person(i, velocity=(50.0, 0.0)) for i in range(10)]
    frames = make_frames([people for _ in range(50)])

    (features,) = list(extractor().extract(frames))

    assert features.flow_direction_variance == pytest.approx(0.0, abs=1e-6)


def test_flow_direction_variance_high_when_crowd_moves_every_which_way():
    """Counter-flow: half moving right, half moving left."""
    people = [make_person(i, velocity=(50.0, 0.0)) for i in range(5)]
    people += [make_person(5 + i, velocity=(-50.0, 0.0)) for i in range(5)]
    frames = make_frames([people for _ in range(50)])

    (features,) = list(extractor().extract(frames))

    assert features.flow_direction_variance == pytest.approx(1.0, abs=1e-6)


def test_flow_direction_variance_ignores_stationary_people():
    """A stationary person has no meaningful heading and must not add variance."""
    moving = [make_person(i, velocity=(50.0, 0.0)) for i in range(3)]
    still = [make_person(10 + i, velocity=(0.0, 0.0)) for i in range(20)]
    frames = make_frames([moving + still for _ in range(50)])

    (features,) = list(extractor().extract(frames))

    # Only the three movers contribute, and they agree.
    assert features.flow_direction_variance == pytest.approx(0.0, abs=1e-6)


def test_flow_direction_variance_is_none_when_nobody_moves():
    people = [make_person(i, velocity=(0.0, 0.0)) for i in range(10)]
    (features,) = list(extractor().extract(make_frames([people for _ in range(50)])))

    assert features.flow_direction_variance is None


# --------------------------------------------------------------------------------------
# stop ratio
# --------------------------------------------------------------------------------------


def test_stop_ratio_counts_near_stationary_tracks():
    # 3 of 10 people are stationary -> 0.3
    stopped = [make_person(i, velocity=(0.0, 0.0)) for i in range(3)]
    moving = [make_person(3 + i, velocity=(50.0, 0.0)) for i in range(7)]
    frames = make_frames([stopped + moving for _ in range(50)])

    (features,) = list(extractor().extract(frames))

    assert features.stop_ratio == pytest.approx(0.3)


def test_stop_ratio_is_one_when_everyone_has_jammed():
    people = [make_person(i, velocity=(0.0, 0.0)) for i in range(8)]
    (features,) = list(extractor().extract(make_frames([people for _ in range(50)])))

    assert features.stop_ratio == pytest.approx(1.0)


def test_stop_ratio_respects_the_threshold_boundary():
    """Speed just under the threshold counts as stopped; just over does not."""
    ext = extractor(stationary_speed=10.0)
    below = [make_person(1, velocity=(9.0, 0.0))]
    above = [make_person(2, velocity=(11.0, 0.0))]
    frames = make_frames([below + above for _ in range(50)])

    (features,) = list(ext.extract(frames))

    assert features.stop_ratio == pytest.approx(0.5)


def test_newly_stopped_ratio_detects_a_crowd_seizing_up():
    """Four people moving in window 0; two of them stop in window 1."""
    window0 = [[make_person(i, velocity=(50.0, 0.0)) for i in range(4)] for _ in range(50)]
    window1 = [
        [make_person(0, velocity=(0.0, 0.0)), make_person(1, velocity=(0.0, 0.0))]
        + [make_person(2, velocity=(50.0, 0.0)), make_person(3, velocity=(50.0, 0.0))]
        for _ in range(50)
    ]
    first, second = list(extractor().extract(make_frames(window0 + window1)))

    # Nothing to compare the first window against.
    assert first.newly_stopped_ratio is None
    # All four tracks carried over; two transitioned from moving to stopped.
    assert second.newly_stopped_ratio == pytest.approx(0.5)
    assert second.stop_ratio == pytest.approx(0.5)


def test_newly_stopped_ratio_ignores_people_who_were_already_stopped():
    """Already-stationary people are not 'newly' stopped."""
    stopped_throughout = [make_person(i, velocity=(0.0, 0.0)) for i in range(2)]
    window0 = [stopped_throughout + [make_person(9, velocity=(50.0, 0.0))] for _ in range(50)]
    window1 = [stopped_throughout + [make_person(9, velocity=(50.0, 0.0))] for _ in range(50)]

    _, second = list(extractor().extract(make_frames(window0 + window1)))

    assert second.stop_ratio == pytest.approx(2 / 3)
    # Nobody changed state.
    assert second.newly_stopped_ratio == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# optical flow entropy
# --------------------------------------------------------------------------------------


def test_shannon_entropy_of_a_single_bin_is_zero():
    assert shannon_entropy(np.array([10.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_shannon_entropy_of_a_uniform_histogram_is_log2_of_bin_count():
    # 8 equally likely outcomes -> exactly 3 bits.
    assert shannon_entropy(np.ones(8)) == pytest.approx(3.0)
    assert shannon_entropy(np.ones(16)) == pytest.approx(4.0)


def test_shannon_entropy_of_two_equal_bins_is_one_bit():
    assert shannon_entropy(np.array([5.0, 5.0])) == pytest.approx(1.0)


def test_shannon_entropy_of_an_empty_histogram_is_zero():
    assert shannon_entropy(np.zeros(8)) == pytest.approx(0.0)


def test_optical_flow_entropy_is_low_for_uniform_motion():
    """A rigid horizontal shift puts all flow in one direction bin -> near-zero entropy."""
    rng = np.random.default_rng(0)
    first = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    # Shift right by 4 px; np.roll keeps the texture identical so flow is unambiguous.
    second = np.roll(first, 4, axis=1)

    entropy = optical_flow_direction_entropy(first, second, bins=16)

    assert entropy < 1.5, f"uniform motion should concentrate direction, got {entropy} bits"


def test_optical_flow_entropy_is_higher_for_disordered_motion():
    """Opposed motion in different regions spreads flow across bins."""
    rng = np.random.default_rng(1)
    first = rng.integers(0, 255, (120, 160), dtype=np.uint8)

    second = first.copy()
    # Top half shifts right, bottom half shifts left: two opposing directions.
    second[:60] = np.roll(first[:60], 4, axis=1)
    second[60:] = np.roll(first[60:], -4, axis=1)

    uniform = optical_flow_direction_entropy(first, np.roll(first, 4, axis=1), bins=16)
    disordered = optical_flow_direction_entropy(first, second, bins=16)

    assert disordered > uniform


def test_optical_flow_entropy_of_a_static_scene_is_zero():
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 255, (120, 160), dtype=np.uint8)

    assert optical_flow_direction_entropy(frame, frame.copy(), bins=16) == pytest.approx(0.0)


def test_entropy_is_none_when_frames_are_unavailable():
    """Synthetic frames carry no image data, so entropy cannot be computed."""
    people = [make_person(1, velocity=(10.0, 0.0))]
    ext = extractor(compute_optical_flow=True)

    (features,) = list(ext.extract(make_frames([people for _ in range(50)])))

    assert features.optical_flow_entropy is None


def test_entropy_is_computed_when_frames_are_present():
    """With real image data attached, the window produces an entropy value."""
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)

    frames = []
    for index in range(20):
        moved = np.roll(base, 3 * index, axis=1)
        frames.append(
            TrackedFrame(
                frame_index=index,
                timestamp=START + timedelta(seconds=index / FPS),
                source_time=index / FPS,
                people=[make_person(1, velocity=(30.0, 0.0))],
                frame=moved,
            )
        )

    (features,) = list(extractor(compute_optical_flow=True).extract(frames))

    assert features.optical_flow_entropy is not None
    assert 0.0 <= features.optical_flow_entropy <= math.log2(16)


# --------------------------------------------------------------------------------------
# historical deviation
# --------------------------------------------------------------------------------------


class _FakeBaselineExtractor(FeatureExtractor):
    """FeatureExtractor with the baseline query stubbed, so no database is needed."""

    def __init__(self, baseline: tuple[float, float] | None, **kwargs):
        self._baseline = baseline
        super().__init__(**kwargs)

    def _load_baseline(self, hour: int, weekday: int):
        self.last_lookup = (hour, weekday)
        return self._baseline

    def _load_camera_settings(self, camera_id: int) -> None:
        # camera_id is passed so the baseline machinery runs, but the tests must not
        # touch the database -- camera settings come from kwargs only.
        return None


def test_historical_deviation_is_the_z_score_against_the_baseline():
    # 30 people / 100 m^2 = 0.30 density. Baseline mean 0.10, stddev 0.05 -> z = 4.0
    ext = _FakeBaselineExtractor(
        baseline=(0.10, 0.05),
        camera_id=1,
        area_sq_meters=AREA,
        compute_optical_flow=False,
    )
    frames = make_frames([[make_person(i) for i in range(30)] for _ in range(50)])

    (features,) = list(ext.extract(frames))

    assert features.historical_deviation == pytest.approx(4.0)


def test_historical_deviation_is_negative_for_a_quiet_window():
    # 0.05 density vs baseline mean 0.10, stddev 0.05 -> z = -1.0
    ext = _FakeBaselineExtractor(
        baseline=(0.10, 0.05), camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    frames = make_frames([[make_person(i) for i in range(5)] for _ in range(50)])

    (features,) = list(ext.extract(frames))

    assert features.historical_deviation == pytest.approx(-1.0)


def test_historical_deviation_uses_the_windows_own_hour_and_weekday():
    ext = _FakeBaselineExtractor(
        baseline=(0.1, 0.05), camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    list(ext.extract(make_frames([[make_person(1)] for _ in range(50)])))

    # START is Wednesday 2026-03-04 at 14:30 UTC -> hour 14, weekday 2 (Mon=0).
    assert ext.last_lookup == (14, 2)


def test_historical_deviation_defaults_to_zero_without_a_baseline():
    """Cold start: a fresh camera has no history, and must not break scoring."""
    ext = _FakeBaselineExtractor(
        baseline=None, camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    (features,) = list(ext.extract(make_frames([[make_person(1)] for _ in range(50)])))

    assert features.historical_deviation == pytest.approx(0.0)


def test_historical_deviation_is_zero_when_baseline_stddev_is_zero():
    """A zero-variance baseline would otherwise make every deviation infinite."""
    ext = _FakeBaselineExtractor(
        baseline=(0.1, 0.0), camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    (features,) = list(ext.extract(make_frames([[make_person(1)] for _ in range(50)])))

    assert features.historical_deviation == pytest.approx(0.0)


def test_historical_deviation_defaults_to_zero_without_a_camera_id():
    (features,) = list(extractor().extract(make_frames([[make_person(1)] for _ in range(50)])))
    assert features.historical_deviation == pytest.approx(0.0)


def test_the_cold_start_default_is_configurable():
    """A deployment that prefers a different neutral can say so."""
    ext = _FakeBaselineExtractor(
        baseline=None,
        camera_id=1,
        area_sq_meters=AREA,
        compute_optical_flow=False,
        no_baseline_deviation=-1.0,
    )
    (features,) = list(ext.extract(make_frames([[make_person(1)] for _ in range(50)])))

    assert features.historical_deviation == pytest.approx(-1.0)


def test_a_real_baseline_still_wins_over_the_default():
    """The fallback must not mask a genuine z-score."""
    ext = _FakeBaselineExtractor(
        baseline=(0.10, 0.05), camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    frames = make_frames([[make_person(i) for i in range(30)] for _ in range(50)])
    (features,) = list(ext.extract(frames))

    assert features.historical_deviation == pytest.approx(4.0)


# --------------------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------------------


def test_stream_is_split_on_window_boundaries():
    # 15 seconds at 10 fps with a 5s window -> exactly 3 windows.
    frames = make_frames([[make_person(1)] for _ in range(150)])
    windows = list(extractor(window_seconds=5.0).extract(frames))

    assert len(windows) == 3
    assert [w.frame_count for w in windows] == [50, 50, 50]


def test_window_length_is_configurable():
    frames = make_frames([[make_person(1)] for _ in range(150)])

    assert len(list(extractor(window_seconds=1.0).extract(frames))) == 15
    assert len(list(extractor(window_seconds=3.0).extract(frames))) == 5


def test_trailing_partial_window_is_emitted():
    """A short clip must still produce features rather than being silently dropped."""
    # 7 seconds -> one full window plus a 2s remainder.
    frames = make_frames([[make_person(1)] for _ in range(70)])
    windows = list(extractor(window_seconds=5.0).extract(frames))

    assert len(windows) == 2
    assert windows[0].frame_count == 50
    assert windows[1].frame_count == 20


def test_empty_stream_yields_nothing():
    assert list(extractor().extract([])) == []


def test_frames_with_no_people_still_produce_a_window():
    """An empty scene is a real measurement -- density zero -- not missing data."""
    frames = make_frames([[] for _ in range(50)])
    (features,) = list(extractor().extract(frames))

    assert features.person_count == 0.0
    assert features.density == 0.0
    assert features.unique_track_count == 0
    assert features.stop_ratio == 0.0
    assert features.flow_direction_variance is None


def test_window_timestamps_track_the_source_clock():
    frames = make_frames([[make_person(1)] for _ in range(100)])
    first, second = list(extractor(window_seconds=5.0).extract(frames))

    assert first.window_start == START
    assert (second.window_start - first.window_start).total_seconds() == pytest.approx(5.0)


# --------------------------------------------------------------------------------------
# construction / validation
# --------------------------------------------------------------------------------------


def test_area_is_required():
    with pytest.raises(ValueError, match="area_sq_meters"):
        FeatureExtractor(area_sq_meters=None)


def test_zero_area_is_rejected():
    with pytest.raises(ValueError, match="area_sq_meters"):
        FeatureExtractor(area_sq_meters=0.0)


def test_window_seconds_must_be_positive():
    with pytest.raises(ValueError, match="window_seconds"):
        FeatureExtractor(area_sq_meters=AREA, window_seconds=0)


def test_thresholds_switch_units_with_calibration():
    """An m/s threshold applied to px/s speeds would misclassify everyone."""
    uncalibrated = extractor()
    calibrated = extractor(pixels_per_meter=50.0)

    assert uncalibrated.calibrated is False
    assert calibrated.calibrated is True
    assert uncalibrated.stationary_speed > calibrated.stationary_speed


def test_model_features_exposes_the_seven_inputs():
    frames = make_frames([[make_person(1, velocity=(50.0, 0.0))] for _ in range(50)])
    (features,) = list(extractor().extract(frames))

    assert set(features.model_features()) == {
        "density",
        "density_rate_of_change",
        "mean_flow_speed",
        "flow_direction_variance",
        "optical_flow_entropy",
        "stop_ratio",
        "historical_deviation",
    }
