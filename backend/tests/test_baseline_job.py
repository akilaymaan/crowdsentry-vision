"""Tests for baseline computation and the cold-start path.

The SQL aggregate itself is exercised against the real database by hand (see the README);
what is pinned here is the slot arithmetic both sides depend on, the cold-start fallback,
and the cache-expiry behaviour that decides whether a running worker ever notices a
newly-written baseline.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.services.baseline_job import BaselineResult
from app.vision.features import FeatureExtractor

from .test_features import AREA, make_frames, make_person


# --------------------------------------------------------------------------------------
# slot key arithmetic
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("moment", "hour", "weekday"),
    [
        # 2026-09-04 is a Friday; Python weekday() has Monday as 0, so Friday is 4.
        (datetime(2026, 9, 4, 11, 30, tzinfo=timezone.utc), 11, 4),
        (datetime(2026, 9, 4, 0, 5, tzinfo=timezone.utc), 0, 4),
        (datetime(2026, 9, 4, 23, 59, tzinfo=timezone.utc), 23, 4),
        # 2026-09-07 is a Monday -> 0; 2026-09-06 a Sunday -> 6.
        (datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc), 8, 0),
        (datetime(2026, 9, 6, 8, 0, tzinfo=timezone.utc), 8, 6),
    ],
)
def test_python_slot_key_matches_the_documented_convention(moment, hour, weekday):
    """The job groups with Postgres `isodow - 1`; this is the other half of that pact.

    If these two ever disagree, every z-score is computed against the wrong slot and
    nothing errors -- the numbers just quietly describe a different hour.
    """
    assert moment.hour == hour
    assert moment.weekday() == weekday


def test_postgres_isodow_convention_matches_python_weekday():
    """isodow is 1=Monday..7=Sunday, so isodow-1 lines up with weekday()."""
    for day_offset in range(7):
        moment = datetime(2026, 9, 7, tzinfo=timezone.utc) + timedelta(days=day_offset)
        iso_dow = moment.isoweekday()          # 1=Mon..7=Sun, same as Postgres isodow
        assert iso_dow - 1 == moment.weekday()


# --------------------------------------------------------------------------------------
# cold start
# --------------------------------------------------------------------------------------


class _StubExtractor(FeatureExtractor):
    """FeatureExtractor with the baseline query stubbed and counted."""

    def __init__(self, baseline=None, **kwargs):
        self._baseline = baseline
        self.loads = 0
        super().__init__(**kwargs)

    def _load_baseline(self, hour, weekday):
        self.loads += 1
        return self._baseline

    def _load_camera_settings(self, camera_id: int) -> None:
        # camera_id is passed so the baseline machinery runs, but the tests must not
        # touch the database -- camera settings come from kwargs only.
        return None


def _window(ext):
    (features,) = list(ext.extract(make_frames([[make_person(1)] for _ in range(50)])))
    return features


def test_cold_start_yields_zero_not_a_crash():
    """A brand-new deployment has no baselines at all; scoring must still work."""
    ext = _StubExtractor(
        baseline=None, camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    assert _window(ext).historical_deviation == pytest.approx(0.0)


def test_cold_start_default_feeds_the_model_a_real_number():
    """model_features() must not hand the model a None for this feature."""
    ext = _StubExtractor(
        baseline=None, camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    value = _window(ext).model_features()["historical_deviation"]

    assert value is not None
    assert value == pytest.approx(0.0)


def test_a_baseline_overrides_the_cold_start_default():
    # 1 person / 100 m^2 = 0.01 density; baseline mean 0.005, sd 0.0025 -> z = 2.0
    ext = _StubExtractor(
        baseline=(0.005, 0.0025),
        camera_id=1,
        area_sq_meters=AREA,
        compute_optical_flow=False,
    )
    assert _window(ext).historical_deviation == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# cache expiry
# --------------------------------------------------------------------------------------


def test_baseline_lookups_are_cached_within_the_ttl():
    ext = _StubExtractor(
        baseline=(0.005, 0.0025),
        camera_id=1,
        area_sq_meters=AREA,
        compute_optical_flow=False,
    )
    for _ in range(3):
        _window(ext)

    # Same slot every time, so one query serves all three windows.
    assert ext.loads == 1


def test_an_expired_cache_entry_is_reloaded(monkeypatch):
    """The important case: the worker started before the nightly job ran.

    Without expiry the miss is cached for the life of the process, and the camera
    reports the cold-start default forever even though real baselines now exist.
    """
    monkeypatch.setattr(settings, "features_baseline_cache_seconds", 0.0)

    ext = _StubExtractor(
        baseline=None, camera_id=1, area_sq_meters=AREA, compute_optical_flow=False
    )
    assert _window(ext).historical_deviation == pytest.approx(0.0)
    loads_after_miss = ext.loads

    # The baseline job runs, and the slot now has data.
    ext._baseline = (0.005, 0.0025)
    time.sleep(0.01)

    assert _window(ext).historical_deviation == pytest.approx(2.0)
    assert ext.loads > loads_after_miss


def test_cache_ttl_is_configurable():
    assert settings.features_baseline_cache_seconds > 0


# --------------------------------------------------------------------------------------
# result reporting
# --------------------------------------------------------------------------------------


def test_result_summary_distinguishes_a_dry_run():
    written = BaselineResult(cameras_processed=3, slots_written=12, observations_considered=500)
    planned = BaselineResult(
        cameras_processed=3, slots_written=12, observations_considered=500, dry_run=True
    )

    assert "wrote 12 slots" in written.summary()
    assert "would write 12 slots" in planned.summary()


def test_empty_result_reports_nothing_written():
    assert "wrote 0 slots" in BaselineResult().summary()


# --------------------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------------------


def test_lookback_default_is_several_weeks():
    """Long enough to cover each weekday slot more than once."""
    assert settings.baseline_lookback_days >= 14


def test_min_samples_default_rules_out_a_degenerate_stddev():
    """A slot needs more than two points before its stddev means anything."""
    assert settings.baseline_min_samples >= 2
