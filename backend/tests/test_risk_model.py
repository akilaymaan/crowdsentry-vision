"""Tests for the Fruin labelling rules and the prediction contract.

The labelling tests are the important ones: they pin the ground truth the whole model is
trained against, so a silent change to a threshold fails here rather than quietly
retraining a different model. The prediction tests are skipped when the model artifacts
have not been built.
"""

from __future__ import annotations

import math

import pytest

from scripts.simulate_crowd_scenarios import (
    CHAOS_ESCALATION_THRESHOLD,
    LOS_DENSITY_BOUNDS,
    RISK_LEVELS,
    chaos_score,
    classify_los,
    label_window,
)

# --------------------------------------------------------------------------------------
# Fruin Level of Service thresholds
# --------------------------------------------------------------------------------------


def test_los_density_bounds_match_fruins_area_modules():
    """Fruin's bands are defined in ft^2/person; ours must be the exact conversion.

    1 m^2 = 10.7639 ft^2, so a 35 ft^2/person module is 10.7639/35 = 0.3075 ped/m^2.
    """
    assert LOS_DENSITY_BOUNDS["A"] == pytest.approx(10.7639 / 35, rel=1e-4)
    assert LOS_DENSITY_BOUNDS["B"] == pytest.approx(10.7639 / 25, rel=1e-4)
    assert LOS_DENSITY_BOUNDS["C"] == pytest.approx(10.7639 / 15, rel=1e-4)
    assert LOS_DENSITY_BOUNDS["D"] == pytest.approx(10.7639 / 10, rel=1e-4)
    assert LOS_DENSITY_BOUNDS["E"] == pytest.approx(10.7639 / 5, rel=1e-4)


def test_los_bounds_are_the_commonly_cited_values():
    """Sanity check against the numbers quoted in the crowd-safety literature."""
    assert LOS_DENSITY_BOUNDS["A"] == pytest.approx(0.31, abs=0.01)
    assert LOS_DENSITY_BOUNDS["B"] == pytest.approx(0.43, abs=0.01)
    assert LOS_DENSITY_BOUNDS["C"] == pytest.approx(0.72, abs=0.01)
    assert LOS_DENSITY_BOUNDS["D"] == pytest.approx(1.08, abs=0.01)
    assert LOS_DENSITY_BOUNDS["E"] == pytest.approx(2.15, abs=0.01)


@pytest.mark.parametrize(
    ("density", "expected"),
    [
        (0.0, "A"),
        (0.20, "A"),
        (0.35, "B"),
        (0.60, "C"),
        (0.90, "D"),
        (1.50, "E"),
        (2.50, "F"),
        (5.00, "F"),
    ],
)
def test_classify_los_assigns_the_right_band(density, expected):
    assert classify_los(density) == expected


def test_classify_los_is_inclusive_at_the_upper_bound():
    """A density exactly on a boundary belongs to the lower (safer) band."""
    assert classify_los(LOS_DENSITY_BOUNDS["A"]) == "A"
    assert classify_los(LOS_DENSITY_BOUNDS["A"] + 1e-9) == "B"
    assert classify_los(LOS_DENSITY_BOUNDS["E"]) == "E"
    assert classify_los(LOS_DENSITY_BOUNDS["E"] + 1e-9) == "F"


def test_classify_los_is_monotonic():
    """Increasing density can never yield a safer band."""
    order = "ABCDEF"
    previous = 0
    for density in [d / 100 for d in range(0, 400, 3)]:
        current = order.index(classify_los(density))
        assert current >= previous
        previous = current


# --------------------------------------------------------------------------------------
# LOS -> risk tier mapping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("density", "expected"),
    [
        (0.10, "LOW"),        # LOS A
        (0.35, "LOW"),        # LOS B
        (0.60, "MODERATE"),   # LOS C
        (0.90, "MODERATE"),   # LOS D
        (1.50, "HIGH"),       # LOS E
        (2.50, "CRITICAL"),   # LOS F
    ],
)
def test_orderly_crowds_are_labelled_by_density_alone(density, expected):
    """With organised movement, the label is exactly the Fruin band."""
    level, _, _, escalated = label_window(density, flow_direction_variance=0.05, stop_ratio=0.0)
    assert level == expected
    assert escalated is False


# --------------------------------------------------------------------------------------
# chaos escalation
# --------------------------------------------------------------------------------------


def test_chaos_score_is_zero_for_perfectly_orderly_movement():
    assert chaos_score(0.0, 0.0) == pytest.approx(0.0)


def test_chaos_score_is_one_for_total_disorder():
    """Weights must sum to 1 so the score stays in [0, 1]."""
    assert chaos_score(1.0, 1.0) == pytest.approx(1.0)


def test_chaos_score_treats_missing_direction_variance_as_orderly():
    """None means nobody was moving, which is not evidence of disorder."""
    assert chaos_score(None, 0.5) == chaos_score(0.0, 0.5)


def test_disorder_escalates_a_moderate_crowd_one_tier():
    """The whole point of the chaos rule: moderate density, chaotic movement."""
    orderly, _, _, _ = label_window(0.60, flow_direction_variance=0.05, stop_ratio=0.0)
    chaotic, _, _, escalated = label_window(0.60, flow_direction_variance=0.95, stop_ratio=0.6)

    assert orderly == "MODERATE"
    assert chaotic == "HIGH"
    assert escalated is True


def test_escalation_moves_exactly_one_tier():
    for density, base in [(0.10, "LOW"), (0.60, "MODERATE"), (1.50, "HIGH")]:
        calm, _, _, _ = label_window(density, 0.0, 0.0)
        wild, _, _, _ = label_window(density, 1.0, 1.0)
        if calm == base and wild != base:
            assert RISK_LEVELS.index(wild) == RISK_LEVELS.index(calm) + 1


def test_escalation_is_capped_at_critical():
    """CRITICAL is the top tier; chaos cannot push past it."""
    level, _, _, _ = label_window(3.0, flow_direction_variance=1.0, stop_ratio=1.0)
    assert level == "CRITICAL"


def test_sparse_crowds_are_never_escalated():
    """One person wandering an empty plaza is not a risk, however erratic."""
    level, _, _, escalated = label_window(0.05, flow_direction_variance=1.0, stop_ratio=1.0)
    assert level == "LOW"
    assert escalated is False


def test_escalation_requires_crossing_the_threshold():
    just_under = CHAOS_ESCALATION_THRESHOLD - 0.02
    just_over = CHAOS_ESCALATION_THRESHOLD + 0.02

    # Pure stop_ratio, so chaos == 0.45 * stop_ratio; solve for the target score.
    _, _, _, low = label_window(0.60, 0.0, just_under / 0.45)
    _, _, _, high = label_window(0.60, 0.0, just_over / 0.45)

    assert low is False
    assert high is True


# --------------------------------------------------------------------------------------
# prediction contract
# --------------------------------------------------------------------------------------

predict = pytest.importorskip("app.ml.predict")


def _model_available() -> bool:
    from pathlib import Path

    return (Path("ml_artifacts") / "risk_model.json").exists()


requires_model = pytest.mark.skipif(
    not _model_available(), reason="model not trained; run app.ml.train_risk_model"
)

CALM = {
    "density": 0.10,
    "density_rate_of_change": 0.0,
    "mean_flow_speed": 1.2,
    "flow_direction_variance": 0.05,
    "optical_flow_entropy": 0.6,
    "stop_ratio": 0.0,
    "historical_deviation": -0.5,
}
CRUSH = {
    "density": 3.20,
    "density_rate_of_change": 0.15,
    "mean_flow_speed": 0.12,
    "flow_direction_variance": 0.85,
    "optical_flow_entropy": 3.2,
    "stop_ratio": 0.80,
    "historical_deviation": 6.0,
}

TIER_VECTORS = {
    "LOW": {**CALM, "density": 0.10},
    "MODERATE": {**CALM, "density": 0.60},
    "HIGH": {**CALM, "density": 1.50},
    "CRITICAL": {**CRUSH, "density": 2.50},
}


@requires_model
@pytest.mark.parametrize("expected_level", ["LOW", "MODERATE", "HIGH", "CRITICAL"])
def test_predict_risk_returns_each_known_risk_tier(expected_level):
    """Known production vectors pin the four-class serving contract."""
    result = predict.predict_risk(TIER_VECTORS[expected_level])
    assert result.risk_level == expected_level
    assert result.probabilities[expected_level] == max(result.probabilities.values())


@requires_model
def test_prediction_separates_a_calm_scene_from_a_crush():
    calm = predict.predict_risk(CALM)
    crush = predict.predict_risk(CRUSH)

    assert calm.risk_level == "LOW"
    assert crush.risk_level == "CRITICAL"
    assert crush.risk_score > calm.risk_score


@requires_model
def test_risk_score_is_within_range_and_probabilities_are_a_distribution():
    result = predict.predict_risk(CRUSH)

    assert 0.0 <= result.risk_score <= 100.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p <= 1.0 for p in result.probabilities.values())


@requires_model
def test_risk_level_is_the_most_likely_class():
    """Score and level must never contradict each other."""
    result = predict.predict_risk(CRUSH)
    assert max(result.probabilities, key=result.probabilities.get) == result.risk_level
    assert result.confidence == result.probabilities[result.risk_level]


@requires_model
def test_score_rises_monotonically_with_density():
    scores = [
        predict.predict_risk({**CALM, "density": density}).risk_score
        for density in (0.1, 0.5, 1.0, 1.6, 2.4, 3.2)
    ]
    # Allow tiny non-monotonic wobble from tree boundaries, but the trend must hold.
    assert scores[-1] > scores[0] + 30
    assert all(later >= earlier - 3.0 for earlier, later in zip(scores, scores[1:]))


@requires_model
def test_missing_features_are_nan_not_zero():
    """Zero is a real measurement; absent is not. They must not be conflated."""
    result = predict.predict_risk({"density": 1.9, "stop_ratio": 0.5})

    assert result.feature_values["mean_flow_speed"] is None
    assert result.feature_values["density"] == 1.9


@requires_model
def test_explanation_ranks_features_by_absolute_contribution():
    result = predict.predict_risk(CRUSH, top_n=4)

    assert len(result.top_features) == 4
    magnitudes = [abs(c.contribution) for c in result.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # Density dominates this model; it should lead the explanation of a crush.
    assert result.top_features[0].feature == "density"


@requires_model
def test_contributing_features_is_json_serialisable():
    """It is written straight into a JSONB column."""
    import json

    blob = predict.predict_risk(CRUSH).contributing_features()
    assert json.loads(json.dumps(blob))["top_features"]


@requires_model
def test_accepts_a_feature_vector_object():
    """The FeatureVector contract from the vision pipeline must work directly."""

    class FakeVector:
        def model_features(self):
            return CRUSH

    assert predict.predict_risk(FakeVector()).risk_level == "CRITICAL"


@requires_model
def test_to_doc_builds_a_persistable_document():
    doc = predict.predict_risk(CRUSH).to_doc(camera_id=7)

    assert doc["camera_id"] == 7
    assert doc["risk_level"] == "CRITICAL"
    assert 0 <= doc["risk_score"] <= 100
    assert "top_features" in doc["contributing_features"]


def test_missing_artifacts_raise_a_helpful_error():
    with pytest.raises(predict.ModelNotTrainedError, match="Train it with"):
        predict.RiskModel("definitely_not_a_real_directory").predict({"density": 1.0})
