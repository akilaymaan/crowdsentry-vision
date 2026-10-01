"""Risk scoring at serving time.

    from app.ml.predict import predict_risk

    risk = predict_risk(feature_vector)
    print(risk.risk_score, risk.risk_level)
    for contribution in risk.top_features:
        print(contribution.feature, contribution.contribution)

Naming note: this module's :class:`RiskScore` is the *prediction result* -- a plain
frozen dataclass. ``app.models.RiskScore`` is the database document. They are deliberately
different objects: a prediction is not yet a persisted record. Use
:meth:`RiskScore.to_doc` to turn a prediction into a storable document when you want to
persist it.

Explanations come from XGBoost's ``pred_contribs``, which computes exact SHAP values for
tree ensembles. That is the correct attribution -- how much each feature moved *this*
prediction -- rather than global feature importance, which is the same for every window
and therefore useless as an operator-facing explanation.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = Path("ml_artifacts")
MODEL_FILENAME = "risk_model.json"
METADATA_FILENAME = "risk_model_metadata.json"


class ModelNotTrainedError(RuntimeError):
    """Raised when the model artifacts are missing."""


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """How much one feature moved this particular prediction."""

    feature: str
    value: float | None
    #: Signed contribution, in the model's log-odds space, toward the predicted class.
    #: Positive pushed the prediction toward this risk level, negative away from it.
    contribution: float

    @property
    def direction(self) -> str:
        return "increased" if self.contribution >= 0 else "decreased"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "contribution": round(self.contribution, 6),
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class RiskScore:
    """A risk prediction for one feature window."""

    #: 0-100. Probability-weighted across risk tiers, so it moves smoothly rather than
    #: jumping at tier boundaries.
    risk_score: float

    #: LOW / MODERATE / HIGH / CRITICAL -- the model's most likely class.
    risk_level: str

    #: Probability per risk level, keyed by name.
    probabilities: dict[str, float]

    #: Features that moved this prediction most, largest absolute contribution first.
    top_features: list[FeatureContribution] = field(default_factory=list)

    #: Every feature value the model actually saw, for auditing.
    feature_values: dict[str, float | None] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted level."""
        return self.probabilities.get(self.risk_level, 0.0)

    def contributing_features(self) -> dict[str, Any]:
        """JSON-ready explanation, for ``risk_scores.contributing_features``."""
        return {
            "top_features": [c.to_dict() for c in self.top_features],
            "probabilities": {k: round(v, 5) for k, v in self.probabilities.items()},
            "feature_values": self.feature_values,
        }

    def to_doc(self, camera_id: int, timestamp: datetime | None = None) -> dict:
        """Build the persistable ``risk_scores`` document for this prediction."""
        return {
            "camera_id": camera_id,
            "timestamp": timestamp or datetime.now(timezone.utc),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "contributing_features": self.contributing_features(),
        }

    def __repr__(self) -> str:
        drivers = ", ".join(c.feature for c in self.top_features[:3])
        return (
            f"<RiskScore {self.risk_level} {self.risk_score:.1f}/100 "
            f"(p={self.confidence:.2f}) driven by {drivers}>"
        )


class RiskModel:
    """Loads the trained booster and scores feature vectors."""

    def __init__(self, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = Path(artifact_dir)
        # One booster shared by every camera worker thread. XGBoost does not document
        # Booster.predict as thread-safe, and predictions are sub-millisecond once per
        # window per camera, so serialising them costs nothing worth measuring.
        self._predict_lock = threading.Lock()
        self.model_path = self.artifact_dir / MODEL_FILENAME
        self.metadata_path = self.artifact_dir / METADATA_FILENAME

    @cached_property
    def metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise ModelNotTrainedError(
                f"No model metadata at {self.metadata_path}. Train it with:\n"
                "    python -m scripts.simulate_crowd_scenarios\n"
                "    python -m app.ml.train_risk_model"
            )
        with self.metadata_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @cached_property
    def feature_names(self) -> list[str]:
        # Read from metadata rather than hardcoded: the column order the booster was
        # trained on is the only order that produces correct predictions, and silently
        # reordering features would give confident nonsense rather than an error.
        return list(self.metadata["feature_names"])

    @cached_property
    def risk_levels(self) -> list[str]:
        return list(self.metadata["risk_levels"])

    @cached_property
    def score_anchors(self) -> np.ndarray:
        return np.array(self.metadata["level_score_anchors"], dtype=float)

    @cached_property
    def booster(self):
        if not self.model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model at {self.model_path}. Train it with:\n"
                "    python -m scripts.simulate_crowd_scenarios\n"
                "    python -m app.ml.train_risk_model"
            )

        import xgboost as xgb

        logger.info("Loading risk model: %s", self.model_path)
        booster = xgb.Booster()
        booster.load_model(str(self.model_path))
        return booster

    # -- input handling ----------------------------------------------------------

    def _extract(self, feature_vector: Any) -> dict[str, float | None]:
        """Pull the model's features out of a FeatureVector, mapping, or object.

        Accepts a :class:`~app.vision.features.FeatureVector` (via its
        ``model_features()`` contract), a plain dict, or anything with the feature names
        as attributes.
        """
        if hasattr(feature_vector, "model_features"):
            raw = feature_vector.model_features()
        elif isinstance(feature_vector, dict):
            raw = feature_vector
        else:
            raw = {
                name: getattr(feature_vector, name, None) for name in self.feature_names
            }

        values: dict[str, float | None] = {}
        for name in self.feature_names:
            value = raw.get(name)
            # None becomes NaN, which is what the booster was trained to branch on for
            # genuinely-absent features. Substituting 0 would assert a real measurement.
            values[name] = None if value is None else float(value)

        unknown = set(raw) - set(self.feature_names)
        if unknown:
            logger.debug("Ignoring features the model was not trained on: %s", sorted(unknown))

        return values

    def predict(self, feature_vector: Any, top_n: int = 3) -> RiskScore:
        """Score one feature vector."""
        import xgboost as xgb

        values = self._extract(feature_vector)
        row = np.array(
            [[np.nan if values[name] is None else values[name] for name in self.feature_names]],
            dtype=float,
        )
        matrix = xgb.DMatrix(row, feature_names=self.feature_names)

        best_iteration = self.metadata.get("best_iteration")
        iteration_range = (0, best_iteration + 1) if best_iteration is not None else None

        with self._predict_lock:
            probabilities = self.booster.predict(
                matrix, iteration_range=iteration_range
            )[0]
        class_index = int(np.argmax(probabilities))
        level = self.risk_levels[class_index]
        score = float(np.dot(probabilities, self.score_anchors))

        return RiskScore(
            risk_score=round(score, 2),
            risk_level=level,
            probabilities={
                name: float(probability)
                for name, probability in zip(self.risk_levels, probabilities)
            },
            top_features=self._explain(
                matrix, class_index, values, top_n, iteration_range
            ),
            feature_values=values,
        )

    def _explain(
        self,
        matrix,
        class_index: int,
        values: dict[str, float | None],
        top_n: int,
        iteration_range: tuple[int, int] | None = None,
    ) -> list[FeatureContribution]:
        """Per-prediction SHAP contributions for the predicted class.

        Uses the same ``iteration_range`` as the probability prediction, so the
        attribution describes the trees that actually produced the score rather than
        the extra rounds early-stopping left behind.
        """
        try:
            with self._predict_lock:
                contributions = self.booster.predict(
                    matrix, pred_contribs=True, iteration_range=iteration_range
                )
        except Exception:
            logger.exception("Could not compute feature contributions")
            return []

        # Shape is (rows, classes, features + 1) for multiclass, where the trailing
        # column is the bias term and is not a feature.
        contributions = np.asarray(contributions)
        if contributions.ndim == 3:
            per_feature = contributions[0, class_index, :-1]
        else:
            per_feature = contributions[0, :-1]

        ranked = sorted(
            (
                FeatureContribution(
                    feature=name,
                    value=values[name],
                    contribution=float(per_feature[index]),
                )
                for index, name in enumerate(self.feature_names)
            ),
            key=lambda contribution: abs(contribution.contribution),
            reverse=True,
        )
        return ranked[:top_n]


# --------------------------------------------------------------------------------------
# Module-level convenience API
# --------------------------------------------------------------------------------------

_default_model: RiskModel | None = None
_model_lock = threading.Lock()


def get_model(artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR) -> RiskModel:
    """Return the process-wide model, creating it on first call."""
    global _default_model

    if _default_model is None:
        with _model_lock:
            if _default_model is None:
                _default_model = RiskModel(artifact_dir)

    return _default_model


def reset_model() -> None:
    """Drop the cached model so the next call reloads from disk."""
    global _default_model

    with _model_lock:
        _default_model = None


def predict_risk(feature_vector: Any, top_n: int = 3) -> RiskScore:
    """Score a feature window, returning level, 0-100 score, and top contributors.

    ``feature_vector`` may be a :class:`~app.vision.features.FeatureVector`, a dict of
    feature names to values, or any object exposing the feature names as attributes.
    Missing features are passed to the model as NaN rather than zero.
    """
    return get_model().predict(feature_vector, top_n=top_n)


__all__ = [
    "FeatureContribution",
    "ModelNotTrainedError",
    "RiskModel",
    "RiskScore",
    "get_model",
    "predict_risk",
    "reset_model",
]
