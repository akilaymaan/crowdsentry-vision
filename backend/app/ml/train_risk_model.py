"""Train the crowd risk classifier on simulated, Fruin-labelled scenario windows.

Run ``scripts/simulate_crowd_scenarios.py`` first to produce the dataset.

    python -m app.ml.train_risk_model
    python -m app.ml.train_risk_model --data data/simulated/crowd_windows.csv

The model is a 4-class XGBoost classifier over the seven production features. The 0-100
risk score is derived from the predicted class probabilities rather than by a separate
regressor, so score and level can never contradict each other and the score inherits the
model's own uncertainty: a window the model is torn between HIGH and CRITICAL scores
between the two, which is the behaviour an operator dashboard wants.

Splitting is **grouped by scenario**. Windows from one simulation run are consecutive
slices of the same crowd and are highly correlated; splitting them at random would put
near-duplicate rows in both train and test and report an accuracy that does not survive
contact with a new scene.

Artifacts are written to backend/ml_artifacts/:

    risk_model.json              the booster
    risk_model_metadata.json     feature order, classes, thresholds, metrics
    feature_importance.png       gain-based importance chart
    confusion_matrix.png         test-set confusion matrix
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("train")

ARTIFACT_DIR = Path("ml_artifacts")
DEFAULT_DATA = Path("data/simulated/crowd_windows.csv")

# The seven production features, in the order the model expects them. predict.py reads
# this order back from the metadata file -- it must never be inferred from a dataframe's
# column order at serving time.
FEATURE_NAMES = [
    "density",
    "density_rate_of_change",
    "mean_flow_speed",
    "flow_direction_variance",
    "optical_flow_entropy",
    "stop_ratio",
    "historical_deviation",
]

RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Score anchors for each class, at the midpoint of its 0-100 band. The reported score is
# the probability-weighted mean of these, so it moves smoothly between tiers.
LEVEL_SCORE_ANCHORS = [12.5, 37.5, 62.5, 87.5]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"No dataset at {path}. Run:\n"
            f"    python -m scripts.simulate_crowd_scenarios"
        )

    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    logger.info("Loaded %d windows from %d scenarios", len(frame), frame.scenario_id.nunique())
    return frame


def split_by_scenario(
    frame: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 train/val/test, grouped so no scenario spans two splits."""
    groups = frame["scenario_id"]

    holdout = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_index, rest_index = next(holdout.split(frame, groups=groups))
    train = frame.iloc[train_index]
    rest = frame.iloc[rest_index]

    halve = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_index, test_index = next(halve.split(rest, groups=rest["scenario_id"]))

    return train, rest.iloc[val_index], rest.iloc[test_index]


def to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Feature matrix in the canonical order.

    Missing values are left as NaN on purpose: XGBoost learns a default branch direction
    for them, which is the right treatment for features that are legitimately absent
    (the first window has no density_rate_of_change; an uncalibrated camera has no
    mean_flow_speed). Imputing a zero would assert "no change" and "not moving", both of
    which are substantive and wrong.
    """
    return frame[FEATURE_NAMES].astype(float).to_numpy()


def risk_scores_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Probability-weighted 0-100 score."""
    return probabilities @ np.array(LEVEL_SCORE_ANCHORS)


def plot_feature_importance(booster, path: Path) -> dict[str, float]:
    """Gain-based importance chart. Returns the importances by feature name."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = booster.get_score(importance_type="gain")
    # get_score omits features the model never split on; report them as 0 rather than
    # dropping them, since "unused" is itself a finding.
    importances = {name: float(raw.get(name, 0.0)) for name in FEATURE_NAMES}
    total = sum(importances.values()) or 1.0
    normalised = {name: value / total for name, value in importances.items()}

    ordered = sorted(normalised.items(), key=lambda item: item[1])
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(names, values, color="#4C78A8")
    axis.set_xlabel("Relative importance (gain, normalised)")
    axis.set_title("CrowdSentry risk model - feature importance")
    axis.set_xlim(0, max(values) * 1.18 if max(values) > 0 else 1)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=9,
        )

    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

    return normalised


def plot_confusion_matrix(actual, predicted, path: Path) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(actual, predicted, labels=range(len(RISK_LEVELS)))
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    axis.set_xticks(range(len(RISK_LEVELS)), RISK_LEVELS, rotation=30, ha="right")
    axis.set_yticks(range(len(RISK_LEVELS)), RISK_LEVELS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Test-set confusion matrix (row-normalised)")

    for row in range(len(RISK_LEVELS)):
        for column in range(len(RISK_LEVELS)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{normalised[row, column]:.0%}",
                ha="center",
                va="center",
                color="white" if normalised[row, column] > 0.5 else "#222222",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the CrowdSentry risk model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    args = parser.parse_args()

    import xgboost as xgb

    args.artifacts.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(args.data)

    frame = frame[frame["risk_level"].isin(RISK_LEVELS)].copy()
    frame["label"] = frame["risk_level"].map(RISK_LEVELS.index)

    train, validation, test = split_by_scenario(frame, args.seed)
    logger.info(
        "Split by scenario: train=%d val=%d test=%d windows (%d/%d/%d scenarios)",
        len(train),
        len(validation),
        len(test),
        train.scenario_id.nunique(),
        validation.scenario_id.nunique(),
        test.scenario_id.nunique(),
    )

    train_matrix = xgb.DMatrix(
        to_matrix(train), label=train["label"], feature_names=FEATURE_NAMES
    )
    validation_matrix = xgb.DMatrix(
        to_matrix(validation), label=validation["label"], feature_names=FEATURE_NAMES
    )
    test_matrix = xgb.DMatrix(
        to_matrix(test), label=test["label"], feature_names=FEATURE_NAMES
    )

    params = {
        "objective": "multi:softprob",
        "num_class": len(RISK_LEVELS),
        "eval_metric": ["mlogloss", "merror"],
        "max_depth": args.max_depth,
        "eta": args.learning_rate,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "seed": args.seed,
    }

    logger.info("Training...")
    booster = xgb.train(
        params,
        train_matrix,
        num_boost_round=args.rounds,
        evals=[(train_matrix, "train"), (validation_matrix, "val")],
        # Stop on the validation set, so the round count is chosen by held-out
        # performance rather than by how long we felt like training.
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    logger.info(
        "Best iteration %d (val mlogloss %.4f)",
        booster.best_iteration,
        booster.best_score,
    )

    def evaluate(matrix, frame_split, name):
        probabilities = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
        predicted = probabilities.argmax(axis=1)
        actual = frame_split["label"].to_numpy()
        return {
            "name": name,
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro")),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted")),
            "actual": actual,
            "predicted": predicted,
            "probabilities": probabilities,
        }

    train_metrics = evaluate(train_matrix, train, "train")
    validation_metrics = evaluate(validation_matrix, validation, "validation")
    test_metrics = evaluate(test_matrix, test, "test")

    importances = plot_feature_importance(
        booster, args.artifacts / "feature_importance.png"
    )
    matrix = plot_confusion_matrix(
        test_metrics["actual"], test_metrics["predicted"], args.artifacts / "confusion_matrix.png"
    )

    model_path = args.artifacts / "risk_model.json"
    booster.save_model(str(model_path))

    report = classification_report(
        validation_metrics["actual"],
        validation_metrics["predicted"],
        labels=range(len(RISK_LEVELS)),
        target_names=RISK_LEVELS,
        output_dict=True,
        zero_division=0,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "risk_levels": RISK_LEVELS,
        "level_score_anchors": LEVEL_SCORE_ANCHORS,
        "best_iteration": int(booster.best_iteration),
        "params": {key: value for key, value in params.items() if key != "eval_metric"},
        "dataset": {
            "path": str(args.data),
            "windows": int(len(frame)),
            "scenarios": int(frame.scenario_id.nunique()),
            "class_counts": {
                level: int((frame["risk_level"] == level).sum()) for level in RISK_LEVELS
            },
        },
        "split": {
            "strategy": "GroupShuffleSplit by scenario_id (60/20/20)",
            "train_windows": int(len(train)),
            "validation_windows": int(len(validation)),
            "test_windows": int(len(test)),
        },
        "metrics": {
            split["name"]: {
                "accuracy": split["accuracy"],
                "macro_f1": split["macro_f1"],
                "weighted_f1": split["weighted_f1"],
            }
            for split in (train_metrics, validation_metrics, test_metrics)
        },
        "per_class_validation": {
            level: {
                "precision": report[level]["precision"],
                "recall": report[level]["recall"],
                "f1": report[level]["f1-score"],
                "support": int(report[level]["support"]),
            }
            for level in RISK_LEVELS
        },
        "feature_importance": importances,
        "training_data": "simulated (social-force model, Fruin LOS labels)",
    }

    with (args.artifacts / "risk_model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # -- summary -----------------------------------------------------------------
    print()
    print("=" * 72)
    print("  CrowdSentry risk model")
    print("=" * 72)
    print(f"  dataset            {len(frame)} windows / {frame.scenario_id.nunique()} scenarios")
    print(f"  split              {len(train)} train / {len(validation)} val / {len(test)} test"
          f"  (grouped by scenario)")
    print(f"  best iteration     {booster.best_iteration} of {args.rounds}")
    print("-" * 72)
    print(f"  {'split':<12}{'accuracy':>12}{'macro F1':>12}{'weighted F1':>14}")
    for split in (train_metrics, validation_metrics, test_metrics):
        print(f"  {split['name']:<12}{split['accuracy']:>12.3f}"
              f"{split['macro_f1']:>12.3f}{split['weighted_f1']:>14.3f}")
    print("-" * 72)
    print("  Per-class (validation)")
    print(f"  {'level':<12}{'precision':>11}{'recall':>10}{'F1':>10}{'support':>10}")
    for level in RISK_LEVELS:
        row = report[level]
        print(f"  {level:<12}{row['precision']:>11.3f}{row['recall']:>10.3f}"
              f"{row['f1-score']:>10.3f}{int(row['support']):>10}")
    print("-" * 72)
    print("  Feature importance (gain)")
    for name, value in sorted(importances.items(), key=lambda item: -item[1]):
        bar = "#" * int(46 * value / max(max(importances.values()), 1e-9))
        print(f"  {name:<26}{value:>7.1%}  {bar}")
    print("-" * 72)
    print("  Test confusion matrix (rows = actual)")
    header = "".join(f"{level[:4]:>9}" for level in RISK_LEVELS)
    print(f"  {'':<11}{header}")
    for index, level in enumerate(RISK_LEVELS):
        cells = "".join(f"{matrix[index, column]:>9}" for column in range(len(RISK_LEVELS)))
        print(f"  {level:<11}{cells}")
    print("-" * 72)
    print(f"  model              {model_path}")
    print(f"  metadata           {args.artifacts / 'risk_model_metadata.json'}")
    print(f"  importance chart   {args.artifacts / 'feature_importance.png'}")
    print(f"  confusion matrix   {args.artifacts / 'confusion_matrix.png'}")
    print("=" * 72)

    gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    if gap > 0.12:
        print()
        print(f"  WARNING: train accuracy exceeds test by {gap:.1%} -- likely overfitting.")

    top_feature, top_value = max(importances.items(), key=lambda item: item[1])
    if top_value > 0.85:
        print()
        print(f"  NOTE: '{top_feature}' carries {top_value:.0%} of the model's gain. The other")
        print("  features are contributing almost nothing, so this is close to a threshold rule.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
