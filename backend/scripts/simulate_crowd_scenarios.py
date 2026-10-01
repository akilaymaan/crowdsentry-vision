"""Generate labelled crowd scenarios with a 2D social-force pedestrian simulation.

We have no incident-labelled crowd footage, so the risk model is trained on simulated
pedestrian dynamics labelled against a published density standard instead of on real
incidents. That is a real limitation and is documented in the README -- the model learns
"what does a dangerous *density and flow pattern* look like", not "what preceded a real
crush".

The simulation is a simplified Helbing & Molnar social force model [1]: each pedestrian
is driven toward a goal, repelled by other pedestrians and by walls, and the resulting
crowd naturally produces congestion, stalling and turbulence at a bottleneck without any
of that being scripted.

    [1] Helbing, D. & Molnar, P. (1995). "Social force model for pedestrian dynamics."
        Physical Review E 51(5), 4282-4286.

Usage (from backend/):

    python -m scripts.simulate_crowd_scenarios                     # 420 scenarios
    python -m scripts.simulate_crowd_scenarios --scenarios 100     # quick run
    python -m scripts.simulate_crowd_scenarios --output data/simulated/windows.csv

Features are computed by running the simulated agents through the *real*
:class:`~app.vision.features.FeatureExtractor` -- the same code that runs in production --
so the training features cannot drift from the serving features. See the module notes on
``optical_flow_entropy`` for the one feature where that parity does not fully hold.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from app.vision.detection import BoundingBox
from app.vision.features import FeatureExtractor, shannon_entropy
from app.vision.tracking import TrackedFrame, TrackedPerson

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("simulate")

DEFAULT_OUTPUT = Path("data/simulated/crowd_windows.csv")

# --------------------------------------------------------------------------------------
# Fruin Level of Service
# --------------------------------------------------------------------------------------
#
# Fruin, J.J. (1971). "Pedestrian Planning and Design". Metropolitan Association of Urban
# Designers and Environmental Planners, New York. Chapter 4 defines Level of Service for
# walkways by "pedestrian area module" -- the floor area available per person, in square
# feet. Those are the canonical thresholds; we invert them to densities (persons/m^2)
# because that is what the vision pipeline measures.
#
#     LOS   area module (ft^2/ped)    density (ped/m^2)     Fruin's description
#     A     >= 35                     <= 0.31               free flow, no conflicts
#     B     25 - 35                   0.31 - 0.43           normal speeds, minor conflicts
#     C     15 - 25                   0.43 - 0.72           speeds restricted, passing hard
#     D     10 - 15                   0.72 - 1.08           speeds restricted, reverse flow
#                                                           causes serious conflict
#     E      5 - 10                   1.08 - 2.15           shuffling, forward movement
#                                                           only, cross-flow near-impossible
#     F     <  5                      > 2.15                contact unavoidable, movement
#                                                           breaks down, crush risk
#
# 1 m^2 = 10.7639 ft^2, so density = 10.7639 / area_module_ft2.
SQ_FT_PER_SQ_M = 10.7639

LOS_AREA_MODULE_SQ_FT = {"A": 35.0, "B": 25.0, "C": 15.0, "D": 10.0, "E": 5.0}
# Upper density bound of each LOS band, in persons/m^2. F is unbounded above.
LOS_DENSITY_BOUNDS = {
    los: SQ_FT_PER_SQ_M / area for los, area in LOS_AREA_MODULE_SQ_FT.items()
}

# Our four operational tiers. Fruin's six bands collapse 2-2-1-1: A/B are both
# comfortable, C/D are both "restricted but functioning", E is where movement degrades to
# shuffling, and F is where it breaks down.
LOS_TO_RISK_LEVEL = {
    "A": "LOW",
    "B": "LOW",
    "C": "MODERATE",
    "D": "MODERATE",
    "E": "HIGH",
    "F": "CRITICAL",
}
RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Disorganised movement escalates the label by one tier. Fruin's LOS is a density-only
# standard, but a crowd at moderate density that has stopped moving and lost a coherent
# direction is the documented precursor to a crush -- density alone does not capture it.
CHAOS_DIRECTION_WEIGHT = 0.55
CHAOS_STOP_WEIGHT = 0.45
# Calibrated against the simulated distribution so escalation is a meaningful but not
# dominant mechanism: it fires on roughly the top decile of eligible windows (~10-12%
# overall). Set too high it never fires and the chaos rule is inert; too low and it
# swamps the density signal it is meant to supplement.
CHAOS_ESCALATION_THRESHOLD = 0.45
# Below LOS B density there is too much free space for disorder to be dangerous; a single
# person wandering an empty plaza should not be escalated.
CHAOS_MIN_DENSITY = LOS_DENSITY_BOUNDS["B"]


def classify_los(density: float) -> str:
    """Fruin Level of Service band for a density in persons/m^2."""
    for los in ("A", "B", "C", "D", "E"):
        if density <= LOS_DENSITY_BOUNDS[los]:
            return los
    return "F"


def chaos_score(flow_direction_variance: float | None, stop_ratio: float) -> float:
    """0-1 measure of how disorganised movement is."""
    variance = 0.0 if flow_direction_variance is None else flow_direction_variance
    return CHAOS_DIRECTION_WEIGHT * variance + CHAOS_STOP_WEIGHT * stop_ratio


def label_window(
    density: float, flow_direction_variance: float | None, stop_ratio: float
) -> tuple[str, str, float, bool]:
    """Ground-truth label for one window.

    Returns ``(risk_level, los, chaos, escalated)``. The base label is Fruin's LOS band
    for the density; a high chaos score pushes it up one tier, capped at CRITICAL.
    """
    los = classify_los(density)
    base_level = LOS_TO_RISK_LEVEL[los]
    chaos = chaos_score(flow_direction_variance, stop_ratio)

    escalated = chaos >= CHAOS_ESCALATION_THRESHOLD and density >= CHAOS_MIN_DENSITY
    if not escalated:
        return base_level, los, chaos, False

    index = min(RISK_LEVELS.index(base_level) + 1, len(RISK_LEVELS) - 1)
    return RISK_LEVELS[index], los, chaos, True


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------

# Social force parameters, following Helbing & Molnar (1995) with the usual values.
RELAXATION_TIME = 0.5           # tau, seconds to reach desired velocity
# A and B are per unit mass. Helbing's Nature (2000) values are A = 2000 N and
# B = 0.08 m for an 80 kg pedestrian, i.e. A/m = 25 m/s^2 over a very short range. A
# weaker, wider field does not jam: agents simply flow past each other at any density.
REPULSION_STRENGTH = 14.0       # A, m/s^2
REPULSION_RANGE = 0.14          # B, metres
# Body compression once people are actually touching. This granular term -- not the
# exponential one -- is what produces standstill in a dense crowd, and without it
# stop_ratio stays near zero even at LOS F.
CONTACT_STIFFNESS = 90.0        # m/s^2 per metre of overlap
# Crowding drag. Helbing's friction term activates only on physical contact, but a
# purely repulsive crowd never reaches contact: it equilibrates at ~0.6 m spacing and
# then behaves like a pressurised gas, where jostling *increases* with density and the
# speed-density relation comes out backwards. This damps velocity in proportion to how
# close an agent's neighbours are, engaging smoothly before contact, and reproduces the
# empirical fundamental diagram (speed falling from ~1.3 m/s in free flow toward zero as
# density approaches ~5 ped/m^2). A deliberate deviation from strict Helbing, in the
# direction of matching observed pedestrian behaviour.
CROWDING_DRAG = 0.95
CROWDING_DRAG_RANGE = 0.25      # metres
# Ceiling on repulsive acceleration, so a momentary deep overlap cannot fling an agent
# across the domain and destabilise the integrator.
MAX_REPULSION = 60.0
WALL_STRENGTH = 14.0
WALL_RANGE = 0.18
PEDESTRIAN_RADIUS = 0.23        # metres
MAX_SPEED = 2.2                 # m/s, cap so the integrator cannot blow up

# Domain: a corridor with a wall across it, pierced by a gate. The measurement zone sits
# in front of the gate, which is where congestion forms.
DOMAIN_LENGTH = 20.0            # metres, x
DOMAIN_WIDTH = 8.0              # metres, y
GATE_X = 12.0
# The zone sits on the gate approach, not the whole corridor: a camera covering the
# bottleneck is what we are modelling, and a deeper zone would average the congested
# queue together with free-flowing people well behind it.
ZONE_X0, ZONE_X1 = 9.0, 12.0    # 3 m deep, right up to the gate
ZONE_Y0, ZONE_Y1 = 2.0, 6.0     # 4 m wide
ZONE_AREA = (ZONE_X1 - ZONE_X0) * (ZONE_Y1 - ZONE_Y0)   # 24 m^2

DT = 0.05                       # seconds per simulation step (20 Hz)
SAMPLE_EVERY = 4                # emit a tracked frame every 4th step -> 5 fps
PIXELS_PER_METER = 40.0         # arbitrary but fixed: sim metres -> synthetic pixels

PATTERNS = ("unidirectional", "bidirectional", "converging", "milling")

# How many more agents a pattern needs to reach the same measured zone density.
# Measured by sweeping agent count per pattern: the funnelling patterns concentrate
# people at the gate, while milling agents spread over the whole domain and only a
# fraction are ever in view. Without this correction the milling scenarios all land in
# LOS A and the dataset has no disordered high-density examples at all.
PATTERN_DENSITY_FACTOR = {
    "unidirectional": 1.0,
    "converging": 1.0,
    "bidirectional": 1.35,
    "milling": 2.7,
}


@dataclass(slots=True)
class Scenario:
    """One simulation run."""

    scenario_id: int
    pattern: str
    n_agents: int
    #: How many agents to seed inside the measurement zone. Seeding the zone directly
    #: gives control over the density actually measured, instead of hoping the spatial
    #: distribution lands where we want it.
    zone_agents: int
    gate_width: float
    desired_speed: float
    duration_seconds: float
    # Synthetic "what is normal here at this hour" baseline, for historical_deviation.
    baseline_density: float
    baseline_stddev: float


def _jittered_grid(
    count: int, x0: float, x1: float, y0: float, y1: float, rng: np.random.Generator
) -> np.ndarray:
    """Lay ``count`` points on a jittered grid inside the rectangle.

    Uniform random seeding puts agents directly on top of each other at high density,
    and the resulting overlap forces blow the crowd apart on the first timestep. A grid
    with jitter reaches the same density without that artefact.
    """
    if count <= 0:
        return np.empty((0, 2))

    aspect = (x1 - x0) / (y1 - y0)
    columns = max(1, int(round(math.sqrt(count * aspect))))
    rows = max(1, math.ceil(count / columns))

    xs = np.linspace(x0, x1, columns, endpoint=False) + (x1 - x0) / (2 * columns)
    ys = np.linspace(y0, y1, rows, endpoint=False) + (y1 - y0) / (2 * rows)
    grid = np.array([(x, y) for y in ys for x in xs][:count], dtype=float)

    jitter = min((x1 - x0) / columns, (y1 - y0) / rows) * 0.25
    grid += rng.uniform(-jitter, jitter, grid.shape)
    grid[:, 0] = np.clip(grid[:, 0], x0 + 0.1, x1 - 0.1)
    grid[:, 1] = np.clip(grid[:, 1], y0 + 0.1, y1 - 0.1)
    return grid


class CrowdSimulation:
    """Vectorised social-force simulation of pedestrians moving through a gate."""

    def __init__(self, scenario: Scenario, rng: np.random.Generator) -> None:
        self.scenario = scenario
        self.rng = rng
        self.n = scenario.n_agents

        self.gate_low = DOMAIN_WIDTH / 2 - scenario.gate_width / 2
        self.gate_high = DOMAIN_WIDTH / 2 + scenario.gate_width / 2

        # Seed the requested number inside the measurement zone, and the remainder
        # upstream feeding into it.
        zone_count = min(scenario.zone_agents, self.n)
        in_zone = _jittered_grid(zone_count, ZONE_X0, ZONE_X1, ZONE_Y0, ZONE_Y1, rng)
        upstream = np.column_stack(
            [
                rng.uniform(0.5, ZONE_X0, self.n - zone_count),
                rng.uniform(0.4, DOMAIN_WIDTH - 0.4, self.n - zone_count),
            ]
        )
        self.positions = np.vstack([in_zone, upstream])
        self.velocities = np.zeros((self.n, 2))
        self._contact_load = np.zeros(self.n)
        self.desired_speeds = np.clip(
            rng.normal(scenario.desired_speed, 0.18, self.n), 0.4, MAX_SPEED
        )

        # Direction of travel: +1 moves toward the gate, -1 is counter-flow.
        if scenario.pattern == "bidirectional":
            self.headed_forward = rng.random(self.n) < 0.6
        else:
            self.headed_forward = np.ones(self.n, dtype=bool)

        self.goals = self._initial_goals()

    def _initial_goals(self) -> np.ndarray:
        gate_centre = np.array([GATE_X + 1.0, DOMAIN_WIDTH / 2])
        goals = np.tile(gate_centre, (self.n, 1))

        if self.scenario.pattern == "bidirectional":
            # Counter-flow agents aim back upstream, through the same gate.
            goals[~self.headed_forward] = np.array([1.0, DOMAIN_WIDTH / 2])
            # Forward agents spread across the gate mouth rather than a single point.
            forward = self.headed_forward
            goals[forward, 1] = self.rng.uniform(
                self.gate_low, self.gate_high, forward.sum()
            )
        elif self.scenario.pattern == "converging":
            # Everyone funnels to the exact gate centre: maximum convergence pressure.
            pass
        elif self.scenario.pattern == "milling":
            # No shared objective: wander toward random points, re-drawn periodically.
            goals = np.column_stack(
                [
                    self.rng.uniform(2.0, GATE_X - 1.0, self.n),
                    self.rng.uniform(1.0, DOMAIN_WIDTH - 1.0, self.n),
                ]
            )
        else:  # unidirectional
            goals[:, 1] = self.rng.uniform(self.gate_low, self.gate_high, self.n)

        return goals

    def _pedestrian_forces(self) -> np.ndarray:
        """Pairwise repulsion between all agents."""
        delta = self.positions[:, None, :] - self.positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)

        # Exponentially decaying repulsion, strongest when bodies overlap.
        overlap = 2 * PEDESTRIAN_RADIUS - distance
        magnitude = REPULSION_STRENGTH * np.exp(overlap / REPULSION_RANGE)
        # Plus a contact force proportional to actual overlap, active only on touch.
        magnitude += CONTACT_STIFFNESS * np.maximum(overlap, 0.0)
        # Ignore anyone far enough away to be irrelevant; keeps the field local.
        magnitude[distance > 1.5] = 0.0

        magnitude = np.minimum(magnitude, MAX_REPULSION)

        direction = delta / distance[:, :, None]
        repulsion = np.nansum(magnitude[:, :, None] * direction, axis=1)

        # Crowding load: how hemmed in each agent is, used to damp its velocity. Uses
        # proximity rather than overlap so it engages before bodies actually touch.
        proximity = np.exp(-(distance - 2 * PEDESTRIAN_RADIUS) / CROWDING_DRAG_RANGE)
        proximity[distance > 1.2] = 0.0
        np.fill_diagonal(proximity, 0.0)
        self._contact_load = np.nansum(np.minimum(proximity, 4.0), axis=1)

        return repulsion

    def _wall_forces(self) -> np.ndarray:
        """Repulsion from the corridor sides and from the gate wall."""
        forces = np.zeros_like(self.positions)
        x, y = self.positions[:, 0], self.positions[:, 1]

        # Corridor sides.
        forces[:, 1] += WALL_STRENGTH * np.exp(-y / WALL_RANGE)
        forces[:, 1] -= WALL_STRENGTH * np.exp(-(DOMAIN_WIDTH - y) / WALL_RANGE)

        # The gate wall spans the corridor except for the opening. Agents aligned with
        # the opening feel nothing; the rest are pushed away from the wall plane and
        # sideways toward the gap, which is what creates the funnel.
        outside_gap = (y < self.gate_low) | (y > self.gate_high)
        dx = x - GATE_X
        near_wall = np.abs(dx) < 1.5

        blocked = outside_gap & near_wall
        # Push away from the wall plane: an agent upstream of the wall (dx < 0) must be
        # pushed further upstream (-x), so the force takes the sign of dx.
        forces[blocked, 0] += WALL_STRENGTH * np.exp(
            -np.abs(dx[blocked]) / WALL_RANGE
        ) * np.sign(dx[blocked] + 1e-9)

        # Nudge blocked agents laterally toward the opening. This must fade with
        # distance from the wall: a constant nudge makes blocked agents slide sideways
        # forever at ~0.75 m/s, which puts a floor under mean speed and stops a jam from
        # ever registering as stopped.
        toward_gap = np.where(y[blocked] < self.gate_low, 1.0, -1.0)
        forces[blocked, 1] += (
            0.6 * toward_gap * np.exp(-np.abs(dx[blocked]) / 0.8)
        )

        return forces

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        to_goal = self.goals - self.positions
        distance = np.linalg.norm(to_goal, axis=1, keepdims=True)
        direction = to_goal / np.maximum(distance, 1e-6)

        driving = (
            self.desired_speeds[:, None] * direction - self.velocities
        ) / RELAXATION_TIME

        repulsion = self._pedestrian_forces()
        acceleration = driving + repulsion + self._wall_forces()
        # Contact friction, applied after _pedestrian_forces has computed the load.
        acceleration -= CROWDING_DRAG * self._contact_load[:, None] * self.velocities
        # Small random jitter: real pedestrians are not deterministic, and without it
        # symmetric configurations lock into unphysical gridlock.
        acceleration += self.rng.normal(0.0, 0.15, self.positions.shape)

        self.velocities += acceleration * DT
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        excessive = speeds[:, 0] > MAX_SPEED
        self.velocities[excessive] *= (MAX_SPEED / speeds[excessive, 0])[:, None]

        self.positions += self.velocities * DT
        self._apply_boundaries()

    def _apply_boundaries(self) -> None:
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.15, DOMAIN_WIDTH - 0.15)
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.15, DOMAIN_LENGTH - 0.15)

        if self.scenario.pattern == "milling":
            # Re-target anyone who reached their wandering goal.
            reached = np.linalg.norm(self.goals - self.positions, axis=1) < 0.6
            if reached.any():
                self.goals[reached] = np.column_stack(
                    [
                        self.rng.uniform(2.0, GATE_X - 1.0, reached.sum()),
                        self.rng.uniform(1.0, DOMAIN_WIDTH - 1.0, reached.sum()),
                    ]
                )
            return

        # Recycle anyone who has passed through, so density upstream stays steady rather
        # than draining away over the run.
        through = self.positions[:, 0] > GATE_X + 2.0
        if through.any():
            # Respawn just upstream of the measurement zone, not at the far end of the
            # domain: from there it takes ~8 s to walk back into view, and the zone
            # drains faster than it refills, so every scenario decays toward LOS A
            # regardless of how densely it started.
            self.positions[through, 0] = self.rng.uniform(
                max(0.5, ZONE_X0 - 3.0), ZONE_X0, through.sum()
            )
            self.positions[through, 1] = self.rng.uniform(
                0.5, DOMAIN_WIDTH - 0.5, through.sum()
            )
            self.velocities[through] = 0.0

        returned = self.positions[:, 0] < 0.3
        if returned.any() and self.scenario.pattern == "bidirectional":
            self.positions[returned, 0] = self.rng.uniform(
                GATE_X + 0.5, GATE_X + 2.0, returned.sum()
            )
            self.velocities[returned] = 0.0

    def agents_in_zone(self) -> np.ndarray:
        """Boolean mask of agents inside the measurement zone (the camera's footprint)."""
        x, y = self.positions[:, 0], self.positions[:, 1]
        return (x >= ZONE_X0) & (x <= ZONE_X1) & (y >= ZONE_Y0) & (y <= ZONE_Y1)


# --------------------------------------------------------------------------------------
# Simulation -> TrackedFrame
# --------------------------------------------------------------------------------------


def to_tracked_frame(
    simulation: CrowdSimulation, frame_index: int, source_time: float, start: datetime
) -> TrackedFrame:
    """Convert the agents currently inside the measurement zone into a TrackedFrame.

    Positions and velocities are converted from metres to synthetic pixels so the frame
    is indistinguishable, to the feature extractor, from one produced by the tracker.
    """
    in_zone = simulation.agents_in_zone()
    indices = np.flatnonzero(in_zone)

    people: list[TrackedPerson] = []
    for agent_index in indices:
        x, y = simulation.positions[agent_index]
        vx, vy = simulation.velocities[agent_index]

        px, py = x * PIXELS_PER_METER, y * PIXELS_PER_METER
        # Roughly person-shaped box around the foot point; only the foot point and the
        # velocity actually matter downstream.
        bbox = BoundingBox(
            x1=px - 12.0, y1=py - 68.0, x2=px + 12.0, y2=py, confidence=0.9
        )
        speed_px = float(math.hypot(vx, vy) * PIXELS_PER_METER)

        people.append(
            TrackedPerson(
                track_id=int(agent_index),
                bbox=bbox,
                centroid=bbox.center,
                foot_point=bbox.foot_point,
                velocity_px_per_s=(
                    float(vx * PIXELS_PER_METER),
                    float(vy * PIXELS_PER_METER),
                ),
                speed_px_per_s=speed_px,
                speed_m_per_s=speed_px / PIXELS_PER_METER,
                age=frame_index + 1,
            )
        )

    return TrackedFrame(
        frame_index=frame_index,
        timestamp=start + timedelta(seconds=source_time),
        source_time=source_time,
        people=people,
    )


def heading_entropy(simulation: CrowdSimulation, bins: int = 16) -> float:
    """Direction-histogram entropy of the agents in the zone, in bits.

    This is the training-time stand-in for ``optical_flow_entropy``. The production
    feature is the entropy of a *dense optical flow* direction histogram; here there are
    no images, so it is the entropy of the *agents'* heading histogram instead. Same
    definition, same units (bits over 16 bins, magnitude-weighted), different source --
    see the README for why that gap matters.
    """
    in_zone = simulation.agents_in_zone()
    velocities = simulation.velocities[in_zone]
    if len(velocities) == 0:
        return 0.0

    speeds = np.linalg.norm(velocities, axis=1)
    moving = speeds > 0.05
    if not moving.any():
        return 0.0

    angles = np.arctan2(velocities[moving, 1], velocities[moving, 0]) % (2 * math.pi)
    histogram, _ = np.histogram(
        angles, bins=bins, range=(0.0, 2 * math.pi), weights=speeds[moving]
    )
    return shannon_entropy(histogram)


def run_scenario(scenario: Scenario, rng: np.random.Generator) -> list[dict]:
    """Simulate one scenario and return one labelled row per feature window."""
    simulation = CrowdSimulation(scenario, rng)
    start = datetime(2026, 3, 4, 14, 0, 0, tzinfo=timezone.utc)

    # Let the crowd settle before measuring; the first second is agents accelerating
    # from a standstill, which is not representative of anything.
    for _ in range(int(2.0 / DT)):
        simulation.step()

    frames: list[TrackedFrame] = []
    entropies: list[tuple[float, float]] = []   # (source_time, entropy)

    total_steps = int(scenario.duration_seconds / DT)
    for step_index in range(total_steps):
        simulation.step()
        if step_index % SAMPLE_EVERY:
            continue

        source_time = step_index * DT
        frames.append(
            to_tracked_frame(simulation, len(frames), source_time, start)
        )
        entropies.append((source_time, heading_entropy(simulation)))

    extractor = FeatureExtractor(
        area_sq_meters=ZONE_AREA,
        pixels_per_meter=PIXELS_PER_METER,
        window_seconds=5.0,
        compute_optical_flow=False,   # no images exist; the proxy is injected below
        persist=False,
    )

    rows: list[dict] = []
    for features in extractor.extract(frames):
        window_start = (features.window_start - start).total_seconds()
        window_end = (features.window_end - start).total_seconds()
        in_window = [e for t, e in entropies if window_start <= t <= window_end]
        entropy = float(np.mean(in_window)) if in_window else 0.0

        density = features.density
        stop_ratio = features.stop_ratio
        direction_variance = features.flow_direction_variance

        level, los, chaos, escalated = label_window(density, direction_variance, stop_ratio)

        # Synthetic historical baseline: what "normal" would be for this venue at this
        # hour. Drawn per scenario so it is not a deterministic function of density --
        # the same density is routine at a busy gate and anomalous at a quiet one.
        deviation = (density - scenario.baseline_density) / scenario.baseline_stddev

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "pattern": scenario.pattern,
                "n_agents": scenario.n_agents,
                "gate_width": round(scenario.gate_width, 2),
                "window_start_s": round(window_start, 2),
                "person_count": round(features.person_count, 3),
                "unique_track_count": features.unique_track_count,
                "density": round(density, 5),
                "density_rate_of_change": (
                    None
                    if features.density_rate_of_change is None
                    else round(features.density_rate_of_change, 6)
                ),
                "mean_flow_speed": (
                    None
                    if features.mean_flow_speed_m_s is None
                    else round(features.mean_flow_speed_m_s, 5)
                ),
                "flow_direction_variance": (
                    None
                    if direction_variance is None
                    else round(direction_variance, 5)
                ),
                "optical_flow_entropy": round(entropy, 5),
                "stop_ratio": round(stop_ratio, 5),
                "newly_stopped_ratio": (
                    None
                    if features.newly_stopped_ratio is None
                    else round(features.newly_stopped_ratio, 5)
                ),
                "historical_deviation": round(deviation, 5),
                "chaos_score": round(chaos, 5),
                "los": los,
                "escalated": int(escalated),
                "risk_level": level,
            }
        )

    return rows


# --------------------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------------------


def build_scenarios(count: int, rng: np.random.Generator) -> list[Scenario]:
    """Spread scenarios across densities and flow patterns.

    Agent counts are drawn to target the whole LOS range rather than a realistic mix:
    a naturally-sampled dataset would be almost entirely LOS A/B, and the model needs
    examples of the dangerous end.
    """
    scenarios: list[Scenario] = []

    # Target zone occupancy bands, chosen so the resulting densities straddle every LOS
    # boundary. ZONE_AREA is 24 m^2, so e.g. 60 agents in zone -> 2.5 ped/m^2 (LOS F).
    # The top bands reach ~4-5.5 ped/m^2, where mean spacing drops below body width and
    # the contact force finally engages -- that is where genuine standstill appears, and
    # it is also the density at which real crowd-crush incidents occur.
    # One band per Fruin LOS letter, giving a roughly balanced dataset instead of the
    # naturally-occurring distribution, which is almost entirely LOS A. Agent counts are
    # calibrated from a density sweep (see PATTERN_DENSITY_FACTOR) for the funnelling
    # patterns; other patterns are scaled by their factor.
    occupancy_bands = [
        (8, 12),      # -> LOS A
        (13, 16),     # -> LOS B
        (17, 21),     # -> LOS C
        (22, 28),     # -> LOS D
        (29, 52),     # -> LOS E
        (58, 155),    # -> LOS F
    ]

    for index in range(count):
        pattern = PATTERNS[index % len(PATTERNS)]
        low, high = occupancy_bands[(index // len(PATTERNS)) % len(occupancy_bands)]
        base_agents = int(rng.integers(low, high + 1))
        total_agents = int(base_agents * PATTERN_DENSITY_FACTOR[pattern])

        scenarios.append(
            Scenario(
                scenario_id=index,
                pattern=pattern,
                n_agents=min(total_agents, 240),
                # Seed roughly half inside the zone so it starts near its equilibrium
                # density rather than spending the first seconds filling up.
                zone_agents=min(total_agents // 2, 80),
                # A narrow gate is what turns a merely dense crowd into a stalled one.
                gate_width=float(rng.uniform(0.9, 3.2)),
                desired_speed=float(rng.uniform(0.9, 1.5)),
                duration_seconds=30.0,
                baseline_density=float(rng.uniform(0.20, 1.30)),
                baseline_stddev=float(rng.uniform(0.08, 0.40)),
            )
        )

    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate labelled synthetic crowd scenario windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenarios", type=int, default=400, help="scenarios to simulate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scenarios = build_scenarios(args.scenarios, rng)

    logger.info(
        "Simulating %d scenarios (zone %.0f m^2, %.0fs each)...",
        len(scenarios),
        ZONE_AREA,
        scenarios[0].duration_seconds,
    )

    rows: list[dict] = []
    started = time.perf_counter()

    for position, scenario in enumerate(scenarios, start=1):
        rows.extend(run_scenario(scenario, rng))
        if position % 25 == 0 or position == len(scenarios):
            elapsed = time.perf_counter() - started
            logger.info(
                "  %d/%d scenarios | %d windows | %.0fs elapsed",
                position,
                len(scenarios),
                len(rows),
                elapsed,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # -- summary ---------------------------------------------------------------
    densities = np.array([r["density"] for r in rows])
    level_counts = {level: sum(1 for r in rows if r["risk_level"] == level) for level in RISK_LEVELS}
    los_counts = {los: sum(1 for r in rows if r["los"] == los) for los in "ABCDEF"}
    escalated = sum(r["escalated"] for r in rows)

    print()
    print("=" * 68)
    print(f"  scenarios           {len(scenarios)}")
    print(f"  windows             {len(rows)}")
    print(f"  measurement zone    {ZONE_AREA:.0f} m^2")
    print(f"  density range       {densities.min():.2f} - {densities.max():.2f} ped/m^2 "
          f"(mean {densities.mean():.2f})")
    print("-" * 68)
    print("  Fruin LOS distribution")
    for los, count in los_counts.items():
        bar = "#" * int(50 * count / max(1, len(rows)))
        bound = (
            f"<= {LOS_DENSITY_BOUNDS[los]:.2f}"
            if los in LOS_DENSITY_BOUNDS
            else f"> {LOS_DENSITY_BOUNDS['E']:.2f}"
        )
        print(f"    {los}  {bound:>8} ped/m^2  {count:>5}  {bar}")
    print("-" * 68)
    print("  risk level distribution (after chaos escalation)")
    for level, count in level_counts.items():
        bar = "#" * int(50 * count / max(1, len(rows)))
        print(f"    {level:<9} {count:>5}  ({100*count/len(rows):4.1f}%)  {bar}")
    print(f"  escalated by chaos  {escalated} windows ({100*escalated/len(rows):.1f}%)")
    print("-" * 68)
    print(f"  written to          {args.output}")
    print(f"  elapsed             {time.perf_counter() - started:.0f}s")
    print("=" * 68)

    if min(level_counts.values()) < 30:
        thin = [level for level, count in level_counts.items() if count < 30]
        print()
        print(f"  WARNING: thin classes {thin} -- increase --scenarios before training.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
