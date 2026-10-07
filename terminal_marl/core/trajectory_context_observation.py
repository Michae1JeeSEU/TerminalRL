"""V2.5 current-state traffic sets, separate from historical flat 90D actors.

Every active neighbor is retained. Forecasts sample existing commitments and
the nominal holder continuation only; no candidate generation or safety query
is performed. UID/order metadata never enters the numeric observation.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import math

import numpy as np

from .config import RingRole
from .observations import (FeatureSpec, ObservationSchema, LOW_OBS_V2_4,
    _slot_vector, _v2_2_scheduler_context, build_low_observation_v2_4)
from .safety_filter import SafetyFilter
from .topology_neighbors import _angle, _slot
from .uav import FlightPhase

VERSION = 'phaseA_low_obs_v2_5_trajectory_context_set'
FORECAST_SECONDS = (5., 15., 30., 60., 90., 120., 150.)
POSITION_SCALE = 1000.  # metres; scaling only, never distance clipping
VELOCITY_SCALE = 25.  # m/s
TIME_SCALE = 150.  # t/(t+150), preserves all finite remaining times
TARGET_PHASES = ('ring_outer', 'ring_middle', 'ring_inner', 'landing')
PHASES = (FlightPhase.INBOUND, FlightPhase.RING_OUTER, FlightPhase.TRANSFER_O2M,
          FlightPhase.RING_MIDDLE, FlightPhase.TRANSFER_M2I, FlightPhase.RING_INNER,
          FlightPhase.FINAL, FlightPhase.LANDING)
TARGET_ROLE = {f'ring_{r.value}': r for r in RingRole}

EGO_SCHEMA = ObservationSchema(tuple(
    FeatureSpec({'target_ring_predecessor': 'target_reference_predecessor',
                 'target_ring_successor': 'target_reference_successor',
                 'nearest_predicted_conflict': 'nearest_linear_cpa'}.get(f.name, f.name), f.size)
    for f in LOW_OBS_V2_4.features) + (
        FeatureSpec('scheduler_context', 6), FeatureSpec('actual_rings_radius_height_speed', 9),
    ))
TRAFFIC_SCHEMA = ObservationSchema((
    FeatureSpec('phase', 8), FeatureSpec('current_ring', 3), FeatureSpec('committed', 1),
    FeatureSpec('committed_target', 4), FeatureSpec('plan_remaining', 1),
    FeatureSpec('relative_position', 3), FeatureSpec('relative_velocity', 3),
    FeatureSpec('plan_endpoint_relative_position', 3), FeatureSpec('plan_endpoint_velocity', 3),
    FeatureSpec('forecast_relative_positions', 21),
))
assert (EGO_SCHEMA.dimension, TRAFFIC_SCHEMA.dimension) == (105, 50)


@dataclass(frozen=True)
class TrajectoryContextObservation:
    ego: np.ndarray
    traffic: np.ndarray
    traffic_mask: np.ndarray
    # Trace metadata only, excluded from model tensors and padding.
    traffic_uids: tuple[int, ...]
    target_uids: tuple[int | None, int | None]
    target_reference_seconds: float


class TrajectoryContextCache:
    """One immutable pre-action state, shared by every ego in that state."""
    def __init__(self, env):
        self.env = env
        if env.config.holding_speed_contract != 'nominal_until_release':
            raise ValueError('V2.5 currently requires the nominal holder contract')
        self.now = float(env.now)
        self.snapshot = SimpleNamespace(now=env.now, active=env.active, airspace=env.airspace)
        # Private prediction caches: never write environment safety/telemetry.
        self.predictor = SafetyFilter(env.config)
        self.uids = tuple(sorted(env.active))
        self.forecasts = {uid: self.positions(env.active[uid], self.now+np.asarray(FORECAST_SECONDS))
                          for uid in self.uids}

    def positions(self, uav, times):
        if uav.committed_plan is not None:
            return np.asarray(self.predictor._committed_positions(uav.committed_plan, self.snapshot, times))
        if uav.ring_role is not None:
            return np.asarray(self.predictor._holder_positions(uav, self.snapshot, times))
        return np.repeat(np.asarray(uav.position)[None, :], len(times), axis=0)

    def target_neighbors(self, ego):
        target = {RingRole.OUTER: RingRole.MIDDLE, RingRole.MIDDLE: RingRole.INNER}.get(ego.ring_role)
        if target is None:
            return None, None, 0.
        source, ring = self.env.airspace.actual_ring(ego.ring_role), self.env.airspace.actual_ring(target)
        # Keep the old baseline horizon so gap correction is explicit. This
        # is a named reference, not the arrival time of any policy candidate.
        horizon = abs(source.radius-ring.radius)/max(1., ring.nominal_speed)+6.
        expected = (_angle(ego)+ring.nominal_speed*horizon/ring.radius) % (2*math.pi)
        candidates = []
        for uid in self.uids:
            other = self.env.active[uid]
            if uid == ego.uid:
                continue
            plan = other.committed_plan
            # A committed departure is no longer an indefinite ring holder.
            if plan is None:
                include = other.ring_role is target
            else:
                include = (TARGET_ROLE.get(plan.metadata.get('target_phase')) is target
                           and plan.end_time <= self.now+horizon+1e-9)
            if include:
                position = self.positions(other, np.asarray([self.now+horizon]))[0]
                angle = math.atan2(position[1], position[0]) % (2*math.pi)
                candidates.append((uid, angle))
        if not candidates:
            return None, None, horizon
        forward, _, predecessor = min(((angle-expected) % (2*math.pi), uid, uid) for uid, angle in candidates)
        backward, _, successor = min(((expected-angle) % (2*math.pi), uid, uid) for uid, angle in candidates)
        def make(uid, gap):
            return _slot(ego, self.env.active[uid], self.env.airspace,
                         path_gap=gap, horizon=self.env.config.shield_horizon_seconds)
        return make(predecessor, forward*ring.radius), make(successor, -backward*ring.radius), horizon

    def build(self, ego):
        if float(self.env.now) != self.now or tuple(sorted(self.env.active)) != self.uids:
            raise ValueError('trajectory context cache belongs to a different state')
        base = build_low_observation_v2_4(ego, self.env).copy()
        predecessor, successor, horizon = self.target_neighbors(ego)
        for name, slot in (('target_ring_predecessor', predecessor), ('target_ring_successor', successor)):
            base[LOW_OBS_V2_4.slices[name]] = _slot_vector(slot, self.env.config) if slot is not None else np.zeros(9)
        geometry = []
        for role in RingRole:
            ring = self.env.airspace.actual_ring(role)
            geometry.extend((ring.radius/POSITION_SCALE, ring.height/POSITION_SCALE, ring.nominal_speed/VELOCITY_SCALE))
        ego_values = np.concatenate((base, _v2_2_scheduler_context(ego, self.env), geometry)).astype(np.float32)
        theta = _angle(ego)
        basis = np.asarray(((math.cos(theta), math.sin(theta), 0.),
                            (-math.sin(theta), math.cos(theta), 0.), (0., 0., 1.)))
        def vector(value, scale):
            return (np.asarray(value) @ basis.T / scale).reshape(-1).tolist()
        rows, uids = [], []
        for uid in self.uids:
            if uid == ego.uid:
                continue
            other = self.env.active[uid]
            plan = other.committed_plan
            phase = FlightPhase.FINAL if other.phase is FlightPhase.EMERGENCY_LANDING else other.phase
            endpoint = other.position if plan is None else plan.points[-1].position
            end_velocity = other.velocity if plan is None else plan.points[-1].velocity
            remaining = 0. if plan is None else max(0., plan.end_time-self.now)
            values = ([float(phase == p) for p in PHASES]
                + [float(other.ring_role is r) for r in RingRole] + [float(plan is not None)]
                + [float(plan is not None and plan.metadata.get('target_phase') == p) for p in TARGET_PHASES]
                + [remaining/(remaining+TIME_SCALE)]
                + vector(other.position-ego.position, POSITION_SCALE)
                + vector(other.velocity-ego.velocity, VELOCITY_SCALE)
                + vector(endpoint-ego.position, POSITION_SCALE)
                + vector(end_velocity, VELOCITY_SCALE)
                + vector(self.forecasts[uid]-ego.position, POSITION_SCALE))
            rows.append(values); uids.append(int(uid))
        traffic = np.asarray(rows, dtype=np.float32).reshape(-1, TRAFFIC_SCHEMA.dimension)
        if ego_values.shape != (105,) or not np.isfinite(ego_values).all() or not np.isfinite(traffic).all():
            raise ValueError('invalid V2.5 observation')
        return TrajectoryContextObservation(ego_values, traffic, np.ones(len(traffic), dtype=bool),
            tuple(uids), (None if predecessor is None else predecessor.uid, None if successor is None else successor.uid), horizon)


def build_low_observation_v2_5(uav, env, *, cache=None):
    return (cache or TrajectoryContextCache(env)).build(uav)


def pad_trajectory_context(observations):
    """Numeric tensors only. No UID, label, snapshot ID or future action."""
    if not observations:
        raise ValueError('empty observation batch')
    count = max(len(o.traffic) for o in observations)
    traffic = np.zeros((len(observations), count, 50), dtype=np.float32)
    mask = np.zeros((len(observations), count), dtype=bool)
    for i, obs in enumerate(observations):
        traffic[i, :len(obs.traffic)] = obs.traffic
        mask[i, :len(obs.traffic)] = obs.traffic_mask
    return np.stack([o.ego for o in observations]), traffic, mask
