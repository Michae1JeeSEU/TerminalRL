from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple
import math
import numpy as np

from ..config import ExperimentMode
from ..core.config import SimConfig
from ..core.baseline import command_for_aircraft
from ..core.arrival_process import ArrivalProcess
from ..control.geometry_event_gate import GateConfig, GateMetrics, GateMode, GeometryEventGate
from ..core.dynamic_airspace import AirspaceBounds, DynamicAirspace, DynamicRingSpec
from ..core.geometry import distance_3d, norm_xy, wrap_angle, clamp
from ..core.lifecycle import UAVLifecycleState, UAVOutcomeRecord
from ..core.observations import AgentObservationBuilder
from ..core.ring_capacity import geometric_ring_capacity
from ..core.rewards import RewardTracker
from ..core.safety_filter import SafetyFilter
from ..core.scenario import build_initial_2x4_queue
from ..core.specs import BoxSpec
from ..core.trajectory_generator import TrajectoryGenerator, TrajectoryPlan
from ..core.uav import UAVState, FlightPhase
from ..core.vtac import VTAC
from ..core.rendering import plot_snapshot
from ..metrics.nominal_schedule import NominalScheduleEstimator


DEFAULT_EPISODE_METRIC_KEYS = [
    "time",
    "completion_time",
    "step",
    "completion_step",
    "finished",
    "sequenced",
    "nmac_count",
    "collision_count",
    "mean_completion_step_finished",
    "mean_completion_time_finished",
    "max_completion_step_finished",
    "max_completion_time_finished",
    "total_path_length",
    "mean_path_length",
    "mean_path_detour_rate",
    "total_waiting_ring_distance",
    "mean_waiting_ring_distance",
    "total_cumulative_turn_angle_deg",
    "mean_cumulative_turn_angle_deg",
    "total_cumulative_height_change",
    "mean_cumulative_height_change",
    "total_residual_action_l1",
    "mean_residual_action_l1",
]


@dataclass
class TerminalParallelEnv:
    config: SimConfig = field(default_factory=SimConfig)
    experiment_mode: ExperimentMode | str | None = None
    action_mode: Optional[str] = None
    safety_enabled: bool = field(init=False)
    vtac: VTAC = field(init=False)
    airspace: DynamicAirspace = field(init=False)
    safety_filter: SafetyFilter = field(init=False)
    geometry_gate: GeometryEventGate = field(init=False)
    trajectory_generator: TrajectoryGenerator = field(init=False)
    arrival_process: ArrivalProcess = field(init=False)
    external_arrival_queue: List[dict] = field(default_factory=list)
    arrival_request_count: int = 0
    admitted_arrival_count: int = 0
    blocked_arrival_count: int = 0
    deferred_arrival_count: int = 0
    dropped_arrival_count: int = 0
    battery_crash_count: int = 0
    external_queue_waiting_time: float = 0.0
    outcome_ledger: dict[int, UAVOutcomeRecord] = field(default_factory=dict)
    active_cap_hit: bool = False
    time_at_active_cap_seconds: float = 0.0
    longest_continuous_cap_seconds: float = 0.0
    time_to_first_cap: Optional[float] = None
    _current_cap_streak_seconds: float = 0.0
    _next_arrival_request_id: int = 0
    aircraft: List[UAVState] = field(default_factory=list)
    time: float = 0.0
    step_count: int = 0
    nmac_count: int = 0
    collision_count: int = 0
    pair_opportunities: int = 0
    raw_pair_conflict_samples: int = 0
    conflict_vehicle_seconds: float = 0.0
    total_waiting_time: float = 0.0
    approach_waiting_time: float = 0.0
    outer_waiting_time: float = 0.0
    inner_waiting_time: float = 0.0
    congestion_pressure_time: float = 0.0
    agents: List[str] = field(default_factory=list)
    possible_agents: List[str] = field(default_factory=list)
    observation_builder: AgentObservationBuilder = field(init=False)
    reward_tracker: RewardTracker = field(init=False)
    last_high_level_info: dict = field(default_factory=dict)
    last_gate_metrics: dict = field(default_factory=dict)
    last_high_level_requested_action: Optional[np.ndarray] = None
    last_high_level_applied_action: Optional[np.ndarray] = None
    last_intention_info: Dict[int, dict] = field(default_factory=dict)
    last_release_approval: Dict[int, dict] = field(default_factory=dict)
    last_landing_approval: Dict[int, dict] = field(default_factory=dict)
    pending_high_level_action: Optional[np.ndarray] = None
    _airspace_transition: Optional[dict] = field(default=None, init=False, repr=False)
    safety_override_count: int = 0
    raw_safety_override_count: int = 0
    final_queue_time: float = 0.0
    min_pairwise_distance: float = math.inf
    _safety_override_keys: set[tuple[int, int, str, str]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _unique_conflict_pairs: set[tuple[int, int]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _terminal_collision_pairs: set[tuple[int, int]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _initial_rings: list = field(init=False, repr=False)
    _initial_headways: tuple[float, float] = field(init=False, repr=False)
    _airspace_bounds: AirspaceBounds = field(init=False, repr=False)

    def __post_init__(self) -> None:
        configured_mode = (
            self.config.experiment_mode
            if self.experiment_mode is None
            else self.experiment_mode
        )
        self.experiment_mode = ExperimentMode.parse(configured_mode)
        self.action_mode = (
            self.config.action_mode
            if self.action_mode is None and self.config.action_mode is not None
            else self.action_mode
        ) or self.experiment_mode.action_mode
        if self.action_mode not in {"legacy_residual", "rule", "intention_residual"}:
            raise ValueError(f"unsupported action_mode: {self.action_mode}")
        self.config.experiment_mode = self.experiment_mode.value
        self.config.action_mode = self.action_mode
        self.safety_enabled = bool(
            self.config.safety_enabled and self.experiment_mode.safety_enabled
        )
        if not self.safety_enabled and not self.config.simulation_only:
            raise ValueError("safety may only be disabled in simulation_only mode")
        if (
            self.experiment_mode is ExperimentMode.CRAZYFLIE_LAB
            and not self.safety_enabled
        ):
            raise ValueError("crazyflie_lab cannot disable safety")
        self.config.safety_enabled = self.safety_enabled

        self._initial_rings = copy.deepcopy(self.config.rings)
        self._initial_headways = (
            float(self.config.ring_to_ring_min_time_gap_s),
            float(self.config.final_min_time_gap_s),
        )
        self._airspace_bounds = self._make_airspace_bounds()
        self.airspace = self._make_airspace()
        self._sync_airspace_to_config()

        boundary = self.config.world_boundary or (
            -self.config.world_xy_limit,
            self.config.world_xy_limit,
            -self.config.world_xy_limit,
            self.config.world_xy_limit,
            0.0,
            self.config.world_z_limit,
        )
        self.safety_filter = SafetyFilter(
            safe_distance=self.config.safe_distance,
            boundary=boundary,
            speed_max=self.config.speed_max,
            acceleration_max=self.config.max_acceleration,
            min_segment_duration=max(1e-3, 0.25 * self.config.dt),
        )
        self.trajectory_generator = TrajectoryGenerator(
            center=self.config.center,
            ring_direction=self.config.ring_direction,
            max_speed=self.config.speed_max,
            max_acceleration=self.config.max_acceleration,
            height_min=0.0,
            height_max=self.config.world_z_limit,
        )
        self.geometry_gate = self._make_geometry_gate()
        self.arrival_process = ArrivalProcess(
            demand_level=self.config.traffic_demand_level,
            model=self.config.arrival_model,
            arrival_rate=self.config.arrival_rate,
            max_active_uavs=self.config.max_active_uavs,
            fixed_schedule=self.config.fixed_arrival_schedule,
            total_requested_uavs=self.config.total_requested_uavs,
        )
        self.vtac = VTAC(self.config)
        observed_airspace = (
            self.airspace
            if self.experiment_mode.includes_airspace_observation
            else None
        )
        self.observation_builder = AgentObservationBuilder(
            self.config,
            airspace=observed_airspace,
        )
        self.reward_tracker = RewardTracker(self.config)
        self.possible_agents = [f"uav_{i+1}" for i in range(self.config.initial_aircraft_count)]
        self.agents = self.possible_agents.copy()

    @staticmethod
    def metric_keys() -> List[str]:
        return list(DEFAULT_EPISODE_METRIC_KEYS)

    def action_spec(self, agent: str) -> BoxSpec:
        size = 3 if self.action_mode == "legacy_residual" else 4
        low = -np.ones((size,), dtype=np.float32)
        high = np.ones((size,), dtype=np.float32)
        return BoxSpec(shape=(size,), low=low, high=high)

    def observation_spec(self, agent: str) -> BoxSpec:
        n = self.observation_builder.observation_length()
        low = -np.ones((n,), dtype=np.float32)
        high = np.ones((n,), dtype=np.float32)
        return BoxSpec(shape=(n,), low=low, high=high)

    def global_state(self) -> np.ndarray:
        vecs = [self._build_obs(ac) for ac in self.aircraft]
        return np.concatenate(vecs, dtype=np.float32)

    def seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            np.random.seed(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.seed(seed)
        self.airspace = self._make_airspace()
        self._sync_airspace_to_config()
        if self.experiment_mode.includes_airspace_observation:
            self.observation_builder.airspace = self.airspace
        self.time = 0.0
        self.step_count = 0
        self.nmac_count = 0
        self.collision_count = 0
        self.pair_opportunities = 0
        self.raw_pair_conflict_samples = 0
        self.conflict_vehicle_seconds = 0.0
        self.total_waiting_time = 0.0
        self.approach_waiting_time = 0.0
        self.outer_waiting_time = 0.0
        self.inner_waiting_time = 0.0
        self.congestion_pressure_time = 0.0
        self._unique_conflict_pairs.clear()
        if hasattr(self, "_terminal_collision_pairs"):
            self._terminal_collision_pairs.clear()
        self.last_high_level_info = {}
        self.last_gate_metrics = {}
        self.last_high_level_requested_action = None
        self.last_high_level_applied_action = None
        self.last_intention_info = {}
        self.last_release_approval = {}
        self.last_landing_approval = {}
        self.pending_high_level_action = None
        self._airspace_transition = None
        self.safety_override_count = 0
        self.raw_safety_override_count = 0
        self.external_arrival_queue = []
        self.arrival_request_count = 0
        self.admitted_arrival_count = 0
        self.blocked_arrival_count = 0
        self.deferred_arrival_count = 0
        self.dropped_arrival_count = 0
        self.battery_crash_count = 0
        self.outcome_ledger = {}
        self.external_queue_waiting_time = 0.0
        self.active_cap_hit = False
        self.time_at_active_cap_seconds = 0.0
        self.longest_continuous_cap_seconds = 0.0
        self.time_to_first_cap = None
        self._current_cap_streak_seconds = 0.0
        self._next_arrival_request_id = 0
        self._safety_override_keys.clear()
        self.final_queue_time = 0.0
        self.min_pairwise_distance = math.inf
        self.vtac.reset()
        self.geometry_gate.reset()
        self.arrival_process.reset(seed)
        if self.config.total_requested_uavs is None:
            candidates = build_initial_2x4_queue(self.config)
            self.aircraft = []
            for ac in candidates:
                arrival_id = int(ac.uid - 1)
                self.outcome_ledger[arrival_id] = UAVOutcomeRecord(
                    arrival_id=arrival_id,
                    request_time=0.0,
                    scenario=self.config.traffic_demand_level,
                    seed=seed,
                    lifecycle_state=UAVLifecycleState.EXTERNAL_QUEUE.value,
                    battery_remaining_seconds=float(self.config.battery_endurance_seconds),
                )
                request = {
                    "arrival_id": arrival_id,
                    "request_time": 0.0,
                    "scenario": self.config.traffic_demand_level,
                    "seed": seed,
                    "burst_index": 0,
                    "burst_size": len(candidates),
                    "first_blocked_time": None,
                }
                if any(
                    distance_3d(ac.position, other.position) < self.config.safe_distance
                    for other in self.aircraft
                ):
                    self.external_arrival_queue.append(request)
                    continue
                ac.arrival_id = arrival_id
                ac.request_time = 0.0
                ac.admission_time = 0.0
                ac.battery_endurance_seconds = float(self.config.battery_endurance_seconds)
                ac.remaining_endurance_seconds = float(self.config.battery_endurance_seconds)
                self.outcome_ledger[arrival_id].lifecycle_state = UAVLifecycleState.ACTIVE_APPROACH.value
                self.outcome_ledger[arrival_id].admission_time = 0.0
                self.outcome_ledger[arrival_id].uid = ac.uid
                self.aircraft.append(ac)
            self._next_arrival_request_id = len(candidates)
        else:
            self.aircraft = []
        self.agents = [self._agent_name(ac.uid) for ac in self.aircraft]
        self.possible_agents = list(self.agents)
        for ac in self.aircraft:
            ac.snapshot()
        self.reward_tracker.reset(self.aircraft)
        obs = {self._agent_name(ac.uid): self._build_obs(ac) for ac in self.aircraft}
        infos = {self._agent_name(ac.uid): self._build_info(ac) for ac in self.aircraft}
        return obs, infos

    def _runtime_request_due(self, due: int) -> None:
        for offset in range(max(0, int(due))):
            if (
                self.config.total_requested_uavs is not None
                and self.arrival_request_count >= self.config.total_requested_uavs
            ):
                break
            arrival_id = int(self._next_arrival_request_id)
            request = {
                "arrival_id": arrival_id,
                "request_time": float(self.time),
                "scenario": self.config.traffic_demand_level,
                "seed": None,
                "burst_index": int(offset),
                "burst_size": int(due),
                "first_blocked_time": None,
            }
            self._next_arrival_request_id += 1
            self.arrival_request_count += 1
            self.outcome_ledger[arrival_id] = UAVOutcomeRecord(
                arrival_id=arrival_id,
                request_time=float(request["request_time"]),
                scenario=str(request["scenario"]),
                seed=request["seed"],
                lifecycle_state=UAVLifecycleState.EXTERNAL_QUEUE.value,
            )
            self.external_arrival_queue.append(request)

    def _admit_one_external_arrival(self, request: dict, offset: int, due: int) -> bool:
        inbound = math.radians(self.config.inbound_bearing_deg)
        base_radius = self.config.approach_threshold_radius + self.config.spawn_margin
        estimator = NominalScheduleEstimator(self.config)
        uid = max((aircraft.uid for aircraft in self.aircraft), default=0) + 1
        lane_count = max(1, int(self.config.gate_count))
        lane_index = int(request.get("arrival_id", uid)) % lane_count
        lateral = (lane_index - (lane_count - 1) / 2.0) * self.config.col_spacing
        position = np.array(
            [
                base_radius * math.cos(inbound) - lateral * math.sin(inbound),
                base_radius * math.sin(inbound) + lateral * math.cos(inbound),
                self.config.spawn_altitude,
            ],
            dtype=float,
        )
        xy_limit = float(self.config.world_xy_limit)
        position[0] = clamp(position[0], -xy_limit, xy_limit)
        position[1] = clamp(position[1], -xy_limit, xy_limit)
        position[2] = clamp(position[2], 0.0, self.config.world_z_limit)
        if any(
            not other.finished
            and distance_3d(position, other.position) < self.config.safe_distance
            for other in self.aircraft
        ):
            self.deferred_arrival_count += 1
            return False
        heading = math.atan2(
            self.config.center[1] - position[1],
            self.config.center[0] - position[0],
        )
        aircraft = UAVState(
            uid=uid,
            position=position,
            speed=self.config.nominal_speed,
            heading=heading,
            spawn_time=float(request.get("request_time", self.time)),
            entry_time=float(self.time),
            request_time=float(request.get("request_time", self.time)),
            admission_time=float(self.time),
            arrival_id=int(request.get("arrival_id", uid)),
            battery_endurance_seconds=float(self.config.battery_endurance_seconds),
            remaining_endurance_seconds=max(
                0.0,
                float(self.config.battery_endurance_seconds)
                - (float(self.time) - float(request.get("request_time", self.time))),
            ),
            planned_nominal_landing_time=estimator.estimate(
                position,
                spawn_time=float(request.get("request_time", self.time)),
                sequence_index=uid,
            ),
        )
        aircraft.snapshot()
        self.aircraft.append(aircraft)
        agent = self._agent_name(uid)
        if agent not in self.possible_agents:
            self.possible_agents.append(agent)
        self.agents.append(agent)
        record = self.outcome_ledger.get(aircraft.arrival_id)
        if record is not None:
            record.uid = uid
            record.admission_time = float(self.time)
            record.lifecycle_state = UAVLifecycleState.ACTIVE_APPROACH.value
        self.admitted_arrival_count += 1
        return True

    def _spawn_runtime_arrivals(self) -> None:
        due = self.arrival_process.arrivals_due(self.time)
        if due > 0:
            self._runtime_request_due(due)
        if not self.external_arrival_queue:
            return
        admitted_this_step = 0
        while self.external_arrival_queue:
            active_count = sum(not aircraft.finished for aircraft in self.aircraft)
            if active_count >= self.config.max_active_uavs:
                for request in self.external_arrival_queue:
                    if request.get("first_blocked_time") is None:
                        request["first_blocked_time"] = float(self.time)
                        self.blocked_arrival_count += 1
                break
            request = self.external_arrival_queue[0]
            if not self._admit_one_external_arrival(
                request,
                admitted_this_step,
                max(1, len(self.external_arrival_queue)),
            ):
                break
            self.external_arrival_queue.pop(0)
            admitted_this_step += 1

    def _track_active_cap_and_external_queue(self) -> None:
        active_count = sum(not aircraft.finished for aircraft in self.aircraft)
        at_cap = active_count >= self.config.max_active_uavs
        if at_cap:
            if not self.active_cap_hit:
                self.time_to_first_cap = float(self.time)
            self.active_cap_hit = True
            self.time_at_active_cap_seconds += self.config.dt
            self._current_cap_streak_seconds += self.config.dt
            self.longest_continuous_cap_seconds = max(
                self.longest_continuous_cap_seconds,
                self._current_cap_streak_seconds,
            )
        else:
            self._current_cap_streak_seconds = 0.0
        self.external_queue_waiting_time += len(self.external_arrival_queue) * self.config.dt

    def _agent_name(self, uid: int) -> str:
        return f"uav_{uid}"

    def _uid_from_agent(self, agent: str) -> int:
        return int(agent.split("_")[-1])

    def _ac_map(self) -> Dict[int, UAVState]:
        return {a.uid: a for a in self.aircraft}

    def _ring_occupancy_by_index(self) -> dict[int, int]:
        return {int(index): len(queue) for index, queue in self.vtac.queues.items() if index != 0}

    def _apply_residual_action(
        self,
        ac: UAVState,
        base_cmd: Tuple[float, float, float],
        action: np.ndarray
    ) -> Tuple[float, float, float]:
        heading, speed, altitude = base_cmd
        a = np.asarray(action, dtype=np.float32).reshape(3)

        if ac.phase == FlightPhase.RING:
            heading_scale = 0.55
            speed_scale = 0.70
            altitude_scale = 0.0
        else:
            heading_scale = 1.0
            speed_scale = 1.0
            altitude_scale = 1.0

        heading += math.radians(self.config.max_action_heading_delta_deg) * float(a[0]) * heading_scale
        speed += self.config.max_action_speed_delta * float(a[1]) * speed_scale
        altitude += self.config.max_action_altitude_delta * float(a[2]) * altitude_scale

        speed = clamp(speed, self.config.speed_min, self.config.speed_max)
        altitude = clamp(altitude, 0.0, self.config.world_z_limit)
        return wrap_angle(heading), speed, altitude

    def _apply_intention_residual(
        self,
        base_cmd: Tuple[float, float, float],
        action: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Apply traffic-intention residuals without free heading/altitude control."""

        heading, speed, altitude = base_cmd
        intention = np.asarray(action, dtype=np.float32).reshape(4)
        speed += self.config.max_action_speed_delta * float(intention[1])
        speed += self.config.max_slot_speed_delta * float(intention[2])
        return heading, clamp(speed, self.config.speed_min, self.config.speed_max), altitude

    def _prepare_intention_actions(
        self,
        actions: Mapping[str, np.ndarray],
    ) -> tuple[Dict[int, np.ndarray], Dict[int, bool]]:
        filtered_actions: Dict[int, np.ndarray] = {}
        release_requests: Dict[int, bool] = {}
        self.last_intention_info = {}
        for ac in self.aircraft:
            if ac.finished:
                continue
            agent = self._agent_name(ac.uid)
            raw = np.asarray(
                actions.get(agent, np.zeros(4, dtype=np.float32)),
                dtype=np.float32,
            )
            filtered, info = self.safety_filter.filter_intention_action(raw)
            effective_release = float(filtered[0]) + (
                self.config.timing_release_bias * float(filtered[3])
            )
            release_requested = effective_release >= self.config.intention_release_threshold
            filtered_actions[ac.uid] = filtered
            release_requests[ac.uid] = release_requested
            self.last_intention_info[ac.uid] = {
                **info,
                "release_request": float(filtered[0]),
                "speed_residual": float(filtered[1]),
                "slot_residual": float(filtered[2]),
                "timing_residual": float(filtered[3]),
                "effective_release_request": effective_release,
                "release_requested": release_requested,
            }
        return filtered_actions, release_requests

    def should_update_high_level(self) -> bool:
        """Return whether the current low-level step is a high-level boundary."""

        return bool(
            self.experiment_mode.dynamic_airspace
            and self.step_count % self.config.high_level_interval_steps == 0
        )

    def should_sample_metrics(self) -> bool:
        """Return whether the 20 s sensing/metrics window should update gate state."""

        steps = max(1, int(round(self.config.metrics_sampling_interval_seconds / self.config.dt)))
        return bool(self.step_count % steps == 0)

    def _geometry_deviation_l1(self) -> float:
        current = self.airspace.state_vector()
        nominal = self._make_airspace().state_vector()
        return float(np.abs(current[:4] - nominal[:4]).sum())

    def _gate_metrics(self) -> GateMetrics:
        metrics = self.metrics()
        active = [aircraft for aircraft in self.aircraft if not aircraft.finished]
        ring_indices = {ring.index for ring in self.config.rings}
        outer_index = max(ring_indices) if ring_indices else 2
        inner_index = min(ring_indices) if ring_indices else 1
        approach_backlog = sum(
            not aircraft.finished and aircraft.sequence_number is None
            for aircraft in self.aircraft
        )
        external_backlog = len(self.external_arrival_queue)
        outer_pressure = len(self.vtac.queues[outer_index]) / max(1, self.vtac.queue_capacity(outer_index))
        inner_pressure = len(self.vtac.queues[inner_index]) / max(1, self.vtac.queue_capacity(inner_index))
        final_pressure = len(self.vtac.queues[0]) / max(1, self.config.final_queue_capacity)
        previous_time = float(self.last_gate_metrics.get("time", self.time))
        window_seconds = max(self.config.dt, self.time - previous_time)
        recent_arrival_count = float(
            self.arrival_request_count
            - float(self.last_gate_metrics.get("arrival_request_count", self.arrival_request_count))
        )
        recent_landing_count = float(
            metrics["landed_count"]
            - float(self.last_gate_metrics.get("landed_count", metrics["landed_count"]))
        )
        recent_conflict_samples = float(
            self.raw_pair_conflict_samples
            - float(self.last_gate_metrics.get("raw_pair_conflict_samples", self.raw_pair_conflict_samples))
        )
        recent_pair_opportunities = float(
            self.pair_opportunities
            - float(self.last_gate_metrics.get("pair_opportunities", self.pair_opportunities))
        )
        arrival_rate = max(0.0, recent_arrival_count) / window_seconds
        landing_rate = max(0.0, recent_landing_count) / window_seconds
        nominal_rate_scale = max(
            1e-9,
            max(1.0, float(self.config.max_active_uavs))
            / max(self.config.dt, float(self.config.metrics_sampling_interval_seconds)),
        )
        recent_arrivals = float(np.clip(arrival_rate / nominal_rate_scale, 0.0, 1.0))
        recent_landings = float(np.clip(landing_rate / nominal_rate_scale, 0.0, 1.0))
        recent_conflict_rate = float(
            np.clip(recent_conflict_samples / max(1.0, recent_pair_opportunities), 0.0, 1.0)
        )
        backlog_ratio = (approach_backlog + external_backlog) / max(1, self.config.max_active_uavs)
        previous_backlog = float(self.last_gate_metrics.get("approach_backlog_ratio", backlog_ratio))
        backlog_growth = float(np.clip(backlog_ratio - previous_backlog, -1.0, 1.0))
        active_ratio = len(active) / max(1, self.config.max_active_uavs)
        imbalance = float(
            np.clip(
                (arrival_rate - landing_rate) / max(arrival_rate + landing_rate, 1.0 / window_seconds),
                -1.0,
                1.0,
            )
        )
        pressure = float(max(outer_pressure, inner_pressure, final_pressure, active_ratio, backlog_ratio))
        gate_metrics = GateMetrics(
            normalized_congestion=float(np.clip(pressure, 0.0, 1.0)),
            approach_backlog_ratio=float(backlog_ratio),
            occupancy_pressure=float(max(outer_pressure, inner_pressure, final_pressure)),
            backlog_growth=float(backlog_growth),
            normalized_conflict_rate=float(recent_conflict_rate),
            recent_arrivals=float(max(0.0, recent_arrivals)),
            recent_landings=float(max(0.0, recent_landings)),
            arrival_service_imbalance=float(imbalance),
            active_capacity_ratio=float(active_ratio),
        )
        self.last_gate_metrics = {
            "time": float(self.time),
            "spawned_count": float(len(self.aircraft)),
            "arrival_request_count": float(self.arrival_request_count),
            "landed_count": float(metrics["landed_count"]),
            "raw_pair_conflict_samples": float(self.raw_pair_conflict_samples),
            "pair_opportunities": float(self.pair_opportunities),
            "approach_backlog_ratio": float(backlog_ratio),
            "backlog_growth": float(backlog_growth),
            "arrival_service_imbalance": float(imbalance),
            "recent_arrivals": float(recent_arrivals),
            "recent_landings": float(recent_landings),
            "recent_conflict_rate": float(recent_conflict_rate),
            "external_queue_pressure": float(external_backlog / max(1, self.config.max_active_uavs)),
        }
        return gate_metrics

    def _update_geometry_gate(self, *, force_sample: bool = False) -> dict:
        if not (force_sample or self.geometry_gate.should_sample(self.time)):
            return self.geometry_gate.info(
                self.time,
                geometry_deviation_l1=self._geometry_deviation_l1(),
            )
        metrics = self._gate_metrics()
        return self.geometry_gate.update(
            metrics,
            time_seconds=self.time,
            geometry_deviation_l1=self._geometry_deviation_l1(),
            force_sample=force_sample,
        )

    def get_high_level_observation(self) -> np.ndarray:
        """Return system-level traffic and airspace features for a future policy."""

        ring_occupancy = []
        for ring in sorted(self.config.rings, key=lambda item: item.index, reverse=True):
            ring_occupancy.append(
                len(self.vtac.queues[ring.index]) / max(1, ring.capacity)
            )
        final_occupancy = len(self.vtac.queues[0]) / max(
            1,
            self.config.final_queue_capacity,
        )
        active_count = sum(not aircraft.finished for aircraft in self.aircraft)
        active_fraction = active_count / max(1, self.config.max_active_uavs)
        active_aircraft = [aircraft for aircraft in self.aircraft if not aircraft.finished]
        pair_count = len(active_aircraft) * (len(active_aircraft) - 1) // 2
        conflict_risk = (
            0.0
            if pair_count <= 0
            else len(self.safety_filter.check_pairwise_distance(active_aircraft))
            / pair_count
        )
        interval_phase = (
            self.step_count % self.config.high_level_interval_steps
        ) / self.config.high_level_interval_steps
        gate_info = self._update_geometry_gate(force_sample=False)
        gate_vector = self.geometry_gate.observation_vector(
            time_seconds=self.time,
            geometry_deviation_l1=self._geometry_deviation_l1(),
            backlog_trend=float(self.last_gate_metrics.get("backlog_growth", 0.0)),
            arrival_service_imbalance=float(
                self.last_gate_metrics.get("arrival_service_imbalance", 0.0)
            ),
        )
        previous_action = (
            np.zeros(8, dtype=np.float32)
            if self.last_high_level_applied_action is None
            else self.last_high_level_applied_action.astype(np.float32)
        )
        total_requested = (
            self.config.total_requested_uavs
            if self.config.total_requested_uavs is not None
            else max(1, len(self.outcome_ledger), self.arrival_request_count)
        )
        remaining_fraction = (
            total_requested - self.terminal_outcome_count()
        ) / max(1, total_requested)
        not_yet_due_fraction = (
            max(0.0, float(total_requested) - float(self.arrival_request_count))
            / max(1.0, float(total_requested))
            if self.config.total_requested_uavs is not None
            else 0.0
        )
        terminal_fraction = self.terminal_outcome_count() / max(1, total_requested)
        battery_values = [
            ac.remaining_endurance_seconds
            for ac in active_aircraft
            if ac.remaining_endurance_seconds is not None
        ]
        battery_risk = 0.0
        if battery_values:
            battery_risk = 1.0 - min(battery_values) / max(1e-9, self.config.battery_endurance_seconds)
        outer_capacity = self.airspace.outer_ring.capacity / max(1, self.config.max_active_uavs)
        inner_capacity = self.airspace.inner_ring.capacity / max(1, self.config.max_active_uavs)
        external_queue_pressure = len(self.external_arrival_queue) / max(1, self.config.max_active_uavs)
        recent_arrivals = float(self.last_gate_metrics.get("recent_arrivals", 0.0))
        recent_landings = float(self.last_gate_metrics.get("recent_landings", 0.0))
        recent_imbalance = float(self.last_gate_metrics.get("arrival_service_imbalance", 0.0))
        recent_conflict = float(self.last_gate_metrics.get("recent_conflict_rate", 0.0))
        trigger_persistence = self.geometry_gate.trigger_persistence_count / max(1, self.geometry_gate.config.k_on)
        recovery_persistence = self.geometry_gate.recovery_persistence_count / max(1, self.geometry_gate.config.k_off)
        return np.asarray(
            [
                *self.airspace.normalized_state_vector(),
                *ring_occupancy,
                final_occupancy,
                active_fraction,
                conflict_risk,
                interval_phase,
                *gate_vector,
                *previous_action,
                outer_capacity,
                inner_capacity,
                external_queue_pressure,
                remaining_fraction,
                not_yet_due_fraction,
                terminal_fraction,
                recent_arrivals,
                recent_landings,
                recent_imbalance,
                recent_conflict,
                battery_risk,
                float(np.clip(trigger_persistence, 0.0, 1.0)),
                float(np.clip(recovery_persistence, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def apply_high_level_action(
        self,
        action: np.ndarray,
        *,
        force: bool = False,
    ) -> dict:
        """Apply at a high-level boundary, otherwise cache the latest action."""

        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (DynamicAirspace.ACTION_SIZE,):
            raise ValueError(
                f"airspace action must have shape ({DynamicAirspace.ACTION_SIZE},)"
            )
        if (
            self.experiment_mode.dynamic_airspace
            and not force
            and not self.should_update_high_level()
        ):
            self.pending_high_level_action = raw.copy()
            info = {
                "applied": False,
                "cached": True,
                "clipped": False,
                "violations": ["high_level_interval"],
                "old_state": self.airspace.state_vector(),
                "new_state": self.airspace.state_vector(),
                "step": self.step_count,
            }
            self.last_high_level_info = info
            return info

        self._update_geometry_gate(force_sample=False)
        gate_action, gate_info = self.geometry_gate.transform_action(
            raw,
            time_seconds=self.time,
            geometry_deviation_l1=self._geometry_deviation_l1(),
        )
        filtered, safety_info = self.safety_filter.filter_airspace_action(
            gate_action,
            self.airspace,
        )
        info = self.airspace.apply_high_level_action(
            filtered,
            occupancy_by_ring=self._ring_occupancy_by_index(),
        )
        self._sync_airspace_to_config()
        geometry_changed = self.geometry_gate.mark_geometry_applied(
            self.time,
            info["old_state"],
            info["new_state"],
        )
        self._start_airspace_transition_if_needed(info)
        combined_violations = list(
            dict.fromkeys(
                [
                    *safety_info["violations"],
                    *info["violations"],
                ]
            )
        )
        info["violations"] = combined_violations
        info["clipped"] = bool(info["clipped"] or safety_info["modified"])
        info["safety_filter"] = safety_info
        info["geometry_gate"] = self.geometry_gate.info(
            self.time,
            geometry_deviation_l1=self._geometry_deviation_l1(),
        )
        info["geometry_gate"].update(gate_info)
        info["requested_action"] = raw.copy()
        info["applied_action"] = filtered.copy()
        info["requested_geometry_action"] = gate_info["requested_geometry_action"]
        info["applied_geometry_action"] = filtered[:4].copy()
        info["geometry_changed"] = bool(geometry_changed)
        info["applied"] = True
        info["cached"] = False
        info["step"] = self.step_count
        self.pending_high_level_action = None
        self.last_high_level_info = info
        self.last_high_level_requested_action = raw.copy()
        self.last_high_level_applied_action = filtered.copy()
        return info

    def _start_airspace_transition_if_needed(self, info: dict) -> None:
        duration = float(self.config.geometry_transition_seconds)
        if duration <= 0.0 or not info.get("dynamic_enabled", True):
            self._airspace_transition = None
            return
        old_state = np.asarray(info["old_state"], dtype=np.float64)
        target_state = np.asarray(info["new_state"], dtype=np.float64)
        if np.allclose(old_state, target_state, rtol=0.0, atol=1e-12):
            self._airspace_transition = None
            return
        actual_state = old_state.copy()
        actual_state[4:] = target_state[4:]
        transition_start_state = actual_state.copy()
        transition_target_state = target_state.copy()
        transition_target_state[4:] = actual_state[4:]
        self.airspace.set_state_vector(actual_state)
        self._sync_airspace_to_config()
        self._airspace_transition = {
            "start_time": float(self.time),
            "duration": duration,
            "start_state": transition_start_state,
            "target_state": transition_target_state,
        }
        info["transition_active"] = True
        info["transition_seconds"] = duration
        info["target_state"] = target_state.copy()
        info["actual_state"] = self.airspace.state_vector()

    def _advance_airspace_transition(self) -> None:
        transition = self._airspace_transition
        if transition is None:
            return
        elapsed = max(0.0, float(self.time) - float(transition["start_time"]))
        ratio = min(1.0, elapsed / max(1e-9, float(transition["duration"])))
        state = (
            np.asarray(transition["start_state"], dtype=np.float64)
            + ratio
            * (
                np.asarray(transition["target_state"], dtype=np.float64)
                - np.asarray(transition["start_state"], dtype=np.float64)
            )
        )
        self.airspace.set_state_vector(state)
        self._sync_airspace_to_config()
        if ratio >= 1.0:
            self._airspace_transition = None

    def _candidate_transition_plan(
        self,
        aircraft: UAVState,
        target_stage: str,
    ) -> TrajectoryPlan:
        rings = {ring.index: ring for ring in self.airspace.to_ring_specs()}
        if target_stage == "outer_to_inner":
            source = rings[aircraft.assigned_ring]
            target = rings[self.vtac.inner_ring_index]
            distance_scale = abs(source.radius - target.radius) + abs(
                source.height - target.height
            )
            duration = max(
                5.0 * self.config.dt,
                2.0 * distance_scale / max(self.config.speed_min, target.speed),
            )
            return self.trajectory_generator.ring_transfer(
                aircraft,
                source,
                target,
                duration,
                self.config.dt,
            )
        if target_stage == "inner_to_final":
            target = self.vtac.final_entry_hold_point(aircraft)
            distance = distance_3d(aircraft.position, target)
            duration = max(
                5.0 * self.config.dt,
                2.0 * distance / max(self.config.speed_min, self.config.final_entry_speed),
            )
            return self.trajectory_generator.final_approach(
                aircraft,
                {"position": target},
                duration,
                self.config.dt,
            )
        if target_stage == "final_to_land":
            target = np.asarray(self.config.center, dtype=float).copy()
            target[2] = max(float(target[2]), float(self.config.touchdown_max_height))
            distance = distance_3d(aircraft.position, target)
            duration = max(
                5.0 * self.config.dt,
                2.0 * distance / max(self.config.final_approach_speed, 1e-6),
            )
            return self.trajectory_generator.landing(
                aircraft,
                {"position": target},
                duration,
                self.config.dt,
            )
        raise ValueError(f"unsupported target_stage: {target_stage}")

    def _transition_stage(self, aircraft: UAVState) -> Optional[str]:
        if aircraft.phase != FlightPhase.RING or aircraft.assigned_ring in (None, 0):
            return None
        if aircraft.assigned_ring == self.vtac.inner_ring_index:
            return "inner_to_final"
        return "outer_to_inner"

    def _stage_target_ring_index(self, aircraft: UAVState, stage: str) -> int:
        if stage == "inner_to_final":
            return 0
        if stage == "outer_to_inner":
            if aircraft.assigned_ring is None:
                raise ValueError("outer_to_inner requires an assigned source ring")
            return int(aircraft.assigned_ring) - 1
        raise ValueError(f"unsupported transition stage: {stage}")

    def _vtac_transition_ready(
        self,
        aircraft: UAVState,
        stage: str,
        active: List[UAVState],
    ) -> bool:
        """Return whether VTAC could consume a safety-approved release now."""

        if aircraft.assigned_ring in (None, 0):
            return False
        source_ring_idx = int(aircraft.assigned_ring)
        target_ring_idx = self._stage_target_ring_index(aircraft, stage)
        if not self.vtac.queue_has_space(target_ring_idx):
            return False
        if not self.vtac.is_head_of_queue(source_ring_idx, aircraft.uid):
            return False
        aircraft_map = {item.uid: item for item in active}
        return self.vtac._eligible_for_inward_transfer(
            aircraft,
            source_ring_idx,
            target_ring_idx,
            aircraft_map,
            self.step_count,
        )

    def _final_occupied_by_other(self, aircraft: UAVState) -> bool:
        return any(uid != aircraft.uid for uid in self.vtac.queues[0])

    def _landing_pad_occupied_by_other(self, aircraft: UAVState) -> bool:
        center = np.asarray(self.config.center, dtype=float)
        return any(
            other.uid != aircraft.uid
            and not other.finished
            and other.phase == FlightPhase.FINAL
            and other.final_entry_fix_reached
            and norm_xy(other.position - center)
            <= max(
                self.config.touchdown_accept_radius,
                self.config.vertiport_physical_radius,
            )
            and other.position[2]
            <= max(
                self.config.touchdown_max_height,
                2.0 * self.config.touchdown_accept_radius,
            )
            for other in self.aircraft
        )

    def _record_safety_override(
        self,
        uid: int,
        target_stage: str,
        reason: str | None,
    ) -> None:
        """Count raw rejects while de-duplicating repeated HOLDs per HRL interval."""

        self.raw_safety_override_count += 1
        interval = self.step_count // max(1, self.config.high_level_interval_steps)
        key = (interval, int(uid), str(target_stage), str(reason or "unknown"))
        if key not in self._safety_override_keys:
            self._safety_override_keys.add(key)
            self.safety_override_count += 1

    def _prepare_release_approvals(
        self,
        requested: Mapping[int, bool],
    ) -> Dict[int, bool]:
        """Run SafetyFilter approval before VTAC or landing progression."""

        approved_requests: Dict[int, bool] = {}
        self.last_release_approval = {}
        self.last_landing_approval = {}
        active = [aircraft for aircraft in self.aircraft if not aircraft.finished]
        for aircraft in active:
            policy_requested = bool(requested.get(aircraft.uid, True))
            stage = self._transition_stage(aircraft)
            if stage is not None:
                if not self.safety_enabled:
                    approval = {
                        "approved": policy_requested,
                        "is_safe": None,
                        "decision": "RELEASE" if policy_requested else "HOLD",
                        "target_stage": stage,
                        "violations": ["safety_disabled"],
                        "reason": "safety_disabled",
                        "filtered_plan": None,
                        "diagnostics": {},
                    }
                elif not policy_requested:
                    approval = {
                        "approved": False,
                        "is_safe": True,
                        "decision": "HOLD",
                        "target_stage": stage,
                        "violations": ["policy_deferred"],
                        "reason": "policy_deferred",
                        "filtered_plan": None,
                        "diagnostics": {},
                    }
                elif not self._vtac_transition_ready(aircraft, stage, active):
                    approval = {
                        "approved": False,
                        "is_safe": True,
                        "decision": "HOLD",
                        "target_stage": stage,
                        "violations": ["vtac_not_eligible"],
                        "reason": "vtac_not_eligible",
                        "filtered_plan": None,
                        "diagnostics": {
                            "ring_slot_active": bool(aircraft.ring_slot_active),
                            "ring_established_step": aircraft.ring_established_step,
                            "assigned_ring": aircraft.assigned_ring,
                        },
                    }
                else:
                    candidate = self._candidate_transition_plan(aircraft, stage)
                    trajectory_states = active
                    if stage == "outer_to_inner":
                        trajectory_states = [
                            other
                            for other in active
                            if other.uid == aircraft.uid
                            or other.assigned_ring != aircraft.assigned_ring
                        ]
                    approval = self.safety_filter.approve_release(
                        aircraft,
                        stage,
                        active,
                        candidate_plan=candidate,
                        trajectory_states=trajectory_states,
                        final_occupied=(
                            stage == "inner_to_final"
                            and self._final_occupied_by_other(aircraft)
                        ),
                    )
                    if not approval["approved"]:
                        self._record_safety_override(
                            aircraft.uid,
                            stage,
                            approval.get("reason"),
                        )
                approved_requests[aircraft.uid] = bool(approval["approved"])
                self.last_release_approval[aircraft.uid] = approval
            else:
                approved_requests[aircraft.uid] = policy_requested

            if aircraft.phase == FlightPhase.FINAL and aircraft.final_entry_fix_reached:
                if not self.safety_enabled:
                    landing_approval = {
                        "approved": policy_requested,
                        "is_safe": None,
                        "decision": "RELEASE" if policy_requested else "HOLD",
                        "target_stage": "final_to_land",
                        "violations": ["safety_disabled"],
                        "reason": "safety_disabled",
                        "filtered_plan": None,
                        "diagnostics": {},
                    }
                elif not policy_requested:
                    landing_approval = {
                        "approved": False,
                        "is_safe": True,
                        "decision": "HOLD",
                        "target_stage": "final_to_land",
                        "violations": ["policy_deferred"],
                        "reason": "policy_deferred",
                        "filtered_plan": None,
                        "diagnostics": {},
                    }
                else:
                    candidate = self._candidate_transition_plan(
                        aircraft,
                        "final_to_land",
                    )
                    landing_approval = self.safety_filter.approve_release(
                        aircraft,
                        "final_to_land",
                        active,
                        candidate_plan=candidate,
                        landing_pad_occupied=self._landing_pad_occupied_by_other(
                            aircraft
                        ),
                    )
                    if not landing_approval["approved"]:
                        self._record_safety_override(
                            aircraft.uid,
                            "final_to_land",
                            landing_approval.get("reason"),
                        )
                self.last_landing_approval[aircraft.uid] = landing_approval
        return approved_requests

    def trajectory_plan_for(
        self,
        aircraft: UAVState,
        *,
        duration: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> TrajectoryPlan:
        """Build and safety-filter a dry-run plan for the aircraft's stage."""

        plan_duration = (
            max(self.config.dt, 5.0 * self.config.dt)
            if duration is None
            else float(duration)
        )
        plan_dt = self.config.dt if dt is None else float(dt)
        if aircraft.phase == FlightPhase.RING and aircraft.assigned_ring not in (None, 0):
            ring = next(
                ring
                for ring in self.airspace.to_ring_specs()
                if ring.index == aircraft.assigned_ring
            )
            plan = self.trajectory_generator.ring_hold(
                aircraft,
                ring,
                plan_duration,
                plan_dt,
            )
        elif aircraft.phase == FlightPhase.FINAL:
            pad = {"position": np.asarray(self.config.center, dtype=float)}
            plan = self.trajectory_generator.final_approach(
                aircraft,
                pad,
                plan_duration,
                plan_dt,
            )
        else:
            plan = self.trajectory_generator.emergency_hover(
                aircraft,
                plan_duration,
                plan_dt,
            )
        filtered_plan, filter_info = self.safety_filter.filter_trajectory_plan(plan)
        if filter_info["modified"]:
            self._record_safety_override(
                aircraft.uid,
                f"trajectory_plan_for_{plan.mode}",
                filter_info.get("reason"),
            )
        filtered_plan.metadata["safety_filter"] = filter_info
        return filtered_plan

    def _make_airspace_bounds(self) -> AirspaceBounds:
        return AirspaceBounds(
            radius_min=max(1e-3, 2.0 * self.config.vertiport_radius),
            radius_max=self.config.approach_threshold_radius,
            height_min=0.0,
            height_max=self.config.world_z_limit,
            speed_min=self.config.speed_min,
            speed_max=self.config.speed_max,
            min_ring_gap=max(1e-3, self.config.ring_capture_radius_tolerance),
            min_height_gap=max(1e-3, self.config.ring_capture_height_tolerance),
            max_radius_change_per_update=self.config.ring_capture_radius_tolerance,
            max_height_change_per_update=self.config.ring_capture_height_tolerance,
            max_speed_change_per_update=self.config.max_action_speed_delta,
            headway_min=max(0.1, self.config.dt),
            headway_max=max(
                60.0,
                3.0 * self.config.ring_to_ring_min_time_gap_s,
                3.0 * self.config.final_min_time_gap_s,
            ),
            max_headway_change_per_update=max(0.1, self.config.dt),
        )

    def _make_airspace(self) -> DynamicAirspace:
        nominal_speeds = [
            self.config.nominal_speed if ring.speed is None else float(ring.speed)
            for ring in self._initial_rings
        ]
        return DynamicAirspace.from_ring_specs(
            self._initial_rings,
            bounds=self._airspace_bounds,
            nominal_speeds=nominal_speeds,
            ring_headway=self._initial_headways[0],
            final_headway=self._initial_headways[1],
            dynamic_enabled=self.experiment_mode.dynamic_airspace,
            required_separation=self.config.safe_distance,
            ring_separation_margin=self.config.ring_separation_margin_m,
        )

    def _make_geometry_gate(self) -> GeometryEventGate:
        mode = str(self.config.geometry_gate_mode or "none").lower()
        if mode == "auto":
            if self.experiment_mode.value in {"event_gated_hrl", "event_rule"}:
                mode = GateMode.EVENT.value
            elif self.experiment_mode.value == "periodic_hrl":
                mode = GateMode.PERIODIC.value
            else:
                mode = GateMode.NONE.value
        return GeometryEventGate(
            config=GateConfig(
                metrics_sampling_interval_seconds=float(
                    self.config.metrics_sampling_interval_seconds
                ),
                geometry_min_dwell_seconds=float(
                    self.config.geometry_min_dwell_seconds
                ),
                theta_on=float(self.config.geometry_gate_theta_on),
                theta_off=float(self.config.geometry_gate_theta_off),
                k_on=int(self.config.geometry_gate_k_on),
                k_off=int(self.config.geometry_gate_k_off),
                recovery_k=int(self.config.geometry_gate_recovery_k),
                nominal_recovery_tolerance=float(
                    self.config.geometry_gate_nominal_recovery_tolerance
                ),
                max_active_uavs=int(self.config.max_active_uavs),
            ),
            mode=GateMode.parse(mode),
        )

    def _sync_airspace_to_config(self) -> None:
        by_index = {ring.index: ring for ring in self.airspace.to_ring_specs()}
        for ring in self.config.rings:
            dynamic_ring = by_index[ring.index]
            ring.radius = dynamic_ring.radius
            ring.height = dynamic_ring.height
            ring.speed = dynamic_ring.speed
            ring.capacity = dynamic_ring.capacity
        headways = self.airspace.get_headways()
        self.config.ring_to_ring_min_time_gap_s = headways["ring"]
        self.config.final_min_time_gap_s = headways["final"]

    def _build_obs(self, ac: UAVState) -> np.ndarray:
        return self.observation_builder.build(ac, self.aircraft, self.vtac)

    def _build_info(self, ac: UAVState) -> dict:
        return {
            "uid": ac.uid,
            "phase": int(ac.phase),
            "assigned_ring": -1 if ac.assigned_ring is None else ac.assigned_ring,
            "sequence_number": -1 if ac.sequence_number is None else ac.sequence_number,
            "finished": ac.finished,
            "time": self.time,
            "step": self.step_count,
            "path_length": float(ac.path_length_total),
            "path_detour_rate": float(ac.path_detour_rate(self.config.center)),
            "waiting_ring_distance": float(ac.waiting_ring_distance),
            "cumulative_turn_angle_deg": float(math.degrees(ac.cumulative_turn_angle_rad)),
            "cumulative_height_change": float(ac.cumulative_altitude_change),
            "residual_action_l1": float(ac.residual_action_l1_total),
            "experiment_mode": self.experiment_mode.value,
            "action_mode": self.action_mode,
            "airspace_state": self.airspace.state_vector().copy(),
            "geometry_gate": self.geometry_gate.info(
                self.time,
                geometry_deviation_l1=self._geometry_deviation_l1(),
            ),
            "requested_high_level_action": self.last_high_level_requested_action,
            "applied_high_level_action": self.last_high_level_applied_action,
            "intention_action": self.last_intention_info.get(ac.uid),
            "release_approval": self.last_release_approval.get(ac.uid),
            "landing_approval": self.last_landing_approval.get(ac.uid),
            "safety_override_count": self.safety_override_count,
            "raw_safety_override_count": self.raw_safety_override_count,
            "safety_enabled": self.safety_enabled,
            "spawn_time": float(ac.spawn_time),
            "entry_time": ac.entry_time,
            "planned_nominal_landing_time": ac.planned_nominal_landing_time,
            "actual_landing_time": ac.actual_landing_time,
            "landing_delay": ac.landing_delay,
            "arrival_id": ac.arrival_id,
            "request_time": float(ac.request_time),
            "admission_time": ac.admission_time,
            "terminal_outcome": ac.terminal_outcome,
            "terminal_reason": ac.terminal_reason,
            "remaining_endurance_seconds": ac.remaining_endurance_seconds,
        }

    #着陆判定
    def _phase_lifecycle_state(self, ac: UAVState) -> str:
        if ac.finished:
            return ac.terminal_outcome or UAVLifecycleState.UNRESOLVED.value
        if ac.phase == FlightPhase.FINAL:
            return UAVLifecycleState.ACTIVE_FINAL.value
        if ac.phase == FlightPhase.RING:
            if ac.assigned_ring == self.vtac.inner_ring_index:
                return UAVLifecycleState.ACTIVE_RING_INNER.value
            return UAVLifecycleState.ACTIVE_RING_OUTER.value
        return UAVLifecycleState.ACTIVE_APPROACH.value

    def _last_position_tuple(self, ac: UAVState) -> tuple[float, float, float]:
        return (float(ac.position[0]), float(ac.position[1]), float(ac.position[2]))

    def _mark_terminal_outcome(
        self,
        ac: UAVState,
        *,
        outcome: str,
        reason: str,
        collision_partner_ids: list[int] | None = None,
    ) -> None:
        if ac.terminal_outcome is not None:
            return
        ac.mark_terminal(
            step=self.step_count,
            time=self.time,
            outcome=outcome,
            reason=reason,
            collision_partner_ids=collision_partner_ids,
        )
        ac.remaining_endurance_seconds = (
            None
            if ac.battery_endurance_seconds is None
            else max(0.0, float(ac.battery_endurance_seconds) - (self.time - ac.request_time))
        )
        self.vtac.release_finished(ac)
        if outcome == UAVLifecycleState.CRASHED_BATTERY.value:
            self.battery_crash_count += 1
        record = self.outcome_ledger.get(ac.arrival_id) if ac.arrival_id is not None else None
        if record is not None:
            record.uid = ac.uid
            record.set_terminal(
                outcome=outcome,
                time=self.time,
                reason=reason,
                last_phase=self._phase_lifecycle_state(ac),
                last_position=self._last_position_tuple(ac),
                last_speed=float(ac.speed),
                battery_remaining_seconds=ac.remaining_endurance_seconds,
                collision_partners=collision_partner_ids,
            )

    def _update_battery_endurance(self) -> None:
        endurance = float(self.config.battery_endurance_seconds)
        retained_queue: list[dict] = []
        for request in self.external_arrival_queue:
            arrival_id = int(request["arrival_id"])
            remaining = endurance - (self.time - float(request["request_time"]))
            record = self.outcome_ledger.get(arrival_id)
            if remaining <= 0.0:
                if record is not None:
                    record.set_terminal(
                        outcome=UAVLifecycleState.CRASHED_BATTERY.value,
                        time=self.time,
                        reason="battery_depleted_in_external_queue",
                        last_phase=UAVLifecycleState.EXTERNAL_QUEUE.value,
                        battery_remaining_seconds=0.0,
                    )
                self.battery_crash_count += 1
            else:
                if record is not None:
                    record.battery_remaining_seconds = float(remaining)
                retained_queue.append(request)
        self.external_arrival_queue = retained_queue

        for ac in self.aircraft:
            if ac.finished:
                continue
            ac.battery_endurance_seconds = endurance
            ac.remaining_endurance_seconds = max(0.0, endurance - (self.time - ac.request_time))
            record = self.outcome_ledger.get(ac.arrival_id) if ac.arrival_id is not None else None
            if record is not None:
                record.lifecycle_state = self._phase_lifecycle_state(ac)
                record.battery_remaining_seconds = ac.remaining_endurance_seconds
                record.last_phase = self._phase_lifecycle_state(ac)
                record.last_position = self._last_position_tuple(ac)
                record.last_speed = float(ac.speed)
            if ac.remaining_endurance_seconds <= 0.0:
                self._mark_terminal_outcome(
                    ac,
                    outcome=UAVLifecycleState.CRASHED_BATTERY.value,
                    reason=f"battery_depleted_in_{self._phase_lifecycle_state(ac).lower()}",
                )

    def _hard_time_guard_reached(self) -> bool:
        max_seconds = self.config.max_simulation_seconds
        if max_seconds is not None and self.time >= float(max_seconds) - 1e-9:
            return True
        return self.step_count >= self.config.max_steps

    def finite_cohort_complete(self) -> bool:
        total = self.config.total_requested_uavs
        if total is None:
            return False
        return bool(
            self.arrival_request_count >= total
            and len(self.external_arrival_queue) == 0
            and sum(not aircraft.finished for aircraft in self.aircraft) == 0
            and self.terminal_outcome_count() >= total
        )

    def terminal_outcome_count(self) -> int:
        return sum(1 for record in self.outcome_ledger.values() if record.is_terminal)

    def _mark_unresolved_at_guard(self) -> None:
        if self.config.total_requested_uavs is None:
            return
        for request in self.external_arrival_queue:
            record = self.outcome_ledger.get(int(request["arrival_id"]))
            if record is not None and not record.is_terminal:
                record.set_terminal(
                    outcome=UAVLifecycleState.UNRESOLVED.value,
                    time=self.time,
                    reason="HARD_TIME_GUARD_WITH_UNRESOLVED_UAVS",
                    last_phase=UAVLifecycleState.EXTERNAL_QUEUE.value,
                    battery_remaining_seconds=record.battery_remaining_seconds,
                )
        self.external_arrival_queue = []
        for ac in self.aircraft:
            if not ac.finished:
                self._mark_terminal_outcome(
                    ac,
                    outcome=UAVLifecycleState.UNRESOLVED.value,
                    reason="HARD_TIME_GUARD_WITH_UNRESOLVED_UAVS",
                )

    def simulation_outcome_summary(self) -> dict:
        total_requested = (
            int(self.config.total_requested_uavs)
            if self.config.total_requested_uavs is not None
            else int(max(len(self.outcome_ledger), self.arrival_request_count))
        )
        landed = sum(
            record.terminal_outcome == UAVLifecycleState.LANDED.value
            for record in self.outcome_ledger.values()
        )
        crashed_collision = sum(
            record.terminal_outcome == UAVLifecycleState.CRASHED_COLLISION.value
            for record in self.outcome_ledger.values()
        )
        crashed_battery = sum(
            record.terminal_outcome == UAVLifecycleState.CRASHED_BATTERY.value
            for record in self.outcome_ledger.values()
        )
        unresolved = sum(
            record.terminal_outcome == UAVLifecycleState.UNRESOLVED.value
            for record in self.outcome_ledger.values()
        )
        terminal_count = int(landed + crashed_collision + crashed_battery + unresolved)
        active_final = int(sum(not aircraft.finished for aircraft in self.aircraft))
        external_final = int(len(self.external_arrival_queue))
        conservation_ok = bool(
            total_requested
            == int(landed + crashed_collision + crashed_battery + unresolved)
        )
        simulation_valid = bool(conservation_ok and unresolved == 0 and external_final == 0 and active_final == 0)
        return {
            "total_requested": int(total_requested),
            "total_admitted": int(self.admitted_arrival_count),
            "landed": int(landed),
            "crashed_collision": int(crashed_collision),
            "crashed_battery": int(crashed_battery),
            "unresolved": int(unresolved),
            "external_queue_final": external_final,
            "internal_active_final": active_final,
            "terminal_outcome_count": terminal_count,
            "last_request_time": float(
                max((record.request_time for record in self.outcome_ledger.values()), default=0.0)
            ),
            "final_simulation_time": float(self.time),
            "simulation_valid": simulation_valid,
            "conservation_ok": conservation_ok,
            "reason": (
                "OK"
                if simulation_valid
                else "HARD_TIME_GUARD_WITH_UNRESOLVED_UAVS"
                if unresolved > 0 or external_final > 0 or active_final > 0
                else "OUTCOME_CONSERVATION_FAILED"
            ),
        }

    def _update_finish_states(self) -> List[int]:
        newly_finished: List[int] = []
        cx, cy, _ = self.config.center

        for ac in self.aircraft:
            if ac.finished:
                continue

            dx = float(ac.position[0] - cx)
            dy = float(ac.position[1] - cy)
            dist_xy = math.hypot(dx, dy)

        # 你这个 UAVState 没有 velocity 向量，用 speed 和 pitch 计算水平速度
            horizontal_speed = float(ac.speed * math.cos(ac.pitch))
            landing_approved = bool(
                self.last_landing_approval.get(ac.uid, {}).get("approved", False)
            )

            in_touchdown_zone = (
                ac.phase == FlightPhase.FINAL
                and landing_approved
                and dist_xy <= self.config.touchdown_accept_radius
                and ac.position[2] <= self.config.touchdown_max_height
                and horizontal_speed <= self.config.touchdown_max_speed
            )

            if in_touchdown_zone:
                ac.touchdown_hold_steps += 1
            else:
                ac.touchdown_hold_steps = 0

            if ac.touchdown_hold_steps >= self.config.touchdown_hold_steps_required:
                self._mark_terminal_outcome(
                    ac,
                    outcome=UAVLifecycleState.LANDED.value,
                    reason="touchdown_accepted",
                )
                newly_finished.append(ac.uid)



        return newly_finished

    def _track_conflicts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        nmac_pairs: Dict[int, int] = {}
        collision_pairs: Dict[int, int] = {}
        active = [aircraft for aircraft in self.aircraft if not aircraft.finished]
        conflict_vehicle_ids: set[int] = set()
        for i, a in enumerate(active):
            if a.finished:
                continue
            for b in active[i + 1 :]:
                if b.finished:
                    continue
                self.pair_opportunities += 1
                d = distance_3d(a.position, b.position)
                self.min_pairwise_distance = min(self.min_pairwise_distance, d)
                if d < self.config.safe_distance:
                    self.raw_pair_conflict_samples += 1
                    pair = tuple(sorted((a.uid, b.uid)))
                    self._unique_conflict_pairs.add(pair)
                    conflict_vehicle_ids.update(pair)
                if d < self.config.collision_distance:
                    pair = tuple(sorted((a.uid, b.uid)))
                    if pair not in self._terminal_collision_pairs:
                        self._terminal_collision_pairs.add(pair)
                        self.collision_count += 1
                        self._mark_terminal_outcome(
                            a,
                            outcome=UAVLifecycleState.CRASHED_COLLISION.value,
                            reason="collision_distance_breach",
                            collision_partner_ids=[b.uid],
                        )
                        self._mark_terminal_outcome(
                            b,
                            outcome=UAVLifecycleState.CRASHED_COLLISION.value,
                            reason="collision_distance_breach",
                            collision_partner_ids=[a.uid],
                        )
                    collision_pairs[a.uid] = collision_pairs.get(a.uid, 0) + 1
                    collision_pairs[b.uid] = collision_pairs.get(b.uid, 0) + 1
                elif d < self.config.nmac_distance:
                    self.nmac_count += 1
                    nmac_pairs[a.uid] = nmac_pairs.get(a.uid, 0) + 1
                    nmac_pairs[b.uid] = nmac_pairs.get(b.uid, 0) + 1
        self.conflict_vehicle_seconds += len(conflict_vehicle_ids) * self.config.dt
        return nmac_pairs, collision_pairs

    def _track_waiting_and_congestion(self) -> None:
        active = [aircraft for aircraft in self.aircraft if not aircraft.finished]
        approach_count = sum(
            aircraft.phase == FlightPhase.APPROACH or aircraft.sequence_number is None
            for aircraft in active
        )
        external_count = len(self.external_arrival_queue)
        ring_indices = {ring.index for ring in self.config.rings}
        outer_index = max(ring_indices)
        inner_index = min(ring_indices)
        outer_count = len(self.vtac.queues[outer_index])
        inner_count = len(self.vtac.queues[inner_index])
        final_count = len(self.vtac.queues[0])
        self.approach_waiting_time += approach_count * self.config.dt
        self.outer_waiting_time += outer_count * self.config.dt
        self.inner_waiting_time += inner_count * self.config.dt
        self.total_waiting_time += (
            external_count + approach_count + outer_count + inner_count + final_count
        ) * self.config.dt

        outer_capacity = max(1, self.vtac.queue_capacity(outer_index))
        inner_capacity = max(1, self.vtac.queue_capacity(inner_index))
        final_capacity = max(1, self.config.final_queue_capacity)
        active_ratio = len(active) / max(1, self.config.max_active_uavs)
        pressure = max(
            active_ratio,
            external_count / max(1, self.config.max_active_uavs),
            approach_count / max(1, self.config.max_active_uavs),
            outer_count / outer_capacity,
            inner_count / inner_capacity,
            final_count / final_capacity,
        )
        self.congestion_pressure_time += pressure * self.config.dt

    def step(
        self,
        actions: Optional[Mapping[str, np.ndarray]] = None,
        high_level_action: Optional[np.ndarray] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        if actions is None:
            actions = {}
        if high_level_action is not None:
            self.apply_high_level_action(high_level_action)
        elif self.pending_high_level_action is not None and self.should_update_high_level():
            self.apply_high_level_action(
                self.pending_high_level_action.copy(),
                force=True,
            )

        prepared_intentions: Dict[int, np.ndarray] = {}
        release_requests: Optional[Dict[int, bool]] = None
        if self.action_mode == "intention_residual":
            prepared_intentions, release_requests = self._prepare_intention_actions(actions)
        else:
            self.last_intention_info = {}
            release_requests = {
                aircraft.uid: True
                for aircraft in self.aircraft
                if not aircraft.finished
            }
        release_requests = self._prepare_release_approvals(release_requests)

        self.vtac.update(
            self.aircraft,
            current_step=self.step_count,
            release_requests=release_requests,
        )
        applied_actions: Dict[int, np.ndarray] = {}
        commands: Dict[int, Tuple[float, float, float]] = {}
        phase_before_step: Dict[int, FlightPhase] = {}
        for ac in self.aircraft:
            if ac.finished:
                continue
            phase_before_step[ac.uid] = ac.phase
            base_cmd = command_for_aircraft(ac, self.aircraft, self.config, vtac=self.vtac)
            agent = self._agent_name(ac.uid)
            if self.action_mode == "legacy_residual":
                residual = np.asarray(
                    actions.get(agent, np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                )
                residual = np.clip(residual, -1.0, 1.0)
                applied_actions[ac.uid] = residual
                commands[ac.uid] = self._apply_residual_action(ac, base_cmd, residual)
            elif self.action_mode == "intention_residual":
                intention = prepared_intentions[ac.uid]
                applied_actions[ac.uid] = intention
                commands[ac.uid] = self._apply_intention_residual(base_cmd, intention)
            else:
                rule_action = np.zeros(4, dtype=np.float32)
                applied_actions[ac.uid] = rule_action
                commands[ac.uid] = base_cmd
            if (
                ac.phase == FlightPhase.FINAL
                and ac.final_entry_fix_reached
                and not self.last_landing_approval.get(ac.uid, {}).get(
                    "approved",
                    False,
                )
            ):
                commands[ac.uid] = (
                    ac.heading,
                    max(self.config.final_speed_min, self.config.speed_min),
                    float(ac.position[2]),
                )

        for ac in self.aircraft:
            if ac.finished:
                continue
            desired_heading, desired_speed, desired_altitude = commands[ac.uid]

            phase_speed_min = self.config.final_speed_min if ac.phase == FlightPhase.FINAL else self.config.speed_min

            telemetry = ac.update_dynamics(
                desired_heading=desired_heading,
                desired_speed=desired_speed,
                desired_altitude=desired_altitude,
                dt=self.config.dt,
                max_turn_rate_deg=self.config.max_turn_rate_deg,
                max_climb_rate=self.config.max_climb_rate,
                max_descent_rate=self.config.max_descent_rate,
                speed_min=phase_speed_min,
                speed_max=self.config.speed_max,
                max_acceleration=self.config.max_acceleration,
            )
            ac.record_step_metrics(
                heading_step_rad=telemetry["heading_step_rad"],
                dz=telemetry["dz"],
                path_distance=telemetry["path_distance"],
                residual_action=applied_actions.get(ac.uid, np.zeros(3, dtype=np.float32)),
                phase_before_step=phase_before_step.get(ac.uid, ac.phase),
            )
            ac.snapshot()

        self.step_count += 1
        self.time += self.config.dt
        self._advance_airspace_transition()
        self._spawn_runtime_arrivals()
        self._update_battery_endurance()
        self._track_active_cap_and_external_queue()
        self.final_queue_time += len(self.vtac.queues[0]) * self.config.dt
        self._track_waiting_and_congestion()
        self._update_finish_states()
        nmac_pairs, collision_pairs = self._track_conflicts()
        if self._hard_time_guard_reached() and not self.finite_cohort_complete():
            self._mark_unresolved_at_guard()
        if self.should_sample_metrics():
            self._update_geometry_gate(force_sample=True)

        rewards_uid = self.reward_tracker.compute(self.aircraft, applied_actions, nmac_pairs, collision_pairs)

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}
        active_agents: List[str] = []

        timed_out = self._hard_time_guard_reached()
        for ac in self.aircraft:
            agent = self._agent_name(ac.uid)
            observations[agent] = self._build_obs(ac)
            rewards[agent] = float(rewards_uid.get(ac.uid, 0.0))
            terminations[agent] = bool(ac.finished or self.finite_cohort_complete())
            truncations[agent] = bool(timed_out and not ac.finished)
            infos[agent] = self._build_info(ac)
            infos[agent]["nmac_count_total"] = self.nmac_count
            infos[agent]["collision_count_total"] = self.collision_count
            if not (terminations[agent] or truncations[agent]):
                active_agents.append(agent)

        self.agents = active_agents
        return observations, rewards, terminations, truncations, infos

    def per_aircraft_metrics(self) -> Dict[str, dict]:
        return {self._agent_name(ac.uid): ac.metrics_dict(self.config.center) for ac in self.aircraft}

    def uav_outcome_rows(self) -> list[dict]:
        return [
            record.to_row()
            for record in sorted(self.outcome_ledger.values(), key=lambda item: item.arrival_id)
        ]

    def metrics(self) -> Dict[str, float]:
        finished = sum(1 for a in self.aircraft if a.finished)
        landed = sum(
            record.terminal_outcome == UAVLifecycleState.LANDED.value
            for record in self.outcome_ledger.values()
        )
        collision_crashes = sum(
            record.terminal_outcome == UAVLifecycleState.CRASHED_COLLISION.value
            for record in self.outcome_ledger.values()
        )
        battery_crashes = sum(
            record.terminal_outcome == UAVLifecycleState.CRASHED_BATTERY.value
            for record in self.outcome_ledger.values()
        )
        unresolved = sum(
            record.terminal_outcome == UAVLifecycleState.UNRESOLVED.value
            for record in self.outcome_ledger.values()
        )
        total_requested = (
            self.config.total_requested_uavs
            if self.config.total_requested_uavs is not None
            else max(1, len(self.outcome_ledger), self.arrival_request_count)
        )
        active = sum(1 for a in self.aircraft if not a.finished)
        sequenced = sum(1 for a in self.aircraft if a.sequence_number is not None)
        ring_counts = {f"queue_{idx}": len(queue) for idx, queue in sorted(self.vtac.queues.items())}
        current_pair_count = active * (active - 1) // 2
        has_pair = current_pair_count > 0
        normalized_conflict_rate = (
            self.raw_pair_conflict_samples / max(1, self.pair_opportunities)
        )
        normalized_waiting = self.total_waiting_time / max(
            self.config.dt,
            self.time * max(1, self.config.max_active_uavs),
        )
        normalized_congestion = self.congestion_pressure_time / max(
            self.config.dt,
            self.time,
        )
        fraction_time_at_active_cap = self.time_at_active_cap_seconds / max(
            self.config.dt,
            self.time,
        )

        per_ac = [ac.metrics_dict(self.config.center) for ac in self.aircraft]
        path_lengths = [m["path_length"] for m in per_ac]
        detour_rates = [m["path_detour_rate"] for m in per_ac]
        ring_distances = [m["waiting_ring_distance"] for m in per_ac]
        turn_angles = [m["cumulative_turn_angle_deg"] for m in per_ac]
        height_changes = [m["cumulative_height_change"] for m in per_ac]
        residual_l1 = [m["residual_action_l1"] for m in per_ac]

        completion_steps = [float(m["completion_step"]) for m in per_ac if m["completion_step"] is not None]
        completion_times = [float(m["completion_time"]) for m in per_ac if m["completion_time"] is not None]

        return {
            "time": float(self.time),
            "completion_time": float(self.time),
            "step": int(self.step_count),
            "completion_step": int(self.step_count),
            "finished": finished,
            "landed_count": float(landed),
            "active_count": float(active),
            "total_spawned_count": float(len(self.aircraft)),
            "total_requested_uavs": float(total_requested),
            "total_landed_count": float(landed),
            "collision_crash_count": float(collision_crashes),
            "battery_crash_count": float(battery_crashes),
            "unresolved_count": float(unresolved),
            "terminal_outcome_count": float(self.terminal_outcome_count()),
            "landing_success_rate": float(landed / max(1, total_requested)),
            "collision_crash_rate": float(collision_crashes / max(1, total_requested)),
            "battery_crash_rate": float(battery_crashes / max(1, total_requested)),
            "mission_failure_rate": float(
                (collision_crashes + battery_crashes + unresolved) / max(1, total_requested)
            ),
            "total_resolution_rate": float(self.terminal_outcome_count() / max(1, total_requested)),
            "arrival_request_count": float(self.arrival_request_count),
            "admitted_arrival_count": float(self.admitted_arrival_count),
            "blocked_arrival_count": float(self.blocked_arrival_count),
            "deferred_arrival_count": float(self.deferred_arrival_count),
            "dropped_arrival_count": float(self.dropped_arrival_count),
            "external_queue_count": float(len(self.external_arrival_queue)),
            "external_queue_waiting_time": float(self.external_queue_waiting_time),
            "internal_active_count": float(active),
            "internal_capacity_ratio": float(active / max(1, self.config.max_active_uavs)),
            "external_backlog": float(len(self.external_arrival_queue)),
            "total_system_backlog": float(
                len(self.external_arrival_queue)
                + sum(
                    not aircraft.finished and aircraft.sequence_number is None
                    for aircraft in self.aircraft
                )
                + len(self.vtac.queues.get(0, []))
            ),
            "active_cap_hit": float(self.active_cap_hit),
            "time_at_active_cap_seconds": float(self.time_at_active_cap_seconds),
            "fraction_time_at_active_cap": float(fraction_time_at_active_cap),
            "longest_continuous_cap_seconds": float(self.longest_continuous_cap_seconds),
            "time_to_first_cap": float(
                -1.0 if self.time_to_first_cap is None else self.time_to_first_cap
            ),
            "sequenced": sequenced,
            "nmac_count": self.nmac_count,
            "collision_count": self.collision_count,
            "raw_pair_conflict_samples": float(self.raw_pair_conflict_samples),
            "unique_conflict_events": float(len(self._unique_conflict_pairs)),
            "nmac_events": float(self.nmac_count),
            "collision_events": float(self.collision_count),
            "conflict_vehicle_seconds": float(self.conflict_vehicle_seconds),
            "pair_opportunities": float(self.pair_opportunities),
            "has_pair": float(has_pair),
            "current_pair_count": float(current_pair_count),
            "normalized_conflict_rate": float(normalized_conflict_rate),
            "mean_completion_step_finished": float(np.mean(completion_steps)) if completion_steps else 0.0,
            "mean_completion_time_finished": float(np.mean(completion_times)) if completion_times else 0.0,
            "max_completion_step_finished": float(np.max(completion_steps)) if completion_steps else 0.0,
            "max_completion_time_finished": float(np.max(completion_times)) if completion_times else 0.0,
            "total_path_length": float(np.sum(path_lengths)),
            "mean_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "mean_path_detour_rate": float(np.mean(detour_rates)) if detour_rates else 0.0,
            "total_waiting_ring_distance": float(np.sum(ring_distances)),
            "mean_waiting_ring_distance": float(np.mean(ring_distances)) if ring_distances else 0.0,
            "total_cumulative_turn_angle_deg": float(np.sum(turn_angles)),
            "mean_cumulative_turn_angle_deg": float(np.mean(turn_angles)) if turn_angles else 0.0,
            "total_cumulative_height_change": float(np.sum(height_changes)),
            "mean_cumulative_height_change": float(np.mean(height_changes)) if height_changes else 0.0,
            "total_residual_action_l1": float(np.sum(residual_l1)),
            "mean_residual_action_l1": float(np.mean(residual_l1)) if residual_l1 else 0.0,
            "airspace_outer_radius": float(self.airspace.outer_ring.radius),
            "airspace_inner_radius": float(self.airspace.inner_ring.radius),
            "airspace_outer_capacity": float(self.airspace.outer_ring.capacity),
            "airspace_inner_capacity": float(self.airspace.inner_ring.capacity),
            "outer_occupancy_ratio": float(
                len(self.vtac.queues.get(self.airspace.outer_ring.index, []))
                / max(1, self.airspace.outer_ring.capacity)
            ),
            "inner_occupancy_ratio": float(
                len(self.vtac.queues.get(self.airspace.inner_ring.index, []))
                / max(1, self.airspace.inner_ring.capacity)
            ),
            "outer_capacity_delta_from_nominal": float(
                self.airspace.outer_ring.capacity
                - geometric_ring_capacity(
                    self._initial_rings[-1].radius,
                    self.config.safe_distance + self.config.ring_separation_margin_m,
                )
            ),
            "inner_capacity_delta_from_nominal": float(
                self.airspace.inner_ring.capacity
                - geometric_ring_capacity(
                    self._initial_rings[0].radius,
                    self.config.safe_distance + self.config.ring_separation_margin_m,
                )
            ),
            "airspace_outer_height": float(self.airspace.outer_ring.height),
            "airspace_inner_height": float(self.airspace.inner_ring.height),
            "airspace_ring_headway": float(self.airspace.ring_headway),
            "airspace_final_headway": float(self.airspace.final_headway),
            "safety_override_count": float(self.safety_override_count),
            "total_safety_override_count": float(self.safety_override_count),
            "raw_safety_override_count": float(self.raw_safety_override_count),
            "total_raw_safety_override_count": float(self.raw_safety_override_count),
            "final_queue_time": float(self.final_queue_time),
            "approach_waiting_time": float(self.approach_waiting_time),
            "outer_waiting_time": float(self.outer_waiting_time),
            "inner_waiting_time": float(self.inner_waiting_time),
            "total_waiting_time": float(self.total_waiting_time),
            "raw_waiting": float(self.total_waiting_time),
            "raw_backlog": float(
                len(self.external_arrival_queue)
                + sum(
                    not aircraft.finished and aircraft.sequence_number is None
                    for aircraft in self.aircraft
                )
            ),
            "normalized_waiting": float(normalized_waiting),
            "congestion_pressure_time": float(self.congestion_pressure_time),
            "raw_congestion": float(self.congestion_pressure_time),
            "normalized_congestion": float(normalized_congestion),
            "min_pairwise_distance": float(self.min_pairwise_distance),
            "safety_enabled": float(self.safety_enabled),
            **ring_counts,
        }

    def render(self, save_path: Optional[str] = None) -> None:
        plot_snapshot(self.config, self.aircraft, save_path=save_path)

    def close(self) -> None:
        return None
