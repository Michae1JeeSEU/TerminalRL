"""A5-v2 preflight and L1 Full-Teacher PPO training.

The L1 runner intentionally keeps the policy action separate from the
teacher-executed action: PPO's rollout buffer always contains the sampled
``requested_action`` and its old log-probability.  Teacher repairs are stored
only as an auxiliary target and as a reward/diagnostic signal.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import itertools
from pathlib import Path
import json
import random

import numpy as np
import torch

from ..algo import RolloutBuffer, SharedPPO, Transition
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.config import SimConfig
from ..core.observations import LOW_OBS_V2
from ..core.safety_trace import SafetyTraceRecorder
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..envs.parallel_env import TerminalParallelEnv
from .audit_low_teacher_dataset import TeacherDatasetAudit, audit_teacher_dataset
from .low_v2_protocol import LOW_V2_TEACHER_CURRICULUM, phase_config, validate_low_v2_config
from .low_v2_seed_partitions import LowV2SeedPartition


@dataclass(frozen=True)
class LowV2TrainingPreflight:
    observation_dimension: int
    action_dimension: int
    phases: tuple[str, ...]
    configs: tuple[dict, ...]


def prepare_low_v2_training(base: SimConfig | None = None, *, manifest_path: str | Path | None = None) -> LowV2TrainingPreflight:
    """Validate and optionally persist a training manifest; does not train."""
    root = base or SimConfig()
    configs = tuple(phase_config(root, phase) for phase in LOW_V2_TEACHER_CURRICULUM)
    for config, phase in zip(configs, LOW_V2_TEACHER_CURRICULUM):
        validate_low_v2_config(config, for_training=phase.name != "teacher_off")
    result = LowV2TrainingPreflight(LOW_OBS_V2.dimension, 3, tuple(phase.name for phase in LOW_V2_TEACHER_CURRICULUM),
                                    tuple({"experiment_mode": config.experiment_mode, "scheduler_mode": config.scheduler_mode,
                                           "teacher_mode": config.teacher_mode, "teacher_enabled": config.teacher_enabled,
                                           "shield_enabled": config.shield_enabled} for config in configs))
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def authorize_low_v2_training(teacher_dataset_path: str | Path, *, min_corrected_fraction: float = 0.05) -> TeacherDatasetAudit:
    """Gate a future optimizer launch on a provenance-complete dataset audit.

    This is intentionally the last operation available before concrete PPO or
    BC training.  A failed audit must be repaired by data collection, never by
    silently lowering the safety/teacher coverage requirement.
    """
    audit = audit_teacher_dataset(teacher_dataset_path, min_corrected_fraction=min_corrected_fraction)
    if not audit.passed:
        raise RuntimeError("low-v2 training is blocked: " + "; ".join(audit.errors))
    return audit


@dataclass(frozen=True)
class L1FullTeacherProtocol:
    """Frozen, intentionally small L1 curriculum protocol.

    ``episodes_per_update`` independent CPU rollouts are collected against a
    frozen policy snapshot, followed by one central PPO update.  This is the
    synchronous on-policy boundary; workers never mutate policy parameters.
    """

    updates: int = 10
    episodes_per_update: int = 4
    rollout_workers: int = 4
    max_steps: int = 1200
    validation_every: int = 5
    rolling_window_updates: int = 5
    validation_min_effective_episodes: int = 2
    validation_min_release_intents: int = 100
    teacher_bc_weight: float = 0.05
    ppo_epochs: int = 2
    ppo_minibatch_size: int = 256
    optimizer_seed: int = 1701
    stop_on_safety_failure: bool = True

    def __post_init__(self) -> None:
        if min(self.updates, self.episodes_per_update, self.rollout_workers, self.max_steps, self.validation_every,
               self.rolling_window_updates, self.validation_min_effective_episodes, self.validation_min_release_intents,
               self.ppo_epochs, self.ppo_minibatch_size) <= 0:
            raise ValueError("L1 protocol counts must be positive")
        if not 0.0 <= self.teacher_bc_weight <= 1.0:
            raise ValueError("teacher_bc_weight must be in [0, 1]")


class L1SafetyGateError(RuntimeError):
    """Raised before another PPO update when an L1 safety hard gate fails."""


@dataclass(frozen=True)
class L1EpisodeSummary:
    update: int
    episode: int
    seed: int
    reset_mode: str
    demand_model: str
    warm_start_uavs: int
    cohort_uavs: int
    rollout_rows: int
    reward: float
    arrivals: int
    requested: int
    admitted: int
    completed: int
    release_intents: int
    teacher_interventions: int
    shield_interventions: int
    correction_magnitude_sum: float
    correction_magnitudes: tuple[float, ...]
    teacher_correction_components: tuple[tuple[str, ...], ...]
    requested_action_saturation_counts: tuple[int, int, int]
    requested_action_count: int
    completion_delays: tuple[float, ...]
    collisions: int
    nmac: int
    hard_separation_violations: int
    steps: int


@dataclass(frozen=True)
class L1TrainingSummary:
    output_dir: str
    checkpoint_path: str
    mu_hat: float
    medium_rate: float
    updates: int
    rollout_episodes: int
    rollout_rows: int
    safety_passed: bool
    teacher_interventions_per_100_release_intents: float
    first_window_teacher_interventions_per_100_release_intents: float | None
    last_window_teacher_interventions_per_100_release_intents: float | None
    l1b_eligible: bool


# The whole L1 validation partition is fixed by L0.  These indices give
# three warm finite, three from-empty finite and six long-horizon Poisson
# cases, respectively.  The latter is deliberately overrepresented to avoid
# treating a zero-arrival single seed as Poisson evidence.
L1_VALIDATION_SCENARIO_INDICES = (0, 1, 2, 3, 9, 13, 4, 8, 14, 18, 24, 28)


def _read_frozen_capacity(path: str | Path) -> tuple[float, tuple[float, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = payload.get("frozen_capacity", {})
    mu_hat = frozen.get("mu_hat")
    medium = frozen.get("traffic_levels", {}).get("medium")
    if not frozen.get("frozen") or not isinstance(mu_hat, (int, float)) or mu_hat <= 0.0:
        raise RuntimeError("L1 requires a frozen positive C0 mu_hat")
    if not isinstance(medium, list) or len(medium) != 2 or not (0.0 < medium[0] <= medium[1]):
        raise RuntimeError("C0 report lacks a valid frozen medium traffic interval")
    return float(mu_hat), (float(medium[0]), float(medium[1]))


def _read_seed_partition(l0_dataset_path: str | Path) -> LowV2SeedPartition:
    path = Path(l0_dataset_path).parent / "seed_partitions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "A5V2-TRAINING-CURRICULUM-v3":
        raise RuntimeError("L1 requires the frozen A5-v2 seed partition from L0")
    try:
        return LowV2SeedPartition(**payload["seed_partition"])
    except (KeyError, TypeError) as error:
        raise RuntimeError("L1 seed partition is malformed") from error


def _finite_schedule(count: int, rate: float, warm_count: int) -> tuple[float, ...]:
    if not 0 <= warm_count <= count:
        raise ValueError("warm count must be within the finite cohort")
    # From-empty episodes must contain an actual lifecycle witness even when
    # the calibrated medium inter-arrival time exceeds a short pilot horizon.
    # Subsequent demand still follows the frozen medium deterministic interval.
    if warm_count == 0:
        return tuple(index / rate for index in range(count))
    return tuple(0.0 for _ in range(warm_count)) + tuple(index / rate for index in range(1, count - warm_count + 1))


def l1_full_teacher_config(
    *, medium_rate: float, reset_mode: str, demand_model: str, warm_start_uavs: int, cohort_uavs: int, horizon_seconds: float,
) -> SimConfig:
    """Return the exact nominal L1 environment prescribed by the curriculum."""
    if reset_mode not in {"warm_start_rings", "flow_from_empty"}:
        raise ValueError("unsupported L1 reset mode")
    if demand_model not in {"finite_cohort", "poisson"}:
        raise ValueError("unsupported L1 demand model")
    if not 8 <= cohort_uavs <= 20:
        raise ValueError("L1 cohorts must contain 8--20 UAVs")
    if reset_mode == "warm_start_rings" and not 8 <= warm_start_uavs <= 16:
        raise ValueError("L1 warm starts require 8--16 initial UAVs")
    if reset_mode == "flow_from_empty" and warm_start_uavs != 0:
        raise ValueError("from-empty L1 episodes cannot have warm-start UAVs")
    if reset_mode == "warm_start_rings" and cohort_uavs < warm_start_uavs:
        raise ValueError("finite L1 cohort cannot be smaller than its warm start")
    schedule = _finite_schedule(cohort_uavs, medium_rate, warm_start_uavs) if demand_model == "finite_cohort" else ()
    return SimConfig(
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode=reset_mode, warm_start_uavs=warm_start_uavs,
        warm_start_steady_motion=False, arrival_model=demand_model,
        fixed_arrival_schedule=schedule, total_requested_uavs=cohort_uavs if demand_model == "finite_cohort" else None,
        arrival_rate_per_second=medium_rate, max_simulation_seconds=horizon_seconds,
        max_active_uavs=32, scheduler_mode="teacher_full", teacher_enabled=True,
        teacher_mode="teacher_full", shield_enabled=True,
        initial_soc_mode="fixed", initial_soc=0.80,
        priority_mode="fixed", initial_priority=1,
        gate_probabilities=(0.25, 0.25, 0.25, 0.25),
        mission_probabilities=(1.0, 0.0, 0.0, 0.0),
    )


def _episode_design(index: int, *, medium_rate: float, horizon_seconds: float) -> tuple[SimConfig, dict]:
    """Deterministic 60/40 reset and finite/Poisson L1 scenario mixture.

    The canonical environment correctly rejects a warm start without a fixed
    arrival schedule.  Consequently the 60% warm branch is finite-cohort,
    while the 40% from-empty branch alternates finite and stationary Poisson;
    both L1 demand types are still trained without changing reset semantics.
    """
    warm = index % 5 < 3  # exact 60% over each five rollouts
    demand = "finite_cohort" if warm or index % 2 else "poisson"
    warm_count = 8 + index % 9 if warm else 0
    cohort = max(warm_count, 8 + (index * 5) % 13)
    config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings" if warm else "flow_from_empty",
                                    demand_model=demand, warm_start_uavs=warm_count, cohort_uavs=cohort,
                                    horizon_seconds=horizon_seconds)
    return config, {"reset_mode": config.reset_mode, "demand_model": demand, "warm_start_uavs": warm_count, "cohort_uavs": cohort}


def _rollout_l1(
    model_state: dict[str, torch.Tensor], config: SimConfig, seed: int, update: int, episode: int, max_steps: int, deterministic: bool = False,
    trace_output_dir: str | Path | None = None,
) -> tuple[list[Transition], L1EpisodeSummary]:
    """Collect one full-teacher rollout, preserving requested action semantics."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = SharedPPO(LOW_OBS_V2.dimension)
    policy.model.load_state_dict(model_state)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    trace_recorder = None if trace_output_dir is None else SafetyTraceRecorder(trace_output_dir, dt=config.dt)
    if trace_recorder is not None:
        trace_recorder.start_episode(env, episode_seed=seed, policy_rng_state={
            "torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
        })
    rows: list[Transition] = []
    reward_sum = 0.0
    release_intents = teacher_interventions = shield_interventions = 0
    correction_magnitude_sum = 0.0
    correction_magnitudes: list[float] = []
    teacher_correction_components: list[tuple[str, ...]] = []
    saturation_counts = np.zeros(3, dtype=int)
    requested_action_count = 0
    trace_row: dict[str, object] | None = None
    try:
        for _ in range(min(max_steps, int(config.max_simulation_seconds / config.dt))):
            ids = list(observations)
            if not ids:
                requested_actions: dict[str, np.ndarray] = {}
                trace_row = None if trace_recorder is None else trace_recorder.record_pre_step(env, observations, requested_actions)
                observations, _, _, _, info = env.step(requested_actions)
                if trace_recorder is not None and trace_row is not None:
                    trace_recorder.record_post_step(env, trace_row, info)
            else:
                actor_obs = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
                actions, log_probs, value = policy.act(actor_obs, actor_obs, deterministic=deterministic)
                requested_actions = {uid: action for uid, action in zip(ids, actions)}
                trace_row = None if trace_recorder is None else trace_recorder.record_pre_step(
                    env, observations, requested_actions, {uid: float(log_prob) for uid, log_prob in zip(ids, log_probs)}
                )
                next_observations, rewards, terminated, truncated, info = env.step(requested_actions)
                if trace_recorder is not None and trace_row is not None:
                    trace_recorder.record_post_step(env, trace_row, info)
                for row_index, uid in enumerate(ids):
                    trace = info.get("action_traces", {}).get(uid, {})
                    requested = np.asarray(trace.get("requested_low_action", actions[row_index]), dtype=np.float32)
                    requested_action_count += 1
                    saturation_counts += (np.abs(requested) >= 0.98).astype(int)
                    teacher = trace.get("teacher_corrected_action")
                    shield = trace.get("shield_applied_action")
                    execution = trace.get("execution_applied_action")
                    applied = shield if shield is not None else execution
                    teacher_used = bool(trace.get("teacher_used", False))
                    shield_used = bool(trace.get("shield_used", False))
                    reward = float(rewards.get(uid, 0.0))
                    rows.append(Transition(f"{update}:{episode}:{seed}:{uid}", actor_obs[row_index].copy(), actor_obs.copy(), requested,
                                           float(log_probs[row_index]), value, reward,
                                           bool(terminated.get(uid, False) or truncated.get(uid, False)),
                                           None if teacher is None else np.asarray(teacher, dtype=np.float32),
                                           None if applied is None else np.asarray(applied, dtype=np.float32), teacher_used, shield_used))
                    reward_sum += reward
                    release_intents += int(requested[0] > config.release_threshold)
                    teacher_interventions += int(teacher_used)
                    shield_interventions += int(shield_used)
                    if teacher_used:
                        correction = float(trace.get("correction_magnitude", 0.0))
                        correction_magnitude_sum += correction
                        correction_magnitudes.append(correction)
                        corrected = np.asarray(teacher, dtype=np.float32) if teacher is not None else requested
                        names = ("release", "speed", "merge_time")
                        teacher_correction_components.append(tuple(
                            name for name, delta in zip(names, np.abs(corrected - requested)) if delta > 1e-7
                        ) or ("hold_reject",))
                observations = next_observations
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
    except SafetyInvariantError as error:
        if trace_recorder is not None:
            trace_recorder.record_safety_invariant(trace_row, error)
        raise
    completion_delays = tuple(float(uav.completion_time - uav.request_time) for uav in env.completed_uavs
                              if uav.completion_time is not None)
    summary = L1EpisodeSummary(update, episode, seed, config.reset_mode, config.arrival_model, config.warm_start_uavs,
                               config.total_requested_uavs or 0, len(rows), reward_sum, int(env.arrivals.generated),
                               len(env.ledger.requested), int(env.telemetry.get("arrivals_admitted", 0)), len(env.completed_uavs),
                               release_intents, teacher_interventions, shield_interventions, correction_magnitude_sum,
                               tuple(correction_magnitudes), tuple(teacher_correction_components),
                               tuple(int(value) for value in saturation_counts), requested_action_count, completion_delays,
                               int(env.telemetry.get("collisions", 0)), int(env.telemetry.get("nmac", 0)),
                               int(env.telemetry.get("hard_separation_violations", 0)), env.step_count)
    return rows, summary


def _rollout_task(args: tuple) -> tuple[list[Transition], L1EpisodeSummary]:
    return _rollout_l1(*args)


def _rate(summaries: list[L1EpisodeSummary]) -> float | None:
    intents = sum(row.release_intents for row in summaries)
    return None if intents == 0 else 100.0 * sum(row.teacher_interventions for row in summaries) / intents


def _scenario_name(summary: L1EpisodeSummary) -> str:
    if summary.reset_mode == "warm_start_rings":
        return "warm_start_finite_cohort"
    return "from_empty_finite_cohort" if summary.demand_model == "finite_cohort" else "poisson"


def _summary_metrics(summaries: list[L1EpisodeSummary]) -> dict:
    """Aggregate teacher, efficiency and safety metrics without zero-intent dilution."""
    effective = [row for row in summaries if row.arrivals > 0 and row.release_intents > 0]
    corrections = [value for row in effective for value in row.correction_magnitudes]
    component_counts = {name: 0 for name in ("release", "speed", "merge_time", "hold_reject")}
    correction_reasons = {name: 0 for name in ("release", "speed", "merge-time", "combined", "hold_reject")}
    for row in effective:
        for components in row.teacher_correction_components:
            for component in components:
                component_counts[component] = component_counts.get(component, 0) + 1
            if len(components) != 1:
                correction_reasons["combined"] += 1
            elif components[0] == "merge_time":
                correction_reasons["merge-time"] += 1
            elif components[0] in correction_reasons:
                correction_reasons[components[0]] += 1
            else:
                correction_reasons["combined"] += 1
    action_count = sum(row.requested_action_count for row in effective)
    saturation = np.sum(np.asarray([row.requested_action_saturation_counts for row in effective], dtype=int), axis=0) \
        if effective else np.zeros(3, dtype=int)
    delays = [value for row in effective for value in row.completion_delays]
    intents = sum(row.release_intents for row in effective)
    steps = sum(row.steps for row in effective)
    requested = sum(row.requested for row in effective)
    completed = sum(row.completed for row in effective)
    return {
        "episodes": len(summaries), "effective_episodes": len(effective), "rollout_rows": sum(row.rollout_rows for row in summaries),
        "arrivals": sum(row.arrivals for row in summaries), "requested": sum(row.requested for row in summaries),
        "admitted": sum(row.admitted for row in summaries), "release_intents": sum(row.release_intents for row in summaries),
        "completed": sum(row.completed for row in summaries), "teacher_interventions": sum(row.teacher_interventions for row in summaries),
        "shield_interventions": sum(row.shield_interventions for row in summaries),
        "teacher_interventions_per_100_release_intents": _rate(effective),
        "correction_magnitude_mean": None if not corrections else float(np.mean(corrections)),
        "correction_magnitude_p95": None if not corrections else float(np.percentile(corrections, 95)),
        "teacher_correction_components": component_counts,
        "teacher_correction_reasons": correction_reasons,
        "requested_action_saturation": {
            name: None if action_count == 0 else float(value / action_count)
            for name, value in zip(("release", "speed", "merge_time"), saturation)
        },
        "release_rate_per_second": None if steps == 0 else intents / steps,
        "completion_ratio": None if requested == 0 else completed / requested,
        "throughput_per_second": None if steps == 0 else completed / steps,
        "delay_mean": None if not delays else float(np.mean(delays)),
        "delay_p95": None if not delays else float(np.percentile(delays, 95)),
        "shield_interventions_per_100_release_intents": None if intents == 0 else 100.0 * sum(row.shield_interventions for row in effective) / intents,
        "collisions": sum(row.collisions for row in summaries), "nmac": sum(row.nmac for row in summaries),
        "hard_separation_violations": sum(row.hard_separation_violations for row in summaries),
    }


def _grouped_metrics(summaries: list[L1EpisodeSummary]) -> dict[str, dict]:
    groups = ("warm_start_finite_cohort", "from_empty_finite_cohort", "poisson")
    return {group: _summary_metrics([row for row in summaries if _scenario_name(row) == group]) for group in groups}


def _validation_sufficiency(grouped: dict[str, dict], protocol: L1FullTeacherProtocol) -> dict[str, bool]:
    return {group: values["effective_episodes"] >= protocol.validation_min_effective_episodes and
            values["release_intents"] >= protocol.validation_min_release_intents
            for group, values in grouped.items()}


def _model_state(policy: SharedPPO) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.model.state_dict().items()}


def _run_fixed_validation(policy: SharedPPO, partition: LowV2SeedPartition, *, medium_rate: float,
                          max_steps: int, rollout_workers: int, update: int,
                          trace_output_dir: str | Path | None = None) -> list[L1EpisodeSummary]:
    specs = list(zip(L1_VALIDATION_SCENARIO_INDICES, partition.validation[:len(L1_VALIDATION_SCENARIO_INDICES)]))
    tasks = []
    state = _model_state(policy)
    for episode, (design_index, seed) in enumerate(specs):
        config, _ = _episode_design(design_index, medium_rate=medium_rate, horizon_seconds=float(max_steps))
        tasks.append((state, config, seed, update, episode, max_steps, False,
                      None if trace_output_dir is None else str(trace_output_dir)))
    if rollout_workers == 1:
        collected = [_rollout_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(rollout_workers, len(tasks))) as executor:
            collected = list(executor.map(_rollout_task, tasks))
    return [summary for _, summary in collected]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def train_l1_full_teacher(
    *, c0_report: str | Path, teacher_dataset_path: str | Path, output_dir: str | Path,
    protocol: L1FullTeacherProtocol | None = None, resume_checkpoint: str | Path | None = None,
) -> L1TrainingSummary:
    """Run the curriculum's L1 nominal Full-Teacher PPO stage.

    This function is deliberately scoped to L1.  It neither anneals the
    teacher nor performs student qualification; those are later curriculum
    stages with separate acceptance gates.
    """
    protocol = protocol or L1FullTeacherProtocol()
    # The central optimizer permutation is part of an ablation's controlled
    # protocol.  Rollout workers retain their own deterministic episode seeds.
    random.seed(protocol.optimizer_seed)
    np.random.seed(protocol.optimizer_seed)
    torch.manual_seed(protocol.optimizer_seed)
    audit = authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    if len(partition.validation) < len(L1_VALIDATION_SCENARIO_INDICES):
        raise RuntimeError("L1 formal validation requires all 12 frozen validation seeds")
    mu_hat, medium_range = _read_frozen_capacity(c0_report)
    medium_rate = 0.5 * (medium_range[0] + medium_range[1])
    root_config = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                         warm_start_uavs=8, cohort_uavs=8, horizon_seconds=float(protocol.max_steps))
    validate_low_v2_config(root_config, for_training=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=protocol.teacher_bc_weight)
    resume_hash = None
    if resume_checkpoint is not None:
        resume = Path(resume_checkpoint)
        load_checkpoint(resume, policy, root_config)
        resume_hash = hashlib.sha256(resume.read_bytes()).hexdigest()
    episode_rows: list[dict] = []
    validation_rows: list[dict] = []
    rolling_rows: list[dict] = []
    validation_history: list[dict] = []
    total_rollout_rows = 0
    design_index = 0
    train_seeds = itertools.cycle(partition.train)
    per_update_summaries: list[list[L1EpisodeSummary]] = []

    def capture_validation(label: str, update: int) -> None:
        summaries = _run_fixed_validation(policy, partition, medium_rate=medium_rate, max_steps=protocol.max_steps,
                                          rollout_workers=protocol.rollout_workers, update=update,
                                          trace_output_dir=destination / "safety_action_tapes" / "validation" / label)
        grouped = _grouped_metrics(summaries)
        record = {"label": label, "update": update, "overall": _summary_metrics(summaries),
                  "by_scenario": grouped, "sufficiency": _validation_sufficiency(grouped, protocol)}
        validation_history.append(record)
        for summary in summaries:
            validation_rows.append({**asdict(summary), "scenario": _scenario_name(summary), "validation": True,
                                    "validation_label": label})
        if protocol.stop_on_safety_failure and any(
            row.collisions or row.nmac or row.hard_separation_violations for row in summaries
        ):
            raise L1SafetyGateError(f"L1 validation safety gate failed at {label}")

    # This is the early L1 reference for the resumed checkpoint.  It is
    # written before any new update, so later trend claims never compare
    # against a post-training value.
    capture_validation("early_baseline", -1)
    for update in range(protocol.updates):
        state = _model_state(policy)
        tasks = []
        designs = []
        for episode in range(protocol.episodes_per_update):
            config, design = _episode_design(design_index, medium_rate=medium_rate, horizon_seconds=float(protocol.max_steps))
            design_index += 1
            tasks.append((state, config, next(train_seeds), update, episode, protocol.max_steps, False,
                          str(destination / "safety_action_tapes")))
            designs.append(design)
        if protocol.rollout_workers == 1:
            collected = [_rollout_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(protocol.rollout_workers, len(tasks))) as executor:
                collected = list(executor.map(_rollout_task, tasks))
        buffer = RolloutBuffer()
        update_summaries: list[L1EpisodeSummary] = []
        for (transitions, summary), design in zip(collected, designs):
            for transition in transitions:
                buffer.add(transition)
            total_rollout_rows += len(transitions)
            update_summaries.append(summary)
            episode_rows.append({**asdict(summary), **design, "scenario": _scenario_name(summary)})
        if protocol.stop_on_safety_failure and any(
            row.collisions or row.nmac or row.hard_separation_violations for row in update_summaries
        ):
            _write_jsonl(destination / "l1_training_metrics.jsonl", episode_rows)
            raise L1SafetyGateError(f"L1 training safety gate failed before PPO update {update}")
        ppo = policy.update(buffer, epochs=protocol.ppo_epochs, minibatch_size=protocol.ppo_minibatch_size)
        per_update_summaries.append(update_summaries)
        window_updates = per_update_summaries[-protocol.rolling_window_updates:]
        rolling_summaries = [summary for summaries in window_updates for summary in summaries]
        current_grouped = _grouped_metrics(update_summaries)
        rolling_grouped = _grouped_metrics(rolling_summaries)
        rolling_record = {"record_type": "rolling_teacher_metrics", "update": update,
                          "window_start_update": update - len(window_updates) + 1, "window_updates": len(window_updates),
                          "overall": _summary_metrics(rolling_summaries), "by_scenario": rolling_grouped}
        rolling_rows.append(rolling_record)
        episode_rows.append({"record_type": "ppo_update", "update": update,
                             "rollout_rows": sum(len(rows) for rows, _ in collected),
                             "current": _summary_metrics(update_summaries), "by_scenario": current_grouped,
                             "rolling": rolling_record, **ppo})
        if (update + 1) % protocol.validation_every == 0 or update + 1 == protocol.updates:
            capture_validation(f"after_update_{update + 1}", update)
    all_episode_summaries = [summary for summaries in per_update_summaries for summary in summaries]
    safety_passed = all(row.collisions == row.nmac == row.hard_separation_violations == 0 for row in all_episode_summaries) and \
                    all(record["overall"]["collisions"] == record["overall"]["nmac"] == record["overall"]["hard_separation_violations"] == 0
                        for record in validation_history)
    first_updates = per_update_summaries[:protocol.rolling_window_updates]
    last_updates = per_update_summaries[-protocol.rolling_window_updates:]
    first = [summary for summaries in first_updates for summary in summaries]
    last = [summary for summaries in last_updates for summary in summaries]
    sufficient_history = [record for record in validation_history if all(record["sufficiency"].values())]
    def decreasing(key: str) -> bool:
        values = [record["overall"].get(key) for record in sufficient_history]
        return len(values) >= 3 and all(value is not None for value in values) and all(right <= left for left, right in zip(values, values[1:])) and values[-1] < values[0]
    stable_validation_trend = decreasing("teacher_interventions_per_100_release_intents") and \
                              decreasing("correction_magnitude_mean") and decreasing("correction_magnitude_p95")
    l1b_eligible = safety_passed and stable_validation_trend and bool(sufficient_history)
    checkpoint = destination / "l1_full_teacher_ppo.pt"
    save_checkpoint(checkpoint, policy, root_config, curriculum_stage="L1_full_teacher", c0_report=str(c0_report),
                    teacher_dataset_path=str(teacher_dataset_path), teacher_dataset_audit=asdict(audit), protocol=asdict(protocol),
                    resumed_from_checkpoint=None if resume_checkpoint is None else str(resume_checkpoint), resume_checkpoint_sha256=resume_hash)
    _write_jsonl(destination / "l1_training_metrics.jsonl", episode_rows)
    _write_jsonl(destination / "l1_validation_metrics.jsonl", validation_rows)
    _write_jsonl(destination / "l1_rolling_teacher_metrics.jsonl", rolling_rows)
    validation_summary = {"stage": "L1_full_teacher_validation", "schema_version": "A5V2-TRAINING-CURRICULUM-v3",
                          "scenario_indices": list(L1_VALIDATION_SCENARIO_INDICES), "records": validation_history,
                          "stable_validation_trend": stable_validation_trend, "l1b_eligible": l1b_eligible}
    (destination / "l1_validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    manifest = {
        "stage": "L1_full_teacher_ppo", "schema_version": "A5V2-TRAINING-CURRICULUM-v3", "evidence_type": "newly_run",
        "c0_report": str(c0_report), "teacher_dataset_path": str(teacher_dataset_path), "teacher_dataset_audit": asdict(audit),
        "seed_partition": asdict(partition), "mu_hat": mu_hat, "medium_rate": medium_rate,
        "medium_interval": list(medium_range), "protocol": asdict(protocol),
        "ppo_bc_provenance": {
            "ppo_behavior_action": "a_requested",
            "ppo_old_log_prob": "log_pi(a_requested|s)",
            "ppo_ratio_uses_a_applied": False,
            "teacher_bc_target": "a_teacher",
            "teacher_bc_mask": "teacher_used && a_teacher is not None",
        },
        "environment": {"teacher": "FULL", "scheduler_mode": "teacher_full", "shield_enabled": True, "high_level_enabled": False,
                        "geometry": "nominal_fixed", "gate_probabilities": [0.25] * 4, "mission_probabilities": [1.0, 0.0, 0.0, 0.0], "initial_soc": 0.80,
                        "priority_mode": "fixed", "initial_priority": 1,
                        "finite_cohort_range": [8, 20], "warm_start_range": [8, 16], "warm_start_fraction": 0.60},
        "checkpoint": str(checkpoint), "resumed_from_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "resume_checkpoint_sha256": resume_hash, "rollout_rows": total_rollout_rows, "safety_passed": safety_passed,
        "training_first_rolling_window": _summary_metrics(first), "training_last_rolling_window": _summary_metrics(last),
        "validation_summary": str(destination / "l1_validation_summary.json"), "stable_validation_trend": stable_validation_trend,
        "l1b_eligible": l1b_eligible,
    }
    (destination / "l1_full_teacher.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return L1TrainingSummary(str(destination), str(checkpoint), mu_hat, medium_rate, protocol.updates,
                             len(all_episode_summaries), total_rollout_rows, safety_passed, _rate(all_episode_summaries) or 0.0,
                             _rate(first), _rate(last), l1b_eligible)


def evaluate_l1_full_teacher_checkpoint(
    *, checkpoint_path: str | Path, c0_report: str | Path, teacher_dataset_path: str | Path,
    output_path: str | Path, max_steps: int, rollout_workers: int = 1,
) -> dict:
    """Evaluate one completed L1 checkpoint on the fixed mixed anchor matrix."""
    if max_steps <= 0 or rollout_workers <= 0:
        raise ValueError("max_steps and rollout_workers must be positive")
    authorize_low_v2_training(teacher_dataset_path)
    partition = _read_seed_partition(teacher_dataset_path)
    if len(partition.validation) < len(L1_VALIDATION_SCENARIO_INDICES):
        raise RuntimeError("L1 formal validation requires all 12 frozen validation seeds")
    _, medium_range = _read_frozen_capacity(c0_report)
    medium_rate = 0.5 * sum(medium_range)
    root = l1_full_teacher_config(medium_rate=medium_rate, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                  warm_start_uavs=8, cohort_uavs=8, horizon_seconds=float(max_steps))
    policy = SharedPPO(LOW_OBS_V2.dimension)
    load_checkpoint(checkpoint_path, policy, root)
    summaries = _run_fixed_validation(policy, partition, medium_rate=medium_rate, max_steps=max_steps,
                                      rollout_workers=rollout_workers, update=-1)
    grouped = _grouped_metrics(summaries)
    protocol = L1FullTeacherProtocol(max_steps=max_steps, rollout_workers=rollout_workers)
    rows = [{**asdict(summary), "scenario": _scenario_name(summary), "validation": True} for summary in summaries]
    report = {
        "stage": "L1_full_teacher_checkpoint_validation", "schema_version": "A5V2-TRAINING-CURRICULUM-v3",
        "checkpoint": str(checkpoint_path), "c0_report": str(c0_report), "medium_rate": medium_rate,
        "seed_split": "validation", "seeds": list(partition.validation[:len(L1_VALIDATION_SCENARIO_INDICES)]),
        "scenario_indices": list(L1_VALIDATION_SCENARIO_INDICES), "rows": rows,
        "overall": _summary_metrics(summaries), "by_scenario": grouped,
        "sufficiency": _validation_sufficiency(grouped, protocol),
        "safety_passed": all(row.collisions == row.nmac == row.hard_separation_violations == 0 for row in summaries),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run A5-v2 L1 nominal FULL-Teacher PPO on CPU rollout workers.")
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--teacher-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=L1FullTeacherProtocol.updates)
    parser.add_argument("--episodes-per-update", type=int, default=L1FullTeacherProtocol.episodes_per_update)
    parser.add_argument("--workers", type=int, default=L1FullTeacherProtocol.rollout_workers)
    parser.add_argument("--max-steps", type=int, default=L1FullTeacherProtocol.max_steps)
    parser.add_argument("--validation-every", type=int, default=L1FullTeacherProtocol.validation_every)
    parser.add_argument("--rolling-window-updates", type=int, default=L1FullTeacherProtocol.rolling_window_updates)
    parser.add_argument("--resume-checkpoint", help="Continue L1 from this compatible FULL-Teacher checkpoint.")
    parser.add_argument("--evaluate-checkpoint", help="Evaluate this completed L1 checkpoint instead of training.")
    parser.add_argument("--eval-output", help="Path for --evaluate-checkpoint JSON report.")
    args = parser.parse_args()
    protocol = L1FullTeacherProtocol(updates=args.updates, episodes_per_update=args.episodes_per_update,
                                     rollout_workers=args.workers, max_steps=args.max_steps,
                                     validation_every=args.validation_every,
                                     rolling_window_updates=args.rolling_window_updates)
    if args.evaluate_checkpoint:
        if not args.eval_output:
            parser.error("--evaluate-checkpoint requires --eval-output")
        print(json.dumps(evaluate_l1_full_teacher_checkpoint(checkpoint_path=args.evaluate_checkpoint, c0_report=args.c0_report,
                                                              teacher_dataset_path=args.teacher_dataset, output_path=args.eval_output,
                                                              max_steps=args.max_steps, rollout_workers=args.workers), indent=2))
        return
    print(json.dumps(asdict(train_l1_full_teacher(c0_report=args.c0_report, teacher_dataset_path=args.teacher_dataset,
                                                   output_dir=args.output, protocol=protocol,
                                                   resume_checkpoint=args.resume_checkpoint)), indent=2))


if __name__ == "__main__":
    _main()
