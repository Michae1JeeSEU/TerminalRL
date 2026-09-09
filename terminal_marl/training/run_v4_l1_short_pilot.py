"""Fresh V4 L1 Full-Teacher joint PPO short pilot.

This is intentionally a bounded diagnostic, not formal L1 training.  It has
no resume argument, uses a newly initialized joint actor/critic, and stops on
the first safety or conservative-WAIT-collapse gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch

from ..algo.joint_ppo import JointPPO, JointRolloutBuffer, JointTransition
from ..core.checkpoints import save_checkpoint
from ..core.low_level_joint_action import ACTION_SCHEMA_V4, JOINT_ACTION_CATALOG, N_JOINT_ACTIONS
from ..core.observations import LOW_OBS_V2
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .low_v2_seed_partitions import LowV2SeedPartition
from .train_low_v2_ppo import _episode_design
from .v4_joint_teacher import component_disagreement, teacher_joint_label


# The import expression above would obscure the frozen pilot constants in an
# artifact; retain them directly here rather than accepting command-line
# overrides.
MU_HAT = .003
LAMBDA = .001575
BASE_COMMIT = "e77831e5c7218e65abeeee81f1b7db5b933a6829"
TRACKED_DIFF_SHA256 = "65F59315D398B8C977A9EEDFDADE2A66E494440246BDF91F5EB201D8170DB97B"
VALIDATION_DESIGN_INDICES = (0, 3, 4)  # warm finite, empty finite, empty Poisson


@dataclass(frozen=True)
class V4L1ShortPilotProtocol:
    updates: int = 6
    episodes_per_update: int = 4
    rollout_workers: int = 4
    max_steps: int = 600
    ppo_epochs: int = 2
    minibatch_size: int = 256
    teacher_bc_weight: float = .05

    def __post_init__(self) -> None:
        frozen = (self.updates, self.episodes_per_update, self.rollout_workers, self.max_steps,
                  self.ppo_epochs, self.minibatch_size, self.teacher_bc_weight)
        if frozen != (6, 4, 4, 600, 2, 256, .05):
            raise ValueError("V4 short pilot is frozen at 6x4, 600 s, 4 workers, PPO 2x256, BC=0.05")


@dataclass(frozen=True)
class V4EpisodeSummary:
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
    joint_decisions: int
    release_intents: int
    release_accepted: int
    teacher_interventions: int
    shield_interventions: int
    release_disagreements: int
    speed_disagreements: int
    merge_disagreements: int
    release_to_wait: int
    wait_to_release: int
    semantic_release_flip: tuple[float, ...]
    semantic_speed_level_difference: tuple[float, ...]
    semantic_merge_level_difference: tuple[float, ...]
    completion_delays: tuple[float, ...]
    collisions: int
    nmac: int
    hard_separation_violations: int
    safety_invariant_errors: int
    uncertified_components: int
    steps: int
    external_queue_final: int
    external_queue_mean: float
    external_queue_max: int


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text("".join(json.dumps(_jsonable(row), allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _v4_config(design_index: int, *, horizon_seconds: float):
    config, design = _episode_design(design_index, medium_rate=LAMBDA, horizon_seconds=horizon_seconds)
    return replace(config, action_schema_version=ACTION_SCHEMA_V4, teacher_release_label_enabled=False), design


def _model_state(policy: JointPPO) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in policy.model.state_dict().items()}


def _level(value: float) -> int:
    # V4's normalized residual levels are exactly {-1, -.5, 0, .5, 1}.
    return int(round((float(value) + 1.0) * 2.0))


def _rollout_v4(
    model_state: dict[str, torch.Tensor], config, seed: int, update: int, episode: int, max_steps: int,
    deterministic: bool = False, trace_output_dir: str | Path | None = None,
) -> tuple[list[JointTransition], V4EpisodeSummary]:
    """Collect one V4 rollout with external Full-Teacher joint labels.

    The environment receives the decoded sampled request.  Teacher labels are
    observed before that transition and retained exclusively for joint CE and
    diagnostics; they cannot replace the behavior action or its logprob.
    """
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = JointPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    policy.model.load_state_dict(model_state)
    env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    tape = None if trace_output_dir is None else SafetyTraceRecorder(trace_output_dir, dt=config.dt)
    if tape is not None:
        tape.start_episode(env, episode_seed=seed, policy_rng_state={"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()})
    rows: list[JointTransition] = []
    reward_sum = 0.0; release_intents = release_accepted = teacher_interventions = shield_interventions = 0
    release_disagreements = speed_disagreements = merge_disagreements = release_to_wait = wait_to_release = 0
    semantic_release: list[float] = []; semantic_speed: list[float] = []; semantic_merge: list[float] = []
    queue_sizes = [len(env.admission.queue.items)]; trace_row = None
    try:
        for _ in range(min(max_steps, int(config.max_simulation_seconds / config.dt))):
            ids = sorted(observations, key=int)
            if not ids:
                trace_row = None if tape is None else tape.record_pre_step(env, observations, {})
                observations, _, _, _, info = env.step({})
                if tape is not None and trace_row is not None:
                    tape.record_post_step(env, trace_row, info)
            else:
                actor_obs = np.stack([observations[uid] for uid in ids]).astype(np.float32, copy=False)
                batch = policy.act_details(actor_obs, actor_obs, deterministic=deterministic)
                # Full Teacher may search only in this label query. Runtime
                # ExecutionScheduler receives/checks exactly one action below.
                teacher_indices = {uid: teacher_joint_label(env, env.active[int(uid)]).index for uid in ids}
                requested_actions = {uid: batch.decoded[index].astype(np.float32) for index, uid in enumerate(ids)}
                trace_row = None if tape is None else tape.record_pre_step(
                    env, observations, requested_actions,
                    {uid: float(batch.joint_log_prob[index]) for index, uid in enumerate(ids)},
                )
                next_observations, rewards, terminated, truncated, info = env.step(requested_actions)
                if tape is not None and trace_row is not None:
                    tape.record_post_step(env, trace_row, info)
                for index, uid in enumerate(ids):
                    trace = info.get("action_traces", {}).get(uid, {})
                    requested_index = int(batch.joint_action_index[index]); teacher_index = int(teacher_indices[uid])
                    # A mismatch here would make PPO provenance invalid.
                    trace_index = int(JOINT_ACTION_CATALOG[requested_index].index)
                    if trace_index != requested_index:
                        raise RuntimeError("PPO requested joint action provenance failure")
                    requested, teacher = JOINT_ACTION_CATALOG[requested_index], JOINT_ACTION_CATALOG[teacher_index]
                    disagreement = component_disagreement(requested, teacher)
                    teacher_used = requested_index != teacher_index
                    shield_used = bool(trace.get("shield_used", False))
                    reward = float(rewards.get(uid, 0.0)); reward_sum += reward
                    rows.append(JointTransition(
                        f"{update}:{episode}:{seed}:{uid}", actor_obs[index].copy(), actor_obs.copy(), requested_index,
                        float(batch.joint_log_prob[index]), batch.value, reward,
                        bool(terminated.get(uid, False) or truncated.get(uid, False)), teacher_joint_action=teacher_index,
                        execution_applied_action=None if trace.get("execution_applied_action") is None else np.asarray(trace["execution_applied_action"], dtype=np.float32),
                        shield_applied_action=None if trace.get("shield_applied_action") is None else np.asarray(trace["shield_applied_action"], dtype=np.float32),
                        teacher_used=teacher_used,
                    ))
                    release_intents += int(requested.release_request > config.release_threshold)
                    release_accepted += int(requested.release_request > config.release_threshold and info.get("release_decisions", {}).get(uid) == "approved")
                    teacher_interventions += int(teacher_used); shield_interventions += int(shield_used)
                    release_disagreements += int(disagreement["release"]); speed_disagreements += int(disagreement["speed"]); merge_disagreements += int(disagreement["merge_time"])
                    release_to_wait += int(requested.release_request > 0.0 and teacher.release_request <= 0.0)
                    wait_to_release += int(requested.release_request <= 0.0 and teacher.release_request > 0.0)
                    if teacher_used:
                        semantic_release.append(float(disagreement["release"]))
                        semantic_speed.append(float(abs(_level(requested.speed_residual) - _level(teacher.speed_residual))))
                        semantic_merge.append(float(abs(_level(requested.merge_time_residual) - _level(teacher.merge_time_residual))))
                observations = next_observations
            queue_sizes.append(len(env.admission.queue.items))
            if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
                break
    except SafetyInvariantError as error:
        if tape is not None:
            tape.record_safety_invariant(trace_row, error)
        raise
    safety = (int(env.telemetry.get("collisions", 0)), int(env.telemetry.get("nmac", 0)), int(env.telemetry.get("hard_separation_violations", 0)))
    if any(safety):
        path = None if tape is None else tape.dump(reason="v4_l1_training_safety_gate")
        raise RuntimeError(f"TRAINING_SAFETY_FAILURE: collision/NMAC/hard={safety}; ActionTape={path}")
    delays = tuple(float(uav.completion_time - uav.request_time) for uav in env.completed_uavs if uav.completion_time is not None)
    summary = V4EpisodeSummary(
        update, episode, seed, config.reset_mode, config.arrival_model, config.warm_start_uavs, config.total_requested_uavs or 0,
        len(rows), reward_sum, int(env.arrivals.generated), len(env.ledger.requested), int(env.telemetry.get("arrivals_admitted", 0)),
        len(env.completed_uavs), len(rows), release_intents, release_accepted, teacher_interventions, shield_interventions,
        release_disagreements, speed_disagreements, merge_disagreements, release_to_wait, wait_to_release,
        tuple(semantic_release), tuple(semantic_speed), tuple(semantic_merge), delays, *safety, 0, 0,
        env.step_count, len(env.admission.queue.items), float(np.mean(queue_sizes)), int(max(queue_sizes)),
    )
    return rows, summary


def _rollout_task(args: tuple) -> tuple[list[JointTransition], V4EpisodeSummary]:
    return _rollout_v4(*args)


def _collect(tasks: list[tuple], workers: int) -> list[tuple[list[JointTransition], V4EpisodeSummary]]:
    if workers == 1:
        return [_rollout_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_rollout_task, tasks))


def _counts(values: np.ndarray) -> list[int]:
    return [int((values == index).sum()) for index in range(N_JOINT_ACTIONS)]


def _joint_metrics(rows: list[JointTransition], model_state: dict[str, torch.Tensor]) -> dict:
    if not rows:
        raise RuntimeError("no V4 joint transitions for metrics")
    requested = np.asarray([row.joint_action_index for row in rows], dtype=np.int64)
    teacher = np.asarray([row.teacher_joint_action for row in rows], dtype=np.int64)
    observation = np.stack([row.observation for row in rows]).astype(np.float32, copy=False)
    policy = JointPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05); policy.model.load_state_dict(model_state); policy.model.eval()
    probability_chunks = []; entropy_chunks = []
    with torch.no_grad():
        for start in range(0, len(observation), 1024):
            distribution = policy.model.actor_distribution(torch.as_tensor(observation[start:start + 1024]))
            probability_chunks.append(distribution.probs.cpu().numpy()); entropy_chunks.append(distribution.entropy().cpu().numpy())
    probabilities = np.concatenate(probability_chunks); entropy = np.concatenate(entropy_chunks)
    decoded_requested = [JOINT_ACTION_CATALOG[int(index)] for index in requested]
    decoded_teacher = [JOINT_ACTION_CATALOG[int(index)] for index in teacher]
    release_dis = np.asarray([left.release_request != right.release_request for left, right in zip(decoded_requested, decoded_teacher)])
    speed_dis = np.asarray([left.speed_residual != right.speed_residual for left, right in zip(decoded_requested, decoded_teacher)])
    merge_dis = np.asarray([left.merge_time_residual != right.merge_time_residual for left, right in zip(decoded_requested, decoded_teacher)])
    changed = requested != teacher
    semantic = {
        "release_flip_mean": None if not changed.any() else float(release_dis[changed].mean()),
        "release_flip_p95": None if not changed.any() else float(np.percentile(release_dis[changed].astype(float), 95)),
        "speed_level_difference_mean": None if not changed.any() else float(np.mean([abs(_level(left.speed_residual) - _level(right.speed_residual)) for left, right, flag in zip(decoded_requested, decoded_teacher, changed) if flag])),
        "speed_level_difference_p95": None if not changed.any() else float(np.percentile([abs(_level(left.speed_residual) - _level(right.speed_residual)) for left, right, flag in zip(decoded_requested, decoded_teacher, changed) if flag], 95)),
        "merge_level_difference_mean": None if not changed.any() else float(np.mean([abs(_level(left.merge_time_residual) - _level(right.merge_time_residual)) for left, right, flag in zip(decoded_requested, decoded_teacher, changed) if flag])),
        "merge_level_difference_p95": None if not changed.any() else float(np.percentile([abs(_level(left.merge_time_residual) - _level(right.merge_time_residual)) for left, right, flag in zip(decoded_requested, decoded_teacher, changed) if flag], 95)),
    }
    requested_release = np.asarray([item.release_request > 0.0 for item in decoded_requested])
    teacher_release = np.asarray([item.release_request > 0.0 for item in decoded_teacher])
    return {
        "joint_decisions": len(rows), "requested_joint_action_distribution": _counts(requested), "teacher_joint_target_distribution": _counts(teacher),
        "joint_teacher_intervention_count": int(changed.sum()), "joint_teacher_intervention_rate": float(changed.mean()),
        "joint_policy_teacher_agreement": float((~changed).mean()), "joint_entropy": float(entropy.mean()),
        "joint_probability_distribution": [float(value) for value in probabilities.mean(axis=0)],
        "decoded": {"requested": {"WAIT": int((~requested_release).sum()), "RELEASE": int(requested_release.sum())},
                    "teacher": {"WAIT": int((~teacher_release).sum()), "RELEASE": int(teacher_release.sum())},
                    "release_disagreement_rate": float(release_dis.mean()), "speed_disagreement_rate": float(speed_dis.mean()),
                    "merge_disagreement_rate": float(merge_dis.mean()),
                    "WAIT_to_RELEASE": int(((~requested_release) & teacher_release).sum()),
                    "RELEASE_to_WAIT": int((requested_release & (~teacher_release)).sum()),
                    "semantic_correction_magnitude": semantic},
    }


def _environment_metrics(summaries: list[V4EpisodeSummary]) -> dict:
    if not summaries:
        return {"episodes": 0}
    def total(name: str) -> int:
        return int(sum(getattr(row, name) for row in summaries))
    decisions = total("joint_decisions"); intents = total("release_intents"); corrections = total("teacher_interventions")
    delays = [item for row in summaries for item in row.completion_delays]
    semantic = {name: [item for row in summaries for item in getattr(row, name)] for name in (
        "semantic_release_flip", "semantic_speed_level_difference", "semantic_merge_level_difference")}
    steps = total("steps")
    return {
        "episodes": len(summaries), "rollout_rows": total("rollout_rows"), "arrivals": total("arrivals"), "requested": total("requested"),
        "admitted": total("admitted"), "completed": total("completed"), "joint_decisions": decisions, "release_intents": intents,
        "teacher_interventions": corrections, "teacher_interventions_per_100_joint_decisions": None if decisions == 0 else 100.0 * corrections / decisions,
        "teacher_interventions_per_100_release_intents": None if intents == 0 else 100.0 * corrections / intents,
        "joint_teacher_agreement": None if decisions == 0 else 1.0 - corrections / decisions,
        "release_acceptance": None if intents == 0 else total("release_accepted") / intents,
        "completion_ratio": None if total("requested") == 0 else total("completed") / total("requested"),
        "throughput_per_second": None if steps == 0 else total("completed") / steps,
        "external_queue_final_mean": float(np.mean([row.external_queue_final for row in summaries])),
        "external_queue_mean": float(np.mean([row.external_queue_mean for row in summaries])),
        "external_queue_max": int(max(row.external_queue_max for row in summaries)),
        "delay_mean": None if not delays else float(np.mean(delays)), "delay_p95": None if not delays else float(np.percentile(delays, 95)),
        "shield_burden_per_100_joint_decisions": None if decisions == 0 else 100.0 * total("shield_interventions") / decisions,
        "shield_burden_per_100_release_intents": None if intents == 0 else 100.0 * total("shield_interventions") / intents,
        "semantic_correction_magnitude": {name: {"mean": None if not values else float(np.mean(values)), "p95": None if not values else float(np.percentile(values, 95))}
                                          for name, values in semantic.items()},
        "collision": total("collisions"), "NMAC": total("nmac"), "hard": total("hard_separation_violations"),
        "SafetyInvariantError": total("safety_invariant_errors"), "uncertified": total("uncertified_components"),
    }


def _records_metrics(records: list[tuple[list[JointTransition], V4EpisodeSummary]], model_state: dict[str, torch.Tensor]) -> dict:
    return {"environment": _environment_metrics([summary for _, summary in records]),
            "joint": _joint_metrics([row for rows, _ in records for row in rows], model_state)}


def _by_scenario(records: list[tuple[list[JointTransition], V4EpisodeSummary]], model_state: dict[str, torch.Tensor]) -> dict:
    groups = {
        "warm_start": [record for record in records if record[1].reset_mode == "warm_start_rings"],
        "from_empty": [record for record in records if record[1].reset_mode == "flow_from_empty"],
        "from_empty_finite": [record for record in records if record[1].reset_mode == "flow_from_empty" and record[1].demand_model == "finite_cohort"],
        "poisson": [record for record in records if record[1].demand_model == "poisson"],
    }
    # A finite cohort can contain no active UAVs by the time its rollout is
    # sampled.  Retain that episode in aggregate environment accounting, but
    # do not attempt categorical metrics for a scenario partition with no
    # joint decisions.  `_joint_metrics` deliberately rejects an empty list
    # because it has no defensible probability/disagreement statistic.
    return {
        name: _records_metrics(values, model_state)
        for name, values in groups.items()
        if any(rows for rows, _ in values)
    }


def _anchor_config(index: int):
    # Held-out qualification seeds, independent of both train and natural-L1 validation seeds.
    config, _ = _v4_config(index, horizon_seconds=40.0)
    return replace(config, reset_mode="warm_start_rings", warm_start_uavs=15, fixed_arrival_schedule=(0.0,) * 15,
                   total_requested_uavs=15, arrival_model="finite_cohort", max_simulation_seconds=40.0)


def _build_anchor_cohort(destination: Path, partition: LowV2SeedPartition) -> dict:
    """Create a fresh held-out conflict anchor; never fed to PPO/BC."""
    rows: list[dict] = []; traces = destination / "anchor_action_tapes"
    for ordinal, seed in enumerate(partition.qualification[:4]):
        config = _anchor_config(ordinal); env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
        tape = SafetyTraceRecorder(traces, dt=config.dt); tape.start_episode(env, episode_seed=seed, policy_rng_state={"anchor": seed})
        trace_row = None
        try:
            for step in range(20):
                actions = {}; labels = {}
                for uid in sorted(observations, key=int):
                    proposal = (seed * 7 + step * 11 + int(uid) * 3) % N_JOINT_ACTIONS
                    actions[uid] = JOINT_ACTION_CATALOG[proposal].normalized
                    labels[uid] = teacher_joint_label(env, env.active[int(uid)]).index
                trace_row = tape.record_pre_step(env, observations, actions)
                next_observations, _, _, _, info = env.step(actions); tape.record_post_step(env, trace_row, info)
                for uid in sorted(observations, key=int):
                    proposal = int((seed * 7 + step * 11 + int(uid) * 3) % N_JOINT_ACTIONS)
                    if proposal != labels[uid]:
                        rows.append({"observation": observations[uid].astype(float).tolist(), "teacher_joint_action": labels[uid],
                                     "anchor_requested_joint_action": proposal, "seed": seed, "step": step, "uid": uid})
                    if len(rows) >= 128:
                        break
                observations = next_observations
                if len(rows) >= 128:
                    break
        except SafetyInvariantError as error:
            tape.record_safety_invariant(trace_row, error)
            raise RuntimeError("held-out V4 anchor SafetyInvariantError") from error
        safety = (int(env.telemetry.get("collisions", 0)), int(env.telemetry.get("nmac", 0)), int(env.telemetry.get("hard_separation_violations", 0)))
        if any(safety):
            path = tape.dump(reason="v4_anchor_safety_gate")
            raise RuntimeError(f"held-out V4 anchor safety failure {safety}; ActionTape={path}")
        if len(rows) >= 128:
            break
    if len(rows) < 128:
        raise RuntimeError(f"held-out V4 joint-conflict anchor contains only {len(rows)} rows")
    cohort = {"evidence_type": "newly_run", "selection": "held-out qualification seeds; proposal != Full-Teacher joint target",
              "rows": rows[:128], "teacher_target_count": 128, "ppo_or_bc_used": False}
    _write_json(destination / "v4_joint_conflict_anchor_cohort.json", cohort)
    return cohort


def _anchor_metrics(policy: JointPPO, cohort: dict) -> dict:
    observations = np.asarray([row["observation"] for row in cohort["rows"]], dtype=np.float32)
    targets = np.asarray([row["teacher_joint_action"] for row in cohort["rows"]], dtype=np.int64)
    state = _model_state(policy)
    rows = [JointTransition(f"anchor:{index}", observation, observations, int(cohort["rows"][index]["anchor_requested_joint_action"]), 0.0, 0.0, 0.0, True,
                            teacher_joint_action=int(target), teacher_used=True) for index, (observation, target) in enumerate(zip(observations, targets))]
    # Replace the synthetic requested action with policy deterministic action so
    # agreement means policy↔Teacher, not source-proposal↔Teacher.
    result = policy.act_details(observations, observations, deterministic=True)
    for row, action in zip(rows, result.joint_action_index):
        row.joint_action_index = int(action)
    return _joint_metrics(rows, state)


def _checkpoint(destination: Path, label: str, policy: JointPPO, root_config) -> str:
    path = destination / f"{label}.pt"
    save_checkpoint(path, policy, root_config, stage="V4_L1_short_pilot", snapshot=label,
                    fresh_initialization=True, resume_checkpoint=None)
    return str(path)


def _healthy_ppo(metrics: dict) -> bool:
    return all(np.isfinite(metrics[key]) for key in ("actor_loss", "value_loss", "entropy", "joint_teacher_bc_loss", "approx_kl", "clip_fraction", "explained_variance", "grad_norm"))


def _safety_ok(metrics: dict) -> bool:
    return all(metrics[key] == 0 for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified"))


def _collapse(updates: list[dict]) -> dict:
    rates = [row["joint"]["decoded"]["requested"] for row in updates]
    wait = [item["WAIT"] / max(1, item["WAIT"] + item["RELEASE"]) for item in rates]
    release = [1.0 - item for item in wait]
    completion = [row["environment"]["completed"] for row in updates]
    throughput = [row["environment"]["throughput_per_second"] or 0.0 for row in updates]
    queue = [row["environment"]["external_queue_mean"] for row in updates]
    persistent = len(wait) >= 3 and all(value >= .95 for value in wait[-3:]) and all(left >= right for left, right in zip(release[-3:], release[-2:]))
    degraded = len(wait) >= 3 and completion[-1] < completion[0] and throughput[-1] < throughput[0] and queue[-1] > queue[0]
    return {"conservative_wait_collapse": bool(persistent and degraded), "policy_wait_rate": wait, "policy_release_rate": release,
            "completion": completion, "throughput": throughput, "external_queue_mean": queue,
            "criteria": {"persistent_wait_ge_0_95_last_3": persistent, "completion_throughput_down_queue_up": degraded}}


def _decision(validation: dict, updates: list[dict], *, safety: bool, ppo_health: bool, collapse: bool) -> tuple[str, str, bool]:
    base, final = validation["baseline"], validation["update6"]
    agreement_up = final["natural_l1"]["joint"]["joint_policy_teacher_agreement"] > base["natural_l1"]["joint"]["joint_policy_teacher_agreement"]
    anchor_up = final["joint_conflict_anchor"]["joint_policy_teacher_agreement"] > base["joint_conflict_anchor"]["joint_policy_teacher_agreement"]
    burden_first = updates[0]["environment"]["teacher_interventions_per_100_joint_decisions"]
    burden_last = updates[-1]["environment"]["teacher_interventions_per_100_joint_decisions"]
    burden_improves = burden_last <= burden_first
    disagreement_base = base["natural_l1"]["joint"]["decoded"]
    disagreement_final = final["natural_l1"]["joint"]["decoded"]
    decoded_not_worse = all(disagreement_final[key] <= disagreement_base[key] + .02 for key in ("release_disagreement_rate", "speed_disagreement_rate", "merge_disagreement_rate"))
    healthy_completion = final["natural_l1"]["environment"]["completed"] >= .5 * base["natural_l1"]["environment"]["completed"] if base["natural_l1"]["environment"]["completed"] else True
    if not (safety and ppo_health and not collapse and healthy_completion):
        return "FAIL", "FAIL", False
    if agreement_up and anchor_up and burden_improves and decoded_not_worse:
        return "PASS", "PASS", True
    return "INCONCLUSIVE", "PASS", False


def _validation_line(name: str, values: dict) -> str:
    env = values.get("environment", {})
    # Natural/warm validation wraps categorical metrics under ``joint`` with
    # environment metrics alongside it.  The static held-out anchor is itself
    # categorical metrics (there is no environment rollout), so accept that
    # second, intentional shape rather than reporting it as zero-decision.
    if "joint" in values:
        joint = values["joint"]
    elif "joint_policy_teacher_agreement" in values:
        joint = values
    else:
        return (f"- {name}: no joint decisions in this fixed validation partition; "
                f"completion={env.get('completed')}; throughput={env.get('throughput_per_second')}; "
                f"queue={env.get('external_queue_mean')}; shield={env.get('shield_burden_per_100_joint_decisions')}")
    decoded = joint["decoded"]
    return (f"- {name}: agreement={joint['joint_policy_teacher_agreement']:.4f}; teacher burden/100 joint={joint['joint_teacher_intervention_rate'] * 100:.2f}; "
            f"WAIT/RELEASE requested={decoded['requested']['WAIT']}/{decoded['requested']['RELEASE']}; teacher={decoded['teacher']['WAIT']}/{decoded['teacher']['RELEASE']}; "
            f"release/speed/merge disagreement={decoded['release_disagreement_rate']:.4f}/{decoded['speed_disagreement_rate']:.4f}/{decoded['merge_disagreement_rate']:.4f}; "
            f"completion={env.get('completed')}; throughput={env.get('throughput_per_second')}; queue={env.get('external_queue_mean')}; shield={env.get('shield_burden_per_100_joint_decisions')}")

def _markdown(report: dict) -> str:
    lines = ["# V4 L1 Full-Teacher PPO Short Pilot", "", "## Experiment Evidence", "",
             "- Status / evidence type: newly_run.", f"- Training samples: {report['training_samples']}; updates/episodes: {len(report['updates'])}/{len(report['episodes'])}.",
             f"- Frozen base commit: `{BASE_COMMIT}`; tracked working-tree diff SHA256: `{TRACKED_DIFF_SHA256}`.",
             "- Fresh V4 joint actor/critic; no V3 checkpoint was loaded or resumed.", ""]
    lines += ["## Safety and PPO health", "", f"- Safety = {'PASS' if report['safety_passed'] else 'FAIL'}; PPO health = {'PASS' if report['ppo_health_passed'] else 'FAIL'}.",
              f"- conservative_wait_collapse = {str(report['collapse']['conservative_wait_collapse']).lower()}", ""]
    for label in ("baseline", "update3", "update6"):
        result = report["validation"][label]; natural, warm, anchor = result["natural_l1"], result["warm_start"], result["joint_conflict_anchor"]
        lines += [f"## {label}", "", _validation_line("natural L1", natural), _validation_line("warm-start", warm), _validation_line("held-out joint-conflict anchor", anchor), ""]
    lines += ["## Per-update joint learning", ""]
    for row in report["updates"]:
        env, joint, ppo = row["environment"], row["joint"], row["ppo"]
        decoded = joint["decoded"]
        lines.append(f"- Update {row['update']}: rows={joint['joint_decisions']}; agreement={joint['joint_policy_teacher_agreement']:.4f}; teacher/100 joint={env['teacher_interventions_per_100_joint_decisions']:.2f}; entropy={joint['joint_entropy']:.4f}; requested WAIT/RELEASE={decoded['requested']['WAIT']}/{decoded['requested']['RELEASE']}; teacher WAIT/RELEASE={decoded['teacher']['WAIT']}/{decoded['teacher']['RELEASE']}; R→W/W→R={decoded['RELEASE_to_WAIT']}/{decoded['WAIT_to_RELEASE']}; release acceptance={env['release_acceptance']}; completion={env['completed']}; throughput={env['throughput_per_second']}; queue={env['external_queue_mean']}; shield/100 joint={env['shield_burden_per_100_joint_decisions']}; actor={ppo['actor_loss']:.6f}; value={ppo['value_loss']:.6f}; KL={ppo['approx_kl']:.6f}; clip={ppo['clip_fraction']:.6f}; entropy={ppo['entropy']:.6f}; EV={ppo['explained_variance']:.6f}; grad={ppo['grad_norm']:.6f}")
    lines += ["", f"V4_joint_learning = {report['V4_joint_learning']}", f"V4_L1_SHORT_PILOT = {report['V4_L1_SHORT_PILOT']}",
              f"longer_L1_continuation_allowed = {str(report['longer_L1_continuation_allowed']).lower()}", "formal_L1_training_allowed = false", "",
              "## Protocol risks", "", "- This is one fresh short pilot, not formal training or a multi-seed generalization claim.", "- The held-out anchor uses disjoint qualification seeds and is never used in PPO/BC; natural validation uses fixed disjoint validation seeds.", "- Reproducibility version freeze remains incomplete because the working tree cannot be committed; the frozen base and diff digest above are recorded.", ""]
    return "\n".join(lines)


def _validate_l0_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    bootstrap = manifest.get("joint_bc_bootstrap", {})
    required = (manifest.get("stage") == "V4_L0_teacher_dataset" and manifest.get("evidence_type") == "newly_run"
                and manifest.get("low_action_schema") == "V4_joint_discrete" and manifest.get("action_schema_version") == ACTION_SCHEMA_V4
                and manifest.get("target_rows", 0) >= 3000 and not any(manifest.get("safety", {}).values())
                and bootstrap.get("joint_ce_after", float("inf")) < bootstrap.get("joint_ce_before", float("-inf")))
    if not required:
        raise RuntimeError("Fresh V4 L1 short pilot requires the passed freshly generated V4 L0 manifest")
    return manifest


def run_v4_l1_short_pilot(*, output_dir: str | Path, l0_manifest: str | Path) -> dict:
    protocol = V4L1ShortPilotProtocol(); destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    l0 = _validate_l0_manifest(l0_manifest)
    partition = LowV2SeedPartition(); anchor = _build_anchor_cohort(destination, partition)
    root, _ = _v4_config(0, horizon_seconds=float(protocol.max_steps))
    torch.manual_seed(6401); np.random.seed(6401); random.seed(6401)
    policy = JointPPO(LOW_OBS_V2.dimension, teacher_bc_weight=protocol.teacher_bc_weight)
    snapshots = {"baseline": _checkpoint(destination, "baseline", policy, root)}

    def validate(label: str, update: int) -> dict:
        state = _model_state(policy); tasks = []
        for episode, (design, seed) in enumerate(zip(VALIDATION_DESIGN_INDICES, partition.validation[:3])):
            config, _ = _v4_config(design, horizon_seconds=float(protocol.max_steps))
            tasks.append((state, config, seed, update, episode, protocol.max_steps, True, str(destination / "validation_action_tapes")))
        records = _collect(tasks, protocol.rollout_workers)
        natural = _records_metrics(records, state)
        warm = _records_metrics([record for record in records if record[1].reset_mode == "warm_start_rings"], state)
        for values in (natural["environment"], warm["environment"]):
            if not _safety_ok(values):
                raise RuntimeError("VALIDATION_SAFETY_FAILURE")
        return {"natural_l1": natural, "warm_start": warm, "joint_conflict_anchor": _anchor_metrics(policy, anchor)}

    validation = {"baseline": validate("baseline", 0)}
    updates: list[dict] = []; episodes: list[dict] = []; stopped_reason = None
    try:
        for update in range(protocol.updates):
            state = _model_state(policy); tasks = []; designs = []
            for episode in range(protocol.episodes_per_update):
                global_episode = update * protocol.episodes_per_update + episode
                config, design = _v4_config(global_episode, horizon_seconds=float(protocol.max_steps))
                seed = partition.train[global_episode]
                tasks.append((state, config, seed, update + 1, episode + 1, protocol.max_steps, False, str(destination / "training_action_tapes")))
                designs.append(design)
            collected = _collect(tasks, protocol.rollout_workers)
            for (_, summary), design in zip(collected, designs):
                if any((summary.collisions, summary.nmac, summary.hard_separation_violations, summary.safety_invariant_errors, summary.uncertified_components)):
                    raise RuntimeError("TRAINING_SAFETY_FAILURE")
                episodes.append({**asdict(summary), **design})
            metrics = _records_metrics(collected, state); by_scenario = _by_scenario(collected, state)
            if not _safety_ok(metrics["environment"]):
                raise RuntimeError("TRAINING_SAFETY_FAILURE")
            buffer = JointRolloutBuffer()
            for rows, _ in collected:
                for row in rows:
                    buffer.add(row)
            ppo = policy.update(buffer, epochs=protocol.ppo_epochs, minibatch_size=protocol.minibatch_size)
            if not _healthy_ppo(ppo):
                raise RuntimeError("PPO_HEALTH_FAILURE")
            record = {"update": update + 1, **metrics, "by_scenario": by_scenario, "ppo": ppo}
            updates.append(record); _write_jsonl(destination / "v4_l1_short_pilot_updates.jsonl", updates); _write_jsonl(destination / "v4_l1_short_pilot_episodes.jsonl", episodes)
            collapse = _collapse(updates)
            if collapse["conservative_wait_collapse"]:
                stopped_reason = "CONSERVATIVE_WAIT_COLLAPSE"; break
            if update + 1 == 3:
                snapshots["update3"] = _checkpoint(destination, "update3", policy, root); validation["update3"] = validate("update3", update + 1)
        if stopped_reason is None and len(updates) == protocol.updates:
            snapshots["update6"] = _checkpoint(destination, "update6", policy, root); validation["update6"] = validate("update6", protocol.updates)
        else:
            # Required snapshots are absent after a deliberate safety/collapse
            # stop; no fabricated update6 validation is emitted.
            raise RuntimeError(stopped_reason or "PILOT_STOPPED_BEFORE_UPDATE6")
    except (SafetyInvariantError, RuntimeError, FloatingPointError) as error:
        partial = {"stage": "V4_L1_short_pilot", "status": "stopped", "evidence_type": "newly_run", "reason": str(error),
                   "updates": updates, "episodes": episodes, "snapshots": snapshots, "validation": validation,
                   "formal_L1_training_allowed": False, "base_commit": BASE_COMMIT, "tracked_worktree_diff_sha256": TRACKED_DIFF_SHA256}
        _write_json(destination / "v4_l1_short_pilot.json", partial)
        (destination / "v4_l1_short_pilot_summary.md").write_text("# V4 L1 Full-Teacher PPO Short Pilot\n\nStatus: STOPPED — " + str(error) + "\n\nformal_L1_training_allowed = false\n", encoding="utf-8")
        raise
    collapse = _collapse(updates); safety = all(_safety_ok(row["environment"]) for row in updates)
    ppo_health = all(_healthy_ppo(row["ppo"]) for row in updates)
    learning, pilot, continuation = _decision(validation, updates, safety=safety, ppo_health=ppo_health, collapse=collapse["conservative_wait_collapse"])
    report = {"stage": "V4_L1_short_pilot", "status": "completed", "evidence_type": "newly_run", "ppo_called": True,
              "base_commit": BASE_COMMIT, "tracked_worktree_diff_sha256": TRACKED_DIFF_SHA256, "v4_l0_manifest": str(l0_manifest),
              "frozen": {"action": "V4_joint_Categorical(30)", "Teacher": "FULL joint-action Teacher", "BC": "joint categorical CE", "teacher_bc_weight": .05,
                         "mu_hat": MU_HAT, "lambda": LAMBDA, "geometry": "nominal", "priority": 1, "mission": "standard", "SOC": .8,
                         "Observation_V2_changed": False, "reward_changed": False, "PPO_hyperparameters_changed": False, "safety_stack_changed": False,
                         "V3_checkpoint_resumed": False, "V3_targeted_release_mixture": False},
              "protocol": asdict(protocol), "training_samples": sum(row["joint"]["joint_decisions"] for row in updates), "updates": updates,
              "episodes": episodes, "snapshots": snapshots, "validation": validation, "safety_passed": safety, "ppo_health_passed": ppo_health,
              "collapse": collapse, "V4_joint_learning": learning, "V4_L1_SHORT_PILOT": pilot,
              "longer_L1_continuation_allowed": continuation, "formal_L1_training_allowed": False}
    _write_json(destination / "v4_l1_short_pilot.json", report)
    (destination / "v4_l1_short_pilot_summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run fresh V4 Full-Teacher joint PPO short pilot; no resume.")
    parser.add_argument("--output", required=True); parser.add_argument("--l0-manifest", required=True)
    args = parser.parse_args()
    report = run_v4_l1_short_pilot(output_dir=args.output, l0_manifest=args.l0_manifest)
    print(json.dumps({key: report[key] for key in ("V4_joint_learning", "V4_L1_SHORT_PILOT", "longer_L1_continuation_allowed")}, indent=2))


if __name__ == "__main__":
    _main()
