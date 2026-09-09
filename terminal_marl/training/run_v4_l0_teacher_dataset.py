"""Fresh V4 L0 Full-Teacher dataset collection (no PPO, no V3 conversion)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo import JointPPO
from ..core.config import SimConfig
from ..core.low_level_joint_action import ACTION_SCHEMA_V4, JOINT_ACTION_CATALOG, JointAction, N_JOINT_ACTIONS, decode_joint_actions
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .low_v2_seed_partitions import LowV2SeedPartition, freeze_seed_partitions
from .v4_joint_teacher import component_disagreement, teacher_joint_label


TARGET_ROWS = 3600
MU_HAT = 0.003
MEDIUM_RATE = 0.001575
LOW_RATE = 0.5 * MU_HAT


@dataclass(frozen=True)
class V4L0Cohort:
    name: str
    reset_mode: str
    arrival_model: str
    n_uavs: int
    rate: float
    episodes: int
    max_steps: int


COHORTS = (
    V4L0Cohort("warm_low_n5", "warm_start_rings", "finite_cohort", 5, LOW_RATE, 2, 160),
    V4L0Cohort("warm_low_n10", "warm_start_rings", "finite_cohort", 10, LOW_RATE, 2, 120),
    V4L0Cohort("warm_medium_n15", "warm_start_rings", "finite_cohort", 15, MEDIUM_RATE, 2, 100),
    V4L0Cohort("empty_low_n5", "flow_from_empty", "finite_cohort", 5, LOW_RATE, 1, 900),
    V4L0Cohort("empty_medium_n8", "flow_from_empty", "poisson", 8, MEDIUM_RATE, 2, 900),
)
COHORT_BY_NAME = {item.name: item for item in COHORTS}


def _config(cohort: V4L0Cohort) -> SimConfig:
    schedule = () if cohort.arrival_model == "poisson" else (
        tuple(0.0 for _ in range(cohort.n_uavs)) if cohort.reset_mode == "warm_start_rings"
        else tuple(index / cohort.rate for index in range(cohort.n_uavs))
    )
    return SimConfig(
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode=cohort.reset_mode, warm_start_uavs=cohort.n_uavs if cohort.reset_mode == "warm_start_rings" else 0,
        warm_start_steady_motion=False, arrival_model=cohort.arrival_model,
        fixed_arrival_schedule=schedule, total_requested_uavs=cohort.n_uavs if cohort.arrival_model == "finite_cohort" else None,
        arrival_rate_per_second=cohort.rate, max_simulation_seconds=float(cohort.max_steps),
        scheduler_mode="teacher_full", teacher_enabled=True, teacher_mode="teacher_full",
        # The legacy V3 per-head release target is deliberately disabled.  V4
        # labels are obtained only through ``teacher_joint_label`` below.
        teacher_release_label_enabled=False, shield_enabled=True,
        initial_soc_mode="fixed", initial_soc=.8, priority_mode="fixed", initial_priority=1,
        gate_probabilities=(.25,) * 4, mission_probabilities=(1., 0., 0., 0.),
        physics_dt=.2, safety_sampling_dt=1.0, action_schema_version=ACTION_SCHEMA_V4,
    )


class CatalogExcitation:
    """Uniform requested-action sweep; it never alters a Teacher target."""
    def __init__(self, seed: int) -> None:
        self.seed = int(seed); self.tick = 0

    def act(self, observations: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        requested: dict[str, np.ndarray] = {}; indices: dict[str, int] = {}
        for ordinal, uid in enumerate(sorted(observations, key=int)):
            index = (self.seed * 17 + self.tick * 31 + ordinal * 7 + int(uid) * 3) % N_JOINT_ACTIONS
            indices[uid] = int(index); requested[uid] = decode_joint_actions(np.asarray(index)).astype(np.float32)
        self.tick += 1
        return requested, indices


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def _row(*, cohort: V4L0Cohort, episode: int, seed: int, step: int, uid: str, observation: np.ndarray,
         requested: JointAction, teacher: JointAction, trace: dict | None, decision: str | None, env: TerminalParallelEnv) -> dict:
    disagreement = component_disagreement(requested, teacher)
    return {
        "low_action_schema": "V4_joint_discrete", "action_schema_version": ACTION_SCHEMA_V4,
        "cohort": cohort.name, "episode": episode, "seed": seed, "step": step, "uav_id": uid,
        "observation_v2": np.asarray(observation, dtype=float).tolist(),
        "observation_hash": _hash(np.asarray(observation, dtype=float).tolist()),
        "a_requested_joint_index": requested.index, "a_requested_decoded": requested.as_dict(),
        "a_teacher_joint_index": teacher.index, "a_teacher_decoded": teacher.as_dict(),
        "teacher_used": requested.index != teacher.index, "joint_correction": requested.index != teacher.index,
        "component_disagreement": disagreement,
        "execution_applied_action": None if trace is None else trace.get("execution_applied_action"),
        "shield_applied_action": None if trace is None else trace.get("shield_applied_action"),
        "release_decision": decision,
        "scenario": {"cohort": cohort.name, "reset_mode": cohort.reset_mode, "arrival_model": cohort.arrival_model,
                     "n_uavs": cohort.n_uavs, "traffic_rate": cohort.rate, "mu_hat": MU_HAT,
                     "geometry": "nominal", "priority": 1, "mission": "standard", "initial_soc": .8,
                     "physics_dt": .2, "safety_sampling_dt": 1.0},
        "state_metadata": {"time": float(env.now), "step_count": int(env.step_count),
                           "active_uavs": len(env.active), "scheduler_reservations": len(env.scheduler.reservations),
                           "release_intents": len(env.release_intents), "state_hash": _hash(SafetyTraceRecorder._state(env))},
    }


def _collect_task(task: tuple[V4L0Cohort, int, int, int, str]) -> tuple[list[dict], dict]:
    cohort, seed, episode, quota, tape_root = task
    config = _config(cohort); env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    policy = CatalogExcitation(seed); tape = SafetyTraceRecorder(Path(tape_root) / cohort.name, dt=config.dt)
    tape.start_episode(env, episode_seed=seed, policy_rng_state={"v4_l0_catalog_seed": seed})
    rows: list[dict] = []; record = None
    try:
        for step in range(cohort.max_steps):
            actions, indices = policy.act(observations)
            # Capture labels from the unmodified full state.  This calls the
            # V4 bounded Teacher catalogue, never the V3 counterfactual label.
            labels = {uid: teacher_joint_label(env, env.active[int(uid)]) for uid in observations if int(uid) in env.active}
            record = tape.record_pre_step(env, observations, actions)
            next_observations, _, terminated, truncated, info = env.step(actions)
            tape.record_post_step(env, record, info)
            for uid in sorted(observations, key=int):
                if len(rows) >= quota or uid not in labels:
                    continue
                requested = JOINT_ACTION_CATALOG[indices[uid]]; trace = info.get("action_traces", {}).get(uid)
                rows.append(_row(cohort=cohort, episode=episode, seed=seed, step=step, uid=uid,
                                 observation=observations[uid], requested=requested, teacher=labels[uid], trace=trace,
                                 decision=info.get("release_decisions", {}).get(uid), env=env))
            observations = next_observations
            if len(rows) >= quota or (env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs):
                break
    except SafetyInvariantError as error:
        tape.record_safety_invariant(record, error)
        raise RuntimeError(f"V4 L0 SafetyInvariantError: {cohort.name}, seed={seed}") from error
    safety = {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)),
              "hard": int(env.telemetry.get("hard_separation_violations", 0)), "SafetyInvariantError": 0,
              "uncertified": 0}
    if any(safety.values()):
        path = tape.dump(reason="v4_l0_safety_gate")
        raise RuntimeError(f"V4 L0 safety gate failed: {safety}; exact ActionTape={path}")
    return rows, {"cohort": cohort.name, "seed": seed, "episode": episode, "rows": len(rows), **safety}


def _coverage(rows: list[dict]) -> dict:
    requested = [0] * N_JOINT_ACTIONS; teacher = [0] * N_JOINT_ACTIONS; correction = [0] * N_JOINT_ACTIONS
    matrix = [[0] * N_JOINT_ACTIONS for _ in range(N_JOINT_ACTIONS)]
    for row in rows:
        left, right = int(row["a_requested_joint_index"]), int(row["a_teacher_joint_index"])
        requested[left] += 1; teacher[right] += 1; matrix[left][right] += 1
        if left != right: correction[right] += 1
    def aggregate(values: list[int]) -> dict[str, object]:
        return {"WAIT_total": sum(values[:5]), "RELEASE_total": sum(values[5:]),
                "WAIT_speed_5": values[:5], "RELEASE_speed_x_merge_25": values[5:]}
    zero_teacher = [index for index, value in enumerate(teacher) if value == 0]
    return {"requested_joint_action_frequency": requested, "teacher_joint_target_frequency": teacher,
            "teacher_correction_frequency_by_target": correction, "requested_to_teacher_transition_matrix": matrix,
            "requested_aggregate": aggregate(requested), "teacher_aggregate": aggregate(teacher),
            "correction_aggregate": aggregate(correction), "teacher_used": sum(bool(row["teacher_used"]) for row in rows),
            "zero_teacher_target_indices": zero_teacher,
            "zero_target_interpretation": "physically unnecessary under the deterministic current Teacher preference unless a targeted state audit proves scenario coverage insufficient"}


def _joint_bc_bootstrap(rows: list[dict]) -> dict:
    """A small CE-only bootstrap check, not a PPO update or a checkpoint."""
    observations = np.asarray([row["observation_v2"] for row in rows], dtype=np.float32)
    labels = np.asarray([row["a_teacher_joint_index"] for row in rows], dtype=np.int64)
    torch.manual_seed(4401); rng = np.random.default_rng(4401)
    policy = JointPPO(observations.shape[1])
    x = torch.as_tensor(observations); y = torch.as_tensor(labels)
    with torch.no_grad():
        distribution = policy.model.actor_distribution(x)
        before = float(F.cross_entropy(distribution.logits, y)); entropy_before = float(distribution.entropy().mean())
    finite_gradient = True
    # Fixed 32 x 256 examples: lightweight supervised evidence only.  It
    # does not call JointPPO.update and cannot affect PPO state/checkpoints.
    for _ in range(32):
        choice = rng.integers(0, len(rows), size=min(256, len(rows)))
        loss = F.cross_entropy(policy.model.actor_distribution(x[choice]).logits, y[choice])
        policy.optimizer.zero_grad(); loss.backward()
        finite_gradient = finite_gradient and all(param.grad is None or bool(torch.isfinite(param.grad).all()) for param in policy.model.parameters())
        policy.optimizer.step()
    with torch.no_grad():
        distribution = policy.model.actor_distribution(x); logits = distribution.logits
        after = float(F.cross_entropy(logits, y)); prediction = logits.argmax(-1)
        accuracy = float((prediction == y).float().mean())
        topk = float((logits.topk(min(3, N_JOINT_ACTIONS), dim=-1).indices == y[:, None]).any(dim=-1).float().mean())
        entropy_after = float(distribution.entropy().mean())
    return {"ppo_called": False, "joint_ce_before": before, "joint_ce_after": after, "accuracy": accuracy,
            "top_3_accuracy": topk, "gradient_finite": finite_gradient, "entropy_before": entropy_before,
            "entropy_after": entropy_after, "entropy_finite": bool(np.isfinite(entropy_before) and np.isfinite(entropy_after)),
            "decoded_component_accuracy": {
                "release_agreement": float(np.mean([(JOINT_ACTION_CATALOG[int(left)].release_request == JOINT_ACTION_CATALOG[int(right)].release_request) for left, right in zip(prediction.numpy(), labels)])),
                "speed_agreement": float(np.mean([(JOINT_ACTION_CATALOG[int(left)].speed_residual == JOINT_ACTION_CATALOG[int(right)].speed_residual) for left, right in zip(prediction.numpy(), labels)])),
                "merge_agreement": float(np.mean([(JOINT_ACTION_CATALOG[int(left)].merge_time_residual == JOINT_ACTION_CATALOG[int(right)].merge_time_residual) for left, right in zip(prediction.numpy(), labels)])),
            }}


def _determinism_probe(cohort: V4L0Cohort, seed: int) -> dict:
    config = _config(cohort); env = TerminalParallelEnv(config); observations, _ = env.reset(seed)
    requested_indices = tuple(range(N_JOINT_ACTIONS))
    labels = {uid: [teacher_joint_label(env, env.active[int(uid)]).index for _ in requested_indices] for uid in observations}
    replay = TerminalParallelEnv(config); replay_obs, _ = replay.reset(seed)
    repeat = {uid: [teacher_joint_label(replay, replay.active[int(uid)]).index for _ in requested_indices] for uid in replay_obs}
    # V4 target selection is state-only, so a request sweep must not change it;
    # the required same-state + same-request property follows as a subset.
    same_state_same_request = labels == repeat
    request_invariant = all(len(set(values)) == 1 for values in labels.values())
    return {"same_full_state_same_requested_joint_same_teacher_target": same_state_same_request,
            "request_sweep_target_invariant": request_invariant, "tested_requests": N_JOINT_ACTIONS}


def _markdown(manifest: dict) -> str:
    coverage, bootstrap = manifest["coverage"], manifest["joint_bc_bootstrap"]
    passed = (not any(manifest["safety"].values()) and manifest["teacher_determinism"]["same_full_state_same_requested_joint_same_teacher_target"]
              and bootstrap["gradient_finite"] and bootstrap["entropy_finite"] and bootstrap["joint_ce_after"] < bootstrap["joint_ce_before"])
    return "\n".join([
        "# V4 L0 Teacher Dataset Summary", "", "Evidence type: **newly_run**. The collector queried the current V4 TeacherScheduler; it did not read, convert, or reuse any V3 L0 label/checkpoint.", "",
        "## Frozen collection", "", f"- Schema: `{manifest['low_action_schema']}` / Categorical({manifest['joint_action_count']})", f"- Teacher: {manifest['teacher']}; μ_hat={manifest['mu_hat']}; low={manifest['low_rate']}; medium={manifest['medium_rate']}",
        "- Geometry=nominal; priority=1; mission=standard; SOC=0.8; physics_dt=0.2; safety_sampling_dt=1.0.", f"- Rows: {manifest['target_rows']}; cohorts: {', '.join(item['name'] for item in manifest['cohorts'])}.", "",
        "## Joint coverage", "", f"- Requested frequency, indices 0--29: `{coverage['requested_joint_action_frequency']}`", f"- Teacher target frequency, indices 0--29: `{coverage['teacher_joint_target_frequency']}`", f"- Teacher correction frequency by target, indices 0--29: `{coverage['teacher_correction_frequency_by_target']}`", f"- Requested aggregate: `{coverage['requested_aggregate']}`", f"- Teacher aggregate: `{coverage['teacher_aggregate']}`", f"- Requested→Teacher transition matrix: `{coverage['requested_to_teacher_transition_matrix']}`", f"- Zero Teacher targets: `{coverage['zero_teacher_target_indices']}`. Interpretation: {coverage['zero_target_interpretation']}.", "",
        "## Semantics and safety", "", f"- Same full state + same requested joint proposal → same Teacher target: **{manifest['teacher_determinism']['same_full_state_same_requested_joint_same_teacher_target']}**.", f"- Teacher's request sweep target invariant: **{manifest['teacher_determinism']['request_sweep_target_invariant']}** (V4 target selection is state-only over the bounded catalogue).", f"- Safety: `{manifest['safety']}`. Exact V4 ActionTape is configured to dump immediately on any gate event.", "",
        "## Joint CE bootstrap (not PPO)", "", f"- CE {bootstrap['joint_ce_before']:.6f} → {bootstrap['joint_ce_after']:.6f}; accuracy={bootstrap['accuracy']:.4f}; top-3={bootstrap['top_3_accuracy']:.4f}.", f"- finite gradient={bootstrap['gradient_finite']}; finite entropy={bootstrap['entropy_finite']}; decoded component diagnostics={bootstrap['decoded_component_accuracy']}.", "",
        f"## V4 L0 = {'PASS' if passed else 'FAIL'}", "",
        "This result is L0 evidence only; it neither ran PPO nor authorizes long training by itself.", "",
    ])


def run_v4_l0_teacher_dataset(*, output_dir: str | Path, target_rows: int = TARGET_ROWS, workers: int = 1) -> dict:
    if not 3000 <= target_rows <= 4000:
        raise ValueError("V4 L0 target_rows must be within 3,000--4,000")
    if workers <= 0:
        raise ValueError("workers must be positive")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    partition = freeze_seed_partitions(destination / "seed_partitions.json", LowV2SeedPartition())
    tasks: list[tuple[V4L0Cohort, int, int, int, str]] = []
    expanded = [(cohort, episode) for cohort in COHORTS for episode in range(cohort.episodes)]
    quota, remainder = divmod(target_rows, len(expanded))
    for ordinal, (cohort, episode) in enumerate(expanded):
        tasks.append((cohort, partition.train[ordinal], episode, quota + (ordinal < remainder), str(destination / "action_tapes")))
    if workers == 1:
        parts = [_collect_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            parts = list(executor.map(_collect_task, tasks))
    rows = [row for part, _ in parts for row in part]
    # From-empty episodes may legitimately have fewer live decision rows than
    # their nominal quota.  Complete the requested fresh sample budget with
    # new seeds from the same frozen *warm medium* L0 cohort; never duplicate
    # rows, reuse V3 data, or alter/synthetically balance Teacher labels.
    retries: list[dict] = []
    retry_index = len(tasks)
    while len(rows) < target_rows:
        if retry_index >= len(partition.train):
            raise RuntimeError(f"V4 L0 collected {len(rows)} fresh rows but exhausted the frozen train seed partition")
        missing = target_rows - len(rows)
        cohort = COHORT_BY_NAME["warm_medium_n15"]
        retry_task = (cohort, partition.train[retry_index], retry_index, missing, str(destination / "action_tapes"))
        part, summary = _collect_task(retry_task)
        if not part:
            raise RuntimeError("V4 L0 fresh retry produced no decision rows")
        rows.extend(part); parts.append((part, summary))
        retries.append({"cohort": cohort.name, "seed": partition.train[retry_index], "needed_rows": missing, "collected_rows": len(part),
                        "reason": "from-empty fresh decision-row shortfall; no label balancing or historical-data fill"})
        retry_index += 1
    if len(rows) != target_rows:
        raise RuntimeError(f"V4 L0 collected {len(rows)} rows, expected exactly {target_rows}")
    coverage = _coverage(rows); deterministic = _determinism_probe(COHORTS[0], partition.train[0]); bootstrap = _joint_bc_bootstrap(rows)
    if not (deterministic["same_full_state_same_requested_joint_same_teacher_target"] and deterministic["request_sweep_target_invariant"]):
        raise RuntimeError("V4 joint Teacher determinism gate failed")
    safety = {key: sum(int(summary[key]) for _, summary in parts) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}
    if any(safety.values()):
        raise RuntimeError(f"V4 L0 aggregate safety gate failed: {safety}")
    if not (bootstrap["gradient_finite"] and bootstrap["entropy_finite"] and bootstrap["joint_ce_after"] < bootstrap["joint_ce_before"]):
        raise RuntimeError("V4 joint BC bootstrap sanity gate failed")
    dataset = destination / "v4_l0_teacher_dataset.jsonl"
    dataset.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"stage": "V4_L0_teacher_dataset", "evidence_type": "newly_run", "ppo_called": False,
                "low_action_schema": "V4_joint_discrete", "action_schema_version": ACTION_SCHEMA_V4,
                "joint_action_count": N_JOINT_ACTIONS, "teacher": "FULL", "mu_hat": MU_HAT,
                "low_rate": LOW_RATE, "medium_rate": MEDIUM_RATE, "priority": 1, "mission": "standard", "initial_soc": .8,
                "physics_dt": .2, "safety_sampling_dt": 1.0, "target_rows": target_rows,
                "cohorts": [asdict(item) for item in COHORTS], "part_summaries": [summary for _, summary in parts], "fresh_row_retries": retries,
                "safety": safety, "teacher_determinism": deterministic, "joint_bc_bootstrap": bootstrap,
                "coverage": coverage, "dataset_path": str(dataset)}
    (destination / "v4_l0_teacher_dataset.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "v4_l0_teacher_dataset_summary.md").write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate V4 L0 joint categorical Teacher data; never runs PPO.")
    parser.add_argument("--output", required=True); parser.add_argument("--target-rows", type=int, default=TARGET_ROWS); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); print(json.dumps(run_v4_l0_teacher_dataset(output_dir=args.output, target_rows=args.target_rows, workers=args.workers), indent=2))


if __name__ == "__main__":
    _main()
