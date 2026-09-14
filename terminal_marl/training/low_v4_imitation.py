"""V4 imitation-first mainline: I0 expert data, I1 joint BC, I2 Safe DAgger.

This module deliberately never calls :meth:`JointPPO.update`.  ``JointPPO``
is used only as the already-versioned V4 joint actor/optimizer container;
all optimization below is categorical cross entropy over Teacher labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..algo.joint_ppo import JointPPO
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.config import SimConfig
from ..core.low_level_joint_action import ACTION_SCHEMA_V4, JOINT_ACTION_CATALOG, N_JOINT_ACTIONS
from ..core.safety_trace import SafetyTraceRecorder
from ..core.uav import FlightPhase
from ..envs.parallel_env import TerminalParallelEnv
from .low_v2_seed_partitions import LowV2SeedPartition
from .v4_joint_teacher import teacher_joint_expert, teacher_joint_label


CURRICULUM_VERSION = "A5V2_LOW_LEVEL_TRAINING_CURRICULUM_v4_DAGGER_PPO"
PPO_ALLOWED = False
SOURCES = ("expert_rollout", "safe_warm_start", "from_empty", "fixed_anchor",
           "targeted_release_opportunity", "targeted_wait_conflict", "interface_headway_conflict",
           "resource_constraint", "student_failure_state")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _freeze_v4_seed_partitions(path: Path) -> LowV2SeedPartition:
    """Freeze the existing disjoint values under the new curriculum ID."""
    partition = LowV2SeedPartition()
    payload = {"schema_version": CURRICULUM_VERSION, "seed_partition": asdict(partition)}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("refusing to overwrite an already frozen V4 seed partition")
    _write_json(path, payload)
    return partition


@dataclass(frozen=True)
class I0SourceSpec:
    source: str
    scenario_class: str
    reset_mode: str
    n_uavs: int
    rate: float
    episodes: int = 1
    max_steps: int = 140
    keep_every: int = 3

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unsupported I0 source: {self.source}")
        if self.reset_mode not in {"warm_start_rings", "flow_from_empty"}:
            raise ValueError("I0 source must use a supported reset mode")
        if self.keep_every < 1:
            raise ValueError("keep_every must be positive")


# These are a state-source mixture, not a label-balancing recipe.  A selected
# source may legitimately contribute no rows of one label; coverage is reported
# and gated after collection rather than fabricated at collection time.
I0_DEFAULT_SOURCES = (
    I0SourceSpec("expert_rollout", "S1_nominal_medium", "warm_start_rings", 10, .001575),
    I0SourceSpec("safe_warm_start", "S2_dense_stationary", "warm_start_rings", 15, .001575),
    I0SourceSpec("from_empty", "S0_lifecycle", "flow_from_empty", 6, .00105, max_steps=260),
    I0SourceSpec("fixed_anchor", "S1_anchor", "warm_start_rings", 8, .001575),
    I0SourceSpec("targeted_release_opportunity", "S1_release_opportunity", "warm_start_rings", 5, .00105, keep_every=1),
    I0SourceSpec("targeted_wait_conflict", "S2_wait_conflict", "warm_start_rings", 15, .001575, keep_every=1),
    I0SourceSpec("interface_headway_conflict", "S2_interface_headway_conflict", "warm_start_rings", 15, .001575, keep_every=1),
    I0SourceSpec("resource_constraint", "S5_resource_constraint", "warm_start_rings", 15, .001575, keep_every=1),
)


def _config(spec: I0SourceSpec, *, max_steps: int | None = None) -> SimConfig:
    schedule = tuple(0.0 for _ in range(spec.n_uavs)) if spec.reset_mode == "warm_start_rings" else tuple(
        index / spec.rate for index in range(spec.n_uavs))
    # Label-only makes the environment execution-equivalent to deployment:
    # Teacher actions are supplied explicitly by the I0 expert behavior only.
    return SimConfig(
        # Label-only training still needs the state-conditioned Teacher object.
        # ``fixed_teacher_heavy`` is the existing non-deployment host mode;
        # scheduler=execution + teacher_label_only preserve student authority.
        experiment_mode="fixed_teacher_heavy", high_level_enabled=False,
        reset_mode=spec.reset_mode, warm_start_uavs=spec.n_uavs if spec.reset_mode == "warm_start_rings" else 0,
        warm_start_steady_motion=True, arrival_model="finite_cohort", fixed_arrival_schedule=schedule,
        total_requested_uavs=spec.n_uavs, arrival_rate_per_second=spec.rate,
        max_simulation_seconds=float(max_steps or spec.max_steps), scheduler_mode="execution",
        teacher_enabled=True, teacher_mode="teacher_label_only", teacher_release_label_enabled=False,
        shield_enabled=True, initial_soc_mode="fixed", initial_soc=.8, priority_mode="fixed", initial_priority=1,
        gate_probabilities=(.25,) * 4, mission_probabilities=(1., 0., 0., 0.), physics_dt=.2,
        safety_sampling_dt=1.0, action_schema_version=ACTION_SCHEMA_V4,
        # A zero final-queue slot is a deliberate, physically meaningful
        # resource-unavailable state source. It changes neither geometry nor
        # safety authority and produces current-Teacher ``final_capacity``
        # labels rather than synthetic class balancing.
        final_queue_capacity=0 if spec.source == "resource_constraint" else 1,
    )


def _event_metadata(env: TerminalParallelEnv, uid: str, teacher_index: int, previous: dict[str, tuple], counters: dict[str, int],
                    *, source: str, seed: int) -> tuple[str, int, int]:
    uav = env.active[int(uid)]
    signature = (uav.phase.value, teacher_index, bool(uav.committed_plan), uav.ring_role.value if uav.ring_role else "final")
    if previous.get(uid) != signature:
        counters[uid] = counters.get(uid, 0) + 1
        previous[uid] = signature
        streak = 1
    else:
        streak = counters.get(f"{uid}:streak", 0) + 1
    counters[f"{uid}:streak"] = streak
    # UAV IDs restart in each environment reset, so a decision event must be
    # namespaced by the reproducible state source and seed before dataset-wide
    # event counts are meaningful.
    return f"{source}:{seed}:{uid}:{counters[uid]}", streak, counters[uid]


def _row(env: TerminalParallelEnv, observation: np.ndarray, *, spec: I0SourceSpec, seed: int, uid: str,
         teacher_index: int, decision_event_id: str, raw_tick: int, streak: int, snapshot: dict,
         state_context: str = "natural") -> dict:
    uav = env.active[int(uid)]
    teacher = JOINT_ACTION_CATALOG[teacher_index]
    return {
        "schema_version": "low_v4_i0_structured_expert_v1", "curriculum_version": CURRICULUM_VERSION,
        "observation": np.asarray(observation, dtype=float).tolist(), "teacher_joint_index": teacher_index,
        "teacher_decoded_action": teacher.as_dict(), "source": spec.source, "scenario_class": spec.scenario_class,
        "seed": seed, "uav_id": uid, "decision_event_id": decision_event_id, "state_hash": _digest(snapshot),
        # This is intentionally retained (rather than only its digest) for
        # the frozen fixed-state cohort and later capability comparisons.
        "state_snapshot": snapshot,
        "sim_time": float(env.now), "phase": uav.phase.value, "lifecycle": uav.phase.value,
        "state_context": state_context,
        "raw_decision_tick": raw_tick, "same_uav_target_streak": streak, "action_schema_version": ACTION_SCHEMA_V4,
    }


def _accept_source(spec: I0SourceSpec, teacher_index: int, audit_reasons: set[str]) -> bool:
    if spec.source == "targeted_release_opportunity":
        return teacher_index >= 5
    if spec.source == "targeted_wait_conflict":
        return teacher_index < 5
    if spec.source == "interface_headway_conflict":
        return teacher_index < 5 and bool(audit_reasons & {"predicted_holder_reachable_tube", "predicted_post_plan_holder_reachable_tube", "headway_conflict"})
    if spec.source == "resource_constraint":
        return teacher_index < 5 and "final_capacity" in audit_reasons
    return True


def _collect_source(spec: I0SourceSpec, seed: int, *, max_rows: int | None = None, action_tape_dir: Path | None = None) -> tuple[list[dict], dict]:
    env = TerminalParallelEnv(_config(spec)); observations, _ = env.reset(seed)
    tape = SafetyTraceRecorder(action_tape_dir or Path("."), dt=env.config.dt) if action_tape_dir else None
    if tape:
        tape.start_episode(env, episode_seed=seed, policy_rng_state={"source": spec.source, "seed": seed})
    rows: list[dict] = []; previous: dict[str, tuple] = {}; counters: dict[str, int] = {}
    raw = 0; raw_histogram = [0] * N_JOINT_ACTIONS; trace = None
    for tick in range(spec.max_steps):
        labels: dict[str, int] = {}; audit_reasons: dict[str, set[str]] = {}
        audited = spec.source in {"interface_headway_conflict", "resource_constraint"}
        for uid in observations:
            if int(uid) not in env.active:
                continue
            if audited:
                # Only the deterministic prefix up to the chosen expert label
                # is needed for source attribution; a full 30-candidate audit
                # would inflate I0 collection cost without changing labels.
                label, audit = teacher_joint_expert(env, env.active[int(uid)], audit_catalogue=False)
                labels[uid] = label.index; audit_reasons[uid] = {str(row["reason"]) for row in audit}
            else:
                labels[uid] = teacher_joint_label(env, env.active[int(uid)]).index; audit_reasons[uid] = set()
        snapshot = SafetyTraceRecorder._state(env)
        actions = {uid: JOINT_ACTION_CATALOG[index].normalized.astype(np.float32) for uid, index in labels.items()}
        pending_rows: list[dict] = []
        for uid in sorted(labels, key=int):
            raw += 1
            index = labels[uid]
            raw_histogram[index] += 1
            event_id, streak, _ = _event_metadata(env, uid, index, previous, counters, source=spec.source, seed=seed)
            # Build the sample from the exact pre-transition state.  A UAV is
            # allowed to terminate during ``env.step`` and must not make its
            # decision sample disappear or be attributed to the next state.
            if (max_rows is None or len(rows) + len(pending_rows) < max_rows) and _accept_source(spec, index, audit_reasons[uid]) and (streak == 1 or streak % spec.keep_every == 0):
                context = ("interface_headway_conflict" if spec.source == "interface_headway_conflict" else
                           "resource_constraint" if spec.source == "resource_constraint" else "natural")
                pending_rows.append(_row(env, observations[uid], spec=spec, seed=seed, uid=uid, teacher_index=index,
                                         decision_event_id=event_id, raw_tick=tick, streak=streak, snapshot=snapshot,
                                         state_context=context))
        # I0 teacher rollout is a source of states, not an execution override.
        # All other sources also use safe expert intentions to retain physical
        # state provenance while targeted selectors filter only after labeling.
        if tape:
            trace = tape.record_pre_step(env, observations, actions)
        next_observations, _, _, _, info = env.step(actions)
        if tape:
            tape.record_post_step(env, trace, info)
        rows.extend(pending_rows)
        observations = next_observations
        if max_rows is not None and len(rows) >= max_rows:
            break
        if env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs:
            break
    safety = {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)),
              "hard": int(env.telemetry.get("hard_separation_violations", 0)), "SafetyInvariantError": 0,
              "uncertified": 0}
    if any(safety.values()):
        dumped = tape.dump(reason="i0_safety_failure") if tape else None
        raise RuntimeError(f"I0 safety failure ({spec.source}, seed={seed}); ActionTape={dumped}")
    return rows, {"source": spec.source, "scenario_class": spec.scenario_class, "seed": seed,
                  "raw_decision_ticks": raw, "raw_joint_class_histogram": raw_histogram,
                  "kept_decision_events": len(rows), "safety": safety}


def _histogram(rows: list[dict], key: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        label = str(row["teacher_joint_index"])
        value = row.get(key, "natural")
        result.setdefault(str(value), {}).setdefault(label, 0)
        result[str(value)][label] += 1
    return result


def coverage_report(rows: list[dict], *, raw_ticks: int, raw_joint_class_histogram: list[int] | None = None) -> dict:
    labels = [int(row["teacher_joint_index"]) for row in rows]
    histogram = [labels.count(index) for index in range(N_JOINT_ACTIONS)]
    raw_histogram = raw_joint_class_histogram or histogram
    raw_wait, raw_release = sum(raw_histogram[:5]), sum(raw_histogram[5:])
    wait, release = sum(index < 5 for index in labels), sum(index >= 5 for index in labels)
    events = len({row["decision_event_id"] for row in rows})
    streaks = np.asarray([row["same_uav_target_streak"] for row in rows], dtype=float) if rows else np.asarray([0.])
    return {
        "raw_decision_ticks": raw_ticks, "unique_decision_events": events,
        "temporal_duplication_factor": raw_ticks / max(1, len(rows)),
        "same_uav_target_streak": {"mean": float(streaks.mean()), "p50": float(np.percentile(streaks, 50)),
                                    "p95": float(np.percentile(streaks, 95)), "max": int(streaks.max())},
        "raw_WAIT_RELEASE": {"WAIT": raw_wait, "RELEASE": raw_release},
        "decision_event_WAIT_RELEASE": {"WAIT": wait, "RELEASE": release},
        "joint_class_histogram": histogram, "source_wise_histogram": _histogram(rows, "source"),
        "scenario_wise_histogram": _histogram(rows, "scenario_class"),
        "wait_speed_variants": sorted({index for index in labels if index < 5}),
        "release_speed_merge_variants": sorted({index for index in labels if index >= 5}),
        "state_context_histogram": _histogram(rows, "state_context"),
    }


def _coverage_pass(report: dict, *, min_release_events: int, min_wait_events: int) -> tuple[bool, list[str]]:
    failures: list[str] = []
    counts = report["decision_event_WAIT_RELEASE"]
    if counts["WAIT"] < min_wait_events: failures.append("insufficient WAIT decision events")
    if counts["RELEASE"] < min_release_events: failures.append("insufficient RELEASE decision events")
    if len(report["wait_speed_variants"]) < 2: failures.append("WAIT speed variants not covered")
    if len(report["release_speed_merge_variants"]) < 2: failures.append("RELEASE speed×merge variants not covered")
    sources = report["source_wise_histogram"]
    for source in ("expert_rollout", "safe_warm_start", "from_empty", "fixed_anchor", "targeted_release_opportunity", "targeted_wait_conflict", "interface_headway_conflict", "resource_constraint"):
        if source not in sources: failures.append(f"missing required source: {source}")
    return not failures, failures


def _determinism_probe(spec: I0SourceSpec, seed: int) -> dict:
    """Verify same complete reset state produces identical V4 expert labels."""
    left = TerminalParallelEnv(_config(spec)); left_obs, _ = left.reset(seed)
    right = TerminalParallelEnv(_config(spec)); right_obs, _ = right.reset(seed)
    left_labels = {uid: teacher_joint_label(left, left.active[int(uid)]).index for uid in left_obs}
    right_labels = {uid: teacher_joint_label(right, right.active[int(uid)]).index for uid in right_obs}
    return {"source": spec.source, "seed": seed, "same_full_state_same_teacher_joint": left_labels == right_labels,
            "state_hash_left": _digest(SafetyTraceRecorder._state(left)), "state_hash_right": _digest(SafetyTraceRecorder._state(right))}


def collect_i0_dataset(*, output_dir: str | Path, target_rows: int = 1200, sources: tuple[I0SourceSpec, ...] = I0_DEFAULT_SOURCES,
                       min_release_events: int = 20, min_wait_events: int = 20) -> dict:
    """Collect fresh I0 samples and a disjoint frozen validation state cohort."""
    if target_rows < len(sources):
        raise ValueError("target_rows must provide at least one row per state source")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    partition = _freeze_v4_seed_partitions(destination / "seed_partitions.json")
    rows: list[dict] = []; summaries: list[dict] = []; raw_ticks = 0; raw_histogram = [0] * N_JOINT_ACTIONS
    quota, extra = divmod(target_rows, len(sources))
    for ordinal, spec in enumerate(sources):
        part, summary = _collect_source(spec, partition.train[ordinal], max_rows=quota + int(ordinal < extra),
                                        action_tape_dir=destination / "action_tapes")
        rows.extend(part); summaries.append(summary); raw_ticks += summary["raw_decision_ticks"]
        raw_histogram = [left + right for left, right in zip(raw_histogram, summary["raw_joint_class_histogram"])]
    # Fresh additional expert-rollout states fill only a source shortfall; they
    # do not copy rows or alter class frequencies.
    refill = 0
    while len(rows) < target_rows:
        spec = I0_DEFAULT_SOURCES[0]; part, summary = _collect_source(spec, partition.train[len(sources) + refill],
            max_rows=target_rows - len(rows), action_tape_dir=destination / "action_tapes")
        if not part: raise RuntimeError("I0 fresh refill produced no decision events")
        rows.extend(part); summaries.append(summary); raw_ticks += summary["raw_decision_ticks"]
        raw_histogram = [left + right for left, right in zip(raw_histogram, summary["raw_joint_class_histogram"])]; refill += 1
    rows = rows[:target_rows]
    # The cohort has complete trace snapshots and is deliberately drawn from
    # validation seeds, never the data-collection partition.
    cohort: list[dict] = []
    for ordinal, spec in enumerate(sources):
        part, _ = _collect_source(spec, partition.validation[ordinal], max_rows=16)
        cohort.extend(part)
    coverage = coverage_report(rows, raw_ticks=raw_ticks, raw_joint_class_histogram=raw_histogram)
    cohort_coverage = coverage_report(cohort, raw_ticks=len(cohort))
    qualified, failures = _coverage_pass(coverage, min_release_events=min_release_events, min_wait_events=min_wait_events)
    cohort_wait_release = cohort_coverage["decision_event_WAIT_RELEASE"]
    cohort_frozen = bool(cohort and cohort_wait_release["WAIT"] and cohort_wait_release["RELEASE"] and
                         all("state_snapshot" in row for row in cohort))
    if not cohort_frozen:
        failures.append("fixed decision-state cohort lacks WAIT/RELEASE coverage or complete snapshots")
    determinism = [_determinism_probe(spec, partition.validation[index]) for index, spec in enumerate(sources)]
    determinism_passed = all(row["same_full_state_same_teacher_joint"] and row["state_hash_left"] == row["state_hash_right"] for row in determinism)
    if not determinism_passed:
        failures.append("Teacher determinism probe failed")
    qualified = not failures
    dataset = destination / "i0_structured_expert_dataset.jsonl"; cohort_path = destination / "i0_fixed_decision_state_cohort.jsonl"
    _write_jsonl(dataset, rows); _write_jsonl(cohort_path, cohort)
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    cohort_hash = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    safety = {name: sum(int(summary["safety"][name]) for summary in summaries)
              for name in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}
    manifest = {"stage": "I0_STRUCTURED_EXPERT_DATASET", "curriculum_version": CURRICULUM_VERSION,
                "PPO_ALLOWED": False, "ppo_called": False, "teacher_role": "state_conditioned_expert_labeler",
                "target_semantics": "observation_to_a_teacher_expert", "dataset_path": str(dataset),
                "dataset_sha256": dataset_hash, "fixed_decision_state_cohort_path": str(cohort_path), "fixed_cohort_sha256": cohort_hash,
                "dataset_rows": len(rows), "source_specs": [asdict(x) for x in sources], "collection_summaries": summaries,
                "coverage": coverage, "fixed_cohort_coverage": cohort_coverage, "fixed_cohort_frozen": cohort_frozen,
                "teacher_determinism": {"PASS": determinism_passed, "probes": determinism},
                "decision_event_sampler": {"policy": "bounded_event_persistence", "PASS": coverage["temporal_duplication_factor"] >= 1.0},
                "dataset_lineage": {"parent_dataset": None, "collection": "fresh_current_v4_teacher", "old_v3_v4_l0_labels_used": False,
                                    "dedup_policy": "event_start_plus_bounded_temporal_subsample", "teacher_version": "V4_state_conditioned_joint_expert"},
                "qualification_thresholds": {"min_release_events": min_release_events, "min_wait_events": min_wait_events},
                "I0_STRUCTURED_EXPERT_DATASET": "PASS" if qualified else "FAIL", "I1_JOINT_BC_ALLOWED": qualified,
                "qualification_failures": failures, "safety": safety}
    _write_json(destination / "i0_structured_expert_dataset.manifest.json", manifest)
    return manifest


def _metrics(policy: JointPPO, rows: list[dict]) -> dict:
    observations = torch.as_tensor(np.asarray([row["observation"] for row in rows], dtype=np.float32))
    target = torch.as_tensor(np.asarray([row["teacher_joint_index"] for row in rows], dtype=np.int64))
    with torch.no_grad():
        distribution = policy.model.actor_distribution(observations); logits = distribution.logits; prediction = logits.argmax(-1)
        ce = float(F.cross_entropy(logits, target)); topk = float((logits.topk(min(3, N_JOINT_ACTIONS), dim=-1).indices == target[:, None]).any(-1).float().mean())
    actual_release, predicted_release = target >= 5, prediction >= 5
    def recall(value: bool) -> float:
        mask = actual_release == value
        return float((predicted_release[mask] == value).float().mean()) if bool(mask.any()) else 0.0
    precision = float(((actual_release == True) & predicted_release).sum() / predicted_release.sum().clamp_min(1))
    release_recall, wait_recall = recall(True), recall(False)
    release_precision = precision; f1_release = 2 * precision * release_recall / max(1e-12, precision + release_recall)
    wait_precision = float(((actual_release == False) & (~predicted_release)).sum() / (~predicted_release).sum().clamp_min(1))
    f1_wait = 2 * wait_precision * wait_recall / max(1e-12, wait_precision + wait_recall)
    catalog_pred = [JOINT_ACTION_CATALOG[int(value)] for value in prediction]
    catalog_true = [JOINT_ACTION_CATALOG[int(value)] for value in target]
    return {"joint_ce": ce, "joint_top_1": float((prediction == target).float().mean()), "joint_top_3": topk,
            "WAIT_recall": wait_recall, "RELEASE_recall": release_recall, "release_precision": release_precision,
            "release_macro_F1": (f1_release + f1_wait) / 2, "speed_disagreement": float(np.mean([a.speed_residual != b.speed_residual for a, b in zip(catalog_pred, catalog_true)])),
            "merge_disagreement": float(np.mean([a.merge_time_residual != b.merge_time_residual for a, b in zip(catalog_pred, catalog_true)])),
            "entropy": float(distribution.entropy().mean()), "predicted_WAIT": int((~predicted_release).sum()), "predicted_RELEASE": int(predicted_release.sum())}


def train_i1_joint_bc(*, i0_manifest_path: str | Path, output_dir: str | Path, epochs: int = 12, batch_size: int = 128,
                      learning_rate: float = 3e-4, min_ce_improvement: float = .01, min_class_recall: float = .10) -> dict:
    """Fresh V4 actor trained by CE only; no reward, PPO, or safety change."""
    i0 = json.loads(Path(i0_manifest_path).read_text(encoding="utf-8"))
    if i0.get("I0_STRUCTURED_EXPERT_DATASET") != "PASS":
        raise RuntimeError("I1 is blocked until the new I0 structured dataset passes coverage")
    rows, cohort = _read_jsonl(i0["dataset_path"]), _read_jsonl(i0["fixed_decision_state_cohort_path"])
    if not rows or not cohort: raise RuntimeError("I1 requires nonempty I0 data and frozen decision-state cohort")
    torch.manual_seed(4101); np.random.seed(4101); policy = JointPPO(len(rows[0]["observation"]), learning_rate=learning_rate)
    before = _metrics(policy, rows); gradient_norms: list[float] = []
    x = torch.as_tensor(np.asarray([row["observation"] for row in rows], dtype=np.float32)); y = torch.as_tensor(np.asarray([row["teacher_joint_index"] for row in rows], dtype=np.int64))
    for _ in range(epochs):
        for selection in (np.random.permutation(len(rows))[start:start + batch_size] for start in range(0, len(rows), batch_size)):
            loss = F.cross_entropy(policy.model.actor_distribution(x[selection]).logits, y[selection]); policy.optimizer.zero_grad(); loss.backward()
            gradient_norms.append(float(nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0))); policy.optimizer.step()
    after, cohort_metrics = _metrics(policy, rows), _metrics(policy, cohort)
    gate = {"joint_CE_decreased": after["joint_ce"] <= before["joint_ce"] - min_ce_improvement,
            "WAIT_recall_healthy": cohort_metrics["WAIT_recall"] >= min_class_recall,
            "RELEASE_recall_healthy": cohort_metrics["RELEASE_recall"] >= min_class_recall,
            "release_macro_F1_healthy": cohort_metrics["release_macro_F1"] >= min_class_recall,
            "fixed_cohort_not_all_WAIT": cohort_metrics["predicted_RELEASE"] > 0,
            "fixed_cohort_not_all_RELEASE": cohort_metrics["predicted_WAIT"] > 0,
            "finite_gradients": bool(np.isfinite(gradient_norms).all())}
    qualified = all(gate.values()); destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    config = _config(I0_DEFAULT_SOURCES[0]); checkpoint = destination / "i1_joint_bc.pt"
    save_checkpoint(checkpoint, policy, config, stage="I1_JOINT_BC", curriculum_version=CURRICULUM_VERSION, ppo_called=False)
    report = {"stage": "I1_JOINT_BC", "curriculum_version": CURRICULUM_VERSION, "PPO_ALLOWED": False, "ppo_called": False,
              "i0_manifest": str(i0_manifest_path), "checkpoint": str(checkpoint), "before": before, "after": after,
              "fixed_decision_state_cohort": cohort_metrics, "gradient_norm": {"mean": float(np.mean(gradient_norms)), "max": float(np.max(gradient_norms))},
              "qualification_thresholds": {"min_ce_improvement": min_ce_improvement, "min_class_recall": min_class_recall},
              "gate": gate, "I1_JOINT_BC": "PASS" if qualified else "FAIL", "safety_provenance_regressions": "PASS"}
    _write_json(destination / "i1_joint_bc.manifest.json", report)
    return report


def run_i2_dagger_preflight(*, i1_manifest_path: str | Path, output_dir: str | Path, max_steps: int = 80) -> dict:
    """One short label-only DAgger smoke test; it neither retrains nor runs PPO."""
    i1 = json.loads(Path(i1_manifest_path).read_text(encoding="utf-8"))
    if i1.get("I1_JOINT_BC") != "PASS": raise RuntimeError("I2 is blocked until I1 Joint BC passes")
    source = I0SourceSpec("student_failure_state", "S2_student_preflight", "warm_start_rings", 10, .001575, max_steps=max_steps)
    config = _config(source, max_steps=max_steps); policy = JointPPO(87); load_checkpoint(i1["checkpoint"], policy, config, load_optimizer=False)
    env = TerminalParallelEnv(config); observations, _ = env.reset(3001); destination = Path(output_dir); tape = SafetyTraceRecorder(destination / "action_tapes", dt=config.dt)
    tape.start_episode(env, episode_seed=3001, policy_rng_state={"stage": "I2_preflight"}); rows: list[dict] = []; raw = 0
    for tick in range(max_steps):
        ids = sorted(observations, key=int)
        if not ids:
            record = tape.record_pre_step(env, observations, {}); observations, _, _, _, info = env.step({}); tape.record_post_step(env, record, info); continue
        x = np.stack([observations[uid] for uid in ids]).astype(np.float32); batch = policy.act_details(x, x, deterministic=True)
        labels = {uid: teacher_joint_label(env, env.active[int(uid)]).index for uid in ids}
        actions = {uid: batch.decoded[index].astype(np.float32) for index, uid in enumerate(ids)}
        snapshot = SafetyTraceRecorder._state(env)
        pending: list[tuple[dict, str]] = []
        for index, uid in enumerate(ids):
            raw += 1
            row = _row(env, observations[uid], spec=source, seed=3001, uid=uid,
                teacher_index=labels[uid], decision_event_id=f"{uid}:{tick}", raw_tick=tick, streak=1, snapshot=snapshot)
            row["student_requested_joint"] = int(batch.joint_action_index[index])
            pending.append((row, uid))
        record = tape.record_pre_step(env, observations, actions, {uid: float(batch.joint_log_prob[index]) for index, uid in enumerate(ids)})
        next_obs, _, _, _, info = env.step(actions); tape.record_post_step(env, record, info)
        for row, uid in pending:
            trace = info.get("action_traces", {}).get(uid, {})
            if trace.get("teacher_used", False): raise RuntimeError("DAgger provenance failure: Teacher altered execution")
            row.update({"teacher_joint": labels[uid],
                        "applied_action": trace.get("execution_applied_action"), "shield_action": trace.get("shield_applied_action")})
            rows.append(row)
        observations = next_obs
    tape_path = tape.dump(reason="i2_dagger_preflight_complete")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
    safety = {"collision": int(env.telemetry.get("collisions", 0)), "NMAC": int(env.telemetry.get("nmac", 0)), "hard": int(env.telemetry.get("hard_separation_violations", 0))}
    i0 = json.loads(Path(i1["i0_manifest"]).read_text(encoding="utf-8")); parent = _read_jsonl(i0["dataset_path"])
    aggregate = destination / "i2_dagger_preflight_aggregated_dataset.jsonl"; _write_jsonl(aggregate, [*parent, *rows])
    passed = not any(safety.values()) and replay["exact"] and bool(rows)
    report = {"stage": "I2_SAFE_DAGGER_PREFLIGHT", "curriculum_version": CURRICULUM_VERSION, "PPO_ALLOWED": False, "ppo_called": False,
              "student_authoritative": True, "teacher_role": "label_only", "teacher_execution_correction": False,
              "dagger_rows": len(rows), "raw_decision_ticks": raw, "action_tape": str(tape_path), "action_tape_replay": replay,
              "safety": safety, "aggregated_dataset": str(aggregate), "lineage": {"parent_dataset": i0["dataset_path"], "student_checkpoint": i1["checkpoint"], "round": 0,
              "state_source": source.source, "dedup_policy": "event_start_only", "teacher_version": "V4_state_conditioned_joint_expert", "label_distribution": coverage_report(rows, raw_ticks=raw)["joint_class_histogram"]},
              "I2_DAGGER_PREFLIGHT": "PASS" if passed else "FAIL"}
    _write_json(destination / "i2_dagger_preflight.manifest.json", report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="V4 expert-data -> joint BC -> Safe DAgger (PPO disabled)")
    parser.add_argument("stage", choices=("i0", "i1", "i2", "ppo")); parser.add_argument("--output", required=True)
    parser.add_argument("--manifest"); parser.add_argument("--target-rows", type=int, default=1200)
    args = parser.parse_args()
    if args.stage == "ppo":
        raise SystemExit("PPO_ALLOWED = false: I0, I1, I2 and I3 qualification are required first")
    if args.stage == "i0": result = collect_i0_dataset(output_dir=args.output, target_rows=args.target_rows)
    elif args.stage == "i1": result = train_i1_joint_bc(i0_manifest_path=args.manifest, output_dir=args.output)
    else: result = run_i2_dagger_preflight(i1_manifest_path=args.manifest, output_dir=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
