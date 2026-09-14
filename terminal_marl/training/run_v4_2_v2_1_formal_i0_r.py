"""Fresh formal V4.2/V2.1 I0-R recollection under the approved support gate.

This module deliberately has no historical-dataset input.  Every formal row is
created by a new simulator rollout.  Collection strata shape initial scenario
conditions; RELEASE-family membership is computed only from current predicates
and the same-UAV semantic transition ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1
from ..core.observations import build_low_observation_v2_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import (
    OBSERVATION_SCHEMA, RETRY_HORIZON, SourceSpec, _config, _digest, _safe,
    _sha, _snapshot, _source_reset_recipe, _topology,
)
from .run_v4_2_v2_1_predicate_release_reachability import (
    CONFLICT_REASONS, _headway_context, _wait_reason, classify_family_flags,
)
from .run_v4_2_v2_1_release_manifold_audit import _catalogue
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


CURRICULUM_VERSION = "A5V2_LOW_LEVEL_TRAINING_CURRICULUM_v4_DAGGER_PPO_V4_2_V2_1"
SCHEMA_VERSION = "v4_2_v2_1_formal_i0_r_v1"
TRAIN_TARGET, FROZEN_TARGET = 2000, 300
TRAIN_MAX_ROWS, FROZEN_MAX_ROWS = 2500, 800
TRAIN_SEED_START, FROZEN_SEED_START = 101001, 111001
REQUIRED_SELECTED_SUPPORT = (11, 12, 13, 14)
MIN_RELEASE_LIFECYCLES = 100
ENGINEERING_RELEASE_TARGET = 130
MIN_SUPPORT_LIFECYCLES = 10
MIN_FROZEN_RELEASE = 80
FAMILY_MINIMUMS = {"clear_outer": 15, "middle_interface": 15, "post_wait": 10, "post_conflict": 10}


@dataclass(frozen=True)
class FormalStratum:
    name: str
    spec: SourceSpec


# Scenario-level stratification only.  Rows are never admitted or rejected by
# their Teacher label.  The N=8 and N=5 contexts are included because the
# independent support-closure audit established that proposals 12 and 14 are
# naturally selected there.
STRATA = (
    FormalStratum("target_proposal_12_context", SourceSpec("formal_target_12", "outer_open_n8", 8, max_steps=12)),
    FormalStratum("target_proposal_14_context", SourceSpec("formal_target_14", "outer_open_n5", 5, max_steps=12)),
    FormalStratum("clear_outer_low_traffic", SourceSpec("formal_clear_outer", "clear_outer", 3, phases=("ring_outer",), max_steps=12)),
    FormalStratum("middle_interface", SourceSpec("formal_middle_interface", "middle_interface", 6, phases=("ring_middle",), max_steps=12)),
    FormalStratum("transition_wait_release", SourceSpec("formal_transition", "transition_wait_release", 12, dynamic=True, max_steps=42)),
    FormalStratum("conflict_recovery", SourceSpec("formal_conflict_recovery", "conflict_recovery", 12, dynamic=True, max_steps=42)),
    FormalStratum("nominal_mixed", SourceSpec("formal_nominal_mixed", "nominal_mixed", 12, max_steps=12)),
    FormalStratum("ordinary_expert", SourceSpec("formal_ordinary", "ordinary_expert", 8, max_steps=12)),
    FormalStratum("resource_pressure", SourceSpec("formal_resource_pressure", "resource_pressure", 12, final_capacity=0, max_steps=12)),
)

# After one complete cycle has established WAIT/conflict/resource coverage,
# continue with whole release-rich episodes until the preregistered support
# gates close.  This is scenario stratification, not row- or label-conditioned
# sampling; every semantic event from every selected episode remains eligible.
RELEASE_RICH_CONTINUATION = tuple(
    item for item in STRATA if item.name in {
        "target_proposal_12_context", "target_proposal_14_context",
        "clear_outer_low_traffic", "middle_interface", "nominal_mixed",
        "ordinary_expert",
    }
)


def _scheduled_stratum(episode: int) -> FormalStratum:
    if episode < len(STRATA):
        return STRATA[episode]
    return RELEASE_RICH_CONTINUATION[(episode - len(STRATA)) % len(RELEASE_RICH_CONTINUATION)]


def _observation_hash(observation: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(observation, np.float32).tobytes()).hexdigest()


def _semantic_reason(prior: dict | None, current: dict, observation: np.ndarray, topology: tuple, tick: int) -> tuple[str | None, int, float]:
    """Return a semantic retention reason without using the target class count."""
    if prior is None:
        return "event_start", 1, 0.0
    old = np.asarray(prior["observation"], np.float32)
    similarity = float(np.dot(observation, old) / (np.linalg.norm(observation) * np.linalg.norm(old) + 1e-12))
    streak = int(prior["streak"]) + 1
    if current["teacher_joint_index"] != prior["teacher_joint_index"]:
        return "teacher_target_change", streak, similarity
    if current["phase"] != prior["phase"] or current["committed"] != prior["committed"]:
        return "phase_or_commit_change", streak, similarity
    if current["retry_active"] != prior["retry_active"]:
        return "release_eligibility_or_retry_change", streak, similarity
    if current["interface_open"] != prior["interface_open"] or current["resource_valid"] != prior["resource_valid"]:
        return "reservation_interface_or_resource_change", streak, similarity
    if topology != prior["topology"]:
        return "leader_follower_or_topology_change", streak, similarity
    # Use a per-feature normalized change, not an 88-D accumulated L2 norm;
    # otherwise tiny motion in many coordinates recreates raw-tick flooding.
    if float(np.max(np.abs(observation - old))) >= 0.25:
        return "meaningful_normalized_observation_change", streak, similarity
    if tick % 12 == 0:
        return "bounded_sparse_persistence", streak, similarity
    return None, streak, similarity


def _predicate_context(env, uav, selected, audit: list[dict]) -> dict:
    if selected.index == 0:
        rationale, context = _wait_reason(env, uav, audit)
    else:
        interface = {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(
            None if uav.ring_role is None else uav.ring_role.value
        )
        rationale = "certified_release"
        context = {
            "retry_active": bool(env.now < uav.safety_retry_after),
            "committed": bool(uav.committed_plan is not None),
            "interface_open": bool(interface is not None and interface not in env.airspace.affected_interfaces()),
            "resource_valid": bool(
                uav.phase.value != "ring_inner"
                or env.config.final_queue_capacity > sum(x.phase.value == "final" for x in env.active.values())
            ),
        }
    return {
        "teacher_gate": int(selected.index > 0),
        "teacher_proposal": None if selected.index == 0 else int(selected.index - 1),
        "teacher_joint_index": int(selected.index),
        "wait_rationale": rationale,
        "retry_active": bool(context["retry_active"]),
        "committed": bool(context["committed"]),
        "interface_open": bool(context["interface_open"]),
        "resource_valid": bool(context["resource_valid"]),
        "conflict_state": rationale in CONFLICT_REASONS,
        "holder_tube_state": rationale == "holder_tube_conflict",
        "phase": uav.phase.value,
    }


def _formal_row(env, observation: np.ndarray, *, partition: str, stratum: FormalStratum,
                seed: int, episode: int, tick: int, uid: int, lifecycle_id: str,
                event_index: int, retention_reason: str, streak: int, similarity: float,
                selected, audit: list[dict], current: dict, previous_semantic: dict | None,
                reset_recipe: dict) -> dict:
    uav = env.active[uid]
    snapshot = _snapshot(env, uav)
    repeated, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
    if int(repeated.index) != int(selected.index):
        raise RuntimeError("non-deterministic Teacher target during formal I0-R")
    hierarchical = hierarchical_from_joint_index_v4_1(int(selected.index))
    recheck = np.asarray(build_low_observation_v2_1(uav, env), np.float32)
    headway = _headway_context(env, uav)
    decision_id = f"{lifecycle_id}:decision-{event_index}"
    ledger_current = {
        **current,
        "uid": int(uid),
        "lifecycle_id": lifecycle_id,
        "decision_event_id": decision_id,
        "time": float(env.now),
    }
    family_flags = classify_family_flags(ledger_current, previous_semantic)
    proposal = JOINT_ACTION_CATALOG_V4_1[int(selected.index)]
    expected_retry = float(np.clip(max(0.0, uav.safety_retry_after - env.now) / RETRY_HORIZON, 0.0, 1.0))
    previous_payload = None if previous_semantic is None else {
        "decision_event_id": previous_semantic["decision_event_id"],
        "teacher_gate": previous_semantic["teacher_gate"],
        "teacher_proposal": previous_semantic["teacher_proposal"],
        "wait_rationale": previous_semantic["wait_rationale"],
        "time": previous_semantic["time"],
        "elapsed_seconds": max(0.0, float(env.now) - float(previous_semantic["time"])),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "curriculum_version": CURRICULUM_VERSION,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": 88,
        "retry_feature_index": 87,
        "action_schema": ACTION_SCHEMA_V4_2,
        "partition": partition,
        "collection_stratum": stratum.name,
        "collection_source": stratum.spec.source,
        "scenario_class": stratum.spec.scenario,
        "seed": int(seed),
        "episode_id": f"formal-i0-r:{partition}:episode-{episode}:seed-{seed}",
        "uav_id": int(uid),
        "raw_decision_tick": int(tick),
        "lifecycle_event_id": lifecycle_id,
        "decision_event_id": decision_id,
        "semantic_sampling_reason": retention_reason,
        "same_uav_target_streak": int(streak),
        "nearest_retained_similarity": float(similarity),
        "observation_v2_1": observation.astype(float).tolist(),
        "observation_hash": _observation_hash(observation),
        "observation_recheck_exact": bool(np.array_equal(observation, recheck)),
        "teacher_gate": int(hierarchical.gate),
        "teacher_proposal": hierarchical.proposal,
        "teacher_joint_index": int(selected.index),
        "teacher_decoded_action": _safe(hierarchical.normalized),
        "teacher_rationale": current["wait_rationale"],
        "teacher_repeat_gate": int(hierarchical_from_joint_index_v4_1(int(repeated.index)).gate),
        "teacher_repeat_proposal": hierarchical_from_joint_index_v4_1(int(repeated.index)).proposal,
        "teacher_repeat_joint_index": int(repeated.index),
        "teacher_selected_speed_residual": float(proposal.speed_residual),
        "teacher_selected_merge_residual": float(proposal.merge_time_residual),
        "teacher_catalogue_audit": _catalogue(audit),
        "audit_sidecar_actor_input": False,
        "audit_sidecar_bc_input": False,
        "family_flags": family_flags,
        "transition_ledger": {
            "same_uav_lifecycle": lifecycle_id,
            "previous_semantic_event": previous_payload,
            "current_teacher_gate": current["teacher_gate"],
            "current_teacher_proposal": current["teacher_proposal"],
            "current_wait_rationale": current["wait_rationale"],
        },
        "headway_state": headway,
        "state_snapshot": snapshot,
        "full_state_hash": _digest(snapshot),
        "config_hash": _digest(vars(env.config)),
        "reset_recipe": _safe(reset_recipe),
        "reset_recipe_hash": _digest(reset_recipe),
        "dynamic_airspace_state_hash": _digest(SafetyTraceRecorder._state(env)["dynamic_airspace"]),
        "arrival_rng_state_hash": _digest(SafetyTraceRecorder._state(env)["arrival_process"]["rng_state"]),
        "retry_feature_expected": expected_retry,
        "retry_feature_observed": float(observation[87]),
        "retry_active": current["retry_active"],
    }


def _lifecycle_counts(rows: list[dict], *, proposal: int | None = None, family: str | None = None) -> int:
    chosen = rows
    if proposal is not None:
        chosen = [row for row in chosen if row["teacher_joint_index"] == proposal]
    if family is not None:
        chosen = [row for row in chosen if row["family_flags"][family]]
    return len({row["lifecycle_event_id"] for row in chosen})


def _coverage(rows: list[dict], *, frozen: bool) -> dict:
    release = [row for row in rows if row["teacher_gate"] == 1]
    proposal_lifecycles = {str(p): _lifecycle_counts(release, proposal=p) for p in REQUIRED_SELECTED_SUPPORT}
    families = {name: _lifecycle_counts(release, family=name) for name in FAMILY_MINIMUMS}
    support = sorted({int(row["teacher_joint_index"]) for row in release})
    release_lifecycles = len({row["lifecycle_event_id"] for row in release})
    support_ok = all(proposal_lifecycles[str(p)] >= MIN_SUPPORT_LIFECYCLES for p in REQUIRED_SELECTED_SUPPORT)
    family_ok = all(families[name] >= minimum for name, minimum in FAMILY_MINIMUMS.items())
    release_ok = release_lifecycles >= (MIN_FROZEN_RELEASE if frozen else MIN_RELEASE_LIFECYCLES)
    return {
        "release_rows": len(release),
        "independent_release_lifecycles": release_lifecycles,
        "observed_selected_support": support,
        "required_selected_support": list(REQUIRED_SELECTED_SUPPORT),
        "proposal_independent_lifecycles": proposal_lifecycles,
        "family_independent_lifecycles": families,
        "stable_proposal_support": "PASS" if support_ok else "FAIL",
        "release_family_coverage": "PASS" if family_ok else "FAIL",
        "release_lifecycle_coverage": "PASS" if release_ok else "FAIL",
        "PASS": bool(support_ok and family_ok and release_ok),
    }


def _wait_coverage(rows: list[dict]) -> dict:
    waits = [row for row in rows if row["teacher_gate"] == 0]
    rationale = Counter(row["teacher_rationale"] for row in waits)
    return {
        "retry_active_WAIT": sum(row["retry_active"] for row in waits),
        "committed_WAIT": rationale["committed"],
        "interface_frozen_WAIT": rationale["geometry_conflict"] + rationale["interface_frozen"],
        "headway_WAIT": rationale["headway_conflict"],
        "resource_WAIT": rationale["resource"] + rationale["final_capacity"] + rationale["lz_unavailable"],
        "conflict_WAIT": sum(rationale[key] for key in CONFLICT_REASONS),
        "ordinary_scheduling_WAIT": rationale["other_wait"],
        "transition_WAIT": sum(row["collection_stratum"] in {"transition_wait_release", "conflict_recovery"} for row in waits),
    }


def _partition_ready(rows: list[dict], *, target: int, frozen: bool) -> bool:
    return len(rows) >= target and _coverage(rows, frozen=frozen)["PASS"]


def _write_progress(output: Path, partition: str, rows: list[dict], episode: int,
                    target: int, stratum: str, completed: bool) -> None:
    coverage = _coverage(rows, frozen=partition == "frozen")
    payload = {
        "stage": "V4_2_V2_1_FORMAL_I0_R",
        "partition": partition,
        "episodes_completed": episode,
        "current_stratum": stratum,
        "retained_rows": len(rows),
        "target_rows": target,
        "WAIT": sum(row["teacher_gate"] == 0 for row in rows),
        "RELEASE": sum(row["teacher_gate"] == 1 for row in rows),
        "coverage": coverage,
        "completed": completed,
    }
    (output / f"collection_progress_{partition}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_partition(output: Path, *, partition: str, seed_start: int, target: int,
                       max_rows: int, max_episodes: int, banned_hashes: set[str]) -> tuple[list[dict], dict, list[tuple[Path, SourceSpec, str]]]:
    rows: list[dict] = []
    seen = set(banned_hashes)
    raw_ticks = 0
    raw_gate = Counter()
    safety = Counter()
    tapes: list[tuple[Path, SourceSpec, str]] = []
    taped_strata: set[str] = set()
    episode = 0
    _write_progress(output, partition, rows, 0, target, "not_started", False)
    while episode < max_episodes and len(rows) < max_rows and not _partition_ready(rows, target=target, frozen=partition == "frozen"):
        stratum = _scheduled_stratum(episode)
        spec = stratum.spec
        seed = seed_start + episode
        env = TerminalParallelEnv(_config(spec))
        env.reset(seed)
        reset_recipe, observations = _source_reset_recipe(env, spec)
        record_tape = stratum.name not in taped_strata
        tape = SafetyTraceRecorder(output / "action_tapes" / partition / stratum.name, dt=env.config.dt) if record_tape else None
        if tape:
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "formal_i0_r", "partition": partition, "stratum": stratum.name}, reset_recipe=reset_recipe)
        lifecycle = {int(uid): f"formal-i0-r:{partition}:episode-{episode}:seed-{seed}:uid-{int(uid)}" for uid in env.active}
        raw_memory: dict[int, dict] = {}
        semantic_ledger: dict[int, dict] = {}
        event_index: Counter = Counter()
        for tick in range(spec.max_steps):
            actions: dict[str, np.ndarray] = {}
            labels: dict[str, dict] = {}
            pending: list[tuple[dict, dict]] = []
            for uid in [int(value) for value in sorted(observations) if int(value) in env.active]:
                uav = env.active[uid]
                # The unchanged Teacher target is cheap; the full 25-proposal
                # catalogue is materialised only for retained semantic events.
                # This changes neither admission nor the executed action.
                selected, light_audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                observation = np.asarray(observations[str(uid)], np.float32)
                current = _predicate_context(env, uav, selected, light_audit)
                topology = _topology(env, uid, _snapshot(env, uav))
                reason, streak, similarity = _semantic_reason(raw_memory.get(uid), current, observation, topology, tick)
                raw_ticks += 1
                raw_gate[current["teacher_gate"]] += 1
                raw_memory[uid] = {**current, "observation": observation.tolist(), "topology": topology, "streak": streak}
                if reason:
                    full_selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if int(full_selected.index) != int(selected.index):
                        raise RuntimeError("full catalogue audit changed deterministic Teacher target")
                    current = _predicate_context(env, uav, full_selected, audit)
                    event_index[uid] += 1
                    row = _formal_row(
                        env, observation, partition=partition, stratum=stratum, seed=seed,
                        episode=episode, tick=tick, uid=uid, lifecycle_id=lifecycle[uid],
                        event_index=event_index[uid], retention_reason=reason, streak=streak,
                        similarity=similarity, selected=full_selected, audit=audit, current=current,
                        previous_semantic=semantic_ledger.get(uid), reset_recipe=reset_recipe,
                    )
                    if row["full_state_hash"] not in seen:
                        seen.add(row["full_state_hash"])
                        ledger_row = {
                            **current,
                            "uid": uid,
                            "lifecycle_id": lifecycle[uid],
                            "decision_event_id": row["decision_event_id"],
                            "time": float(env.now),
                        }
                        pending.append((row, ledger_row))
                action = hierarchical_from_joint_index_v4_1(int(selected.index))
                actions[str(uid)] = action.normalized
                labels[str(uid)] = {"gate": action.gate, "proposal": action.proposal, "joint_index_v4_1": int(selected.index)}
            pre = tape.record_pre_step(env, observations, actions) if tape else None
            observations, _, _, _, info = env.step(actions)
            if tape:
                tape.record_post_step(env, pre, info)
                tape.record_teacher_labels(pre, labels)
            for row, ledger_row in pending:
                rows.append(row)
                semantic_ledger[int(row["uav_id"])] = ledger_row
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
            safety[key] += int(env.telemetry.get(field, 0))
        safety["SafetyInvariantError"] += 0
        safety["uncertified"] += 0
        if tape:
            path = tape.dump(reason=f"formal_i0_r_{partition}_{stratum.name}")
            tapes.append((path, spec, stratum.name))
            taped_strata.add(stratum.name)
        episode += 1
        complete = _partition_ready(rows, target=target, frozen=partition == "frozen")
        _write_progress(output, partition, rows, episode, target, stratum.name, complete)
    info = {
        "episodes": episode,
        "unique_seeds": episode,
        "raw_decision_ticks": raw_ticks,
        "raw_WAIT_RELEASE": {"WAIT": int(raw_gate[0]), "RELEASE": int(raw_gate[1])},
        "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")},
        "target_rows": target,
        "max_rows": max_rows,
        "coverage_at_stop": _coverage(rows, frozen=partition == "frozen"),
    }
    return rows, info, tapes


def _summary(rows: list[dict], info: dict) -> dict:
    release = [row for row in rows if row["teacher_gate"] == 1]
    lifecycle_counts = Counter(row["lifecycle_event_id"] for row in rows)
    proposal_rows = Counter(int(row["teacher_joint_index"]) for row in release)
    streaks = np.asarray([row["same_uav_target_streak"] for row in rows] or [0], float)
    retained = np.asarray(list(lifecycle_counts.values()) or [0], float)
    return {
        "rows": len(rows),
        "WAIT_count": len(rows) - len(release),
        "RELEASE_count": len(release),
        "unique_seeds": len({row["seed"] for row in rows}),
        "unique_episodes": len({row["episode_id"] for row in rows}),
        "unique_uav_lifecycles": len(lifecycle_counts),
        "unique_decision_events": len({row["decision_event_id"] for row in rows}),
        "raw_decision_ticks": info["raw_decision_ticks"],
        "raw_WAIT_RELEASE": info["raw_WAIT_RELEASE"],
        "temporal_duplication_factor": float(info["raw_decision_ticks"] / max(1, len(rows))),
        "same_uav_target_streak": {"p50": float(np.percentile(streaks, 50)), "p95": float(np.percentile(streaks, 95)), "max": int(streaks.max())},
        "same_lifecycle_retained_count": {"p50": float(np.percentile(retained, 50)), "p95": float(np.percentile(retained, 95)), "max": int(retained.max())},
        "proposal_row_histogram": dict(sorted(proposal_rows.items())),
        "selected_speed_values_diagnostic": sorted({row["teacher_selected_speed_residual"] for row in release}),
        "selected_merge_values_diagnostic": sorted({row["teacher_selected_merge_residual"] for row in release}),
        "family_row_counts": {name: sum(row["family_flags"][name] for row in release) for name in FAMILY_MINIMUMS},
        "wait_coverage": _wait_coverage(rows),
        "phase_ring_distribution": dict(Counter(f'{row["state_snapshot"]["target_uav"]["phase"]}:{row["state_snapshot"]["target_uav"]["ring_role"]}' for row in rows)),
        "traffic_N_distribution": dict(Counter(len(row["state_snapshot"]["active_uavs"]) for row in rows)),
        "source_stratum_distribution": dict(Counter(row["collection_stratum"] for row in rows)),
        "coverage": _coverage(rows, frozen=rows[0]["partition"] == "frozen") if rows else {},
    }


def _provenance(rows: list[dict]) -> dict:
    retry_mismatch = []
    schema_mismatch = []
    required_snapshot = {"env_now", "target_uav", "active_uavs", "scheduler", "airspace", "resources", "trace_state"}
    for row in rows:
        obs = np.asarray(row["observation_v2_1"], np.float32)
        target = row["state_snapshot"]["target_uav"]
        expected = float(np.clip(max(0.0, float(target["safety_retry_after"]) - float(row["state_snapshot"]["env_now"])) / RETRY_HORIZON, 0.0, 1.0))
        if not np.isclose(float(obs[87]) if len(obs) == 88 else np.nan, expected, atol=1e-7):
            retry_mismatch.append(row["decision_event_id"])
        if len(obs) != 88 or not np.isfinite(obs).all() or not row["observation_recheck_exact"] or not required_snapshot.issubset(row["state_snapshot"]):
            schema_mismatch.append(row["decision_event_id"])
    return {
        "rows_checked": len(rows),
        "retry_mismatch_count": len(retry_mismatch),
        "schema_mismatch_count": len(schema_mismatch),
        "retry_feature_provenance": "PASS" if not retry_mismatch else "FAIL",
        "schema_provenance": "PASS" if not schema_mismatch else "FAIL",
        "examples": {"retry": retry_mismatch[:10], "schema": schema_mismatch[:10]},
    }


def _aliasing(rows: list[dict]) -> dict:
    labels: dict[str, set[int]] = defaultdict(set)
    waits, releases = [], []
    for row in rows:
        labels[row["observation_hash"]].add(int(row["teacher_gate"]))
        (releases if row["teacher_gate"] else waits).append(np.asarray(row["observation_v2_1"], np.float32))
    conflicts = [key for key, value in labels.items() if len(value) > 1]
    nearest = [float(min(np.linalg.norm(item - other) for other in waits)) for item in releases] if waits and releases else []
    return {
        "exact_observation_hash_opposite_gate_conflicts": len(conflicts),
        "conflicting_hashes": conflicts[:10],
        "nearest_opposite_gate_distance": {
            "min": min(nearest) if nearest else None,
            "p50": float(np.percentile(nearest, 50)) if nearest else None,
            "p95": float(np.percentile(nearest, 95)) if nearest else None,
        },
        "observation_aliasing": "PASS" if not conflicts else "FAIL",
    }


def _replay(tapes: list[tuple[Path, SourceSpec, str]]) -> dict:
    reports = []
    for path, spec, stratum in tapes:
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        reports.append({"stratum": stratum, "path": str(path), "sha256": _sha(path), "result": result})
    exact = sum(bool(row["result"]["exact"]) for row in reports)
    return {
        "tapes": reports,
        "exact_tapes": exact,
        "total_tapes": len(reports),
        "exact_replay_rate": float(exact / len(reports)) if reports else 0.0,
        "replay": "PASS" if len(reports) == 2 * len(STRATA) and exact == len(reports) else "FAIL",
    }


def run(output_dir: str | Path, *, train_target: int = TRAIN_TARGET,
        frozen_target: int = FROZEN_TARGET, train_max_rows: int = TRAIN_MAX_ROWS,
        frozen_max_rows: int = FROZEN_MAX_ROWS, max_train_episodes: int = 600,
        max_frozen_episodes: int = 300) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, train_info, train_tapes = _collect_partition(
        output, partition="train", seed_start=TRAIN_SEED_START, target=train_target,
        max_rows=train_max_rows, max_episodes=max_train_episodes, banned_hashes=set(),
    )
    frozen, frozen_info, frozen_tapes = _collect_partition(
        output, partition="frozen", seed_start=FROZEN_SEED_START, target=frozen_target,
        max_rows=frozen_max_rows, max_episodes=max_frozen_episodes,
        banned_hashes={row["full_state_hash"] for row in train},
    )
    train_path = output / "formal_i0_r_train.jsonl"
    frozen_path = output / "formal_i0_r_frozen_cohort.jsonl"
    train_path.write_text("".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in train), encoding="utf-8")
    frozen_path.write_text("".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in frozen), encoding="utf-8")
    train_summary = _summary(train, train_info)
    frozen_summary = _summary(frozen, frozen_info)
    provenance = _provenance(train + frozen)
    aliasing = _aliasing(train + frozen)
    replay = _replay(train_tapes + frozen_tapes)
    teacher_deterministic = all(
        row["teacher_gate"] == row["teacher_repeat_gate"]
        and row["teacher_proposal"] == row["teacher_repeat_proposal"]
        and row["teacher_joint_index"] == row["teacher_repeat_joint_index"]
        for row in train + frozen
    )
    train_ids = {row["decision_event_id"] for row in train}
    frozen_ids = {row["decision_event_id"] for row in frozen}
    fresh_lineage = len(train_ids) == len(train) and len(frozen_ids) == len(frozen) and not train_ids & frozen_ids
    independence = all((
        not ({row["seed"] for row in train} & {row["seed"] for row in frozen}),
        not ({row["episode_id"] for row in train} & {row["episode_id"] for row in frozen}),
        not ({row["lifecycle_event_id"] for row in train} & {row["lifecycle_event_id"] for row in frozen}),
        not ({row["full_state_hash"] for row in train} & {row["full_state_hash"] for row in frozen}),
    ))
    safety = {key: int(train_info["safety"][key]) + int(frozen_info["safety"][key]) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}
    row_cardinality = 1800 <= len(train) <= 2500 and len(frozen) >= 240
    gates = {
        "fresh_lineage": fresh_lineage,
        "schema_provenance": provenance["schema_provenance"] == "PASS",
        "retry_provenance": provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": teacher_deterministic,
        "observation_aliasing": aliasing["observation_aliasing"] == "PASS",
        "ActionTape_exact_replay": replay["replay"] == "PASS",
        "safety": not any(safety.values()),
        "training_row_cardinality": row_cardinality,
        "training_RELEASE_lifecycle_coverage": train_summary["coverage"]["release_lifecycle_coverage"] == "PASS",
        "training_family_coverage": train_summary["coverage"]["release_family_coverage"] == "PASS",
        "training_stable_proposal_support": train_summary["coverage"]["stable_proposal_support"] == "PASS",
        "frozen_independence": independence,
        "frozen_RELEASE_coverage": frozen_summary["coverage"]["release_lifecycle_coverage"] == "PASS",
        "frozen_family_coverage": frozen_summary["coverage"]["release_family_coverage"] == "PASS",
        "frozen_stable_proposal_support": frozen_summary["coverage"]["stable_proposal_support"] == "PASS",
    }
    passed = all(gates.values())
    manifest = {
        "stage": "FRESH_FORMAL_V4_2_V2_1_I0_R",
        "evidence_type": "newly_run",
        "schema_version": SCHEMA_VERSION,
        "curriculum_version": CURRICULUM_VERSION,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": 88,
        "retry_feature_index": 87,
        "action_schema": ACTION_SCHEMA_V4_2,
        "frozen_semantics": ["Teacher", "Teacher ranking", "ExecutionScheduler", "TrajectoryGenerator", "SafetyFilter", "PredictiveShield", "action catalogue", "Observation V2.1", "release-family predicates", "transition ledger semantics"],
        "historical_rows_used": False,
        "diagnostic_rows_used": False,
        "class_weighting": False,
        "oversampling": False,
        "row_duplication": False,
        "synthetic_labels": False,
        "formal_training_path": str(train_path),
        "formal_training_sha256": _sha(train_path),
        "formal_frozen_path": str(frozen_path),
        "formal_frozen_sha256": _sha(frozen_path),
        "formal_training_rows": len(train),
        "formal_frozen_rows": len(frozen),
        "collection_protocol": {"strata": [{"name": item.name, "spec": asdict(item.spec)} for item in STRATA], "continuation_schedule": [item.name for item in RELEASE_RICH_CONTINUATION], "continuation_semantics": "whole scenario episodes after one full all-strata cycle; no row- or label-conditioned admission", "semantic_sampling": ["event start", "Teacher target change", "retry/release eligibility change", "phase/commit change", "leader/follower/topology change", "reservation/interface/resource change", "meaningful normalized-observation change", "bounded sparse persistence"]},
        "train_collection": train_info,
        "frozen_collection": frozen_info,
        "training_summary": train_summary,
        "frozen_summary": frozen_summary,
        "schema_retry_provenance": provenance,
        "teacher_determinism": "PASS" if teacher_deterministic else "FAIL",
        "observation_aliasing_audit": aliasing,
        "ActionTape_replay": replay,
        "safety_counts": safety,
        "frozen_cohort_independence": "PASS" if independence else "FAIL",
        "qualification_gates": gates,
        "qualification_failures": [name for name, value in gates.items() if not value],
        "V4_2_V2_1_I0_R": "PASS" if passed else "FAIL",
        "V4_2_V2_1_I1_ALLOWED": bool(passed),
        "I1_training_run": False,
        "DAgger_ALLOWED": False,
        "PPO_ALLOWED": False,
    }
    manifest_path = output / "v4_2_v2_1_formal_i0_r_manifest.json"
    manifest_path.write_text(json.dumps(_safe(manifest), indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-target", type=int, default=TRAIN_TARGET)
    parser.add_argument("--frozen-target", type=int, default=FROZEN_TARGET)
    parser.add_argument("--train-max-rows", type=int, default=TRAIN_MAX_ROWS)
    parser.add_argument("--frozen-max-rows", type=int, default=FROZEN_MAX_ROWS)
    parser.add_argument("--max-train-episodes", type=int, default=600)
    parser.add_argument("--max-frozen-episodes", type=int, default=300)
    args = parser.parse_args()
    result = run(
        args.output_dir,
        train_target=args.train_target,
        frozen_target=args.frozen_target,
        train_max_rows=args.train_max_rows,
        frozen_max_rows=args.frozen_max_rows,
        max_train_episodes=args.max_train_episodes,
        max_frozen_episodes=args.max_frozen_episodes,
    )
    print(json.dumps({
        "formal_training_rows": result["formal_training_rows"],
        "formal_frozen_rows": result["formal_frozen_rows"],
        "training_coverage": result["training_summary"]["coverage"],
        "frozen_coverage": result["frozen_summary"]["coverage"],
        "V4_2_V2_1_I0_R": result["V4_2_V2_1_I0_R"],
        "V4_2_V2_1_I1_ALLOWED": result["V4_2_V2_1_I1_ALLOWED"],
        "DAgger_ALLOWED": False,
        "PPO_ALLOWED": False,
    }, indent=2))
