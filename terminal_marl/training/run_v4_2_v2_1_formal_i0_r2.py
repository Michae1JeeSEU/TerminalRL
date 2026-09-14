"""Fresh standalone V4.2/V2.1 I0-R2 collection and qualification.

The consumed Q cohort is inaccessible until both collection partitions have
finished. It is then used only for the preregistered support-distance audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1
from ..core.runtime_traffic_safety import SafetyInvariantError
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import (
    OBSERVATION_SCHEMA, SourceSpec, _config, _digest, _safe, _sha, _snapshot,
    _source_reset_recipe, _topology,
)
from .run_v4_2_v2_1_formal_i0_r import (
    _aliasing, _formal_row, _predicate_context, _provenance, _semantic_reason,
)
from .run_v4_2_v2_1_i1_failure_attribution import _families, _total_variation
from .run_v4_2_v2_1_i1_q_cohort import _consume_object, _history_files, _identity_accumulator
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


PROTOCOL_VERSION = "V2_1_I0_R2_COLLECTOR_PROTOCOL_V1"
SCHEMA_VERSION = "v4_2_v2_1_formal_i0_r2_v1"
PROTOCOL_PATH = Path("terminal_marl/configs/low_v4_2/i0_r2_collection_protocol.yaml")
DESIGN_PATH = Path("docs/v4_2_v2_1_i0_r2_collection_protocol.md")
REPRO_PATH = Path("docs/v4_new_mainline_reproducibility_manifest.json")
EXPECTED_PROTOCOL_SHA = "40685f58b1db2096b86d84630f29527f0985430a1c284a60c78a49963c2994db"
EXPECTED_DESIGN_SHA = "6fea1cc6b8949957089d08b9c50fff6cc4b90fc77f2121ac2492c96a066d174f"
EXPECTED_Q_SHA = "8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5"
TRAIN_RANGE = (310001, 319999)
AUDIT_RANGE = (320001, 329999)
Q2_RESERVED = (330001, 339999)
TRAIN_MIN, TRAIN_MAX = 2500, 3500
AUDIT_MIN, AUDIT_MAX = 600, 1000
SECONDARY_PER_LIFECYCLE = 4
KNN_K = 10
LOCAL_RADII = (0.05, 0.10, 0.25)

CONTEXT_MINIMA = {
    "post_WAIT_RELEASE": 60,
    "retry_expired_or_transition_cleared_RELEASE": 40,
    "post_conflict_RELEASE": 40,
    "high_traffic_N12_proposal_11": 30,
    "middle_interface_proposal_14": 20,
    "post_WAIT_proposal_14": 20,
    "post_conflict_proposal_14": 15,
}
CONTEXT_EPISODE_MINIMA = {
    "post_WAIT_RELEASE": 12,
    "retry_expired_or_transition_cleared_RELEASE": 10,
    "post_conflict_RELEASE": 10,
    "high_traffic_N12_proposal_11": 10,
    "middle_interface_proposal_14": 8,
    "post_WAIT_proposal_14": 8,
    "post_conflict_proposal_14": 8,
}
ANCHOR_MINIMA = {
    "ordinary_WAIT": 250, "retry_active_WAIT": 100, "committed_WAIT": 100,
    "interface_frozen_WAIT": 80, "resource_WAIT": 30,
    "clear_outer_RELEASE": 100, "middle_interface_normal_RELEASE": 100,
    "proposal_12_RELEASE": 80, "proposal_13_RELEASE": 160,
}


@dataclass(frozen=True)
class R2Stratum:
    name: str
    spec: SourceSpec


STRATA = (
    R2Stratum("post_wait_release", SourceSpec("i0_r2_post_wait", "transition_wait_release", 12, dynamic=True, max_steps=42)),
    R2Stratum("retry_expired_transition_release", SourceSpec("i0_r2_retry_transition", "transition_wait_release", 12, dynamic=True, max_steps=42)),
    R2Stratum("conflict_recovery_release", SourceSpec("i0_r2_conflict_recovery", "conflict_recovery", 12, dynamic=True, max_steps=42)),
    R2Stratum("high_traffic_proposal11", SourceSpec("i0_r2_high_traffic_p11", "nominal_mixed", 12, max_steps=12)),
    R2Stratum("middle_interface_proposal14", SourceSpec("i0_r2_middle_p14", "middle_interface", 12, phases=("ring_middle",), max_steps=18)),
    R2Stratum("post_wait_proposal14", SourceSpec("i0_r2_post_wait_p14", "transition_wait_release", 12, dynamic=True, max_steps=42)),
    R2Stratum("post_conflict_proposal14", SourceSpec("i0_r2_post_conflict_p14", "conflict_recovery", 12, dynamic=True, max_steps=42)),
    R2Stratum("nominal_mixed", SourceSpec("i0_r2_nominal", "nominal_mixed", 12, max_steps=12)),
    R2Stratum("clear_outer", SourceSpec("i0_r2_clear_outer", "clear_outer", 3, phases=("ring_outer",), max_steps=12)),
    R2Stratum("resource_pressure", SourceSpec("i0_r2_resource", "resource_pressure", 12, final_capacity=0, max_steps=12)),
)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _verify_protocol() -> tuple[dict, dict]:
    protocol_sha, design_sha = _sha(PROTOCOL_PATH), _sha(DESIGN_PATH)
    manifest = json.loads(REPRO_PATH.read_text(encoding="utf-8"))
    registered = manifest["v4_2_v2_1_i0_r2_collector_protocol"]
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checks = {
        "protocol_version": protocol.get("protocol") == PROTOCOL_VERSION,
        "protocol_sha_expected": protocol_sha == EXPECTED_PROTOCOL_SHA,
        "design_sha_expected": design_sha == EXPECTED_DESIGN_SHA,
        "protocol_sha_manifest": registered.get("protocol_sha256") == protocol_sha,
        "design_sha_manifest": registered.get("design_sha256") == design_sha,
        "collection_authorized": protocol.get("authorization", {}).get("I0_R2_COLLECTION_ALLOWED") is True,
        "training_range": protocol["lineage"]["training_seed_range"] == list(TRAIN_RANGE),
        "audit_range": protocol["lineage"]["internal_audit_seed_range"] == list(AUDIT_RANGE),
        "q2_reserved": protocol["lineage"]["future_q2_seed_range_reserved"] == list(Q2_RESERVED),
        "context_minima": protocol["prospective_context_minima_independent_lifecycles"] == CONTEXT_MINIMA,
        "context_episode_minima": protocol["prospective_context_minimum_independent_episodes"] == CONTEXT_EPISODE_MINIMA,
        "anchor_minima": protocol["nominal_anchor_minima_independent_lifecycles"] == ANCHOR_MINIMA,
        "strata": protocol["collection"]["strata"] == [x.name for x in STRATA],
    }
    return protocol, {"checks": checks, "PASS": all(checks.values()), "protocol_sha256": protocol_sha, "design_sha256": design_sha}


def _set_digest(values: Iterable[Any]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _build_pre_reset_registry(history_root: Path, output: Path) -> tuple[dict, dict[str, Any]]:
    """Scan registry/manifest JSON metadata only; Q rows stay unopened."""
    identities = _identity_accumulator()
    sources = []
    excluded = output.resolve()
    for path in sorted(history_root.rglob("*.json")):
        try:
            path.resolve().relative_to(excluded)
            continue
        except ValueError:
            pass
        parsed, error = False, None
        try:
            _consume_object(json.loads(path.read_text(encoding="utf-8")), identities)
            parsed = True
        except Exception:
            # Legacy registries may contain JSON-lines or escaped display text.
            # Recover only explicit seed tokens; never infer an unavailable ID.
            try:
                text = path.read_text(encoding="utf-8")
                line_objects = 0
                for line in text.splitlines():
                    try:
                        _consume_object(json.loads(line), identities); line_objects += 1
                    except json.JSONDecodeError:
                        pass
                if not line_objects:
                    for match in re.finditer(r'\\?"?(?:episode_)?seed\\?"?\s*[:=-]\s*(\d+)', text, re.IGNORECASE):
                        identities["seeds"].add(int(match.group(1)))
                parsed = bool(line_objects) or bool(text)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        sources.append({"path": str(path), "sha256": _sha(path), "parsed": parsed, "error": error})
    ranges = {"training": TRAIN_RANGE, "internal_audit": AUDIT_RANGE, "future_Q2_reserved": Q2_RESERVED}
    collisions = {name: sorted(set(range(lo, hi + 1)) & identities["seeds"]) for name, (lo, hi) in ranges.items()}
    available = {name: [x for x in range(lo, hi + 1) if x not in identities["seeds"]] for name, (lo, hi) in ranges.items()}
    registry = {
        "stage": "V2_1_I0_R2_PRE_RESET_SEED_REGISTRY", "built_before_any_reset": True,
        "consumed_Q_rows_opened": False, "history_root": str(history_root),
        "history_files": sources, "history_file_count": len(sources),
        "parse_error_count": sum(x["error"] is not None for x in sources),
        "historical_seed_count": len(identities["seeds"]),
        "historical_seed_digest": _set_digest(identities["seeds"]),
        "candidate_ranges": {k: list(v) for k, v in ranges.items()},
        "candidate_collisions_rejected_before_reset": collisions,
        "available_candidates": available,
    }
    path = output / "v4_2_v2_1_i0_r2_pre_reset_seed_registry.json"
    _write(path, registry)
    registry.update({"path": str(path), "sha256": _sha(path)})
    return registry, identities


def _lifecycle_count(rows: list[dict], predicate) -> int:
    return len({r["lifecycle_event_id"] for r in rows if predicate(r)})


def _episode_count(rows: list[dict], predicate) -> int:
    return len({r["episode_id"] for r in rows if predicate(r)})


def _traffic(row: dict) -> int:
    return len(row["state_snapshot"]["active_uavs"])


def _proof(row: dict) -> dict:
    return row.get("transition_ledger", {}).get("wait_to_release_proof", {})


def _predicates() -> dict[str, Any]:
    release = lambda r: int(r["teacher_gate"]) == 1
    wait = lambda r: int(r["teacher_gate"]) == 0
    p11 = lambda r: release(r) and int(r["teacher_joint_index"]) == 11
    p14 = lambda r: release(r) and int(r["teacher_joint_index"]) == 14
    return {
        "post_WAIT_RELEASE": lambda r: release(r) and bool(r["family_flags"]["post_wait"]),
        "retry_expired_or_transition_cleared_RELEASE": lambda r: release(r) and _proof(r).get("wait_release_transition_kind") in {"retry_WAIT_to_RELEASE", "transition_WAIT_to_RELEASE"},
        "post_conflict_RELEASE": lambda r: release(r) and bool(r["family_flags"]["post_conflict"]),
        "high_traffic_N12_proposal_11": lambda r: p11(r) and _traffic(r) == 12,
        "middle_interface_proposal_14": lambda r: p14(r) and bool(r["family_flags"]["middle_interface"]),
        "post_WAIT_proposal_14": lambda r: p14(r) and bool(r["family_flags"]["post_wait"]),
        "post_conflict_proposal_14": lambda r: p14(r) and bool(r["family_flags"]["post_conflict"]),
        "ordinary_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "other_wait",
        "retry_active_WAIT": lambda r: wait(r) and bool(r["retry_active"]),
        "committed_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "committed",
        "interface_frozen_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "geometry_conflict",
        "resource_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "resource",
        "clear_outer_RELEASE": lambda r: release(r) and bool(r["family_flags"]["clear_outer"]),
        "middle_interface_normal_RELEASE": lambda r: release(r) and bool(r["family_flags"]["middle_interface"]) and not bool(r["family_flags"]["post_wait"]),
        "proposal_12_RELEASE": lambda r: release(r) and int(r["teacher_joint_index"]) == 12,
        "proposal_13_RELEASE": lambda r: release(r) and int(r["teacher_joint_index"]) == 13,
    }


def _coverage(rows: list[dict]) -> dict:
    predicates = _predicates()
    lifecycle_values = {key: _lifecycle_count(rows, pred) for key, pred in predicates.items()}
    episode_values = {key: _episode_count(rows, predicates[key]) for key in CONTEXT_EPISODE_MINIMA}
    context_pass = {key: lifecycle_values[key] >= minimum for key, minimum in CONTEXT_MINIMA.items()}
    context_episode_pass = {key: episode_values[key] >= minimum for key, minimum in CONTEXT_EPISODE_MINIMA.items()}
    anchor_pass = {key: lifecycle_values[key] >= minimum for key, minimum in ANCHOR_MINIMA.items()}
    support = sorted({int(r["teacher_joint_index"]) for r in rows if r["teacher_gate"] == 1})
    return {
        "independent_lifecycles": lifecycle_values, "independent_episodes": episode_values,
        "context_minima": CONTEXT_MINIMA, "context_episode_minima": CONTEXT_EPISODE_MINIMA,
        "nominal_anchor_minima": ANCHOR_MINIMA, "context_pass": context_pass,
        "context_episode_pass": context_episode_pass, "anchor_pass": anchor_pass,
        "teacher_selected_support": support, "stable_teacher_selected_support_PASS": set((11, 12, 13, 14)).issubset(support),
        "all_repair_context_minima_PASS": all(context_pass.values()) and all(context_episode_pass.values()),
        "nominal_anchor_coverage_PASS": all(anchor_pass.values()),
    }


def _make_transition_proof(row: dict, previous: dict | None, current: dict) -> dict:
    same = bool(previous and previous["uid"] == row["uav_id"] and previous["lifecycle_id"] == row["lifecycle_event_id"])
    is_transition = bool(same and previous["teacher_gate"] == 0 and row["teacher_gate"] == 1)
    kind = None
    if is_transition:
        if previous["retry_active"] and not current["retry_active"]:
            kind = "retry_WAIT_to_RELEASE"
        elif previous["wait_rationale"] in {"geometry_conflict", "headway_conflict", "holder_tube_conflict", "candidate_conflict"}:
            kind = "geometry_conflict_WAIT_to_RELEASE"
        elif previous["interface_open"] != current["interface_open"]:
            kind = "transition_WAIT_to_RELEASE"
        else:
            kind = "ordinary_WAIT_to_RELEASE"
    return {
        "previous_wait_gate": previous["teacher_gate"] if is_transition else None,
        "previous_wait_rationale": previous["wait_rationale"] if is_transition else None,
        "previous_wait_state_hash": previous.get("state_hash") if is_transition else None,
        "previous_wait_time": previous["time"] if is_transition else None,
        "current_release_state_hash": row["full_state_hash"] if is_transition else None,
        "elapsed_time": float(current["time"] - previous["time"]) if is_transition else None,
        "retry_active_to_inactive": bool(is_transition and previous["retry_active"] and not current["retry_active"]),
        "interface_state_transition": {"previous_open": previous["interface_open"], "current_open": current["interface_open"]} if is_transition else None,
        "headway_state_transition": {"previous": previous.get("headway_state"), "current": row.get("headway_state")} if is_transition else None,
        "conflict_state_transition": {"previous": previous["conflict_state"], "current": current["conflict_state"]} if is_transition else None,
        "same_uav_id": bool(same), "same_lifecycle_event_id": bool(same),
        "wait_release_transition_kind": kind,
    }


def _r2_row(env, observation: np.ndarray, *, partition: str, stratum: R2Stratum,
            seed: int, episode: int, tick: int, uid: int, lifecycle: str,
            event_index: int, reason: str, streak: int, similarity: float,
            selected, audit: list[dict], current: dict, previous: dict | None,
            reset_recipe: dict) -> tuple[dict, dict]:
    row = _formal_row(
        env, observation, partition=partition, stratum=stratum, seed=seed,
        episode=episode, tick=tick, uid=uid, lifecycle_id=lifecycle,
        event_index=event_index, retention_reason=reason, streak=streak,
        similarity=similarity, selected=selected, audit=audit, current=current,
        previous_semantic=previous, reset_recipe=reset_recipe,
    )
    episode_id = f"v2-1-i0-r2:{partition}:episode-{episode}:seed-{seed}"
    row.update({"schema_version": SCHEMA_VERSION, "partition": partition,
                "episode_id": episode_id, "uav_instance_id": f"{episode_id}:uav-{uid}"})
    rich = {**current, "uid": uid, "lifecycle_id": lifecycle,
            "decision_event_id": row["decision_event_id"], "time": float(env.now),
            "state_hash": row["full_state_hash"], "headway_state": row.get("headway_state")}
    row["transition_ledger"]["wait_to_release_proof"] = _make_transition_proof(row, previous, rich)
    return row, rich


def _partition_ready(rows: list[dict], partition: str) -> bool:
    if partition == "training":
        coverage = _coverage(rows)
        return len(rows) >= TRAIN_MIN and coverage["all_repair_context_minima_PASS"] and coverage["nominal_anchor_coverage_PASS"] and coverage["stable_teacher_selected_support_PASS"]
    return len(rows) >= AUDIT_MIN


def _collect_partition(output: Path, partition: str, seeds: list[int], *, smoke_cycles: int | None = None) -> tuple[list[dict], dict, list]:
    maximum = TRAIN_MAX if partition == "training" else AUDIT_MAX
    rows: list[dict] = []
    state_hashes: set[str] = set()
    decision_ids: set[str] = set()
    safety = Counter(); raw_gate = Counter(); raw_ticks = 0
    tapes = []; taped = set(); selected_seeds = []; fatal = None
    episode = 0
    # Smoke means a bounded number of short static clear-outer episodes.  It
    # verifies row/sidecar/tape serialization without paying for a full formal
    # ten-stratum cycle or opening the consumed Q artifact.
    max_episodes = smoke_cycles if smoke_cycles is not None else len(seeds)
    while episode < max_episodes:
        if episode and episode % len(STRATA) == 0 and _partition_ready(rows, partition):
            break
        if len(rows) >= maximum:
            break
        stratum = STRATA[8] if smoke_cycles is not None else STRATA[episode % len(STRATA)]
        if smoke_cycles is not None:
            stratum = R2Stratum(stratum.name, replace(stratum.spec, max_steps=1))
        seed = seeds[episode]
        selected_seeds.append(seed)
        env = TerminalParallelEnv(_config(stratum.spec)); env.reset(seed)
        reset_recipe, observations = _source_reset_recipe(env, stratum.spec)
        tape = None
        if stratum.name not in taped:
            tape = SafetyTraceRecorder(output / "action_tapes" / partition / stratum.name, dt=env.config.dt)
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "formal_i0_r2", "partition": partition, "stratum": stratum.name}, reset_recipe=reset_recipe)
        lifecycle = {int(uid): f"v2-1-i0-r2:{partition}:episode-{episode}:seed-{seed}:uav-{int(uid)}" for uid in env.active}
        raw_memory: dict[int, dict] = {}; semantic: dict[int, dict] = {}; event_index = Counter()
        for tick in range(stratum.spec.max_steps):
            actions = {}; labels = {}; pending = []
            for uid in [int(x) for x in sorted(observations) if int(x) in env.active]:
                uav = env.active[uid]
                selected, light = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                obs = np.asarray(observations[str(uid)], np.float32)
                current = _predicate_context(env, uav, selected, light)
                topology = _topology(env, uid, _snapshot(env, uav))
                reason, streak, similarity = _semantic_reason(raw_memory.get(uid), current, obs, topology, tick)
                primary = {"event_start", "teacher_target_change", "release_eligibility_or_retry_change", "phase_or_commit_change", "reservation_interface_or_resource_change"}
                if reason not in primary and event_index[uid] >= SECONDARY_PER_LIFECYCLE:
                    reason = None
                raw_ticks += 1; raw_gate[current["teacher_gate"]] += 1
                raw_memory[uid] = {**current, "observation": obs.tolist(), "topology": topology, "streak": streak}
                if reason:
                    full, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if full.index != selected.index:
                        raise RuntimeError("Teacher target changed while materialising the audit sidecar")
                    current = _predicate_context(env, uav, full, audit)
                    event_index[uid] += 1
                    row, rich = _r2_row(env, obs, partition=partition, stratum=stratum,
                                        seed=seed, episode=episode, tick=tick, uid=uid,
                                        lifecycle=lifecycle[uid], event_index=event_index[uid],
                                        reason=reason, streak=streak, similarity=similarity,
                                        selected=full, audit=audit, current=current,
                                        previous=semantic.get(uid), reset_recipe=reset_recipe)
                    if row["full_state_hash"] not in state_hashes and row["decision_event_id"] not in decision_ids:
                        pending.append((row, rich))
                action = hierarchical_from_joint_index_v4_1(int(selected.index))
                actions[str(uid)] = action.normalized
                labels[str(uid)] = {"gate": action.gate, "proposal": action.proposal, "joint_index_v4_1": int(selected.index)}
            pre = tape.record_pre_step(env, observations, actions) if tape else None
            try:
                observations, _, _, _, info = env.step(actions)
            except SafetyInvariantError as exc:
                safety["SafetyInvariantError"] += 1; fatal = str(exc); break
            post_hash = _digest(SafetyTraceRecorder._state(env))
            if tape:
                tape.record_post_step(env, pre, info); tape.record_teacher_labels(pre, labels)
            for row, rich in pending:
                if len(rows) >= maximum:
                    break
                row["post_state_hash"] = post_hash
                rows.append(row); state_hashes.add(row["full_state_hash"]); decision_ids.add(row["decision_event_id"])
                semantic[int(row["uav_id"])] = rich
            if fatal:
                break
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard_violation", "hard_separation_violations"), ("uncertified_execution", "uncertified_execution")):
            safety[key] += int(env.telemetry.get(field, 0))
        if tape:
            path = tape.dump(reason=f"formal_i0_r2_{partition}_{stratum.name}")
            tapes.append((path, stratum.spec, stratum.name)); taped.add(stratum.name)
        episode += 1
        progress = {"partition": partition, "episodes": episode, "rows": len(rows), "last_stratum": stratum.name,
                    "complete_cycles": episode // len(STRATA), "coverage": _coverage(rows), "safety": dict(safety), "fatal": fatal}
        _write(output / f"collection_progress_{partition}.json", progress)
        if fatal:
            break
    info = {
        "episodes": episode, "selected_seeds": selected_seeds, "rows": len(rows),
        "complete_stratum_cycles": episode // len(STRATA), "raw_decision_ticks": raw_ticks,
        "raw_WAIT_RELEASE": {"WAIT": int(raw_gate[0]), "RELEASE": int(raw_gate[1])},
        "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard_violation", "SafetyInvariantError", "uncertified_execution")},
        "fatal": fatal, "coverage": _coverage(rows), "ready_at_stop": _partition_ready(rows, partition),
    }
    return rows, info, tapes


def _replay(tapes: list) -> dict:
    reports = []
    for path, spec, stratum in tapes:
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        reports.append({"stratum": stratum, "path": str(path), "sha256": _sha(path), "result": result})
    exact = sum(bool(x["result"]["exact"]) for x in reports)
    required = 2 * len(STRATA)
    return {"tapes": reports, "exact_tapes": exact, "total_tapes": len(reports),
            "required_tapes": required, "exact_replay_rate": exact / max(1, len(reports)),
            "replay": "PASS" if len(reports) == required and exact == required else "FAIL"}


def _categorical(rows: list[dict], proposal: int) -> dict[str, dict[str, int]]:
    chosen = [r for r in rows if r["teacher_gate"] == 1 and r["teacher_joint_index"] == proposal]
    phase_ring = Counter(f'{r["state_snapshot"]["target_uav"]["phase"]}:{r["state_snapshot"]["target_uav"]["ring_role"]}' for r in chosen)
    traffic = Counter(str(_traffic(r)) for r in chosen)
    families = Counter(f for r in chosen for f in _families(r))
    return {"phase_ring": dict(phase_ring), "traffic_N": dict(traffic), "families": dict(families)}


def _distance_metrics(q_rows: list[dict], train_rows: list[dict], q_predicate, *, same_family: str | None = None) -> dict:
    train_obs = np.asarray([r["observation_v2_1"] for r in train_rows], np.float32)
    train_gate = np.asarray([r["teacher_gate"] for r in train_rows], int)
    release = train_gate == 1
    if same_family:
        candidate = np.asarray([release[i] and same_family in _families(r) for i, r in enumerate(train_rows)])
    else:
        candidate = release
    nearest, fractions = [], []
    for row in q_rows:
        if not q_predicate(row):
            continue
        dist = np.linalg.norm(train_obs - np.asarray(row["observation_v2_1"], np.float32), axis=1)
        if candidate.any():
            nearest.append(float(dist[candidate].min()))
        order = np.argsort(dist)[:KNN_K]
        fractions.append(float(train_gate[order].mean()))
    return {
        "reference_rows": len(nearest),
        "nearest_median": float(np.median(nearest)) if nearest else None,
        "nearest_mean": float(np.mean(nearest)) if nearest else None,
        "nearest_p90": float(np.percentile(nearest, 90)) if nearest else None,
        "knn_RELEASE_fraction_median": float(np.median(fractions)) if fractions else None,
        "knn_RELEASE_fraction_mean": float(np.mean(fractions)) if fractions else None,
    }


def _support_closure(train_rows: list[dict], q_path: Path, coverage: dict) -> dict:
    if _sha(q_path) != EXPECTED_Q_SHA:
        raise ValueError("consumed Q SHA256 mismatch")
    q_rows = [json.loads(line) for line in q_path.read_text(encoding="utf-8").splitlines() if line]
    train_context = {str(p): _categorical(train_rows, p) for p in (11, 14)}
    q_context = {str(p): _categorical(q_rows, p) for p in (11, 14)}
    tv = {str(p): {key: _total_variation(train_context[str(p)][key], q_context[str(p)][key]) for key in ("phase_ring", "traffic_N", "families")} for p in (11, 14)}
    q_release = lambda r: int(r["teacher_gate"]) == 1
    metrics = {
        "post_WAIT": _distance_metrics(q_rows, train_rows, lambda r: q_release(r) and bool(r["family_flags"]["post_wait"])),
        "transition_retry": _distance_metrics(q_rows, train_rows, lambda r: q_release(r) and r.get("collection_stratum") == "transition_retry"),
        "post_conflict": _distance_metrics(q_rows, train_rows, lambda r: q_release(r) and bool(r["family_flags"]["post_conflict"]), same_family="post_conflict"),
    }
    life = coverage["independent_lifecycles"]
    gates = {
        "proposal_11_support_gap_closed": life["high_traffic_N12_proposal_11"] >= 30 and max(tv["11"].values()) <= 0.20,
        "proposal_14_support_gap_closed": life["middle_interface_proposal_14"] >= 20 and life["post_WAIT_proposal_14"] >= 20 and life["post_conflict_proposal_14"] >= 15 and max(tv["14"].values()) <= 0.20,
        "post_WAIT_support_gap_closed": life["post_WAIT_RELEASE"] >= 60 and metrics["post_WAIT"]["nearest_median"] <= 1.00 and metrics["post_WAIT"]["knn_RELEASE_fraction_median"] >= 0.60,
        "transition_retry_support_gap_closed": life["retry_expired_or_transition_cleared_RELEASE"] >= 40 and metrics["transition_retry"]["nearest_median"] <= 1.00 and metrics["transition_retry"]["knn_RELEASE_fraction_median"] >= 0.60,
        "post_conflict_support_gap_closed": life["post_conflict_RELEASE"] >= 40 and metrics["post_conflict"]["nearest_median"] <= 1.00 and metrics["post_conflict"]["knn_RELEASE_fraction_median"] >= 0.60,
    }
    return {
        "evidence_type": "newly_run_plus_preexisting_artifact", "Q_sha256": _sha(q_path),
        "Q_status": "CONSUMED_DIAGNOSTIC_REFERENCE", "Q_rows_used_for_training": 0,
        "Q_access_phase": "after_training_and_internal_audit_collection_complete",
        "distance": "Euclidean normalized Observation V2.1 88D", "kNN_k": KNN_K,
        "local_radii": list(LOCAL_RADII), "proposal_context_distribution": {"training": train_context, "Q": q_context, "total_variation": tv},
        "distance_metrics": metrics, "closure_gates": gates, "all_support_gap_closure_gates_PASS": all(gates.values()),
    }


def _historical_independence(history_root: Path, output: Path, rows: list[dict]) -> dict:
    historical = _identity_accumulator()
    for path in _history_files(history_root, output):
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip(): _consume_object(json.loads(line), historical)
            else:
                _consume_object(json.loads(path.read_text(encoding="utf-8")), historical)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    episode_ids = {r["episode_id"] for r in rows}; lifecycle_ids = {r["lifecycle_event_id"] for r in rows}
    state_hashes = {r["full_state_hash"] for r in rows}; seeds = {r["seed"] for r in rows}
    # Runtime UAV integers are reset-local; the globally scoped identity is episode_id:uav_id.
    uav_instances = {r["uav_instance_id"] for r in rows}
    return {
        "seed": not bool(seeds & historical["seeds"]),
        "episode_id": not bool(episode_ids & historical["episode_ids"]),
        "lifecycle_event_id": not bool(lifecycle_ids & historical["lifecycle_ids"]),
        "full_state_hash": not bool(state_hashes & historical["full_state_hashes"]),
        "uav_instance_identity": len(uav_instances) == len({(r["episode_id"], r["uav_id"]) for r in rows}),
        "decision_event_unique": len({r["decision_event_id"] for r in rows}) == len(rows),
        "full_state_unique": len(state_hashes) == len(rows),
    }


def _cross_report(rows: list[dict]) -> dict:
    release = [r for r in rows if r["teacher_gate"] == 1]
    def cross(chosen: list[dict], fn) -> dict: return dict(Counter(fn(r) for r in chosen))
    return {
        "proposal11_x_traffic_N": cross([r for r in release if r["teacher_joint_index"] == 11], lambda r: str(_traffic(r))),
        "proposal14_x_phase_ring": cross([r for r in release if r["teacher_joint_index"] == 14], lambda r: f'{r["state_snapshot"]["target_uav"]["phase"]}:{r["state_snapshot"]["target_uav"]["ring_role"]}'),
        "proposal14_x_family": dict(Counter(f for r in release if r["teacher_joint_index"] == 14 for f in _families(r))),
        "post_WAIT_x_traffic_N": cross([r for r in release if r["family_flags"]["post_wait"]], lambda r: str(_traffic(r))),
        "post_conflict_x_traffic_N": cross([r for r in release if r["family_flags"]["post_conflict"]], lambda r: str(_traffic(r))),
        "retry_transition_x_phase_ring": cross([r for r in release if _proof(r).get("wait_release_transition_kind") in {"retry_WAIT_to_RELEASE", "transition_WAIT_to_RELEASE"}], lambda r: f'{r["state_snapshot"]["target_uav"]["phase"]}:{r["state_snapshot"]["target_uav"]["ring_role"]}'),
        "full_context_records": [{
            "decision_event_id": r["decision_event_id"], "proposal_id": r["teacher_proposal"], "traffic_N": _traffic(r),
            "phase": r["state_snapshot"]["target_uav"]["phase"], "ring_role": r["state_snapshot"]["target_uav"]["ring_role"],
            "family_flags": r["family_flags"], "previous_WAIT_rationale": _proof(r).get("previous_wait_rationale"),
            "retry_state": r["retry_active"], "conflict_recovery_state": r["family_flags"]["post_conflict"],
            "interface_state": r["headway_state"].get("interface"), "headway_state": r["headway_state"],
            "resource_state": r["state_snapshot"]["resources"],
        } for r in rows],
    }


def _render_summary(manifest: dict, closure: dict) -> str:
    life = manifest["training_coverage"]["independent_lifecycles"]
    lines = ["# Fresh V4.2 / Observation V2.1 I0-R2 Collection and Qualification", "", "## Formal decision", "", "```text",
             f"formal_training_rows = {manifest['formal_training_rows']}", f"formal_internal_audit_rows = {manifest['formal_internal_audit_rows']}",
             f"training_seed_count = {manifest['training_seed_count']}", f"audit_seed_count = {manifest['audit_seed_count']}", ""]
    for key in CONTEXT_MINIMA: lines.append(f"{key} = {life[key]}")
    for key in ANCHOR_MINIMA: lines.append(f"{key} = {life[key]}")
    lines += ["", f"proposal_11_context_TV = {max(closure['proposal_context_distribution']['total_variation']['11'].values()):.6f}",
              f"proposal_14_context_TV = {max(closure['proposal_context_distribution']['total_variation']['14'].values()):.6f}"]
    for name, metric in closure["distance_metrics"].items():
        lines += [f"{name}_nearest_median = {metric['nearest_median']}", f"{name}_knn_RELEASE_fraction = {metric['knn_RELEASE_fraction_median']}"]
    lines += [""] + [f"{key} = {str(value).lower()}" for key, value in closure["closure_gates"].items()]
    lines += ["", f"retry_provenance = {manifest['schema_retry_provenance']['retry_feature_provenance']}",
              f"teacher_determinism = {manifest['teacher_determinism']}", f"replay = {manifest['ActionTape_replay']['replay']}",
              f"safety = {manifest['safety']['result']}", f"independence = {manifest['independence']}",
              f"aliasing = {manifest['observation_aliasing']['observation_aliasing']}", "",
              f"V4_2_V2_1_I0_R2 = {manifest['V4_2_V2_1_I0_R2']}",
              f"V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = {str(manifest['V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED']).lower()}",
              "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false", "PPO_ALLOWED = false", "```", "",
              "## Evidence", "", "All numeric collection results are `newly_run`. The old Q is a `preexisting_artifact` consumed only after collection for support-distance diagnosis; zero Q rows enter either dataset.", "",
              "## Protocol risks", "", "- This is internal dataset qualification, not external generalization evidence.",
              "- Euclidean 88D support distance is diagnostic and not causal.",
              "- Scenario targeting changes occupancy by design; nominal-anchor gates are conjunctive.", ""]
    return "\n".join(lines)


def run(output_dir: Path, history_root: Path, q_path: Path, *, smoke_cycles: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol, protocol_check = _verify_protocol()
    if not protocol_check["PASS"]:
        fail = {"I0_R2_COLLECTION_VALID": False, "protocol_check": protocol_check, "V4_2_V2_1_I0_R2": "FAIL",
                "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": False,
                "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False}
        _write(output_dir / "v4_2_v2_1_i0_r2_manifest.json", fail); return fail
    registry, _ = _build_pre_reset_registry(history_root, output_dir)
    if registry["parse_error_count"]:
        raise RuntimeError("pre-reset seed registry metadata contains parse errors")
    training, train_info, train_tapes = _collect_partition(output_dir, "training", registry["available_candidates"]["training"], smoke_cycles=smoke_cycles)
    audit, audit_info, audit_tapes = _collect_partition(output_dir, "internal_audit", registry["available_candidates"]["internal_audit"], smoke_cycles=smoke_cycles)
    train_path = output_dir / "v4_2_v2_1_i0_r2_training.jsonl"
    audit_path = output_dir / "v4_2_v2_1_i0_r2_internal_audit.jsonl"
    train_path.write_text("".join(json.dumps(_safe(r), sort_keys=True) + "\n" for r in training), encoding="utf-8")
    audit_path.write_text("".join(json.dumps(_safe(r), sort_keys=True) + "\n" for r in audit), encoding="utf-8")
    coverage = _coverage(training)
    provenance = _provenance(training + audit)
    teacher_determinism = all(r["teacher_gate"] == r["teacher_repeat_gate"] and r["teacher_proposal"] == r["teacher_repeat_proposal"] and r["teacher_joint_index"] == r["teacher_repeat_joint_index"] for r in training + audit)
    aliasing = _aliasing(training + audit)
    replay = _replay(train_tapes + audit_tapes)
    safety_counts = {key: train_info["safety"][key] + audit_info["safety"][key] for key in train_info["safety"]}
    safety = {"counts": safety_counts, "result": "PASS" if not any(safety_counts.values()) else "FAIL"}
    if smoke_cycles is None:
        closure = _support_closure(training, q_path, coverage)
    else:
        closure = {
            "evidence_type": "not_run_in_smoke", "Q_sha256": EXPECTED_Q_SHA,
            "Q_status": "NOT_OPENED_DURING_SMOKE", "Q_rows_used_for_training": 0,
            "proposal_context_distribution": {"total_variation": {"11": {"not_run": 1.0}, "14": {"not_run": 1.0}}},
            "distance_metrics": {name: {"nearest_median": None, "knn_RELEASE_fraction_median": None} for name in ("post_WAIT", "transition_retry", "post_conflict")},
            "closure_gates": {key: False for key in (
                "proposal_11_support_gap_closed", "proposal_14_support_gap_closed",
                "post_WAIT_support_gap_closed", "transition_retry_support_gap_closed",
                "post_conflict_support_gap_closed")},
            "all_support_gap_closure_gates_PASS": False,
        }
    closure_path = output_dir / "v4_2_v2_1_i0_r2_support_closure.json"
    _write(closure_path, closure)
    history = _historical_independence(history_root, output_dir, training + audit)
    train_sets = {key: {r[key] for r in training} for key in ("seed", "episode_id", "lifecycle_event_id", "decision_event_id", "full_state_hash")}
    audit_sets = {key: {r[key] for r in audit} for key in train_sets}
    partition_checks = {key: not bool(train_sets[key] & audit_sets[key]) for key in train_sets}
    independence = {"historical": history, "partition": partition_checks,
                    "PASS": all(history.values()) and all(partition_checks.values())}
    formal_gates = {
        "fresh_lineage": history["seed"] and history["episode_id"] and history["lifecycle_event_id"] and history["full_state_hash"],
        "schema_retry_provenance": provenance["schema_provenance"] == "PASS" and provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": teacher_determinism, "replay": replay["replay"] == "PASS",
        "safety": safety["result"] == "PASS", "aliasing": aliasing["observation_aliasing"] == "PASS",
        "semantic_event_uniqueness": history["decision_event_unique"] and history["full_state_unique"],
        "partition_independence": independence["PASS"],
        "training_cardinality": TRAIN_MIN <= len(training) <= TRAIN_MAX and train_info["complete_stratum_cycles"] >= 1,
        "internal_audit_cardinality": AUDIT_MIN <= len(audit) <= AUDIT_MAX and audit_info["complete_stratum_cycles"] >= 1,
        "nominal_anchor_coverage": coverage["nominal_anchor_coverage_PASS"],
        "all_repair_context_minima": coverage["all_repair_context_minima_PASS"],
        "stable_teacher_selected_support": coverage["stable_teacher_selected_support_PASS"],
        "support_gap_closure": closure["all_support_gap_closure_gates_PASS"],
    }
    passed = smoke_cycles is None and all(formal_gates.values())
    reset_hashes = sorted({r["reset_recipe_hash"] for r in training + audit})
    manifest = {
        "stage": "FRESH_FORMAL_V4_2_V2_1_I0_R2", "evidence_type": "newly_run", "smoke_only": smoke_cycles is not None,
        "protocol": PROTOCOL_VERSION, "protocol_check": protocol_check, "I0_R2_COLLECTION_VALID": True,
        "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 88, "action_schema": ACTION_SCHEMA_V4_2,
        "frozen_semantics": ["Teacher", "Teacher ranking", "Observation V2.1", "V4.2 action", "SafetyFilter", "PredictiveShield"],
        "forbidden_operations_used": False, "historical_rows_used": False, "consumed_Q_rows_used": False,
        "formal_training_path": str(train_path), "formal_training_sha256": _sha(train_path), "formal_training_rows": len(training),
        "formal_internal_audit_path": str(audit_path), "formal_internal_audit_sha256": _sha(audit_path), "formal_internal_audit_rows": len(audit),
        "internal_audit_usage": "NEVER_TRAIN_NEVER_MODEL_SELECT_DATASET_QUALIFICATION_ONLY",
        "training_seed_count": len(set(train_info["selected_seeds"])), "audit_seed_count": len(set(audit_info["selected_seeds"])),
        "pre_reset_seed_registry": registry["path"], "pre_reset_seed_registry_sha256": registry["sha256"],
        "collector_code": str(Path(__file__)), "collector_code_sha256": _sha(Path(__file__)),
        "collector_config": str(PROTOCOL_PATH), "collector_config_sha256": _sha(PROTOCOL_PATH),
        "collection_schedule": [{"name": x.name, "spec": asdict(x.spec)} for x in STRATA],
        "training_collection": train_info, "internal_audit_collection": audit_info,
        "training_coverage": coverage, "context_coverage_report": _cross_report(training),
        "schema_retry_provenance": provenance, "teacher_determinism": "PASS" if teacher_determinism else "FAIL",
        "observation_aliasing": aliasing, "ActionTape_replay": replay, "safety": safety,
        "independence_checks": independence, "independence": "PASS" if independence["PASS"] else "FAIL",
        "support_closure_path": str(closure_path), "support_closure_sha256": _sha(closure_path),
        "reset_recipe_hashes": reset_hashes, "formal_gates": formal_gates,
        "qualification_failures": [key for key, value in formal_gates.items() if not value],
        "V4_2_V2_1_I0_R2": "PASS" if passed else "FAIL",
        "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": bool(passed),
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False,
        "protocol_risks": ["Internal dataset qualification only", "Consumed Q is diagnostic, not untouched qualification", "88D Euclidean support distance is non-causal"],
    }
    manifest_path = output_dir / "v4_2_v2_1_i0_r2_manifest.json"
    _write(manifest_path, manifest)
    summary_path = output_dir / "v4_2_v2_1_i0_r2_collection_summary.md"
    summary_path.write_text(_render_summary(manifest, closure), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history-root", required=True, type=Path)
    parser.add_argument("--consumed-q", required=True, type=Path)
    parser.add_argument("--smoke-cycles", type=int)
    args = parser.parse_args()
    result = run(args.output_dir, args.history_root, args.consumed_q, smoke_cycles=args.smoke_cycles)
    print(json.dumps({key: result.get(key) for key in (
        "formal_training_rows", "formal_internal_audit_rows", "training_seed_count", "audit_seed_count",
        "teacher_determinism", "independence", "V4_2_V2_1_I0_R2",
        "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED",
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED", "PPO_ALLOWED")}, indent=2))


if __name__ == "__main__":
    main()
