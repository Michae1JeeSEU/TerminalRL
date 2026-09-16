"""Fresh standalone V4.2/V2.1 I0-R3 collection and qualification.

The fixed protocol/config hashes and the historical seed registry are checked
before the first simulator reset.  Consumed Q and failure-consensus artifacts
are opened only after both fresh collection partitions have finished and are
used exclusively for read-only support diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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
from .run_v4_2_v2_1_formal_i0_r2 import _historical_independence
from .run_v4_2_v2_1_i0_r2_support_closure_attribution import (
    _coverage_summary, _families, _is_transition_retry, _prior_audit,
    _q_intrinsic, _records, _tv_decomposition,
)
from .run_v4_2_v2_1_i1_q_cohort import _consume_object, _identity_accumulator
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


PROTOCOL_VERSION = "V2_1_I0_R3_REPAIR_PROTOCOL_V2"
SCHEMA_VERSION = "v4_2_v2_1_formal_i0_r3_v1"
PROTOCOL_PATH = Path("terminal_marl/configs/low_v4_2/i0_r3_repair_protocol.yaml")
DESIGN_PATH = Path("docs/evidence/v4_2/v4_2_v2_1_i0_r3_repair_protocol.md")
REPRO_PATH = Path("docs/evidence/v4/v4_new_mainline_reproducibility_manifest.json")
EXPECTED_PROTOCOL_SHA = "9daf8a2b279a517fb763a06cd011ae088b4000fd31b2a144767bd7bbfa4deb20"
EXPECTED_DESIGN_SHA = "7a4d0a3801d29109d5fbff229544ae05e8ba7fe751774bbcf0e3f57d8abf5d19"
EXPECTED_Q_SHA = "8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5"
EXPECTED_CONSENSUS_SHA = "934d12a1b58589661846b5924d5d4e95f342e80b2eec4c67b0b9f6e6fcbc0a01"
EXPECTED_FAILED_COVERAGE_SHA = "6910e573e5c8691e78b99050315a7a9e1002f32734a86bd1fd17c6ae3ea2137a"
TRAIN_RANGE = (410001, 419999)
AUDIT_RANGE = (420001, 429999)
Q2_RESERVED = (430001, 439999)
TRAIN_TARGET = (3000, 4000)
TRAIN_CEILING = 4200
AUDIT_TARGET = (800, 1200)
AUDIT_CEILING = 1400
SECONDARY_PER_LIFECYCLE = 4
RADII = (0.05, 0.10, 0.25, 1.00)

ANCHOR_MINIMA = {
    "ordinary_WAIT": 250, "retry_active_WAIT": 100, "committed_WAIT": 100,
    "interface_frozen_WAIT": 80, "resource_WAIT": 30,
    "clear_outer_RELEASE": 100, "middle_interface_RELEASE": 100,
    "proposal_11_RELEASE": 60, "proposal_12_RELEASE": 80,
    "proposal_13_RELEASE": 160, "proposal_14_RELEASE": 60,
}


@dataclass(frozen=True)
class R3Stratum:
    name: str
    spec: SourceSpec


@dataclass(frozen=True)
class FixedCollectionSchedule:
    """Prospective fixed episode budget; historical R3 defaults stay frozen."""
    strata: tuple[R3Stratum, ...]
    cycles: int
    row_ceiling: int
    namespace: str

    def __post_init__(self):
        if not self.strata or self.cycles < 1 or self.row_ceiling < 1:
            raise ValueError("positive fixed collection budget required")


STRATA = (
    R3Stratum("proposal14_N5_outer", SourceSpec("i0_r3_p14_n5_outer", "outer_open_n5", 5, phases=("ring_outer",), max_steps=12)),
    R3Stratum("proposal14_N5_middle_interface", SourceSpec("i0_r3_p14_n5_middle", "middle_interface", 5, phases=("ring_middle",), max_steps=18)),
    R3Stratum("proposal14_N5_post_wait", SourceSpec("i0_r3_p14_n5_post_wait", "transition_wait_release", 5, dynamic=True, max_steps=42)),
    R3Stratum("proposal14_N5_post_conflict", SourceSpec("i0_r3_p14_n5_post_conflict", "conflict_recovery", 5, dynamic=True, max_steps=42)),
    R3Stratum("proposal14_N6_outer", SourceSpec("i0_r3_p14_n6_outer", "outer_open_n6", 6, phases=("ring_outer",), max_steps=12)),
    R3Stratum("proposal14_N6_middle_interface", SourceSpec("i0_r3_p14_n6_middle", "middle_interface", 6, phases=("ring_middle",), max_steps=18)),
    R3Stratum("proposal14_N6_post_wait", SourceSpec("i0_r3_p14_n6_post_wait", "transition_wait_release", 6, dynamic=True, max_steps=42)),
    R3Stratum("proposal14_N6_post_conflict", SourceSpec("i0_r3_p14_n6_post_conflict", "conflict_recovery", 6, dynamic=True, max_steps=42)),
    R3Stratum("proposal14_N12_mixed", SourceSpec("i0_r3_p14_n12_mixed", "nominal_mixed", 12, max_steps=12)),
    R3Stratum("residual_p11_N12_outer_o2m", SourceSpec("i0_r3_p11_n12_outer", "clear_outer", 12, phases=("ring_outer",), max_steps=18)),
    R3Stratum("residual_p13_N12_middle_m2i", SourceSpec("i0_r3_p13_n12_middle", "middle_interface", 12, phases=("ring_middle",), max_steps=18)),
    R3Stratum("residual_middle_N12_m2i", SourceSpec("i0_r3_middle_n12", "middle_interface", 12, phases=("ring_middle",), max_steps=18)),
    R3Stratum("residual_inner_N12_i2f_transition", SourceSpec("i0_r3_inner_transition_n12", "transition_wait_release", 12, dynamic=True, phases=("ring_inner",), max_steps=42)),
    R3Stratum("residual_post_wait_N8_outer", SourceSpec("i0_r3_post_wait_n8_outer", "transition_wait_release", 8, dynamic=True, phases=("ring_outer",), max_steps=42)),
    R3Stratum("residual_post_wait_N8_middle", SourceSpec("i0_r3_post_wait_n8_middle", "transition_wait_release", 8, dynamic=True, phases=("ring_middle",), max_steps=42)),
    R3Stratum("nominal_mixed", SourceSpec("i0_r3_nominal", "nominal_mixed", 12, max_steps=12)),
    R3Stratum("nominal_clear_outer", SourceSpec("i0_r3_clear_outer", "clear_outer", 3, phases=("ring_outer",), max_steps=12)),
    R3Stratum("nominal_middle_interface", SourceSpec("i0_r3_middle", "middle_interface", 6, phases=("ring_middle",), max_steps=18)),
    R3Stratum("nominal_retry_wait", SourceSpec("i0_r3_retry", "transition_wait_release", 12, dynamic=True, max_steps=42)),
    R3Stratum("nominal_committed_interface_resource", SourceSpec("i0_r3_resource", "resource_pressure", 12, final_capacity=0, dynamic=True, max_steps=24)),
)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _verify_protocol() -> tuple[dict, dict]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    repro = json.loads(REPRO_PATH.read_text(encoding="utf-8"))
    registered = repro["v4_2_v2_1_i0_r3_repair_protocol"]
    psha, dsha = _sha(PROTOCOL_PATH), _sha(DESIGN_PATH)
    checks = {
        "version": protocol.get("protocol") == PROTOCOL_VERSION,
        "protocol_sha_expected": psha == EXPECTED_PROTOCOL_SHA,
        "design_sha_expected": dsha == EXPECTED_DESIGN_SHA,
        "protocol_sha_manifest": registered.get("protocol_sha256") == psha,
        "design_sha_manifest": registered.get("design_sha256") == dsha,
        "authorized": protocol["authorization"]["I0_R3_COLLECTION_ALLOWED"] is True,
        "training_range": protocol["lineage"]["training_seed_range"] == list(TRAIN_RANGE),
        "audit_range": protocol["lineage"]["internal_audit_seed_range"] == list(AUDIT_RANGE),
        "q2_reserved": protocol["lineage"]["future_q2_seed_range_reserved"] == list(Q2_RESERVED),
        "strata": protocol["collection"]["strata"] == [x.name for x in STRATA],
        "anchors": protocol["nominal_anchor_minima_independent_lifecycles"] == ANCHOR_MINIMA,
    }
    return protocol, {"checks": checks, "PASS": all(checks.values()), "protocol_sha256": psha, "design_sha256": dsha}


def _set_digest(values: Iterable[Any]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _build_pre_reset_registry(history_root: Path, output: Path) -> tuple[dict, dict[str, Any]]:
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
        "stage": "V2_1_I0_R3_PRE_RESET_SEED_REGISTRY", "built_before_any_reset": True,
        "consumed_Q_rows_opened": False, "history_root": str(history_root),
        "history_files": sources, "history_file_count": len(sources),
        "parse_error_count": sum(x["error"] is not None for x in sources),
        "historical_seed_count": len(identities["seeds"]), "historical_seed_digest": _set_digest(identities["seeds"]),
        "candidate_ranges": {k: list(v) for k, v in ranges.items()},
        "candidate_collisions_rejected_before_reset": collisions, "available_candidates": available,
    }
    path = output / "v4_2_v2_1_i0_r3_pre_reset_seed_registry.json"
    _write(path, registry)
    registry.update({"path": str(path), "sha256": _sha(path)})
    return registry, identities


def _traffic(row: dict) -> int:
    return len(row["state_snapshot"]["active_uavs"])


def _proof(row: dict) -> dict:
    return row.get("transition_ledger", {}).get("wait_to_release_proof", {})


def _is_release(row: dict) -> bool:
    return int(row["teacher_gate"]) == 1


def _interface(row: dict) -> str | None:
    return row.get("headway_state", {}).get("interface")


def _phase(row: dict) -> str:
    return str(row["state_snapshot"]["target_uav"]["phase"])


def _has_family(row: dict, *names: str) -> bool:
    flags = row.get("family_flags", {})
    return any(bool(flags.get(name)) for name in names)


def _cell_predicates() -> dict[str, Any]:
    return {
        "C01": lambda r: _is_release(r) and r["teacher_joint_index"] == 11 and _traffic(r) == 12 and _phase(r) == "ring_outer" and _interface(r) == "o2m" and _has_family(r, "clear_outer", "post_wait", "post_conflict"),
        "C02": lambda r: _is_release(r) and r["teacher_joint_index"] == 14 and _traffic(r) == 12 and _phase(r) == "ring_middle" and _interface(r) == "m2i" and _has_family(r, "middle_interface", "post_wait", "post_conflict"),
        "C03": lambda r: _is_release(r) and _traffic(r) == 12 and _phase(r) == "ring_inner" and _interface(r) == "i2f" and (_has_family(r, "post_wait", "post_conflict") or _is_transition_retry(r)),
        "C04": lambda r: _is_release(r) and _traffic(r) == 12 and _phase(r) == "ring_middle" and _interface(r) == "m2i" and _has_family(r, "middle_interface"),
        "C05": lambda r: _is_release(r) and r["teacher_joint_index"] == 14 and _traffic(r) == 5 and _phase(r) == "ring_outer" and _interface(r) == "o2m" and _has_family(r, "clear_outer"),
        "C06": lambda r: _is_release(r) and r["teacher_joint_index"] == 14 and _traffic(r) == 6 and _phase(r) == "ring_outer" and _interface(r) == "o2m" and _has_family(r, "clear_outer"),
        "C07": lambda r: _is_release(r) and _traffic(r) == 8 and _phase(r) == "ring_outer" and _interface(r) == "o2m" and _has_family(r, "clear_outer", "post_wait"),
        "C08": lambda r: _is_release(r) and _traffic(r) == 8 and _phase(r) == "ring_middle" and _interface(r) == "m2i" and _has_family(r, "middle_interface", "post_wait"),
        "C09": lambda r: _is_release(r) and r["teacher_joint_index"] == 13 and _traffic(r) == 12 and _phase(r) == "ring_middle" and _interface(r) == "m2i" and _has_family(r, "middle_interface", "post_wait", "post_conflict"),
    }


def _lifecycle_count(rows: list[dict], pred) -> int:
    return len({r["lifecycle_event_id"] for r in rows if pred(r)})


def _episode_count(rows: list[dict], pred) -> int:
    return len({r["episode_id"] for r in rows if pred(r)})


def _coverage(rows: list[dict], protocol: dict) -> dict:
    release = lambda r: _is_release(r)
    wait = lambda r: not _is_release(r)
    anchors = {
        "ordinary_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "other_wait",
        "retry_active_WAIT": lambda r: wait(r) and bool(r["retry_active"]),
        "committed_WAIT": lambda r: wait(r) and r["teacher_rationale"] == "committed",
        "interface_frozen_WAIT": lambda r: wait(r) and r["teacher_rationale"] in {"geometry_conflict", "interface_frozen"},
        "resource_WAIT": lambda r: wait(r) and r["teacher_rationale"] in {"resource", "final_capacity", "lz_unavailable"},
        "clear_outer_RELEASE": lambda r: release(r) and _has_family(r, "clear_outer"),
        "middle_interface_RELEASE": lambda r: release(r) and _has_family(r, "middle_interface"),
        "proposal_11_RELEASE": lambda r: release(r) and r["teacher_joint_index"] == 11,
        "proposal_12_RELEASE": lambda r: release(r) and r["teacher_joint_index"] == 12,
        "proposal_13_RELEASE": lambda r: release(r) and r["teacher_joint_index"] == 13,
        "proposal_14_RELEASE": lambda r: release(r) and r["teacher_joint_index"] == 14,
    }
    anchor_counts = {k: _lifecycle_count(rows, p) for k, p in anchors.items()}
    cells = {}
    specs = protocol["residual_error_context_cells"]
    for item, (short, pred) in zip(specs, _cell_predicates().items()):
        life, eps = _lifecycle_count(rows, pred), _episode_count(rows, pred)
        cells[short] = {"id": item["id"], "lifecycles": life, "episodes": eps,
                        "minimum_lifecycles": item["minimum_release_lifecycles"], "minimum_episodes": item["minimum_episodes"],
                        "PASS": life >= item["minimum_release_lifecycles"] and eps >= item["minimum_episodes"]}
    traffic = {}
    for key, n in (("N5", 5), ("N6", 6), ("N12", 12)):
        pred = lambda r, n=n: release(r) and r["teacher_joint_index"] == 14 and _traffic(r) == n
        life, eps = _lifecycle_count(rows, pred), _episode_count(rows, pred)
        minimum = protocol["proposal14_traffic_minima"][key]
        traffic[key] = {"lifecycles": life, "episodes": eps,
                        "minimum_lifecycles": minimum["independent_teacher_release_lifecycles"], "minimum_episodes": minimum["independent_episodes"],
                        "PASS": life >= minimum["independent_teacher_release_lifecycles"] and eps >= minimum["independent_episodes"]}
    support = sorted({int(r["teacher_joint_index"]) for r in rows if release(r)})
    return {"nominal_anchor_counts": anchor_counts,
            "nominal_anchor_PASS": all(anchor_counts[k] >= v for k, v in ANCHOR_MINIMA.items()),
            "residual_context_cells": cells, "all_residual_context_cells_PASS": all(x["PASS"] for x in cells.values()),
            "proposal14_traffic": traffic, "proposal14_traffic_PASS": all(x["PASS"] for x in traffic.values()),
            "teacher_selected_support": support, "stable_support_PASS": set((11, 12, 13, 14)).issubset(support)}


def _make_transition_proof(row: dict, previous: dict | None, current: dict) -> dict:
    same = bool(previous and previous["uid"] == row["uav_id"] and previous["lifecycle_id"] == row["lifecycle_event_id"])
    transition = bool(same and previous["teacher_gate"] == 0 and row["teacher_gate"] == 1)
    kind = None
    if transition:
        if previous["retry_active"] and not current["retry_active"]: kind = "retry_WAIT_to_RELEASE"
        elif previous["interface_open"] != current["interface_open"]: kind = "transition_WAIT_to_RELEASE"
        elif previous["wait_rationale"] in {"geometry_conflict", "headway_conflict", "holder_tube_conflict", "candidate_conflict"}: kind = "geometry_conflict_WAIT_to_RELEASE"
        else: kind = "ordinary_WAIT_to_RELEASE"
    return {"previous_wait_gate": previous["teacher_gate"] if transition else None,
            "previous_wait_rationale": previous["wait_rationale"] if transition else None,
            "previous_wait_state_hash": previous.get("state_hash") if transition else None,
            "previous_wait_time": previous["time"] if transition else None,
            "current_release_state_hash": row["full_state_hash"] if transition else None,
            "elapsed_time": float(current["time"] - previous["time"]) if transition else None,
            "retry_active_to_inactive": bool(transition and previous["retry_active"] and not current["retry_active"]),
            "interface_state_transition": {"previous_open": previous["interface_open"], "current_open": current["interface_open"]} if transition else None,
            "conflict_state_transition": {"previous": previous["conflict_state"], "current": current["conflict_state"]} if transition else None,
            "same_uav_id": same, "same_lifecycle_event_id": same, "wait_release_transition_kind": kind}


def _r3_row(env, observation: np.ndarray, *, partition: str, stratum: R3Stratum,
            seed: int, episode: int, tick: int, uid: int, lifecycle: str,
            event_index: int, reason: str, streak: int, similarity: float,
            selected, audit: list[dict], current: dict, previous: dict | None,
            reset_recipe: dict, namespace: str = "v2-1-i0-r3") -> tuple[dict, dict]:
    row = _formal_row(env, observation, partition=partition, stratum=stratum, seed=seed,
                      episode=episode, tick=tick, uid=uid, lifecycle_id=lifecycle,
                      event_index=event_index, retention_reason=reason, streak=streak,
                      similarity=similarity, selected=selected, audit=audit, current=current,
                      previous_semantic=previous, reset_recipe=reset_recipe)
    episode_id = f"{namespace}:{partition}:episode-{episode}:seed-{seed}"
    row.update({"schema_version": SCHEMA_VERSION, "partition": partition,
                "episode_id": episode_id, "uav_instance_id": f"{episode_id}:uav-{uid}"})
    rich = {**current, "uid": uid, "lifecycle_id": lifecycle,
            "decision_event_id": row["decision_event_id"], "time": float(env.now),
            "state_hash": row["full_state_hash"], "headway_state": row.get("headway_state")}
    row["transition_ledger"]["wait_to_release_proof"] = _make_transition_proof(row, previous, rich)
    return row, rich


def _collect_partition(output: Path, partition: str, seeds: list[int], protocol: dict, *, smoke_cycles: int | None = None,
                       schedule: FixedCollectionSchedule | None = None) -> tuple[list[dict], dict, list]:
    minimum, ceiling = TRAIN_TARGET[0], TRAIN_CEILING
    if partition == "internal_audit": minimum, ceiling = AUDIT_TARGET[0], AUDIT_CEILING
    strata = STRATA if schedule is None else schedule.strata
    namespace = "v2-1-i0-r3" if schedule is None else schedule.namespace
    if schedule is not None:
        if smoke_cycles is not None:
            raise ValueError("use either legacy smoke or a fixed schedule")
        ceiling = schedule.row_ceiling
        if len(seeds) != schedule.cycles * len(strata):
            raise ValueError("fixed schedule requires exactly one seed per episode")
    rows: list[dict] = []; state_hashes: set[str] = set(); decision_ids: set[str] = set()
    safety = Counter(); raw_gate = Counter(); raw_ticks = 0
    tapes = []; taped = set(); selected_seeds = []; fatal = None; episode = 0
    max_episodes = smoke_cycles if smoke_cycles is not None else len(seeds)
    while episode < max_episodes:
        if schedule is None and smoke_cycles is None and episode and episode % len(strata) == 0:
            coverage = _coverage(rows, protocol)
            ready = len(rows) >= minimum and (partition == "internal_audit" or (coverage["nominal_anchor_PASS"] and coverage["all_residual_context_cells_PASS"] and coverage["proposal14_traffic_PASS"] and coverage["stable_support_PASS"]))
            if ready or len(rows) >= ceiling: break
        stratum = strata[0] if smoke_cycles is not None else strata[episode % len(strata)]
        if smoke_cycles is not None: stratum = R3Stratum(stratum.name, replace(stratum.spec, max_steps=1))
        seed = seeds[episode]; selected_seeds.append(seed)
        env = TerminalParallelEnv(_config(stratum.spec)); env.reset(seed)
        reset_recipe, observations = _source_reset_recipe(env, stratum.spec)
        tape = None
        if stratum.name not in taped:
            tape = SafetyTraceRecorder(output / "action_tapes" / partition / stratum.name, dt=env.config.dt)
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": namespace, "partition": partition, "stratum": stratum.name}, reset_recipe=reset_recipe)
        lifecycle = {int(uid): f"{namespace}:{partition}:episode-{episode}:seed-{seed}:uav-{int(uid)}" for uid in env.active}
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
                if reason not in primary and event_index[uid] >= SECONDARY_PER_LIFECYCLE: reason = None
                raw_ticks += 1; raw_gate[current["teacher_gate"]] += 1
                raw_memory[uid] = {**current, "observation": obs.tolist(), "topology": topology, "streak": streak}
                if reason:
                    full, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if full.index != selected.index: raise RuntimeError("Teacher target changed while materialising audit sidecar")
                    current = _predicate_context(env, uav, full, audit); event_index[uid] += 1
                    row, rich = _r3_row(env, obs, partition=partition, stratum=stratum, seed=seed, episode=episode,
                                        tick=tick, uid=uid, lifecycle=lifecycle[uid], event_index=event_index[uid],
                                        reason=reason, streak=streak, similarity=similarity, selected=full, audit=audit,
                                        current=current, previous=semantic.get(uid), reset_recipe=reset_recipe, namespace=namespace)
                    if schedule is not None:
                        row["schema_version"] = namespace + "_event_v1"
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
                row["post_state_hash"] = post_hash; rows.append(row)
                state_hashes.add(row["full_state_hash"]); decision_ids.add(row["decision_event_id"])
                semantic[int(row["uav_id"])] = rich
            if fatal: break
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard_violation", "hard_separation_violations"), ("uncertified_execution", "uncertified_execution")):
            safety[key] += int(env.telemetry.get(field, 0))
        if tape:
            path = tape.dump(reason=f"{namespace}_{partition}_{stratum.name}")
            tapes.append((path, stratum.spec, stratum.name)); taped.add(stratum.name)
        episode += 1
        progress = {"partition": partition, "episodes": episode, "rows": len(rows), "last_stratum": stratum.name,
                    "complete_cycles": episode // len(strata), "coverage": _coverage(rows, protocol),
                    "safety": dict(safety), "fatal": fatal, "consumed_Q_rows_opened": False}
        _write(output / f"collection_progress_{partition}.json", progress)
        if schedule is not None and len(rows) > ceiling:
            fatal = "fixed collection row ceiling exceeded after complete episode"
        if fatal: break
    info = {"episodes": episode, "selected_seeds": selected_seeds, "rows": len(rows),
            "complete_stratum_cycles": episode // len(strata), "raw_decision_ticks": raw_ticks,
            "raw_WAIT_RELEASE": {"WAIT": int(raw_gate[0]), "RELEASE": int(raw_gate[1])},
            "safety": {k: int(safety[k]) for k in ("collision", "NMAC", "hard_violation", "SafetyInvariantError", "uncertified_execution")},
            "fatal": fatal, "coverage": _coverage(rows, protocol), "ceiling_exceeded": len(rows) > ceiling}
    return rows, info, tapes


def _replay(tapes: list) -> dict:
    reports = []
    for path, spec, stratum in tapes:
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        reports.append({"stratum": stratum, "path": str(path), "sha256": _sha(path), "result": result})
    exact = sum(bool(x["result"]["exact"]) for x in reports)
    return {"tapes": reports, "exact_tapes": exact, "total_tapes": len(reports),
            "required_tapes": 2 * len(STRATA), "exact_replay_rate": exact / max(1, len(reports)),
            "replay": "PASS" if len(reports) == 2 * len(STRATA) and exact == len(reports) else "FAIL"}


def _load_q_and_consensus(q_path: Path, consensus_path: Path, failed_coverage_path: Path) -> tuple[list[dict], dict, dict]:
    for path, expected in ((q_path, EXPECTED_Q_SHA), (consensus_path, EXPECTED_CONSENSUS_SHA), (failed_coverage_path, EXPECTED_FAILED_COVERAGE_SHA)):
        if _sha(path) != expected: raise ValueError(f"frozen diagnostic SHA mismatch: {path}")
    q = [json.loads(line) for line in q_path.read_text(encoding="utf-8").splitlines() if line]
    return q, json.loads(consensus_path.read_text(encoding="utf-8")), json.loads(failed_coverage_path.read_text(encoding="utf-8"))


def _reference_ids(consensus: dict) -> tuple[list[int], list[int]]:
    fn, proposal = [], []
    for row in consensus["rows"]:
        if int(row["teacher_gate"]) != 1: continue
        predictions = list(row["checkpoint_predictions"].values())
        if all(int(x["gate_prediction"]) == 0 for x in predictions): fn.append(int(row["q_index"]))
        if all(int(x["proposal_prediction_joint"]) != int(row["teacher_joint_index"]) for x in predictions): proposal.append(int(row["q_index"]))
    return fn, proposal


def _radius_coverage(reference: list[dict], training: list[dict], *, candidate_mode: str) -> dict:
    train_obs = np.asarray([r["observation_v2_1"] for r in training], np.float32)
    distances = []
    for q in reference:
        mask = np.asarray([_is_release(r) for r in training])
        if candidate_mode == "same_proposal": mask &= np.asarray([r["teacher_joint_index"] == q["teacher_joint_index"] for r in training])
        elif candidate_mode == "post_conflict": mask &= np.asarray([_has_family(r, "post_conflict") for r in training])
        dist = np.linalg.norm(train_obs - np.asarray(q["observation_v2_1"], np.float32), axis=1)
        distances.append(float(dist[mask].min()) if mask.any() else float("inf"))
    coverage = {f"coverage_{r:.2f}": float(np.mean(np.asarray(distances) <= r)) if distances else None for r in RADII}
    return {"reference_states": len(reference), "candidate_mode": candidate_mode,
            "nearest_distance": {"median": float(np.median(distances)) if distances else None,
                                 "p90": float(np.percentile(distances, 90)) if distances else None,
                                 "max": float(np.max(distances)) if distances else None}, **coverage,
            "PASS": bool(distances and coverage["coverage_0.25"] >= .75 and coverage["coverage_1.00"] >= .90)}


def _diagnostics(training: list[dict], q: list[dict], consensus: dict, failed: dict) -> tuple[dict, dict]:
    fn_ids, prop_ids = _reference_ids(consensus)
    if len(fn_ids) != 52 or len(prop_ids) != 30: raise RuntimeError("frozen shared-error reference count changed")
    q_by_id = {i: row for i, row in enumerate(q)}
    q_release = [r for r in q if _is_release(r)]
    groups = {
        "shared_FN": ([q_by_id[i] for i in fn_ids], "same_gate"),
        "shared_proposal_error": ([q_by_id[i] for i in prop_ids], "same_proposal"),
        "post_WAIT": ([r for r in q_release if _has_family(r, "post_wait")], "same_gate"),
        "transition_retry": ([r for r in q_release if _is_transition_retry(r)], "same_gate"),
        "post_conflict": ([r for r in q_release if _has_family(r, "post_conflict")], "post_conflict"),
        "proposal11": ([r for r in q_release if r["teacher_joint_index"] == 11], "same_proposal"),
        "proposal14": ([r for r in q_release if r["teacher_joint_index"] == 14], "same_proposal"),
    }
    radius = {name: _radius_coverage(rows, training, candidate_mode=mode) for name, (rows, mode) in groups.items()}
    recs = _records(q, training)
    intrinsic_records, intrinsic_summary = _q_intrinsic(q)
    diagnostic = {"evidence_type": "newly_run_plus_preexisting_artifact", "Q_status": "CONSUMED_DIAGNOSTIC_REFERENCE",
                  "Q_rows_used_for_training": 0, "Q_access_phase": "after_training_and_internal_audit_collection_complete",
                  "failed_coverage_reference_sha256": EXPECTED_FAILED_COVERAGE_SHA,
                  "R3_to_reference": _coverage_summary(recs),
                  "Q_intrinsic": {"summary": intrinsic_summary, "records": intrinsic_records},
                  "class_priors": _prior_audit(training, q, recs),
                  "proposal14_TV": _tv_decomposition(training, q),
                  "knn_purity_role": "DIAGNOSTIC_ONLY", "radius_coverage_role": "HARD"}
    return radius, diagnostic


def _summary(manifest: dict, radius: dict, diagnostics: dict) -> str:
    c = manifest["training_coverage"]
    lines = ["# Fresh V4.2 / Observation V2.1 I0-R3 Collection and Qualification", "", "## Formal result", "", "```text",
             f"formal_training_rows = {manifest['formal_training_rows']}", f"formal_internal_audit_rows = {manifest['formal_internal_audit_rows']}",
             f"training_seed_count = {manifest['training_seed_count']}", f"audit_seed_count = {manifest['audit_seed_count']}", ""]
    for key in (f"C{i:02d}" for i in range(1, 10)):
        x = c["residual_context_cells"][key]; lines.append(f"{key} = {x['lifecycles']} lifecycles / {x['episodes']} episodes")
    for key in ("N5", "N6", "N12"):
        x = c["proposal14_traffic"][key]; lines.append(f"proposal14_{key}_lifecycles/episodes = {x['lifecycles']}/{x['episodes']}")
    lines.append("")
    for key in ("shared_FN", "shared_proposal_error", "post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14"):
        lines += [f"{key}_coverage_0_25 = {radius[key]['coverage_0.25']}", f"{key}_coverage_1_00 = {radius[key]['coverage_1.00']}"]
    lines += ["", f"retry_provenance = {manifest['schema_retry_provenance']['retry_feature_provenance']}",
              f"teacher_determinism = {manifest['teacher_determinism']}", f"replay = {manifest['ActionTape_replay']['replay']}",
              f"safety = {manifest['safety']['result']}", f"independence = {manifest['independence']}",
              f"aliasing = {manifest['observation_aliasing']['observation_aliasing']}", "",
              f"V4_2_V2_1_I0_R3 = {manifest['V4_2_V2_1_I0_R3']}",
              f"V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = {str(manifest['V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED']).lower()}",
              "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false", "PPO_ALLOWED = false", "```", "",
              "## Diagnostic-only metrics", "", f"kNN/class-prior/TV artifact: `{manifest['diagnostics_path']}`.",
              "These values are reported only; none is a qualification threshold.", "", "## Evidence and risks", "",
              "Collection results are `newly_run`; consumed Q and failure-state references are `preexisting_artifact` and never enter training.",
              "The internal audit is `NEVER_TRAIN / NEVER_MODEL_SELECT`. Euclidean 88D radius coverage is descriptive, and future Q2 remains uncollected.", ""]
    return "\n".join(lines)


def run(output_dir: Path, history_root: Path, q_path: Path, consensus_path: Path, failed_coverage_path: Path, *, smoke_cycles: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol, protocol_check = _verify_protocol()
    if not protocol_check["PASS"]:
        fail = {"I0_R3_COLLECTION_VALID": False, "protocol_check": protocol_check, "V4_2_V2_1_I0_R3": "FAIL",
                "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": False,
                "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False}
        _write(output_dir / "v4_2_v2_1_i0_r3_manifest.json", fail); return fail
    registry, _ = _build_pre_reset_registry(history_root, output_dir)
    if registry["parse_error_count"]: raise RuntimeError("pre-reset seed registry metadata contains parse errors")
    training, train_info, train_tapes = _collect_partition(output_dir, "training", registry["available_candidates"]["training"], protocol, smoke_cycles=smoke_cycles)
    audit, audit_info, audit_tapes = _collect_partition(output_dir, "internal_audit", registry["available_candidates"]["internal_audit"], protocol, smoke_cycles=smoke_cycles)
    train_path = output_dir / "v4_2_v2_1_i0_r3_training.jsonl"; audit_path = output_dir / "v4_2_v2_1_i0_r3_internal_audit.jsonl"
    train_path.write_text("".join(json.dumps(_safe(r), sort_keys=True) + "\n" for r in training), encoding="utf-8")
    audit_path.write_text("".join(json.dumps(_safe(r), sort_keys=True) + "\n" for r in audit), encoding="utf-8")
    coverage = _coverage(training, protocol); provenance = _provenance(training + audit)
    determinism = all(r["teacher_gate"] == r["teacher_repeat_gate"] and r["teacher_proposal"] == r["teacher_repeat_proposal"] and r["teacher_joint_index"] == r["teacher_repeat_joint_index"] for r in training + audit)
    aliasing = _aliasing(training + audit); replay = _replay(train_tapes + audit_tapes)
    safety_counts = {k: train_info["safety"][k] + audit_info["safety"][k] for k in train_info["safety"]}
    safety = {"counts": safety_counts, "result": "PASS" if not any(safety_counts.values()) else "FAIL"}
    if smoke_cycles is None:
        q, consensus, failed = _load_q_and_consensus(q_path, consensus_path, failed_coverage_path)
        radius, diagnostics = _diagnostics(training, q, consensus, failed)
    else:
        radius = {k: {"coverage_0.25": None, "coverage_1.00": None, "PASS": False} for k in ("shared_FN", "shared_proposal_error", "post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14")}
        diagnostics = {"Q_status": "NOT_OPENED_DURING_SMOKE", "Q_rows_used_for_training": 0, "knn_purity_role": "DIAGNOSTIC_ONLY"}
    radius_path = output_dir / "v4_2_v2_1_i0_r3_radius_coverage.json"; diagnostic_path = output_dir / "v4_2_v2_1_i0_r3_diagnostics.json"
    _write(radius_path, radius); _write(diagnostic_path, diagnostics)
    history = _historical_independence(history_root, output_dir, training + audit)
    keys = ("seed", "episode_id", "lifecycle_event_id", "decision_event_id", "full_state_hash")
    partition = {k: not ({r[k] for r in training} & {r[k] for r in audit}) for k in keys}
    independence_ok = all(history.values()) and all(partition.values())
    formal_gates = {
        "fresh_lineage": history["seed"] and history["episode_id"] and history["lifecycle_event_id"] and history["full_state_hash"],
        "training_cardinality": TRAIN_TARGET[0] <= len(training) <= TRAIN_CEILING and train_info["complete_stratum_cycles"] >= 1,
        "audit_cardinality": AUDIT_TARGET[0] <= len(audit) <= AUDIT_CEILING and audit_info["complete_stratum_cycles"] >= 1,
        "nominal_anchors": coverage["nominal_anchor_PASS"], "all_residual_context_cells": coverage["all_residual_context_cells_PASS"],
        "proposal14_N5_N6_N12": coverage["proposal14_traffic_PASS"], "stable_support": coverage["stable_support_PASS"],
        "shared_FN_radius": radius["shared_FN"]["PASS"], "shared_proposal_error_radius": radius["shared_proposal_error"]["PASS"],
        "all_family_radius": all(radius[k]["PASS"] for k in ("post_WAIT", "transition_retry", "post_conflict", "proposal11", "proposal14")),
        "schema_retry_provenance": provenance["schema_provenance"] == "PASS" and provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": determinism, "replay": replay["replay"] == "PASS", "aliasing": aliasing["observation_aliasing"] == "PASS",
        "safety": safety["result"] == "PASS", "independence": independence_ok,
    }
    passed = smoke_cycles is None and all(formal_gates.values())
    manifest = {"stage": "FRESH_FORMAL_V4_2_V2_1_I0_R3", "evidence_type": "newly_run", "smoke_only": smoke_cycles is not None,
                "protocol": PROTOCOL_VERSION, "protocol_check": protocol_check, "I0_R3_COLLECTION_VALID": True,
                "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 88, "action_schema": ACTION_SCHEMA_V4_2,
                "forbidden_operations_used": False, "historical_rows_used": False, "consumed_Q_rows_used": False,
                "formal_training_path": str(train_path), "formal_training_sha256": _sha(train_path), "formal_training_rows": len(training),
                "formal_internal_audit_path": str(audit_path), "formal_internal_audit_sha256": _sha(audit_path), "formal_internal_audit_rows": len(audit),
                "internal_audit_usage": "NEVER_TRAIN_NEVER_MODEL_SELECT", "training_seed_count": len(set(train_info["selected_seeds"])), "audit_seed_count": len(set(audit_info["selected_seeds"])),
                "pre_reset_seed_registry": registry["path"], "pre_reset_seed_registry_sha256": registry["sha256"],
                "collector_code": str(Path(__file__)), "collector_code_sha256": _sha(Path(__file__)),
                "collector_config": str(PROTOCOL_PATH), "collector_config_sha256": _sha(PROTOCOL_PATH),
                "collection_schedule": [{"name": x.name, "spec": asdict(x.spec)} for x in STRATA],
                "training_collection": train_info, "internal_audit_collection": audit_info, "training_coverage": coverage,
                "schema_retry_provenance": provenance, "teacher_determinism": "PASS" if determinism else "FAIL",
                "observation_aliasing": aliasing, "ActionTape_replay": replay, "safety": safety,
                "independence_checks": {"historical": history, "partition": partition}, "independence": "PASS" if independence_ok else "FAIL",
                "radius_coverage_path": str(radius_path), "radius_coverage_sha256": _sha(radius_path),
                "diagnostics_path": str(diagnostic_path), "diagnostics_sha256": _sha(diagnostic_path),
                "reset_recipe_hashes": sorted({r["reset_recipe_hash"] for r in training + audit}),
                "formal_gates": formal_gates, "qualification_failures": [k for k, v in formal_gates.items() if not v],
                "V4_2_V2_1_I0_R3": "PASS" if passed else "FAIL",
                "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": bool(passed),
                "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False,
                "protocol_risks": ["Consumed Q is diagnostic, not untouched qualification", "88D Euclidean radius is descriptive", "Internal audit is not future Q2"]}
    manifest_path = output_dir / "v4_2_v2_1_i0_r3_manifest.json"; _write(manifest_path, manifest)
    summary_path = output_dir / "v4_2_v2_1_i0_r3_collection_summary.md"; summary_path.write_text(_summary(manifest, radius, diagnostics), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history-root", required=True, type=Path)
    parser.add_argument("--consumed-q", required=True, type=Path)
    parser.add_argument("--error-consensus", required=True, type=Path)
    parser.add_argument("--failed-coverage", required=True, type=Path)
    parser.add_argument("--smoke-cycles", type=int)
    args = parser.parse_args()
    result = run(args.output_dir, args.history_root, args.consumed_q, args.error_consensus, args.failed_coverage, smoke_cycles=args.smoke_cycles)
    print(json.dumps({k: result.get(k) for k in ("formal_training_rows", "formal_internal_audit_rows", "training_seed_count", "audit_seed_count", "teacher_determinism", "independence", "V4_2_V2_1_I0_R3", "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED", "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED", "PPO_ALLOWED")}, indent=2))


if __name__ == "__main__":
    main()
