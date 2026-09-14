"""Collect and freeze the untouched V2.1 I1 qualification cohort.

The collector has no policy/checkpoint input.  It creates a complete historical
identity registry before the first reset, runs only whole episodes from the
preregistered natural scenario schedule, and fails closed at 1,200 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1
from ..core.observations import build_low_observation_v2_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import (
    OBSERVATION_SCHEMA, RETRY_HORIZON, SourceSpec, _config, _digest, _safe,
    _sha, _snapshot, _source_reset_recipe, _topology,
)
from .run_v4_2_v2_1_formal_i0_r import (
    _aliasing, _formal_row, _predicate_context, _provenance, _semantic_reason,
)
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


SCHEMA_VERSION = "v4_2_v2_1_i1_q_cohort_v1"
PROTOCOL_VERSION = "V4_2_V2_1_I1_QUALIFICATION_V1"
SEED_START, SEED_END = 210001, 210999
MIN_ROWS, MAX_ROWS = 650, 1200
MAX_SECONDARY_EVENTS_PER_LIFECYCLE = 4
PROTOCOL_CONFIG = Path("terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol.yaml")
COLLECTION_CONFIG = Path("terminal_marl/configs/low_v4_2/i1_v2_1_q_cohort.yaml")

MINIMUMS = {
    "WAIT_rows": 300, "WAIT_lifecycles": 250,
    "RELEASE_rows": 160, "RELEASE_lifecycles": 150,
    "proposal_11_lifecycles": 15, "proposal_12_lifecycles": 15,
    "proposal_13_lifecycles": 30, "proposal_14_lifecycles": 15,
    "clear_outer": 30, "middle_interface": 30, "post_WAIT": 30,
    "post_conflict": 30, "retry_active_WAIT": 60,
    "retry_expired_RELEASE": 60,
}


@dataclass(frozen=True)
class QStratum:
    name: str
    spec: SourceSpec
    closes: tuple[str, ...]


STRATA = (
    QStratum("clear_outer_low_traffic", SourceSpec("i1_q_clear_outer", "clear_outer", 3, phases=("ring_outer",), max_steps=12),
             ("clear_outer", "proposal_11_lifecycles", "RELEASE_lifecycles")),
    QStratum("middle_interface", SourceSpec("i1_q_middle_interface", "middle_interface", 6, phases=("ring_middle",), max_steps=12),
             ("middle_interface", "proposal_13_lifecycles", "RELEASE_lifecycles")),
    QStratum("post_WAIT_release", SourceSpec("i1_q_post_wait", "transition_wait_release", 12, dynamic=True, max_steps=42),
             ("post_WAIT", "retry_expired_RELEASE", "RELEASE_lifecycles")),
    QStratum("conflict_recovery", SourceSpec("i1_q_conflict_recovery", "conflict_recovery", 12, dynamic=True, max_steps=42),
             ("post_conflict", "RELEASE_lifecycles")),
    QStratum("transition_retry", SourceSpec("i1_q_transition_retry", "transition_wait_release", 12, dynamic=True, max_steps=42),
             ("retry_active_WAIT", "WAIT_lifecycles")),
    QStratum("nominal_mixed", SourceSpec("i1_q_nominal_mixed", "nominal_mixed", 12, max_steps=12),
             ("WAIT_rows", "WAIT_lifecycles", "RELEASE_rows", "RELEASE_lifecycles")),
    QStratum("resource_pressure", SourceSpec("i1_q_resource_pressure", "resource_pressure", 12, final_capacity=0, max_steps=12),
             ("WAIT_rows", "WAIT_lifecycles")),
    QStratum("proposal_12_context", SourceSpec("i1_q_proposal_12", "outer_open_n8", 8, max_steps=12),
             ("proposal_12_lifecycles", "RELEASE_lifecycles")),
    QStratum("proposal_14_context", SourceSpec("i1_q_proposal_14", "outer_open_n5", 5, max_steps=12),
             ("proposal_14_lifecycles", "RELEASE_lifecycles")),
)


def _iter_objects(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_objects(item)


def _identity_accumulator() -> dict[str, Any]:
    return {
        "seeds": set(), "episode_ids": set(), "lifecycle_ids": set(),
        "full_state_hashes": set(), "observation_labels": defaultdict(set),
    }


def _consume_object(value: Any, identities: dict[str, Any]) -> None:
    for item in _iter_objects(value):
        for key, field in (("seed", "seeds"), ("episode_seed", "seeds")):
            raw = item.get(key)
            if isinstance(raw, (int, float)) and float(raw).is_integer():
                identities[field].add(int(raw))
        for key in ("episode_id",):
            if isinstance(item.get(key), str): identities["episode_ids"].add(item[key])
        for key in ("lifecycle_event_id", "lifecycle_id"):
            if isinstance(item.get(key), str): identities["lifecycle_ids"].add(item[key])
        for key in ("full_state_hash", "state_hash"):
            if isinstance(item.get(key), str): identities["full_state_hashes"].add(item[key])
        obs_hash = item.get("observation_hash") or item.get("obs_hash")
        gate = item.get("teacher_gate")
        if isinstance(obs_hash, str) and gate in (0, 1):
            identities["observation_labels"][obs_hash].add(int(gate))


def _history_files(history_root: Path, excluded_root: Path) -> list[Path]:
    files = []
    excluded = excluded_root.resolve()
    for path in history_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            path.resolve().relative_to(excluded)
            continue
        except ValueError:
            pass
        files.append(path)
    return sorted(files)


def build_seed_registry(history_root: Path, output: Path) -> tuple[dict, dict[str, Any]]:
    identities = _identity_accumulator(); sources = []
    for path in _history_files(history_root, output):
        parsed = 0; errors = 0; parse_mode = "jsonl" if path.suffix.lower() == ".jsonl" else "json"
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip(): continue
                        try:
                            _consume_object(json.loads(line), identities); parsed += 1
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            errors += 1
            else:
                text = path.read_text(encoding="utf-8")
                try:
                    _consume_object(json.loads(text), identities); parsed = 1
                except json.JSONDecodeError:
                    # Some historical artifacts use one JSON object per line
                    # despite a .json suffix.  Preserve them in the registry.
                    parse_mode = "json_lines_fallback"
                    for line in text.splitlines():
                        if not line.strip(): continue
                        try:
                            _consume_object(json.loads(line), identities); parsed += 1
                        except json.JSONDecodeError:
                            pass
                    if parsed == 0:
                        # A few legacy status files are escaped display text,
                        # not JSON.  Scan identity tokens before marking them
                        # as safely non-row metadata.
                        parse_mode = "identity_regex_fallback"
                        for match in re.finditer(r'\\?"?(?:episode_)?seed\\?"?\s*[:=-]\s*(\d+)', text, re.IGNORECASE):
                            identities["seeds"].add(int(match.group(1)))
                        for key, field in (("episode_id", "episode_ids"), ("lifecycle_event_id", "lifecycle_ids"),
                                           ("lifecycle_id", "lifecycle_ids"), ("full_state_hash", "full_state_hashes"),
                                           ("state_hash", "full_state_hashes")):
                            for match in re.finditer(rf'\\?"?{key}\\?"?\s*:\s*\\?"([^"\\]+)', text):
                                identities[field].add(match.group(1))
            errors = 0
        except (OSError, UnicodeDecodeError):
            errors = 1
        sources.append({"path": str(path), "sha256": _sha(path), "records_parsed": parsed,
                        "parse_mode": parse_mode, "parse_errors": errors})
    candidates = list(range(SEED_START, SEED_END + 1))
    collisions = sorted(seed for seed in candidates if seed in identities["seeds"])
    available = [seed for seed in candidates if seed not in identities["seeds"]]
    def set_digest(values: Iterable[Any]) -> str:
        return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()
    registry = {
        "stage": "V2_1_I1_Q_PRE_RESET_SEED_REGISTRY",
        "built_before_any_reset": True,
        "history_root": str(history_root),
        "history_files": sources,
        "history_file_count": len(sources),
        "history_parse_error_count": sum(row["parse_errors"] for row in sources),
        "historical_counts": {"seeds": len(identities["seeds"]), "episode_ids": len(identities["episode_ids"]),
                              "lifecycle_ids": len(identities["lifecycle_ids"]), "full_state_hashes": len(identities["full_state_hashes"]),
                              "observation_hashes": len(identities["observation_labels"])},
        "historical_identity_digests": {
            "seeds": set_digest(identities["seeds"]), "episode_ids": set_digest(identities["episode_ids"]),
            "lifecycle_ids": set_digest(identities["lifecycle_ids"]), "full_state_hashes": set_digest(identities["full_state_hashes"]),
            "observation_hashes": set_digest(identities["observation_labels"]),
        },
        "candidate_range": [SEED_START, SEED_END],
        "candidate_collisions_rejected_before_reset": collisions,
        "available_candidates": available,
    }
    path = output / "v2_1_i1_q_seed_registry_pre_reset.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    registry["path"] = str(path); registry["sha256"] = _sha(path)
    return registry, identities


def _lifecycles(rows: list[dict], predicate) -> int:
    return len({row["lifecycle_event_id"] for row in rows if predicate(row)})


def support(rows: list[dict]) -> dict:
    waits = [row for row in rows if int(row["teacher_gate"]) == 0]
    releases = [row for row in rows if int(row["teacher_gate"]) == 1]
    values = {
        "WAIT_rows": len(waits),
        "WAIT_lifecycles": len({row["lifecycle_event_id"] for row in waits}),
        "RELEASE_rows": len(releases),
        "RELEASE_lifecycles": len({row["lifecycle_event_id"] for row in releases}),
        **{f"proposal_{proposal}_lifecycles": _lifecycles(releases, lambda row, p=proposal: int(row["teacher_joint_index"]) == p)
           for proposal in (11, 12, 13, 14)},
        "clear_outer": _lifecycles(releases, lambda row: bool(row["family_flags"]["clear_outer"])),
        "middle_interface": _lifecycles(releases, lambda row: bool(row["family_flags"]["middle_interface"])),
        "post_WAIT": _lifecycles(releases, lambda row: bool(row["family_flags"]["post_wait"])),
        "post_conflict": _lifecycles(releases, lambda row: bool(row["family_flags"]["post_conflict"])),
        "retry_active_WAIT": _lifecycles(waits, lambda row: bool(row["retry_active"])),
        "retry_expired_RELEASE": _lifecycles(releases, lambda row: not bool(row["retry_active"])),
    }
    return {"values": values, "minimums": MINIMUMS,
            "passed": {key: values[key] >= minimum for key, minimum in MINIMUMS.items()},
            "all_minima_PASS": all(values[key] >= minimum for key, minimum in MINIMUMS.items())}


def _next_stratum(rows: list[dict], episode: int, cursor: int) -> tuple[QStratum, int]:
    if episode < len(STRATA):
        return STRATA[episode], cursor
    current = support(rows)
    for offset in range(len(STRATA)):
        index = (cursor + offset) % len(STRATA)
        stratum = STRATA[index]
        if any(not current["passed"][key] for key in stratum.closes):
            return stratum, (index + 1) % len(STRATA)
    return STRATA[cursor], (cursor + 1) % len(STRATA)


def _q_row(env, observation, *, stratum, seed, episode, tick, uid, lifecycle_id,
           event_index, reason, streak, similarity, selected, audit, current,
           previous_semantic, reset_recipe) -> dict:
    row = _formal_row(
        env, observation, partition="i1_q", stratum=stratum, seed=seed,
        episode=episode, tick=tick, uid=uid, lifecycle_id=lifecycle_id,
        event_index=event_index, retention_reason=reason, streak=streak,
        similarity=similarity, selected=selected, audit=audit, current=current,
        previous_semantic=previous_semantic, reset_recipe=reset_recipe,
    )
    row.update({"schema_version": SCHEMA_VERSION, "partition": "I1_Q",
                "episode_id": f"v2-1-i1-q:episode-{episode}:seed-{seed}",
                "qualification_usage": "NEVER_TRAIN_NEVER_MODEL_SELECT_NEVER_THRESHOLD_TUNE"})
    return row


def collect(output: Path, registry: dict, historical: dict[str, Any], *, smoke_episodes: int | None = None) -> tuple[list[dict], dict, list]:
    available = registry["available_candidates"]
    rows: list[dict] = []; seen_state = set(historical["full_state_hashes"])
    seen_decisions = set(); raw_ticks = 0; raw_gate = Counter(); safety = Counter()
    tapes = []; taped = set(); selected_seeds = []; episode = 0; cursor = 0
    stopped_for_opposite_observation = None
    episode_limit = smoke_episodes if smoke_episodes is not None else len(available)
    while episode < episode_limit and len(rows) <= MAX_ROWS:
        if smoke_episodes is None and len(rows) >= MIN_ROWS and support(rows)["all_minima_PASS"]:
            break
        stratum, cursor = _next_stratum(rows, episode, cursor)
        seed = available[episode]
        if seed in historical["seeds"]:
            raise RuntimeError("candidate seed collision was not rejected before reset")
        selected_seeds.append(seed)
        env = TerminalParallelEnv(_config(stratum.spec)); env.reset(seed)
        reset_recipe, observations = _source_reset_recipe(env, stratum.spec)
        record_tape = stratum.name not in taped
        tape = SafetyTraceRecorder(output / "action_tapes" / stratum.name, dt=env.config.dt) if record_tape else None
        if tape:
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "v2_1_i1_q", "stratum": stratum.name}, reset_recipe=reset_recipe)
        lifecycle = {int(uid): f"v2-1-i1-q:episode-{episode}:seed-{seed}:uid-{int(uid)}" for uid in env.active}
        raw_memory = {}; semantic_ledger = {}; event_index = Counter()
        for tick in range(stratum.spec.max_steps):
            actions = {}; labels = {}; pending = []
            for uid in [int(value) for value in sorted(observations) if int(value) in env.active]:
                uav = env.active[uid]
                selected, light_audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                observation = np.asarray(observations[str(uid)], np.float32)
                current = _predicate_context(env, uav, selected, light_audit)
                topology = _topology(env, uid, _snapshot(env, uav))
                reason, streak, similarity = _semantic_reason(raw_memory.get(uid), current, observation, topology, tick)
                # The preregistered collector requires bounded sparse
                # persistence and forbids stable-lifecycle flooding.  Always
                # preserve decision-semantic transitions; cap only secondary
                # topology/continuous-observation/persistence snapshots.
                primary = {
                    "event_start", "teacher_target_change",
                    "release_eligibility_or_retry_change", "phase_or_commit_change",
                    "reservation_interface_or_resource_change",
                }
                if reason not in primary and event_index[uid] >= MAX_SECONDARY_EVENTS_PER_LIFECYCLE:
                    reason = None
                raw_ticks += 1; raw_gate[current["teacher_gate"]] += 1
                raw_memory[uid] = {**current, "observation": observation.tolist(), "topology": topology, "streak": streak}
                if reason:
                    full_selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if int(full_selected.index) != int(selected.index):
                        raise RuntimeError("Teacher target changed when catalogue audit was materialised")
                    current = _predicate_context(env, uav, full_selected, audit)
                    event_index[uid] += 1
                    row = _q_row(env, observation, stratum=stratum, seed=seed, episode=episode,
                                 tick=tick, uid=uid, lifecycle_id=lifecycle[uid], event_index=event_index[uid],
                                 reason=reason, streak=streak, similarity=similarity, selected=full_selected,
                                 audit=audit, current=current, previous_semantic=semantic_ledger.get(uid),
                                 reset_recipe=reset_recipe)
                    old_labels = historical["observation_labels"].get(row["observation_hash"], set())
                    if old_labels and int(row["teacher_gate"]) not in old_labels:
                        stopped_for_opposite_observation = {"decision_event_id": row["decision_event_id"],
                                                            "observation_hash": row["observation_hash"],
                                                            "historical_labels": sorted(old_labels),
                                                            "new_label": int(row["teacher_gate"])}
                        break
                    if row["full_state_hash"] not in seen_state and row["decision_event_id"] not in seen_decisions:
                        seen_state.add(row["full_state_hash"]); seen_decisions.add(row["decision_event_id"])
                        ledger = {**current, "uid": uid, "lifecycle_id": lifecycle[uid],
                                  "decision_event_id": row["decision_event_id"], "time": float(env.now)}
                        pending.append((row, ledger))
                action = hierarchical_from_joint_index_v4_1(int(selected.index))
                actions[str(uid)] = action.normalized
                labels[str(uid)] = {"gate": action.gate, "proposal": action.proposal, "joint_index_v4_1": int(selected.index)}
            if stopped_for_opposite_observation: break
            pre = tape.record_pre_step(env, observations, actions) if tape else None
            observations, _, _, _, info = env.step(actions)
            post_hash = _digest(SafetyTraceRecorder._state(env))
            if tape:
                tape.record_post_step(env, pre, info); tape.record_teacher_labels(pre, labels)
            for row, ledger in pending:
                row["post_state_hash"] = post_hash
                rows.append(row); semantic_ledger[int(row["uav_id"])] = ledger
        if stopped_for_opposite_observation: break
        for key, telemetry in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
            safety[key] += int(env.telemetry.get(telemetry, 0))
        safety["SafetyInvariantError"] += 0; safety["uncertified"] += int(env.telemetry.get("uncertified_execution", 0))
        if tape:
            path = tape.dump(reason=f"v2_1_i1_q_{stratum.name}")
            tapes.append((path, stratum.spec, stratum.name)); taped.add(stratum.name)
        episode += 1
        progress = {"episodes_completed": episode, "rows": len(rows), "last_stratum": stratum.name,
                    "support": support(rows), "safety": dict(safety)}
        (output / "collection_progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        if len(rows) > MAX_ROWS: break
    return rows, {"episodes": episode, "selected_seeds": selected_seeds, "raw_decision_ticks": raw_ticks,
                  "raw_WAIT_RELEASE": {"WAIT": int(raw_gate[0]), "RELEASE": int(raw_gate[1])},
                  "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")},
                  "opposite_observation_stop": stopped_for_opposite_observation}, tapes


def _replay(tapes: list) -> dict:
    reports = []
    for path, spec, stratum in tapes:
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        reports.append({"stratum": stratum, "path": str(path), "sha256": _sha(path), "result": result})
    exact = sum(bool(row["result"]["exact"]) for row in reports)
    return {"tapes": reports, "exact_tapes": exact, "total_tapes": len(reports),
            "exact_replay_rate": exact / max(1, len(reports)),
            "replay": "PASS" if len(reports) == len(STRATA) and exact == len(reports) else "FAIL"}


def run(output_dir: Path, history_root: Path, *, smoke_episodes: int | None = None) -> dict:
    output = output_dir; output.mkdir(parents=True, exist_ok=True)
    registry, historical = build_seed_registry(history_root, output)
    if registry["history_parse_error_count"]:
        raise RuntimeError("historical identity registry contains parse errors")
    rows, info, tapes = collect(output, registry, historical, smoke_episodes=smoke_episodes)
    dataset = output / ("v2_1_i1_q_smoke.jsonl" if smoke_episodes is not None else "v2_1_i1_q_cohort.jsonl")
    dataset.write_text("".join(json.dumps(_safe(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    current_support = support(rows); provenance = _provenance(rows); aliasing = _aliasing(rows)
    teacher_determinism = all(row["teacher_gate"] == row["teacher_repeat_gate"] and
                              row["teacher_proposal"] == row["teacher_repeat_proposal"] and
                              row["teacher_joint_index"] == row["teacher_repeat_joint_index"] for row in rows)
    replay = _replay(tapes)
    q_seeds = set(info["selected_seeds"]); q_episodes = {row["episode_id"] for row in rows}
    q_lifecycles = {row["lifecycle_event_id"] for row in rows}; q_states = {row["full_state_hash"] for row in rows}
    independence_checks = {
        "seed": not bool(q_seeds & historical["seeds"]),
        "episode_id": not bool(q_episodes & historical["episode_ids"]),
        "lifecycle_event_id": not bool(q_lifecycles & historical["lifecycle_ids"]),
        "full_state_hash": not bool(q_states & historical["full_state_hashes"]),
        "decision_event_unique": len({row["decision_event_id"] for row in rows}) == len(rows),
        "full_state_hash_unique": len(q_states) == len(rows),
    }
    safety_pass = not any(info["safety"].values())
    cardinality = MIN_ROWS <= len(rows) <= MAX_ROWS
    gates = {
        "row_cardinality": cardinality,
        "all_support_minima": current_support["all_minima_PASS"],
        "schema_provenance": provenance["schema_provenance"] == "PASS",
        "retry_provenance": provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": teacher_determinism,
        "replay": replay["replay"] == "PASS",
        "safety": safety_pass,
        "independence": all(independence_checks.values()),
        "opposite_gate_exact_observation_conflict": info["opposite_observation_stop"] is None and aliasing["exact_observation_hash_opposite_gate_conflicts"] == 0,
        "all_preregistered_strata_executed": len(tapes) == len(STRATA),
    }
    frozen = smoke_episodes is None and all(gates.values())
    final_registry = {
        "pre_reset_registry_path": registry["path"], "pre_reset_registry_sha256": registry["sha256"],
        "selected_seeds": info["selected_seeds"], "selected_seed_count": len(info["selected_seeds"]),
        "rejected_candidate_collisions": registry["candidate_collisions_rejected_before_reset"],
        "seed_independence": independence_checks["seed"],
    }
    final_registry_path = output / "v2_1_i1_q_seed_registry_final.json"
    final_registry_path.write_text(json.dumps(final_registry, indent=2), encoding="utf-8")
    reset_hashes = sorted({row["reset_recipe_hash"] for row in rows})
    manifest = {
        "stage": "V2_1_I1_Q_COHORT_COLLECTION_AND_FREEZE",
        "evidence_type": "newly_run", "smoke_only": smoke_episodes is not None,
        "protocol": PROTOCOL_VERSION, "observation_schema": OBSERVATION_SCHEMA,
        "observation_dim": 88, "action_schema": ACTION_SCHEMA_V4_2,
        "dataset_path": str(dataset), "dataset_sha256": _sha(dataset), "Q_rows": len(rows),
        "seed_registry_pre_reset_path": registry["path"], "seed_registry_pre_reset_sha256": registry["sha256"],
        "seed_registry_final_path": str(final_registry_path), "seed_registry_final_sha256": _sha(final_registry_path),
        "collection_config": str(COLLECTION_CONFIG), "collection_config_sha256": _sha(COLLECTION_CONFIG),
        "qualification_protocol_config": str(PROTOCOL_CONFIG), "qualification_protocol_sha256": _sha(PROTOCOL_CONFIG),
        "collection_code": str(Path(__file__)), "collection_code_sha256": _sha(Path(__file__)),
        "reset_recipe_hashes": reset_hashes, "reset_recipe_hash_count": len(reset_hashes),
        "collection": info, "support": current_support, "schema_retry_provenance": provenance,
        "teacher_determinism": "PASS" if teacher_determinism else "FAIL",
        "ActionTape_replay": replay, "safety": {"counts": info["safety"], "result": "PASS" if safety_pass else "FAIL"},
        "independence_checks": independence_checks, "independence": "PASS" if all(independence_checks.values()) else "FAIL",
        "observation_aliasing": aliasing, "freeze_gates": gates,
        "qualification_failures": [key for key, value in gates.items() if not value],
        "usage": {"IMMUTABLE": frozen, "NEVER_TRAIN": True, "NEVER_MODEL_SELECT": True,
                  "NEVER_THRESHOLD_TUNE": True, "HIDDEN_FROM_TRAINING_CODE": True},
        "I1_Q_COHORT_FROZEN": frozen,
        "I1_FRESH_MULTI_SEED_TRAINING_ALLOWED": frozen,
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False,
    }
    manifest_path = output / ("v2_1_i1_q_smoke_manifest.json" if smoke_episodes is not None else "v2_1_i1_q_cohort_manifest.json")
    manifest_path.write_text(json.dumps(_safe(manifest), indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history-root", required=True, type=Path)
    parser.add_argument("--smoke-episodes", type=int)
    args = parser.parse_args()
    result = run(args.output_dir, args.history_root, smoke_episodes=args.smoke_episodes)
    print(json.dumps({"Q_rows": result["Q_rows"], "support": result["support"],
                      "retry_provenance": result["schema_retry_provenance"]["retry_feature_provenance"],
                      "teacher_determinism": result["teacher_determinism"],
                      "replay": result["ActionTape_replay"]["replay"], "safety": result["safety"],
                      "independence": result["independence"],
                      "I1_Q_COHORT_FROZEN": result["I1_Q_COHORT_FROZEN"],
                      "I1_FRESH_MULTI_SEED_TRAINING_ALLOWED": result["I1_FRESH_MULTI_SEED_TRAINING_ALLOWED"],
                      "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False, "PPO_ALLOWED": False}, indent=2))
