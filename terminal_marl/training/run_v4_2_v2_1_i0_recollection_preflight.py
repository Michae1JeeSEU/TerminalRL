"""Collector-only V4.2/V2.1 I0 repair and fresh recollection preflight.

This module never reads formal training rows.  It uses only new rollout seeds
to qualify reset provenance and natural scenario-level RELEASE coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import (
    SOURCES, _aliasing, _collect_partition, _config, _digest, _provenance,
    _sha, _source_reset_recipe, _summary,
)
from .v4_1_joint_teacher import teacher_joint_expert_v4_1
from ..core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1


PREFLIGHT_SEEDS = tuple(range(73001, 75001))
DYNAMIC_TAPES_PER_FAMILY = 10
FAMILY_QUOTAS = {
    "clear_outer_RELEASE": 15, "middle_interface_RELEASE": 15,
    "post_conflict_RELEASE": 10, "post_WAIT_RELEASE": 10,
}


def _state_diff(before, after, prefix=""):
    """Bounded field-level provenance diff for the historic dynamic tapes."""
    rows = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            rows.extend(_state_diff(before.get(key, "<missing>"), after.get(key, "<missing>"), f"{prefix}.{key}"))
    elif isinstance(before, list) and isinstance(after, list):
        if before != after:
            rows.append({"field": prefix, "before": before, "after": after})
    elif before != after:
        rows.append({"field": prefix, "before": before, "after": after})
    return rows[:80]


def audit_historic_dynamic_tapes() -> dict:
    """Diagnose only: failed 1800/300 tapes remain read-only external input."""
    historic = Path.home() / "training_outputs/terminal/v4_2_v2_1_formal_i0_20260912/action_tapes"
    source_by_seed = {51004: "transition_retry_wait", 51007: "interface_headway",
                      61004: "transition_retry_wait", 61007: "interface_headway"}
    specs = {item.source: item for item in SOURCES}
    reports = []
    for seed, source in source_by_seed.items():
        tape = next(historic.glob(f"safety_trace_seed_{seed}_step_*.json"), None)
        if tape is None:
            reports.append({"seed": seed, "source": source, "status": "missing_historic_tape"}); continue
        payload = json.loads(tape.read_text(encoding="utf-8")); env = TerminalParallelEnv(_config(specs[source])); env.reset(seed)
        generic = SafetyTraceRecorder._state(env); recipe, _ = _source_reset_recipe(env, specs[source]); configured = SafetyTraceRecorder._state(env)
        legacy = payload["reset_state"]
        reports.append({"seed": seed, "source": source, "tape_sha256": _sha(tape),
                        "legacy_reset_hash_matches_generic": _digest({k: generic[k] for k in legacy}) == payload["reset_state_hash"],
                        "legacy_reset_hash_matches_configured": _digest({k: configured[k] for k in legacy}) == payload["reset_state_hash"],
                        "generic_to_predecision_dynamic_diff": _state_diff(generic["dynamic_airspace"], configured["dynamic_airspace"]),
                        "generic_to_predecision_event_diff": _state_diff(generic["event_scheduler"], configured["event_scheduler"]),
                        "reset_recipe": recipe, "reset_recipe_hash": _digest(recipe),
                        "dynamic_airspace_state_hash": _digest(configured["dynamic_airspace"]),
                        "arrival_rng_state_hash": _digest(configured["arrival_process"]["rng_state"]),
                        "legacy_state_omitted_dynamic_airspace": "dynamic_airspace" not in legacy})
    return {"historic_artifact": str(historic), "artifact_role": "READ_ONLY_FAILED_DIAGNOSTIC_ARTIFACT", "failed_tapes": reports}


def _dynamic_replay_closure(output: Path) -> dict:
    reports = {}
    for spec in (item for item in SOURCES if item.dynamic):
        family = []
        for offset in range(DYNAMIC_TAPES_PER_FAMILY):
            seed = 76000 + 100 * list(item.source for item in SOURCES).index(spec.source) + offset
            env = TerminalParallelEnv(_config(spec)); env.reset(seed); recipe, observations = _source_reset_recipe(env, spec)
            tape = SafetyTraceRecorder(output / "dynamic_replay_tapes" / spec.source, dt=env.config.dt)
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "collector_replay_closure", "source": spec.source}, reset_recipe=recipe)
            for _ in range(2):
                actions = {}
                labels = {}
                for uid in sorted(observations):
                    if int(uid) not in env.active: continue
                    label, _ = teacher_joint_expert_v4_1(env, env.active[int(uid)], audit_catalogue=False)
                    actions[uid] = hierarchical_from_joint_index_v4_1(int(label.index)).normalized
                    target = hierarchical_from_joint_index_v4_1(int(label.index))
                    labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": int(label.index)}
                record = tape.record_pre_step(env, observations, actions)
                observations, _, _, _, info = env.step(actions)
                tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
            path = tape.dump(reason=f"collector_replay_closure_{spec.source}")
            result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
            family.append({"seed": seed, "path": str(path), "sha256": _sha(path), "result": result})
        reports[spec.source] = family
    all_reports = [item for family in reports.values() for item in family]
    return {"minimum_tapes_per_dynamic_source": DYNAMIC_TAPES_PER_FAMILY, "families": reports,
            "exact_replay_rate": sum(bool(item["result"]["exact"]) for item in all_reports) / max(1, len(all_reports)),
            "DYNAMIC_SOURCE_REPLAY_CLOSURE": "PASS" if all_reports and all(item["result"]["exact"] for item in all_reports) else "FAIL"}


def _family_counts(rows: list[dict]) -> dict[str, int]:
    groups = {
        "clear_outer_RELEASE": [row for row in rows if row["source"] == "clear_release_opportunity" and row["teacher_gate"] == 1],
        "middle_interface_RELEASE": [row for row in rows if row["source"] == "middle_interface_release" and row["teacher_gate"] == 1],
        "post_conflict_RELEASE": [row for row in rows if row["source"] == "post_conflict_recovery" and row["teacher_gate"] == 1 and row["previous_teacher_gate"] == 0],
        "post_WAIT_RELEASE": [row for row in rows if row["source"] == "post_wait_accumulated_release" and row["teacher_gate"] == 1 and row["previous_teacher_gate"] == 0],
    }
    return {name: len({row["lifecycle_event_id"] for row in values}) for name, values in groups.items()} | {
        "other_valid_RELEASE": len({row["lifecycle_event_id"] for row in rows if row["teacher_gate"] == 1 and row["source"] not in {"clear_release_opportunity", "middle_interface_release", "post_conflict_recovery", "post_wait_accumulated_release"}})
    }


def run(output_dir: str | Path, *, target_rows: int = 600, max_seeds: int = len(PREFLIGHT_SEEDS)) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    historic = audit_historic_dynamic_tapes()
    replay = _dynamic_replay_closure(output)
    rows, collection, tapes = _collect_partition(output, name="collector_preflight", seeds=PREFLIGHT_SEEDS[:max_seeds], target=target_rows, banned_hashes=set())
    path = output / "collector_preflight_rows.jsonl"; path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = _summary(rows, collection["raw_decision_ticks"], Counter({int(k): int(v) for k, v in collection["raw_joint_histogram"].items()}))
    provenance, aliasing = _provenance(rows), _aliasing(rows)
    determinism = all(row["teacher_gate"] == row["teacher_repeat_gate"] and row["teacher_proposal"] == row["teacher_repeat_proposal"] for row in rows)
    family_counts = _family_counts(rows)
    family_pass = all(family_counts[name] >= quota for name, quota in FAMILY_QUOTAS.items())
    diversity = {"release_joint_classes": summary["release_joint_classes"], "speed_values": summary["release_speed_values"], "merge_values": summary["release_merge_values"]}
    diversity_pass = len(diversity["release_joint_classes"]) >= 8 and len(diversity["speed_values"]) >= 4 and len(diversity["merge_values"]) == 5
    preflight_replay = {"sampled_collector_tapes": [{"path": str(path), "result": SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)} for path, spec in tapes]}
    preflight_replay["PRELIGHT_ACTIONTAPE_EXACT_REPLAY"] = "PASS" if preflight_replay["sampled_collector_tapes"] and all(item["result"]["exact"] for item in preflight_replay["sampled_collector_tapes"]) else "FAIL"
    safety = {key: int(collection["safety"].get(key, 0)) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}
    allowed = all((replay["DYNAMIC_SOURCE_REPLAY_CLOSURE"] == "PASS", family_pass, diversity_pass,
                   provenance["retry_feature_provenance"] == "PASS", determinism,
                   aliasing["observation_aliasing_check"] == "PASS", not any(safety.values()),
                   preflight_replay["PRELIGHT_ACTIONTAPE_EXACT_REPLAY"] == "PASS"))
    manifest = {"stage": "V4_2_V2_1_I0_COLLECTOR_REPAIR_PREFLIGHT", "evidence_type": "newly_run",
                "historical_failed_formal_i0": historic, "dynamic_replay_closure": replay,
                "preflight_rows_path": str(path), "preflight_rows_sha256": _sha(path), "preflight_rows": len(rows),
                "new_seed_pool": [PREFLIGHT_SEEDS[0], PREFLIGHT_SEEDS[min(max_seeds, len(PREFLIGHT_SEEDS)) - 1]],
                "collection": collection, "summary": summary, "release_family_quotas": FAMILY_QUOTAS,
                "release_family_lifecycle_counts": family_counts,
                "COLLECTOR_RELEASE_FAMILY_COVERAGE": "PASS" if family_pass else "FAIL",
                "proposal_diversity": diversity, "NATURALLY_REACHABLE_PROPOSAL_SET": diversity["release_joint_classes"],
                "COLLECTOR_PROPOSAL_DIVERSITY": "PASS" if diversity_pass else "FAIL",
                "retry_provenance": provenance["retry_feature_provenance"], "teacher_determinism": "PASS" if determinism else "FAIL",
                "observation_aliasing": aliasing, "safety": safety, "preflight_action_tape_replay": preflight_replay,
                "formal_protocol_if_allowed": {"training_unique_RELEASE_lifecycles_target": 130, "frozen_Teacher_RELEASE_target": 80,
                                                "qualification_minimums_unchanged": {"training_unique_RELEASE_lifecycles": 100, "frozen_Teacher_RELEASE": 60,
                                                                                       "release_joint_classes": 8, "speed_values": 4, "merge_values": 5}},
                "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED": bool(allowed), "I1_ALLOWED": False,
                "DAgger_ALLOWED": False, "PPO_ALLOWED": False}
    (output / "v4_2_v2_1_i0_recollection_preflight_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--target-rows", type=int, default=600); parser.add_argument("--max-seeds", type=int, default=len(PREFLIGHT_SEEDS))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, target_rows=args.target_rows, max_seeds=args.max_seeds), indent=2))
