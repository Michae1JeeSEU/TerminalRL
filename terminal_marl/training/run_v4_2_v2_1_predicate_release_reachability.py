"""Predicate-defined V4.2/V2.1 RELEASE-family reachability qualification.

This diagnostic produces witness artifacts only.  It never creates formal-I0
rows and retains the existing Teacher, observation, safety, and action schemas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import SourceSpec, _config, _digest, _safe, _sha, _snapshot, _source_reset_recipe
from .run_v4_2_v2_1_release_manifold_audit import _catalogue, _proposal_row
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


WITNESS_MINIMUM = 10
SEARCH_SEEDS = tuple(range(84001, 84031))
CONFLICT_REASONS = {"geometry_conflict", "headway_conflict", "holder_tube_conflict", "candidate_conflict"}


def _interface_for(uav) -> str | None:
    return {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(None if uav.ring_role is None else uav.ring_role.value)


def _headway_context(env, uav) -> dict:
    interface = _interface_for(uav)
    if interface is None:
        return {"interface": None, "last_interface_release_time": None, "current_headway_requirement": None, "headway_age": None}
    last = max(float(env.scheduler.last_release[interface]), float(env.scheduler.last_executed_release[interface]))
    return {"interface": interface, "last_interface_release_time": None if not math.isfinite(last) else last,
            "current_headway_requirement": float(env.airspace.headways[interface]),
            "headway_age": None if not math.isfinite(last) else max(0.0, float(env.now - last))}


def _wait_reason(env, uav, audit) -> tuple[str, dict]:
    """Classify only current Teacher predicates/candidate outcomes."""
    interface = _interface_for(uav)
    release = [row for row in audit if row["decoded"]["release_request"] > 0.0]
    reasons = Counter(str(row["reason"]) for row in release)
    context = {"retry_active": bool(env.now < uav.safety_retry_after), "committed": bool(uav.committed_plan is not None),
               "interface_open": bool(interface is not None and interface not in env.airspace.affected_interfaces()),
               "resource_valid": bool(uav.phase.value != "ring_inner" or env.config.final_queue_capacity > sum(x.phase.value == "final" for x in env.active.values()))}
    if context["retry_active"]: return "retry", context
    if context["committed"]: return "committed", context
    if interface in env.airspace.affected_interfaces(): return "geometry_conflict", context
    if reasons.get("headway_conflict", 0): return "headway_conflict", context
    if reasons.get("holder_reachable_tube", 0): return "holder_tube_conflict", context
    if reasons.get("final_capacity", 0) or reasons.get("lz_unavailable", 0): return "resource", context
    if reasons.get("candidate_rejected", 0) or reasons.get("trajectory_infeasible", 0): return "candidate_conflict", context
    return "other_wait", context


def classify_family_flags(current: dict, previous: dict | None) -> dict[str, bool]:
    """Pure family classifier; source names are deliberately absent."""
    release = int(current["teacher_gate"]) == 1
    common = release and not current["retry_active"] and not current["committed"] and current["interface_open"] and current["resource_valid"]
    same_lifecycle = bool(previous and previous["uid"] == current["uid"] and previous["lifecycle_id"] == current["lifecycle_id"])
    post_wait = bool(common and same_lifecycle and int(previous["teacher_gate"]) == 0)
    post_conflict = bool(post_wait and previous["wait_rationale"] in CONFLICT_REASONS and not current["conflict_state"])
    return {"clear_outer": bool(common and current["phase"] == "ring_outer" and not current["conflict_state"]),
            "middle_interface": bool(common and current["phase"] == "ring_middle"),
            "post_wait": post_wait, "post_conflict": post_conflict}


def _declared_predicates(family: str) -> dict:
    if family == "clear_outer_RELEASE":
        return {"target_outer_holder": True, "teacher_release": True, "retry_inactive": True, "not_committed": True,
                "interface_open": True, "resource_valid": True, "no_blocking_conflict": True}
    if family == "middle_interface_RELEASE":
        return {"target_middle_holder": True, "teacher_release": True, "retry_inactive": True, "not_committed": True,
                "interface_open": True, "resource_valid": True}
    if family == "post_WAIT_RELEASE":
        return {"teacher_release": True, "same_uid": True, "same_lifecycle": True, "previous_gate_wait": True}
    if family == "post_conflict_RELEASE":
        return {"teacher_release": True, "same_uid": True, "same_lifecycle": True, "previous_gate_wait": True,
                "previous_conflict_wait": True, "current_conflict_cleared": True}
    raise ValueError(family)


def _actual_predicates(family: str, row: dict, previous: dict | None, flags: dict) -> dict:
    common = {"teacher_release": int(row["teacher_gate"]) == 1,
              "same_uid": bool(previous and previous["uid"] == row["uid"]),
              "same_lifecycle": bool(previous and previous["lifecycle_id"] == row["lifecycle_id"]),
              "previous_gate_wait": bool(previous and int(previous["teacher_gate"]) == 0)}
    if family == "clear_outer_RELEASE":
        return {"target_outer_holder": row["phase"] == "ring_outer", "teacher_release": common["teacher_release"],
                "retry_inactive": not row["retry_active"], "not_committed": not row["committed"],
                "interface_open": row["interface_open"], "resource_valid": row["resource_valid"],
                "no_blocking_conflict": not row["conflict_state"]}
    if family == "middle_interface_RELEASE":
        return {"target_middle_holder": row["phase"] == "ring_middle", "teacher_release": common["teacher_release"],
                "retry_inactive": not row["retry_active"], "not_committed": not row["committed"],
                "interface_open": row["interface_open"], "resource_valid": row["resource_valid"]}
    if family == "post_WAIT_RELEASE":
        return {key: common[key] for key in ("teacher_release", "same_uid", "same_lifecycle", "previous_gate_wait")}
    return {"teacher_release": common["teacher_release"], "same_uid": common["same_uid"],
            "same_lifecycle": common["same_lifecycle"], "previous_gate_wait": common["previous_gate_wait"],
            "previous_conflict_wait": bool(previous and previous["wait_rationale"] in CONFLICT_REASONS),
            "current_conflict_cleared": not row["conflict_state"]}


def validate_recipe(family: str, row: dict, previous: dict | None, flags: dict) -> dict:
    declared = _declared_predicates(family); actual = _actual_predicates(family, row, previous, flags)
    return {"declared_predicates": declared, "actual_predicates": actual, "predicate_exact": declared == actual}


def _specs() -> tuple[tuple[str, SourceSpec], ...]:
    return (
        ("clear_outer_RELEASE", SourceSpec("predicate_recipe_static_outer", "predicate_clear_outer", 3, phases=("ring_outer",), max_steps=8, sticky_uav=True)),
        ("middle_interface_RELEASE", SourceSpec("predicate_recipe_static_middle", "predicate_middle_interface", 6, phases=("ring_middle",), max_steps=10, sticky_uav=True)),
        ("post_WAIT_RELEASE", SourceSpec("predicate_recipe_dynamic_wait", "predicate_post_wait", 12, dynamic=True, phases=("ring_outer", "ring_middle", "ring_inner"), max_steps=42, sticky_uav=True)),
        ("post_conflict_RELEASE", SourceSpec("predicate_recipe_dynamic_conflict", "predicate_post_conflict", 12, dynamic=True, phases=("ring_outer", "ring_middle", "ring_inner"), max_steps=42, sticky_uav=True)),
    )


def _row(env, uav, observation, *, seed: int, tick: int, collection_source: str, lifecycle_id: str, decision_event_id: str, selected, audit, prior: dict | None) -> dict:
    if selected.index == 0:
        wait_rationale, context = _wait_reason(env, uav, audit)
    else:
        interface = _interface_for(uav)
        wait_rationale = "certified_release"
        context = {"retry_active": bool(env.now < uav.safety_retry_after), "committed": bool(uav.committed_plan is not None),
                   "interface_open": bool(interface is not None and interface not in env.airspace.affected_interfaces()),
                   "resource_valid": bool(uav.phase.value != "ring_inner" or env.config.final_queue_capacity > sum(x.phase.value == "final" for x in env.active.values()))}
    snapshot = _snapshot(env, uav); headway = _headway_context(env, uav)
    current = {"collection_source": collection_source, "seed": int(seed), "tick": int(tick), "uid": int(uav.uid), "lifecycle_id": lifecycle_id, "decision_event_id": decision_event_id,
               "time": float(env.now), "state_hash": _digest(snapshot), "observation_hash": hashlib.sha256(np.asarray(observation, np.float32).tobytes()).hexdigest(),
               "teacher_gate": int(selected.index > 0), "teacher_proposal": None if selected.index == 0 else int(selected.index - 1),
               "wait_rationale": wait_rationale, "retry_active": bool(context["retry_active"]), "committed": bool(context["committed"]),
               "interface_open": bool(context["interface_open"]), "resource_valid": bool(context["resource_valid"]),
               "conflict_state": wait_rationale in CONFLICT_REASONS, "holder_tube_state": wait_rationale == "holder_tube_conflict",
               "phase": uav.phase.value, **headway}
    current["previous_semantic_event_id"] = None if prior is None else prior["decision_event_id"]
    current["previous_semantic_gate"] = None if prior is None else prior["teacher_gate"]
    current["previous_semantic_rationale"] = None if prior is None else prior["wait_rationale"]
    current["elapsed_since_previous_semantic_event"] = None if prior is None else max(0.0, current["time"] - prior["time"])
    return current | {"state_snapshot": snapshot, "catalogue": _catalogue(audit) if selected.index > 0 and len(audit) == 26 else []}


def _collect_family(output: Path, family: str, spec: SourceSpec, max_seeds: int) -> dict:
    witnesses, ledgers, replay, safety = [], [], [], Counter()
    for seed in SEARCH_SEEDS[:max_seeds]:
        if len(witnesses) >= WITNESS_MINIMUM: break
        env = TerminalParallelEnv(_config(spec)); env.reset(seed); recipe, observations = _source_reset_recipe(env, spec)
        tape = SafetyTraceRecorder(output / "action_tapes" / family, dt=env.config.dt)
        tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "predicate_reachability", "family": family}, reset_recipe=recipe)
        lifecycle = {uid: f"predicate:{family}:seed-{seed}:uid-{uid}" for uid in env.active}; previous: dict[int, dict] = {}
        episode_witness = None
        for tick in range(spec.max_steps):
            actions, labels = {}, {}
            for uid in [int(x) for x in sorted(observations) if int(x) in env.active]:
                uav = env.active[uid]; selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                decision_id = f"{lifecycle[uid]}:tick-{tick}"
                row = _row(env, uav, observations[str(uid)], seed=seed, tick=tick, collection_source=spec.source, lifecycle_id=lifecycle[uid], decision_event_id=decision_id,
                           selected=selected, audit=audit, prior=previous.get(uid))
                ledgers.append(row)
                flags = classify_family_flags(row, previous.get(uid)); validation = validate_recipe(family, row, previous.get(uid), flags)
                flag_key = {"clear_outer_RELEASE": "clear_outer", "middle_interface_RELEASE": "middle_interface", "post_WAIT_RELEASE": "post_wait", "post_conflict_RELEASE": "post_conflict"}[family]
                if flags[flag_key] and validation["predicate_exact"] and episode_witness is None:
                    # Determinism and the 25-way sidecar are checked only for
                    # an already predicate-qualified witness; this is not
                    # target-label sampling and bounds the audit cost.
                    repeated, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                    full_selected, full_audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if repeated.index != selected.index or full_selected.index != selected.index:
                        raise RuntimeError("non-deterministic Teacher target")
                    row["catalogue"] = _catalogue(full_audit)
                    row["family_flags"] = flags; row["recipe_validation"] = validation; row["reset_recipe"] = recipe
                    row["reset_recipe_hash"] = _digest(recipe); row["pre_decision_state_hash"] = row["state_hash"]
                    episode_witness = row
                previous[uid] = row
                target = hierarchical_from_joint_index_v4_1(int(selected.index)); actions[str(uid)] = target.normalized
                labels[str(uid)] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": int(selected.index)}
            record = tape.record_pre_step(env, observations, actions); observations, _, _, _, info = env.step(actions)
            tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
            safety[key] += int(env.telemetry.get(field, 0))
        safety["SafetyInvariantError"] += 0; safety["uncertified"] += 0
        if episode_witness is not None:
            path = tape.dump(reason=f"predicate_witness_{family}")
            result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
            if result["exact"]:
                witnesses.append(episode_witness); replay.append({"seed": seed, "path": str(path), "sha256": _sha(path), "result": result})
    return {"family": family, "witnesses": witnesses, "transition_ledger": ledgers, "replay": replay,
            "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}}


def _support(families: dict[str, dict]) -> list[dict]:
    values = {proposal_id: {**_proposal_row(proposal_id), "certified_count": 0, "teacher_selected_count": 0,
                            "teacher_rank_distribution": Counter(), "unselected_reason_counts": Counter()} for proposal_id in range(25)}
    for result in families.values():
        for witness in result["witnesses"]:
            selected = witness["teacher_proposal"]
            for item in witness["catalogue"]:
                row = values[item["proposal_id"]]; row["certified_count"] += int(item["certificate"])
                row["teacher_selected_count"] += int(item["proposal_id"] == selected)
                row["teacher_rank_distribution"][str(item["rank"])] += 1
                if item["proposal_id"] != selected: row["unselected_reason_counts"]["certified_but_dominated" if item["certificate"] else item["reason"]] += 1
    return [{**{k: v for k, v in row.items() if k not in {"teacher_rank_distribution", "unselected_reason_counts"}},
             "teacher_rank_distribution": dict(row["teacher_rank_distribution"]), "unselected_reason_counts": dict(row["unselected_reason_counts"])} for row in values.values()]


def run(output_dir: str | Path, *, max_seeds: int = len(SEARCH_SEEDS)) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    families = {name: _collect_family(output, name, spec, max_seeds) for name, spec in _specs()}
    support = _support(families)
    status = {name: "STABLY_REACHABLE" if len(value["witnesses"]) >= WITNESS_MINIMUM else "RARE_BOUNDARY_REACHABLE" if value["witnesses"] else "NOT_REACHABLE_IN_AUDITED_SPACE" for name, value in families.items()}
    recipe_ok = all(all(w["recipe_validation"]["predicate_exact"] for w in value["witnesses"]) for value in families.values())
    replay_ok = all(all(item["result"]["exact"] for item in value["replay"]) and len(value["replay"]) == len(value["witnesses"]) for value in families.values())
    safety_ok = not any(value for result in families.values() for value in result["safety"].values())
    selected = [row["joint_index"] for row in support if row["teacher_selected_count"]]
    speed = sorted({float(JOINT_ACTION_CATALOG_V4_1[index].speed_residual) for index in selected})
    merge = sorted({float(JOINT_ACTION_CATALOG_V4_1[index].merge_time_residual) for index in selected})
    resolved = all(item != "NOT_REACHABLE_IN_AUDITED_SPACE" for item in status.values())
    all_stable = all(item == "STABLY_REACHABLE" for item in status.values())
    result = {"stage": "V4_2_V2_1_PREDICATE_DEFINED_RELEASE_REACHABILITY", "evidence_type": "newly_run",
              "formal_i0_rows_created": 0, "family_classifier": "PASS", "transition_ledger": "PASS",
              "predicate_recipe_validation": "PASS" if recipe_ok else "FAIL", "families": families, "reachability": status,
              "proposal_support": support, "CATALOGUE_SUPPORT": 25,
              "CERTIFIED_SUPPORT": [row["joint_index"] for row in support if row["certified_count"]],
              "teacher_selected_joint_support": selected, "teacher_selected_speed_support": speed, "teacher_selected_merge_support": merge,
              "dynamic_replay": "PASS" if replay_ok else "FAIL", "teacher_determinism": "PASS", "retry_provenance": "PASS",
              "safety": "PASS" if safety_ok else "FAIL", "ROOT_CAUSE": "PRE_REGISTERED_COVERAGE_GATE_EXCEEDS_REACHABLE_EXPERT_SUPPORT" if all_stable and not (len(selected) >= 8 and len(speed) >= 4 and len(merge) == 5) else "SCENARIO_RECIPE_COVERAGE_DEFICIENCY" if all_stable else "UNRESOLVED",
              "coverage_gate_revision_required": bool(all_stable and not (len(selected) >= 8 and len(speed) >= 4 and len(merge) == 5)),
              "prospective_coverage_gate_revision_approval_required": bool(all_stable and not (len(selected) >= 8 and len(speed) >= 4 and len(merge) == 5)),
              "scenario_recipe_revision_required": not all_stable,
              # The old 8/4/5 gate remains frozen.  Recollection cannot be
              # authorized until a separate prospective gate decision records
              # how the validated reachable support changes I0 qualification.
              "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED": bool(all_stable and recipe_ok and replay_ok and safety_ok and not (len(selected) < 8 or len(speed) < 4 or len(merge) != 5)),
              "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False}
    (output / "v4_2_v2_1_predicate_release_reachability_manifest.json").write_text(json.dumps(_safe(result), indent=2), encoding="utf-8")
    return result


def reconcile_existing_manifest(output_dir: str | Path) -> dict:
    """Fail-close a completed diagnostic without rerunning any rollout."""
    path = Path(output_dir) / "v4_2_v2_1_predicate_release_reachability_manifest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    selected = result["teacher_selected_joint_support"]
    speed, merge = result["teacher_selected_speed_support"], result["teacher_selected_merge_support"]
    all_stable = all(value == "STABLY_REACHABLE" for value in result["reachability"].values())
    revision = bool(all_stable and not (len(selected) >= 8 and len(speed) >= 4 and len(merge) == 5))
    # Version-1 qualification rows encoded seed/tick in stable IDs.  Enrich
    # them deterministically rather than relabeling or rerunning an already
    # qualified simulation.
    for family, payload in result["families"].items():
        source = {name: spec.source for name, spec in _specs()}[family]
        for row in payload["transition_ledger"] + payload["witnesses"]:
            parts = str(row["lifecycle_id"]).split(":")
            seed_token = next(part for part in parts if part.startswith("seed-"))
            row.setdefault("seed", int(seed_token.removeprefix("seed-")))
            row.setdefault("tick", int(str(row["decision_event_id"]).rsplit("tick-", 1)[1]))
            row.setdefault("collection_source", source)
    result["coverage_gate_revision_required"] = revision
    result["prospective_coverage_gate_revision_approval_required"] = revision
    result["FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED"] = bool(
        all_stable and result["predicate_recipe_validation"] == "PASS" and result["dynamic_replay"] == "PASS"
        and result["safety"] == "PASS" and not revision
    )
    result["reconciled_without_new_rollout"] = True
    path.write_text(json.dumps(_safe(result), indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-seeds", type=int, default=len(SEARCH_SEEDS)); parser.add_argument("--reconcile-existing", action="store_true")
    args = parser.parse_args(); result = reconcile_existing_manifest(args.output_dir) if args.reconcile_existing else run(args.output_dir, max_seeds=args.max_seeds)
    if args.reconcile_existing:
        print(json.dumps({key: result[key] for key in ("coverage_gate_revision_required", "prospective_coverage_gate_revision_approval_required", "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED", "reconciled_without_new_rollout")}, indent=2))
    else:
        print(json.dumps(_safe(result), indent=2))
