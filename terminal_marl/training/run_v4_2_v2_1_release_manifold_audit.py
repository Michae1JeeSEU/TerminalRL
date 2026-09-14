"""Read-only V4.2/V2.1 RELEASE-manifold and reachable-support audit.

This is diagnostic-only: new rollouts are witnesses, never formal-I0 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from ..core.observations import build_low_observation_v2_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import (
    SourceSpec, _config, _digest, _safe, _sha, _snapshot, _source_reset_recipe,
)
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


PREFLIGHT_ROWS = Path.home() / "training_outputs/terminal/v4_2_v2_1_i0_collector_preflight_20260912/collector_preflight_rows.jsonl"
# A predeclared bounded nominal-state search. A zero witness is interpreted
# only as not reachable within this audited space, never as a global proof.
SEARCH_SEEDS = tuple(range(82001, 82016))
WITNESS_MINIMUM = 10
CONFLICT_WAIT_REASONS = {"geometry_frozen", "headway_conflict", "candidate_rejected", "trajectory_infeasible", "holder_reachable_tube"}
PREFERENCE_RANK = {item.index - 1: rank for rank, item in enumerate(
    sorted(JOINT_ACTION_CATALOG_V4_1, key=lambda row: (row.release_request <= 0.0,
          abs(float(row.speed_residual)), abs(float(row.merge_time_residual)), row.index))
) if item.index > 0}


def _proposal_row(proposal_id: int) -> dict:
    item = JOINT_ACTION_CATALOG_V4_1[int(proposal_id) + 1]
    return {"proposal_id": int(proposal_id), "joint_index": int(item.index),
            "speed_residual": float(item.speed_residual), "merge_residual": float(item.merge_time_residual)}


def _canonical_preference_catalogue() -> list[dict]:
    """Teacher's static catalogue order, recorded even for historical sidecars."""
    return [{**_proposal_row(proposal_id), "teacher_preference_rank": PREFERENCE_RANK[proposal_id]}
            for proposal_id in range(25)]


def _catalogue(audit: tuple[dict, ...]) -> list[dict]:
    """Join certificates and preference ranks by proposal ID, never list offset."""
    records = []
    for item in audit:
        if item["decoded"]["release_request"] <= 0.0:
            continue
        proposal_id = int(item["joint_index"] - 1)
        records.append({**_proposal_row(proposal_id), "rank": int(item["rank"]),
                        "certificate": bool(item["feasible"]), "reason": str(item["reason"]),
                        "teacher_preference": _safe(item["candidate_score"]),
                        "selected": False})
    if len(records) != 25 or len({row["proposal_id"] for row in records}) != 25:
        raise RuntimeError("audit catalogue must expose exactly one row per release proposal")
    return records


def _history_key(row: dict) -> tuple[str, int, int]:
    return str(row["source"]), int(row["seed"]), int(row["uav_id"])


def _reclassify_existing(rows: list[dict]) -> dict:
    """Audit all existing release lifecycles without treating them as training data."""
    rows = sorted(rows, key=lambda item: (str(item["source"]), int(item["seed"]), int(item["uav_id"]), int(item["raw_decision_tick"])))
    history: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    entries = []
    misclassified = overlap = 0
    detector_families = {"clear_outer_RELEASE", "middle_interface_RELEASE", "post_conflict_RELEASE", "post_WAIT_RELEASE"}
    physical_seen = Counter(); selected_support = Counter(); certified_support = Counter()
    for row in rows:
        key = _history_key(row); prior = list(history[key]); history[key].append(row)
        if int(row["teacher_gate"]) != 1:
            continue
        target = row["state_snapshot"]["target_uav"]
        phase = target["phase"]; ring = target["ring_role"]
        prior_waits = [item for item in prior if int(item["teacher_gate"]) == 0]
        wait_reasons = [str(item["teacher_rationale"]) for item in prior_waits]
        physical = set()
        if phase == "ring_outer": physical.add("clear_outer_RELEASE")
        if phase == "ring_middle": physical.add("middle_interface_RELEASE")
        if prior_waits: physical.add("post_WAIT_RELEASE")
        if any(reason in CONFLICT_WAIT_REASONS for reason in wait_reasons): physical.add("post_conflict_RELEASE")
        for name in physical: physical_seen[name] += 1
        detector = set()
        if row["source"] == "clear_release_opportunity": detector.add("clear_outer_RELEASE")
        if row["source"] == "middle_interface_release": detector.add("middle_interface_RELEASE")
        if row["source"] == "post_conflict_recovery" and int(row.get("previous_teacher_gate") or 1) == 0: detector.add("post_conflict_RELEASE")
        if row["source"] == "post_wait_accumulated_release" and int(row.get("previous_teacher_gate") or 1) == 0: detector.add("post_WAIT_RELEASE")
        if physical != detector:
            misclassified += 1
        if len(physical) > 1:
            overlap += 1
        selected_id = int(row["teacher_proposal"])
        selected_support[selected_id + 1] += 1
        catalogue = {int(item["proposal_id"]): item for item in row["teacher_catalogue_audit"]}
        for proposal_id, proposal in catalogue.items():
            if bool(proposal.get("final_certificate")):
                certified_support[proposal_id + 1] += 1
        selected = catalogue.get(selected_id)
        entries.append({"source": row["source"], "seed": row["seed"], "uav_id": row["uav_id"],
                        "lifecycle_event_id": row["lifecycle_event_id"], "phase_before_release": phase,
                        "ring": ring, "previous_semantic_decision": None if not prior else int(prior[-1]["teacher_gate"]),
                        "time_since_previous_WAIT": None if not prior_waits else float(row["state_snapshot"]["env_now"] - prior_waits[-1]["state_snapshot"]["env_now"]),
                        "retry_history": [bool(item["retry_active"]) for item in prior],
                        "interface": {"outer": "o2m", "middle": "m2i", "inner": "i2f"}.get(ring),
                        "headway_age": None, "conflict_state": any(reason in CONFLICT_WAIT_REASONS for reason in wait_reasons),
                        "recovery_state": bool(prior_waits), "detector_families": sorted(detector), "physical_families": sorted(physical),
                        "teacher_selected_proposal": selected_id, "selected_proposal": selected,
                        "all_25_teacher_preference_ranking": _canonical_preference_catalogue(),
                        "all_25_certificates": row["teacher_catalogue_audit"], "state_hash": row["full_state_hash"]})
    unreachable = sum(1 for name in detector_families if not physical_seen[name])
    return {"release_lifecycle_entries": entries, "family_detector_misclassification_count": misclassified,
            "family_definition_overlap_count": overlap, "family_definition_unreachable_count": unreachable,
            "physical_family_event_counts": dict(physical_seen),
            # These are support observations over all 109 historical
            # preflight RELEASE lifecycles, keyed by canonical joint index.
            # They are not a claim that every action is simultaneously
            # certified in every state.
            "preexisting_teacher_selected_joint_counts": dict(selected_support),
            "preexisting_certified_joint_counts": dict(certified_support)}


def _scenario_specs() -> tuple[tuple[str, SourceSpec], ...]:
    return (
        ("clear_outer_RELEASE", SourceSpec("audit_clear_outer", "clear_outer_RELEASE", 3, phases=("ring_outer",), max_steps=1)),
        ("middle_interface_RELEASE", SourceSpec("audit_middle_interface", "middle_interface_RELEASE", 6, phases=("ring_middle",), max_steps=1)),
        # Dynamic transition witnesses deliberately evaluate the same naturally
        # selected UAV before and after the frozen-interface transition. Empty
        # intermediate env steps advance only legal simulator time.
        ("post_conflict_RELEASE", SourceSpec("audit_post_conflict", "post_conflict_RELEASE", 12, dynamic=True, phases=("ring_outer", "ring_middle", "ring_inner"), max_steps=33)),
        ("post_WAIT_RELEASE", SourceSpec("audit_post_wait", "post_WAIT_RELEASE", 12, dynamic=True, phases=("ring_outer", "ring_middle", "ring_inner"), max_steps=33)),
    )


def _search(output: Path, *, max_seeds: int) -> dict:
    witnesses: dict[str, list[dict]] = defaultdict(list); support: dict[int, dict] = {item.index - 1: {**_proposal_row(item.index - 1), "certified_count": 0, "teacher_selected_count": 0, "best_score_count": 0, "rank_distribution": Counter(), "reasons": Counter()} for item in JOINT_ACTION_CATALOG_V4_1[1:]}
    replay_reports, safety = [], Counter()
    for family, spec in _scenario_specs():
        for seed in SEARCH_SEEDS[:max_seeds]:
            if len({item["seed"] for item in witnesses[family]}) >= WITNESS_MINIMUM:
                break
            env = TerminalParallelEnv(_config(spec)); env.reset(seed); recipe, observations = _source_reset_recipe(env, spec)
            tape = SafetyTraceRecorder(output / "witness_tapes" / family, dt=env.config.dt)
            tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "release_manifold_audit", "family": family}, reset_recipe=recipe)
            history: dict[int, list[dict]] = defaultdict(list); episode_witness = False
            for tick in range(spec.max_steps):
                actions = {}; labels = {}; release_rows = []
                evaluate = tick == 0 or (spec.dynamic and tick in {31, 32})
                candidates = [uid for uid in sorted(observations) if int(uid) in env.active and env.active[int(uid)].phase.value in spec.phases]
                if not candidates:
                    candidates = [uid for uid in sorted(observations) if int(uid) in env.active]
                # Deterministic seed/tick rotation is independent of the
                # Teacher result. One candidate per tick is sufficient for a
                # state-search witness and bounds the 25-proposal audit cost.
                selected_uids = [] if not candidates or not evaluate else [candidates[seed % len(candidates)]]
                for uid in selected_uids:
                    uav = env.active[int(uid)]
                    # A WAIT result means no RELEASE proposal was certified:
                    # obtain all 25 certificates only for a naturally selected
                    # RELEASE state, where support evidence is informative.
                    selected, wait_audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=(family == "post_conflict_RELEASE"))
                    if selected.index > 0:
                        selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                        catalogue = _catalogue(audit)
                    else:
                        catalogue = []
                    for item in catalogue:
                        item["selected"] = item["proposal_id"] == int(selected.index - 1)
                        item_support = support[item["proposal_id"]]
                        item_support["certified_count"] += int(item["certificate"])
                        item_support["teacher_selected_count"] += int(item["selected"])
                        item_support["best_score_count"] += int(item["selected"])
                        item_support["rank_distribution"][str(item["rank"])] += 1
                        if not item["selected"]: item_support["reasons"]["certified_but_dominated" if item["certificate"] else item["reason"]] += 1
                    prior = list(history[int(uid)]); prior_waits = [item for item in prior if item["gate"] == 0]
                    wait_reasons = [item["rationale"] for item in prior_waits]
                    physical = set()
                    if uav.phase.value == "ring_outer": physical.add("clear_outer_RELEASE")
                    if uav.phase.value == "ring_middle": physical.add("middle_interface_RELEASE")
                    if prior_waits: physical.add("post_WAIT_RELEASE")
                    if any(reason in CONFLICT_WAIT_REASONS for reason in wait_reasons): physical.add("post_conflict_RELEASE")
                    rationale = ("certified_release" if selected.index else
                                 next((str(item["reason"]) for item in wait_audit if str(item["reason"]) != "wait_legal"), "wait"))
                    history[int(uid)].append({"gate": int(selected.index > 0), "time": float(env.now), "rationale": rationale})
                    if selected.index > 0 and family in physical and len({item["seed"] for item in witnesses[family]}) < WITNESS_MINIMUM:
                        snapshot = _snapshot(env, uav); observation = build_low_observation_v2_1(uav, env)
                        release_rows.append({"family": family, "seed": seed, "tick": tick, "uav_id": int(uid),
                                             "lifecycle_key": f"{family}:seed-{seed}:uav-{uid}", "full_state_hash": _digest(snapshot),
                                             "observation_hash": hashlib.sha256(observation.tobytes()).hexdigest(), "state_snapshot": snapshot,
                                             "reset_recipe": recipe, "reset_recipe_hash": _digest(recipe), "teacher_selected_proposal": int(selected.index - 1),
                                             "all_25_proposals": catalogue, "physical_families": sorted(physical),
                                             "prior_wait_reasons": wait_reasons})
                    target = hierarchical_from_joint_index_v4_1(int(selected.index)); actions[uid] = target.normalized
                    labels[uid] = {"gate": target.gate, "proposal": target.proposal, "joint_index_v4_1": int(selected.index)}
                record = tape.record_pre_step(env, observations, actions); observations, _, _, _, info = env.step(actions)
                tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
                for item in release_rows:
                    if seed not in {row["seed"] for row in witnesses[family]}:
                        witnesses[family].append(item); episode_witness = True
            for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
                safety[key] += int(env.telemetry.get(field, 0))
            if episode_witness:
                path = tape.dump(reason=f"release_manifold_witness_{family}")
                replay_reports.append({"family": family, "seed": seed, "path": str(path), "sha256": _sha(path),
                                       "result": SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)})
    normalized_support = []
    for proposal_id, item in sorted(support.items()):
        normalized_support.append({**{key: value for key, value in item.items() if key not in {"rank_distribution", "reasons"}},
                                   "rank_distribution": dict(item["rank_distribution"]), "unselected_reason_counts": dict(item["reasons"])})
    return {"witness_minimum_per_family": WITNESS_MINIMUM, "witnesses": dict(witnesses), "proposal_support": normalized_support,
            "replay": replay_reports, "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard")}}


def run(output_dir: str | Path, *, max_seeds: int = len(SEARCH_SEEDS)) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in PREFLIGHT_ROWS.read_text(encoding="utf-8").splitlines() if line]
    existing = _reclassify_existing(rows)
    search = _search(output, max_seeds=max_seeds)
    support = search["proposal_support"]
    # The 109 read-only preflight RELEASE lifecycles are the complete existing
    # support evidence.  A fresh targeted search with no RELEASE witness is a
    # recipe failure, not evidence that its support is empty.
    certified = sorted(int(index) for index in existing["preexisting_certified_joint_counts"])
    selected = sorted(int(index) for index in existing["preexisting_teacher_selected_joint_counts"])
    def values(indices, key): return sorted({float(getattr(JOINT_ACTION_CATALOG_V4_1[index], key)) for index in indices})
    reachability = {}
    for family, _ in _scenario_specs():
        count = len({item["seed"] for item in search["witnesses"].get(family, [])})
        reachability[family] = "STABLY_REACHABLE" if count >= WITNESS_MINIMUM else "RARE_BOUNDARY" if count else "NOT_REACHABLE"
    selected_diverse = len(selected) >= 8 and len(values(selected, "speed_residual")) >= 4 and len(values(selected, "merge_time_residual")) >= 5
    taxonomy_error = existing["family_detector_misclassification_count"] > 0
    targeted_witness_count = sum(len(rows) for rows in search["witnesses"].values())
    # A zero-witness probe cannot lower a coverage gate while the historical
    # bank demonstrably contains RELEASE states.  It first diagnoses the
    # declared recipe/family construction.  Source-vs-physical disagreement
    # takes priority because it invalidates the family count itself.
    root = ("COLLECTOR_SEMANTIC_CLASSIFICATION_ERROR" if taxonomy_error else
            "SCENARIO_RECIPE_COVERAGE_DEFICIENCY" if targeted_witness_count == 0 else
            "PRE_REGISTERED_I0_COVERAGE_GATE_EXCEEDS_REACHABLE_EXPERT_SUPPORT" if not selected_diverse else "UNRESOLVED")
    manifest = {"stage": "V4_2_V2_1_RELEASE_MANIFOLD_REACHABLE_ACTION_AUDIT", "evidence_type": "newly_run",
                "preflight_input": str(PREFLIGHT_ROWS), "preflight_rows_read": len(rows), "existing_release_reclassification": existing,
                "nominal_legal_search_variables": ["initial UAV spacing", "ring progress / relative progress", "occupancy", "active N", "headway age", "arrival timing", "reservation context", "retry state", "lifecycle phase", "conflict/recovery state"],
                "search": search, "reachability": reachability, "catalogue_joint_support": 25, "certified_joint_support": certified,
                "teacher_selected_joint_support": selected, "certified_speed_support": values(certified, "speed_residual"),
                "teacher_selected_speed_support": values(selected, "speed_residual"), "certified_merge_support": values(certified, "merge_time_residual"),
                "teacher_selected_merge_support": values(selected, "merge_time_residual"), "ROOT_CAUSE": root,
                "targeted_witness_count": targeted_witness_count,
                "coverage_gate_revision_required": False,
                "coverage_gate_decision": "PENDING_VALID_REACHABILITY_EVIDENCE",
                "scenario_recipe_revision_required": targeted_witness_count == 0 or not all(value == "STABLY_REACHABLE" for value in reachability.values()),
                "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED": False, "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False}
    (output / "v4_2_v2_1_release_manifold_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-seeds", type=int, default=len(SEARCH_SEEDS))
    args = parser.parse_args(); print(json.dumps(run(args.output_dir, max_seeds=args.max_seeds), indent=2))
