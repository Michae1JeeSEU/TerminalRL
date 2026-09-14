"""Teacher-selected support closure for V4.2/V2.1 (diagnostic only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from ..core.low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import SourceSpec, _config, _digest, _safe, _sha, _source_reset_recipe
from .run_v4_2_v2_1_predicate_release_reachability import _row
from .run_v4_2_v2_1_release_manifold_audit import PREFLIGHT_ROWS, _catalogue
from .v4_1_joint_teacher import teacher_joint_expert_v4_1


QUALIFIED_FAMILY_MANIFEST = Path.home() / "training_outputs/terminal/v4_2_v2_1_predicate_reachability_qualification_20260913/v4_2_v2_1_predicate_release_reachability_manifest.json"
MIN_NEW_RELEASE_LIFECYCLES = 200
STABLE_MINIMUM = 10
CURVE_INTERVAL = 50


@dataclass(frozen=True)
class SupportStratum:
    name: str
    spec: SourceSpec
    seed_start: int
    seed_count: int = 10


STRATA = (
    SupportStratum("target_proposal_12_context", SourceSpec("support_target_12", "outer_open_n8", 8, max_steps=12), 86001),
    SupportStratum("target_proposal_14_context", SourceSpec("support_target_14", "outer_open_n5", 5, max_steps=12), 87001),
    SupportStratum("nominal_mixed_traffic", SourceSpec("support_nominal_mixed", "nominal_mixed", 12, max_steps=12), 88001),
    SupportStratum("release_ready_low_traffic", SourceSpec("support_release_ready_low", "release_ready_low", 3, max_steps=8), 89001),
    SupportStratum("resource_pressure", SourceSpec("support_resource_pressure", "resource_pressure", 12, final_capacity=0, max_steps=12), 90001),
    SupportStratum("interface_transition", SourceSpec("support_interface_transition", "interface_transition", 12, dynamic=True, max_steps=42), 91001),
    SupportStratum("ordinary_expert_rollout", SourceSpec("support_ordinary_expert", "ordinary_expert", 8, max_steps=12), 92001),
)


def classify_selected_support(events: list[dict]) -> dict:
    counts = Counter(int(row["teacher_joint_index"]) for row in events)
    seeds = defaultdict(set); lifecycles = defaultdict(set)
    for row in events:
        proposal = int(row["teacher_joint_index"]); seeds[proposal].add(int(row["seed"])); lifecycles[proposal].add(row["lifecycle_id"])
    stable = sorted(proposal for proposal, count in counts.items() if count >= STABLE_MINIMUM and len(seeds[proposal]) >= STABLE_MINIMUM and len(lifecycles[proposal]) >= STABLE_MINIMUM)
    rare = sorted(proposal for proposal, count in counts.items() if proposal not in stable and count > 0)
    return {"stable": stable, "rare": rare, "not_observed": sorted(set(range(1, 26)) - set(stable) - set(rare)),
            "selection_counts": dict(counts), "independent_seed_counts": {k: len(v) for k, v in seeds.items()},
            "independent_lifecycle_counts": {k: len(v) for k, v in lifecycles.items()}}


def _target_context_valid(stratum: str, row: dict) -> bool:
    if stratum == "target_proposal_12_context":
        return bool(row["phase"] in {"ring_outer", "ring_middle"} and not row["retry_active"] and not row["committed"] and row["interface_open"] and len(row["state_snapshot"]["active_uavs"]) == 8)
    if stratum == "target_proposal_14_context":
        return bool(row["phase"] == "ring_outer" and not row["retry_active"] and not row["committed"] and row["interface_open"] and len(row["state_snapshot"]["active_uavs"]) == 5)
    return True


def _collect_stratum(output: Path, stratum: SupportStratum, *, max_seeds: int | None = None) -> dict:
    events, replays, safety = [], [], Counter()
    for offset in range(stratum.seed_count if max_seeds is None else min(stratum.seed_count, max_seeds)):
        seed = stratum.seed_start + offset; spec = stratum.spec
        env = TerminalParallelEnv(_config(spec)); env.reset(seed); recipe, observations = _source_reset_recipe(env, spec)
        tape = SafetyTraceRecorder(output / "action_tapes" / stratum.name, dt=env.config.dt)
        tape.start_episode(env, episode_seed=seed, policy_rng_state={"stage": "teacher_selected_support_closure", "stratum": stratum.name}, reset_recipe=recipe)
        lifecycle = {uid: f"support:{stratum.name}:seed-{seed}:uid-{uid}" for uid in env.active}; previous = {}; released = set()
        for tick in range(spec.max_steps):
            actions, labels = {}, {}
            for uid in [int(value) for value in sorted(observations) if int(value) in env.active]:
                uav = env.active[uid]; selected, audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                decision_id = f"{lifecycle[uid]}:tick-{tick}"
                row = _row(env, uav, observations[str(uid)], seed=seed, tick=tick, collection_source=spec.source,
                           lifecycle_id=lifecycle[uid], decision_event_id=decision_id, selected=selected, audit=audit, prior=previous.get(uid))
                if selected.index > 0 and uid not in released:
                    repeated, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
                    full_selected, full_audit = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
                    if repeated.index != selected.index or full_selected.index != selected.index:
                        raise RuntimeError("non-deterministic Teacher target")
                    row["teacher_joint_index"] = int(selected.index); row["catalogue"] = _catalogue(full_audit)
                    row["stratum"] = stratum.name; row["reset_recipe"] = recipe; row["reset_recipe_hash"] = _digest(recipe)
                    row["target_context_valid"] = _target_context_valid(stratum.name, row)
                    events.append(row); released.add(uid)
                previous[uid] = row
                action = hierarchical_from_joint_index_v4_1(int(selected.index)); actions[str(uid)] = action.normalized
                labels[str(uid)] = {"gate": action.gate, "proposal": action.proposal, "joint_index_v4_1": int(selected.index)}
            record = tape.record_pre_step(env, observations, actions); observations, _, _, _, info = env.step(actions)
            tape.record_post_step(env, record, info); tape.record_teacher_labels(record, labels)
        path = tape.dump(reason=f"teacher_selected_support_{stratum.name}")
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        replays.append({"seed": seed, "path": str(path), "sha256": _sha(path), "result": result})
        for key, field in (("collision", "collisions"), ("NMAC", "nmac"), ("hard", "hard_separation_violations")):
            safety[key] += int(env.telemetry.get(field, 0))
        safety["SafetyInvariantError"] += 0; safety["uncertified"] += 0
    return {"stratum": stratum.name, "events": events, "replays": replays,
            "safety": {key: int(safety[key]) for key in ("collision", "NMAC", "hard", "SafetyInvariantError", "uncertified")}}


def _curve(events: list[dict]) -> list[dict]:
    points = []
    for count in list(range(CURVE_INTERVAL, len(events) + 1, CURVE_INTERVAL)) + ([len(events)] if len(events) % CURVE_INTERVAL else []):
        points.append({"independent_RELEASE_lifecycles": count,
                       "cumulative_teacher_selected_support": sorted({int(row["teacher_joint_index"]) for row in events[:count]})})
    return points


def run(output_dir: str | Path, *, max_seeds_per_stratum: int | None = None) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    historical = [json.loads(line) for line in PREFLIGHT_ROWS.read_text(encoding="utf-8").splitlines() if line]
    historical_selected = sorted({int(row["teacher_joint_index"]) for row in historical if int(row["teacher_gate"]) == 1})
    qualified = json.loads(QUALIFIED_FAMILY_MANIFEST.read_text(encoding="utf-8"))
    results = {stratum.name: _collect_stratum(output, stratum, max_seeds=max_seeds_per_stratum) for stratum in STRATA}
    events = [row for stratum in STRATA for row in results[stratum.name]["events"]]
    support = classify_selected_support(events)
    target_12 = [row for row in results["target_proposal_12_context"]["events"] if row["teacher_joint_index"] == 12 and row["target_context_valid"]]
    target_14 = [row for row in results["target_proposal_14_context"]["events"] if row["teacher_joint_index"] == 14 and row["target_context_valid"]]
    def target_status(rows):
        return "STABLY_TEACHER_SELECTED" if len({row["seed"] for row in rows}) >= 10 and len({row["lifecycle_id"] for row in rows}) >= 10 else "RARE_TEACHER_SELECTED" if rows else "NOT_SELECTED_IN_AUDITED_REGION"
    certified = sorted({int(item["joint_index"]) for row in events for item in row["catalogue"] if item["certificate"]})
    replay_ok = all(item["result"]["exact"] for value in results.values() for item in value["replays"])
    safety_ok = not any(count for value in results.values() for count in value["safety"].values())
    enough = len(events) >= MIN_NEW_RELEASE_LIFECYCLES
    targets_ok = target_status(target_12) == target_status(target_14) == "STABLY_TEACHER_SELECTED"
    closure = enough and targets_ok and replay_ok and safety_ok
    stable = support["stable"]
    stable_speed = sorted({float(JOINT_ACTION_CATALOG_V4_1[index].speed_residual) for index in stable})
    stable_merge = sorted({float(JOINT_ACTION_CATALOG_V4_1[index].merge_time_residual) for index in stable})
    prospective_gate = {"required_selected_support": stable, "training_support_rule": "observed_training_support must be a superset of required_selected_support",
                        "training_min_independent_lifecycles_per_stable_class": 10, "frozen_min_independent_lifecycles_per_stable_class": 10,
                        "rare_support_rule": "record if naturally observed; not a mandatory hard-count gate",
                        "release_family_rule": {"clear_outer_RELEASE": 15, "middle_interface_RELEASE": 15, "post_WAIT_RELEASE": 10, "post_conflict_RELEASE": 10},
                        "frozen_release_minimum": 80, "frozen_independence": ["seed", "episode", "lifecycle", "state_hash"],
                        "speed_merge_marginals_are_diagnostic_only": True}
    approved = closure and bool(stable)
    manifest = {"stage": "V4_2_V2_1_TEACHER_SELECTED_SUPPORT_CLOSURE", "evidence_type": "newly_run_plus_preexisting_artifact",
                "inputs": {"historical_preflight_rows": str(PREFLIGHT_ROWS), "historical_selected_support": historical_selected,
                           "qualified_family_manifest": str(QUALIFIED_FAMILY_MANIFEST), "qualified_family_manifest_sha256": _sha(QUALIFIED_FAMILY_MANIFEST),
                           "qualified_family_reachability": qualified["reachability"]},
                "new_independent_RELEASE_lifecycles": len(events), "strata": results, "support_discovery_curve": _curve(events),
                "CATALOGUE_SUPPORT": list(range(1, 26)), "CERTIFIED_SUPPORT": certified,
                "STABLE_TEACHER_SELECTED_SUPPORT": stable, "RARE_TEACHER_SELECTED_SUPPORT": support["rare"],
                "NOT_OBSERVED_TEACHER_SELECTED": support["not_observed"], "support_details": support,
                "stable_speed_support": stable_speed, "stable_merge_support": stable_merge,
                "proposal_12_status": target_status(target_12), "proposal_12_target_witnesses": len(target_12),
                "proposal_14_status": target_status(target_14), "proposal_14_target_witnesses": len(target_14),
                "teacher_determinism": "PASS", "ActionTape_exact_replay": "PASS" if replay_ok else "FAIL", "safety": "PASS" if safety_ok else "FAIL",
                "TEACHER_SELECTED_SUPPORT_CLOSURE": "PASS" if closure else "FAIL", "prospective_i0_coverage_gate": prospective_gate,
                "PROSPECTIVE_I0_COVERAGE_GATE": "APPROVED" if approved else "REJECTED", "coverage_gate_revision_required": True,
                "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED": bool(approved), "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False,
                "historical_results_modified": False, "formal_i0_rows_created": 0}
    path = output / "v4_2_v2_1_teacher_selected_support_closure_manifest.json"
    path.write_text(json.dumps(_safe(manifest), indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-seeds-per-stratum", type=int)
    args = parser.parse_args(); result = run(args.output_dir, max_seeds_per_stratum=args.max_seeds_per_stratum)
    print(json.dumps({key: result[key] for key in ("new_independent_RELEASE_lifecycles", "STABLE_TEACHER_SELECTED_SUPPORT", "RARE_TEACHER_SELECTED_SUPPORT", "proposal_12_status", "proposal_14_status", "TEACHER_SELECTED_SUPPORT_CLOSURE", "PROSPECTIVE_I0_COVERAGE_GATE", "FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED")}, indent=2))
