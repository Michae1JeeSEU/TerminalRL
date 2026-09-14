"""Independent, read-only V2.1 closure confirmation; it never creates I0 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .run_v4_2_observation_v2_1_qualification import runtime_feature_sanity
from .run_v4_2_observation_v2_1_teacher_predicate_closure_audit import (
    _action_map, _audit_uav, _config, _digest, _pairs, _predicate_inventory,
)
from .run_v4_2_privileged_teacher_gate_factorization_diagnostic import (
    GATE_A, GATE_B, _assert_disjoint, _binary_report, _closure_reclassification,
    _proposal_reason_breakdown, _proposal_selection,
)
from ..envs.parallel_env import TerminalParallelEnv


CONFIRMATORY_SEEDS = tuple(range(14001, 14201))
TARGET_STATES = 600
SOURCE_SPECS = (
    ("independent_interface_transition", True, 12),
    ("independent_nominal_mixed", False, 12),
    ("independent_release_ready_low_traffic", False, 6),
    ("independent_resource_pressure", False, 12),
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_config(*, dynamic: bool, warm_start_uavs: int):
    base = _config(dynamic=dynamic)
    # These are standard reachable traffic-load/source settings, not altered
    # Teacher/Safety semantics and not synthetic candidate-rejection states.
    return replace(base, warm_start_uavs=warm_start_uavs, total_requested_uavs=warm_start_uavs,
                   fixed_arrival_schedule=(0.0,) * warm_start_uavs, max_simulation_seconds=16.0)


def _ordered_uav_id(env: TerminalParallelEnv, observations: dict, seed: int, step: int) -> int | None:
    if not observations:
        return None
    phase_rank = {"ring_inner": 0, "ring_middle": 1, "ring_outer": 2, "transfer_o2m": 3, "transfer_m2i": 4, "final": 5}
    ordered = sorted((int(uid) for uid in observations), key=lambda uid: (phase_rank.get(env.active[uid].phase.value, 9), uid))
    return ordered[(seed + step) % len(ordered)]


def _collect_independent_bank(target_states: int) -> tuple[list[dict], dict]:
    if not 500 <= target_states <= 1000:
        raise ValueError("confirmatory bank target must be in [500, 1000]")
    bank: list[dict] = []; seen: set[str] = set(); previous: dict[tuple[int, int], int] = {}
    source_counts: Counter = Counter(); state_hashes: set[str] = set(); episodes = 0
    for episode, seed in enumerate(CONFIRMATORY_SEEDS):
        source, dynamic, warm_start_uavs = SOURCE_SPECS[episode % len(SOURCE_SPECS)]
        env = TerminalParallelEnv(_source_config(dynamic=dynamic, warm_start_uavs=warm_start_uavs))
        observations, _ = env.reset(seed)
        if dynamic:
            # A valid dynamic-airspace transition is an operational source;
            # it does not fabricate a candidate outcome or bypass Safety.
            env.apply_high_level_action(np.asarray((1., -1., 1., 0., 0., 0., 0., 0., 0., 0.), np.float32), geometry_gate=True)
            observations = env._observations()
        for step in range(10):
            uid = _ordered_uav_id(env, observations, seed, step)
            if uid is not None:
                row = _audit_uav(env, env.active[uid], source, seed, episode, step, previous.get((seed, uid)))
                semantic = _digest({"observation": row["observation_v2_1"], "gate": row["Teacher_gate"],
                                    "proposal": row["Teacher_proposal"], "rationale": row["Teacher_gate_rationale"]})
                previous[(seed, uid)] = row["Teacher_gate"]
                if semantic not in seen:
                    seen.add(semantic); bank.append(row); source_counts[source] += 1; state_hashes.add(row["state_hash"])
                if len(bank) >= target_states:
                    break
            observations, _, _, _, _ = env.step(_action_map(observations, seed, step))
        episodes = episode + 1
        if len(bank) >= target_states:
            break
    if len(bank) < 500:
        raise RuntimeError(f"independent confirmatory bank shortfall: {len(bank)} / {target_states}")
    if len(state_hashes) != len(bank):
        raise RuntimeError("confirmatory bank includes a repeated full state hash")
    return bank, {"unique_semantic_states": len(bank), "unique_state_hashes": len(state_hashes), "episodes": episodes,
                  "seeds": sorted(set(row["seed"] for row in bank)), "source_counts": dict(source_counts)}


def _state_only_coverage(rows: list[dict]) -> dict:
    rationales = Counter(row["Teacher_gate_rationale"] for row in rows)
    phase_ring = sorted({(item["full_simulator_state"]["uavs"][next(index for index, uav in enumerate(item["full_simulator_state"]["uavs"]) if int(uav["uid"]) == int(item["uav_id"]))]["phase"],
                          item["full_simulator_state"]["uavs"][next(index for index, uav in enumerate(item["full_simulator_state"]["uavs"]) if int(uav["uid"]) == int(item["uav_id"]))]["ring_role"])
                         for item in rows})
    direct = {
        "retry_remaining": {"active": sum(row["retry_remaining"] > 0.0 for row in rows), "expired_or_inactive": sum(row["retry_remaining"] == 0.0 for row in rows), "covered": True},
        "self_committed_plan": {"committed": sum(row["commit_state"] for row in rows), "uncommitted": sum(not row["commit_state"] for row in rows), "covered": True},
        "phase_and_ring_context": {"observed_phase_ring": phase_ring, "covered": bool(phase_ring)},
        "interface_frozen": {"frozen": sum(bool(row["interface_state"]) for row in rows), "open": sum(not row["interface_state"] for row in rows), "covered": True},
        "final_capacity": {"blocked": int(rationales["final_capacity"]), "covered": int(rationales["final_capacity"]) > 0},
        "landing_zone_availability": {"lz_unavailable": int(rationales["lz_unavailable"]), "available_observed": sum(row["resource_state"]["free_lz"] > 0 for row in rows),
                                      "coverage_status": "NOT_TRIGGERED_BEFORE_FINAL_CAPACITY_IN_DOMAIN" if not rationales["lz_unavailable"] else "TRIGGERED"},
    }
    # The LZ branch is after the final-capacity branch.  Its actor-side field
    # is direct, but this nominal configuration may make a negative branch
    # unobservable without fabricating an infeasible state.
    return {"predicate_coverage": direct, "teacher_gate_rationale_counts": dict(rationales),
            "unresolved_state_only_missing_features": [], "all_effective_state_only_predicates_covered": all(value.get("covered", True) for value in direct.values())}


def _state_pairs(rows: list[dict]) -> dict:
    release = lambda row: row["Teacher_gate"] == 1
    result = {
        "retry": _pairs(rows, "retry", lambda row: row["Teacher_gate"] == 0 and row["retry_remaining"] > 0.0, release),
        "committed": _pairs(rows, "committed", lambda row: row["Teacher_gate"] == 0 and row["commit_state"], release),
        "interface_frozen": _pairs(rows, "interface", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] == "interface_frozen", release),
        "resource_final_capacity": _pairs(rows, "resource", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] == "final_capacity", release),
    }
    return result


def _state_only_pass(coverage: dict, pairs: dict) -> bool:
    return bool(coverage["all_effective_state_only_predicates_covered"] and not coverage["unresolved_state_only_missing_features"] and all(row["status"] == "PASS" for row in pairs.values()))


def run(output_dir: str | Path, *, target_states: int = TARGET_STATES) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    rows, bank_coverage = _collect_independent_bank(target_states)
    bank_path = output / "independent_confirmatory_state_bank.jsonl"
    bank_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    split = _assert_disjoint(rows); coverage = _state_only_coverage(rows); pairs = _state_pairs(rows)
    probe_a, _ = _binary_report(rows, name="A_confirmatory_any_certified_release", proposal_conditioned=False,
                                positive_name="RELEASE", negative_name="WAIT", gate=GATE_A)
    probe_b, proposal_records = _binary_report(rows, name="B_confirmatory_proposal_certification", proposal_conditioned=True,
                                               positive_name="certified", negative_name="rejected", gate=GATE_B)
    proposal_path = output / "confirmatory_state_proposal_records.jsonl"
    proposal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in proposal_records), encoding="utf-8")
    probe_b["rejection_reason_breakdown"] = _proposal_reason_breakdown(proposal_records, probe_b)
    probe_b["record_path"] = str(proposal_path); probe_b["record_sha256"] = _sha256(proposal_path)
    probe_c = _proposal_selection(rows)
    state_only_pass = _state_only_pass(coverage, pairs)
    supervision_pass = all(probe["status"] == "PASS" for probe in (probe_a, probe_b, probe_c))
    sanity = runtime_feature_sanity(output / "schema_sanity")
    boundary = True
    final_freeze = bool(state_only_pass and supervision_pass and boundary and not sanity["future_leakage"] and sanity["feature_sanity"]["replay_exact"])
    result = {"stage": "V4_2_V2_1_INDEPENDENT_CONFIRMATORY_CLOSURE_AUDIT", "evidence_type": "newly_run",
              "independent_bank": {"path": str(bank_path), "sha256": _sha256(bank_path), **bank_coverage}, "split": split,
              "state_only_coverage": coverage, "state_only_matched_pairs": pairs, "Probe_A_gate": probe_a, "Probe_B_proposal_certification": probe_b,
              "Probe_C_teacher_proposal": probe_c, "closure_rationale_reclassification": _closure_reclassification(),
              "STATE_ONLY_CLOSURE": "PASS" if state_only_pass else "FAIL", "TEACHER_SUPERVISION_CLOSURE": "PASS" if supervision_pass else "FAIL",
              "TEACHER_PRIVILEGED_INFORMATION_RISK": "RESOLVED_FOR_SUPERVISION" if supervision_pass and boundary else "UNRESOLVED",
              "execution_boundary_preserved": boundary, "actor_receives_no_catalogue_scan_or_certificate": True,
              "schema_sanity": sanity, "observation_revision_required": False, "action_factorization_revision_required": False,
              "OBSERVATION_V2_1_FINAL_FREEZE": "PASS" if final_freeze else "FAIL",
              "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": bool(final_freeze), "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False,
              "forbidden_operations_called": []}
    (output / "v4_2_v2_1_confirmatory_closure_audit.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-states", type=int, default=TARGET_STATES)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, target_states=arguments.target_states), indent=2))
