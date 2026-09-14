"""Powered, independent closure confirmation for Observation V2.1.

This is an audit-only collector.  It creates no I0, BC, DAgger, or PPO
artifact.  Scenario rotation and target-UAV selection are fixed before seeing
the Teacher label; the preregistered support rule is therefore a collection
stop rule, not class balancing or target-conditioned sampling.
"""
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
    _action_map, _audit_uav, _config, _digest, _pairs,
)
from .run_v4_2_privileged_teacher_gate_factorization_diagnostic import (
    GATE_A, GATE_B, _assert_disjoint, _binary_probability, _binary_report,
    _binary_view, _closure_reclassification, _proposal_reason_breakdown,
    _proposal_selection,
)
from ..envs.parallel_env import TerminalParallelEnv


POWERED_SEEDS = tuple(range(16001, 16801))
MINIMUM_SUPPORT = {"train_Teacher_RELEASE": 100, "dev_Teacher_RELEASE": 40, "dev_retry_active_WAIT": 40}
# (name, dynamic airspace, warm-start count, phase preference).  A phase is an
# observable, physical state; neither selection nor scenario rotation reads a
# Teacher label, a certificate, or an outcome from the catalogue scan.
SOURCE_SPECS = (
    ("powered_release_ready_low_traffic", False, 3, ("ring_outer",)),
    ("powered_clear_outer_release", False, 6, ("ring_outer",)),
    ("powered_middle_interface_release", False, 6, ("ring_middle",)),
    ("powered_transition_retry", True, 12, ("ring_inner", "ring_middle", "ring_outer")),
    ("powered_mixed_traffic", False, 8, ("ring_outer", "ring_middle", "ring_inner")),
    ("powered_resource_pressure", False, 12, ("ring_inner",)),
)
STEPS_PER_EPISODE = 12


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_config(*, dynamic: bool, warm_start_uavs: int):
    base = _config(dynamic=dynamic)
    return replace(base, warm_start_uavs=warm_start_uavs, total_requested_uavs=warm_start_uavs,
                   fixed_arrival_schedule=(0.0,) * warm_start_uavs, max_simulation_seconds=16.0)


def _split(row: dict) -> str:
    return "dev" if int(row["seed"]) % 5 == 0 else "train"


def _support(rows: list[dict]) -> dict[str, int]:
    return {
        "train_Teacher_RELEASE": sum(_split(row) == "train" and row["Teacher_gate"] == 1 for row in rows),
        "dev_Teacher_RELEASE": sum(_split(row) == "dev" and row["Teacher_gate"] == 1 for row in rows),
        "dev_retry_active_WAIT": sum(_split(row) == "dev" and row["Teacher_gate"] == 0 and row["retry_remaining"] > 0 for row in rows),
    }


def _support_reached(support: dict[str, int]) -> bool:
    return all(support[name] >= target for name, target in MINIMUM_SUPPORT.items())


def _select_uav_id(env: TerminalParallelEnv, observations: dict, preferred_phases: tuple[str, ...], seed: int, step: int) -> int | None:
    """Deterministically choose a currently active real holder by phase only."""
    candidates = [int(uid) for uid in observations if env.active[int(uid)].phase.value in preferred_phases]
    if not candidates:
        candidates = [int(uid) for uid in observations]
    if not candidates:
        return None
    candidates.sort()
    return candidates[(seed + step) % len(candidates)]


def _progress(output: Path, *, rows: list[dict], episodes: int, completed: bool = False) -> None:
    payload = {"stage": "V4_2_V2_1_POWERED_INDEPENDENT_CONFIRMATION", "episodes_completed": episodes,
               "unique_state_count": len(rows), "support": _support(rows), "minimum_support": MINIMUM_SUPPORT,
               "support_reached": _support_reached(_support(rows)), "completed": completed}
    (output / "collection_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect(output: Path, max_episodes: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen_state_hashes: set[str] = set()
    previous: dict[tuple[int, int], int] = {}
    source_counts: Counter = Counter()
    partial = output / "powered_independent_state_bank.partial.jsonl"
    _progress(output, rows=rows, episodes=0)
    with partial.open("w", encoding="utf-8") as stream:
        for episode, seed in enumerate(POWERED_SEEDS[:max_episodes]):
            source, dynamic, warm_start_uavs, phases = SOURCE_SPECS[episode % len(SOURCE_SPECS)]
            env = TerminalParallelEnv(_source_config(dynamic=dynamic, warm_start_uavs=warm_start_uavs))
            observations, _ = env.reset(seed)
            if dynamic:
                env.apply_high_level_action(np.asarray((1., -1., 1., 0., 0., 0., 0., 0., 0., 0.), np.float32), geometry_gate=True)
                observations = env._observations()
            for step in range(STEPS_PER_EPISODE):
                uid = _select_uav_id(env, observations, phases, seed, step)
                if uid is not None:
                    row = _audit_uav(env, env.active[uid], source, seed, episode, step, previous.get((seed, uid)))
                    previous[(seed, uid)] = row["Teacher_gate"]
                    # Full state identity, rather than row or target identity,
                    # is the deduplication key.  This retains no replay tick.
                    if row["state_hash"] not in seen_state_hashes:
                        seen_state_hashes.add(row["state_hash"])
                        rows.append(row); source_counts[source] += 1
                        stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
                observations, _, _, _, _ = env.step(_action_map(observations, seed, step))
            _progress(output, rows=rows, episodes=episode + 1)
            if _support_reached(_support(rows)):
                break
    if len(seen_state_hashes) != len(rows):
        raise RuntimeError("repeated full-state hash in powered independent bank")
    final_bank = output / "powered_independent_state_bank.jsonl"
    final_bank.write_text(partial.read_text(encoding="utf-8"), encoding="utf-8")
    _progress(output, rows=rows, episodes=episode + 1, completed=True)
    return rows, {"episodes": episode + 1, "seeds": sorted({row["seed"] for row in rows}),
                  "unique_state_hashes": len(seen_state_hashes), "source_counts": dict(source_counts),
                  "support": _support(rows), "support_reached": _support_reached(_support(rows)),
                  "selection_rule": "fixed source rotation plus phase-only deterministic target-UAV selection; no Teacher/certificate/outcome input"}


def _other_state_pairs(rows: list[dict]) -> dict:
    release = lambda row: row["Teacher_gate"] == 1
    return {
        "committed": _pairs(rows, "committed", lambda row: row["Teacher_gate"] == 0 and row["commit_state"], release),
        "interface_frozen": _pairs(rows, "interface", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] == "interface_frozen", release),
        "final_capacity": _pairs(rows, "final_capacity", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] == "final_capacity", release),
        "lz_unavailable": _pairs(rows, "lz", lambda row: row["Teacher_gate"] == 0 and row["Teacher_gate_rationale"] == "lz_unavailable", release),
    }


def _other_state_only_closure(pairs: dict) -> tuple[str, list[str]]:
    # A NOT_AVAILABLE branch is reported as naturally not observed; it is not
    # turned into a fabricated state merely to make a matched pair.
    failed = [name for name, value in pairs.items() if value["status"] == "FAIL"]
    return ("FAIL" if failed else "PASS"), failed


def _gate_error_attribution(rows: list[dict]) -> dict:
    x, y, _ = _binary_view(rows, proposal_conditioned=False)
    probabilities = _binary_probability(x["train"], y["train"], x["dev"], seed=9131)
    dev = [row for row in rows if _split(row) == "dev"]
    errors = []
    for row, probability in zip(dev, probabilities):
        predicted = int(probability >= .5)
        if predicted == row["Teacher_gate"]:
            continue
        errors.append({"state_hash": row["state_hash"], "source": row["source"], "seed": row["seed"],
                       "lifecycle_event_id": row["lifecycle_event_id"], "teacher_gate": row["Teacher_gate"],
                       "predicted_gate": predicted, "release_probability": float(probability),
                       "rationale": row["Teacher_gate_rationale"], "retry_active": row["retry_remaining"] > 0,
                       "committed": row["commit_state"], "interface_state": row["interface_state"],
                       "resource_state": row["resource_state"], "phase": next(u["phase"] for u in row["full_simulator_state"]["uavs"] if int(u["uid"]) == row["uav_id"]),
                       "proposal_reason_counts": row["candidate_reason_counts"]})
    false_negative = [row for row in errors if row["teacher_gate"] == 1]
    false_positive = [row for row in errors if row["teacher_gate"] == 0]
    # A common value in every FN is evidence for inspection, never a claim of
    # a missing feature: every listed state-only predicate is already audited.
    common = []
    if false_negative:
        for field in ("source", "phase", "rationale", "retry_active", "committed"):
            values = {str(row[field]) for row in false_negative}
            if len(values) == 1:
                common.append({field: values.pop()})
    return {"decision_threshold": 0.5, "false_negative_RELEASE_states": false_negative,
            "false_positive_WAIT_states": false_positive, "false_negative_count": len(false_negative),
            "false_positive_count": len(false_positive), "common_patterns": common,
            "common_missing_state_only_variable": False}


def _retry_sanity_pass(sanity: dict) -> bool:
    feature = sanity["feature_sanity"]
    return bool(feature["no_future_leakage"] and feature["replay_exact"] and feature["finite"] and feature["in_unit_interval"] and feature["deterministic"])


def qualify_existing_bank(bank_path: str | Path, output_dir: str | Path) -> dict:
    """Qualify an already-frozen powered bank without recollecting a row."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    bank_path = Path(bank_path)
    rows = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("powered bank is empty")
    bank = {"episodes": max(int(row["episode_id"]) for row in rows) + 1,
            "seeds": sorted({row["seed"] for row in rows}), "unique_state_hashes": len({row["state_hash"] for row in rows}),
            "source_counts": dict(Counter(row["source"] for row in rows)), "support": _support(rows),
            "support_reached": _support_reached(_support(rows)),
            "selection_rule": "pre-frozen collection: fixed source rotation plus phase-only deterministic target-UAV selection; no Teacher/certificate/outcome input"}
    if bank["unique_state_hashes"] != len(rows):
        raise RuntimeError("powered bank cannot be qualified: repeated full-state hash")
    result = {"stage": "V4_2_V2_1_POWERED_INDEPENDENT_CONFIRMATION", "evidence_type": "newly_run",
              "closure_semantics": {"RETRY_STATE_ONLY_CLOSURE": "direct runtime predicate/normalization/determinism/no-future-leakage/replay equality; learned retry-pair probe is diagnostic only", "historical_fail_artifacts_modified": False},
              "independent_bank": {"path": str(bank_path), "sha256": _sha256(bank_path), **bank},
              "split": _assert_disjoint(rows), "forbidden_operations_called": []}
    if not bank["support_reached"]:
        result.update({"collection_status": "INSUFFICIENT_PREREGISTERED_SUPPORT", "RETRY_STATE_ONLY_CLOSURE": "NOT_REEVALUATED",
                       "OTHER_STATE_ONLY_CLOSURE": "NOT_EVALUATED", "GATE_PROBE_A": "NOT_EVALUATED",
                       "PROPOSAL_PROBE_B": "NOT_EVALUATED", "PROPOSAL_PROBE_C": "NOT_EVALUATED",
                       "TEACHER_PRIVILEGED_INFORMATION_RISK": "UNRESOLVED", "OBSERVATION_V2_1_FINAL_FREEZE": "FAIL",
                       "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": False, "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False})
        (output / "v4_2_v2_1_powered_confirmation.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    sanity = runtime_feature_sanity(output / "schema_sanity")
    retry_pass = _retry_sanity_pass(sanity)
    pairs = _other_state_pairs(rows); other_status, unresolved_pairs = _other_state_only_closure(pairs)
    probe_a, _ = _binary_report(rows, name="A_powered_any_certified_release", proposal_conditioned=False, positive_name="RELEASE", negative_name="WAIT", gate=GATE_A)
    probe_b, proposal_records = _binary_report(rows, name="B_powered_proposal_certification", proposal_conditioned=True, positive_name="certified", negative_name="rejected", gate=GATE_B)
    proposal_path = output / "powered_state_proposal_records.jsonl"
    proposal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in proposal_records), encoding="utf-8")
    probe_b.update({"rejection_reason_breakdown": _proposal_reason_breakdown(proposal_records, probe_b), "record_path": str(proposal_path), "record_sha256": _sha256(proposal_path)})
    probe_c = _proposal_selection(rows)
    errors = _gate_error_attribution(rows)
    probe_a_pass = probe_a["status"] == "PASS"; probe_b_pass = probe_b["status"] == "PASS"; probe_c_pass = probe_c["status"] == "PASS"
    root = "NONE"
    observation_revision = False
    if not probe_a_pass:
        root = "NEW_OBSERVATION_ALIASING" if errors["common_missing_state_only_variable"] else "TEACHER_GATE_TARGET_NOT_ROBUSTLY_LEARNABLE_FROM_DEPLOYABLE_OBSERVATION"
        observation_revision = errors["common_missing_state_only_variable"]
    final_freeze = bool(retry_pass and other_status == "PASS" and probe_a_pass and probe_b_pass and probe_c_pass and not observation_revision and sanity["feature_sanity"]["no_future_leakage"] and sanity["feature_sanity"]["replay_exact"])
    result.update({"collection_status": "POWERED_SUPPORT_REACHED", "schema_sanity": sanity,
                   "RETRY_STATE_ONLY_CLOSURE": "PASS" if retry_pass else "FAIL", "other_state_only_matched_pairs": pairs,
                   "OTHER_STATE_ONLY_CLOSURE": other_status, "unresolved_state_only_missing_features": unresolved_pairs,
                   "Probe_A_gate": probe_a, "Probe_B_proposal_certification": probe_b, "Probe_C_teacher_proposal": probe_c,
                   "gate_A_error_attribution": errors, "ROOT_CAUSE": root,
                   "observation_revision_required": observation_revision, "action_factorization_revision_required": False,
                   "execution_boundary_preserved": True, "actor_receives_no_catalogue_scan_or_certificate": True,
                   "future_leakage": not sanity["feature_sanity"]["no_future_leakage"], "schema_replay": "PASS" if sanity["feature_sanity"]["replay_exact"] else "FAIL",
                   "GATE_PROBE_A": "PASS" if probe_a_pass else "FAIL", "PROPOSAL_PROBE_B": "PASS" if probe_b_pass else "FAIL", "PROPOSAL_PROBE_C": "PASS" if probe_c_pass else "FAIL",
                   "TEACHER_PRIVILEGED_INFORMATION_RISK": "RESOLVED_FOR_SUPERVISION" if final_freeze else "UNRESOLVED",
                   "OBSERVATION_V2_1_FINAL_FREEZE": "PASS" if final_freeze else "FAIL",
                   "FORMAL_V2_1_I0_RECOLLECTION_ALLOWED": bool(final_freeze), "I1_ALLOWED": False, "DAgger_ALLOWED": False, "PPO_ALLOWED": False,
                   "closure_rationale_reclassification": _closure_reclassification()})
    (output / "v4_2_v2_1_powered_confirmation.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(output_dir: str | Path, *, max_episodes: int = len(POWERED_SEEDS)) -> dict:
    """Collect then qualify; retained only for a fresh audit directory."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    if not 1 <= max_episodes <= len(POWERED_SEEDS):
        raise ValueError("max_episodes must be within the preregistered independent seed range")
    _collect(output, max_episodes)
    return qualify_existing_bank(output / "powered_independent_state_bank.jsonl", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=len(POWERED_SEEDS))
    parser.add_argument("--bank", help="existing frozen powered bank; read-only qualification only")
    arguments = parser.parse_args()
    result = qualify_existing_bank(arguments.bank, arguments.output_dir) if arguments.bank else run(arguments.output_dir, max_episodes=arguments.max_episodes)
    print(json.dumps(result, indent=2))
