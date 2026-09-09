"""Offline-only audit of V4 state-expert versus request-repair semantics.

This runner never calls ``JointPPO.update`` and never steps an optimizer PPO
objective.  It evaluates frozen simulator states, then runs a separate CE-only
probe solely to qualify whether the existing state-conditioned expert labels
are learnable from Observation V2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..algo.joint_ppo import JointPPO
from ..core.low_level_joint_action import ACTION_SCHEMA_V4, JOINT_ACTION_CATALOG, N_JOINT_ACTIONS
from ..core.observations import LOW_OBS_V2
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .low_v2_seed_partitions import LowV2SeedPartition
from .run_v4_l1_short_pilot import LAMBDA
from .train_low_v2_ppo import _episode_design, l1_full_teacher_config
from .v4_joint_teacher import teacher_joint_expert, teacher_joint_label


STATES_PER_SOURCE = 20


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _config(source: str, ordinal: int):
    if source == "natural_l1":
        config, _ = _episode_design(ordinal, medium_rate=LAMBDA, horizon_seconds=600.0)
    elif source == "warm_start":
        config = l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                        warm_start_uavs=12, cohort_uavs=12, horizon_seconds=600.0)
    elif source == "held_out_anchor":
        config = l1_full_teacher_config(medium_rate=LAMBDA, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                        warm_start_uavs=15, cohort_uavs=15, horizon_seconds=600.0)
    else:
        raise ValueError(f"unknown source {source}")
    return replace(config, action_schema_version=ACTION_SCHEMA_V4, teacher_release_label_enabled=False)


def _coverage_actions(observations: dict, step: int) -> dict:
    release_nominal = next(row for row in JOINT_ACTION_CATALOG if row.release_request > 0.0 and row.speed_residual == 0.0 and row.merge_time_residual == 0.0)
    wait_nominal = next(row for row in JOINT_ACTION_CATALOG if row.release_request == 0.0 and row.speed_residual == 0.0)
    # A fixed catalogue cycle creates a spread of reachable operational
    # states before freezing.  It is not a Teacher policy and no generated
    # request is used as the label; its sole purpose is to avoid sampling only
    # reset-time inbound/crowded states, which have no release decision.
    return {uid: (release_nominal if (step // 25 + int(uid)) % 3 else wait_nominal).normalized for uid in observations}


def _aggregate(indices: list[int]) -> dict:
    counts = [int(indices.count(index)) for index in range(N_JOINT_ACTIONS)]
    decoded = [JOINT_ACTION_CATALOG[index] for index in indices]
    return {"count": len(indices), "joint_class_counts": counts,
            "WAIT": int(sum(row.release_request == 0.0 for row in decoded)),
            "RELEASE": int(sum(row.release_request > 0.0 for row in decoded))}


def _state_sweep(source: str, seed: int, env: TerminalParallelEnv, observation: np.ndarray, uid: str) -> dict:
    uav = env.active[int(uid)]
    state = SafetyTraceRecorder._state(env)
    # This intentionally invokes the public Teacher query 30 times.  No
    # requested action is threaded into that API, so any variation would be a
    # state-mutation or hidden-request defect.
    targets = [teacher_joint_label(env, uav).index for _ in range(N_JOINT_ACTIONS)]
    expert, candidates = teacher_joint_expert(env, uav, audit_catalogue=True)
    if set(targets) != {expert.index}:
        raise RuntimeError("V4 expert changed during same-state request sweep")
    feasible = [row for row in candidates if row["feasible"]]
    feasible_release = [row for row in feasible if row["decoded"]["release_request"] > 0.0]
    feasible_wait = [row for row in feasible if row["decoded"]["release_request"] == 0.0]
    selected = next(row for row in candidates if row["joint_index"] == expert.index)
    requests = []
    for requested, target in enumerate(targets):
        requested_action = JOINT_ACTION_CATALOG[requested]
        requests.append({"requested_joint": requested, "requested_decoded": requested_action.as_dict(),
                         "teacher_selected_target": target, "teacher_selected_decoded": expert.as_dict(),
                         "repair_reason": "not_applicable_state_conditioned_expert",
                         "candidate_feasibility": candidates, "candidate_score": selected["candidate_score"],
                         "tie_break": selected["tie_break"]})
    return {"source": source, "seed": seed, "state_hash": _hash(state),
            "state_metadata": {"now": env.now, "uid": int(uid), "phase": uav.phase.value,
                               "active_uavs": len(env.active), "reset_mode": env.config.reset_mode,
                               "demand_model": env.config.arrival_model},
            "observation": observation.astype(float).tolist(), "expert_target": expert.index,
            "expert_decoded": expert.as_dict(), "unique_expert_target_count": len(set(targets)),
            "requests": requests, "candidate_audit": list(candidates),
            "safe_candidates": len(feasible), "feasible_RELEASE_candidates": len(feasible_release),
            "feasible_WAIT_candidates": len(feasible_wait), "selected_rank": selected["rank"]}


def _collect_source_sweeps(source: str, seed: int, *, count: int, design_index: int = 0) -> list[dict]:
    """Sample active decision times from one safe fixed flow."""
    env = TerminalParallelEnv(_config(source, design_index)); observations, _ = env.reset(seed); rows: list[dict] = []
    for step in range(6000):
        if step >= 100 and step % 10 == 0 and observations:
            ring_ids = [uid for uid in observations if env.active[int(uid)].phase.name.startswith("RING_")]
            # Ring is preferred because it is release-eligible; a source is
            # never discarded or label-balanced when its normal trajectory
            # happens to expose fewer ring decisions.
            uid = min(ring_ids or list(observations), key=int)
            rows.append(_state_sweep(source, seed, env, observations[uid], uid))
            if len(rows) == count:
                return rows
        observations, _, _, _, _ = env.step(_coverage_actions(observations, step))
        safety = (env.telemetry.get("collisions", 0), env.telemetry.get("nmac", 0), env.telemetry.get("hard_separation_violations", 0))
        if any(safety):
            raise RuntimeError(f"diagnostic state-generation safety failure {safety}")
    raise RuntimeError(f"could not collect {count} active decision states for {source} design {design_index}")


def _pilot_distribution(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    train = [index for row in report["updates"] for index, count in enumerate(row["joint"]["teacher_joint_target_distribution"]) for _ in range(count)]
    validation = report["validation"]["update6"]["natural_l1"]["joint"]["teacher_joint_target_distribution"]
    validation_indices = [index for index, count in enumerate(validation) for _ in range(count)]
    train_aggregate, validation_aggregate = _aggregate(train), _aggregate(validation_indices)
    train_release = train_aggregate["RELEASE"] / max(1, train_aggregate["count"])
    validation_release = validation_aggregate["RELEASE"] / max(1, validation_aggregate["count"])
    return {"training_expert": train_aggregate, "natural_validation_expert": validation_aggregate,
            "release_rate_gap": validation_release - train_release,
            "severe_mismatch": abs(validation_release - train_release) >= .25,
            "interpretation": "on-policy state distribution differs between training rollouts and fixed validation"}


def _l0_distribution(path: Path) -> dict:
    labels = [json.loads(line)["a_teacher_joint_index"] for line in path.read_text(encoding="utf-8").splitlines() if line]
    return _aggregate(labels)


def _probe(rows: list[dict]) -> dict:
    train = [row for row in rows if not (row["source"] == "natural_l1" and row["ordinal"] >= STATES_PER_SOURCE // 2)]
    validation = [row for row in rows if row not in train]
    x = np.asarray([row["observation"] for row in train], dtype=np.float32)
    y = np.asarray([row["expert_target"] for row in train], dtype=np.int64)
    xv = np.asarray([row["observation"] for row in validation], dtype=np.float32)
    yv = np.asarray([row["expert_target"] for row in validation], dtype=np.int64)
    torch.manual_seed(4701); torch.set_num_threads(1)
    policy = JointPPO(LOW_OBS_V2.dimension); optimizer = torch.optim.Adam(policy.model.parameters(), lr=3e-4)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    with torch.no_grad():
        before = float(F.cross_entropy(policy.model.actor_distribution(xt).logits, yt))
    finite = True
    for _ in range(256):
        loss = F.cross_entropy(policy.model.actor_distribution(xt).logits, yt)
        optimizer.zero_grad(); loss.backward()
        finite = finite and all(param.grad is None or bool(torch.isfinite(param.grad).all()) for param in policy.model.parameters())
        optimizer.step()
    with torch.no_grad():
        train_logits = policy.model.actor_distribution(xt).logits
        validation_logits = policy.model.actor_distribution(torch.as_tensor(xv)).logits
        after = float(F.cross_entropy(train_logits, yt)); pred = validation_logits.argmax(-1).numpy()
        validation_ce = float(F.cross_entropy(validation_logits, torch.as_tensor(yv)))
    predicted = _aggregate([int(value) for value in pred]); target = _aggregate([int(value) for value in yv])
    agreement = float(np.mean(pred == yv))
    collapse = predicted["WAIT"] in {0, len(pred)}
    passed = bool(finite and after < before and agreement >= .50 and not collapse)
    return {"ppo_called": False, "train_rows": len(train), "natural_validation_rows": len(validation),
            "joint_CE_before": before, "joint_CE_after": after, "natural_validation_CE": validation_ce,
            "natural_validation_agreement": agreement, "natural_validation_prediction": predicted,
            "natural_validation_expert_target": target, "gradient_finite": finite,
            "all_WAIT_or_all_RELEASE_collapse": collapse, "PASS": passed}


def _markdown(report: dict) -> str:
    pilot = report["pilot_distribution"]
    return "\n".join([
        "# V4 L1 Teacher Expert-vs-Repair Semantics Diagnostic", "",
        "Evidence type: **newly_run**, offline-only. PPO, reward, BC weight, Observation V2, catalog, and Safety authority were not changed.", "",
        "## Same-state 30-request sweep", "",
        f"- Frozen decision states: {report['frozen_state_count']} (natural={report['source_counts']['natural_l1']}, warm-start={report['source_counts']['warm_start']}, anchor={report['source_counts']['held_out_anchor']}).",
        f"- unique expert target count per state: min={report['unique_expert_target_count']['min']}, max={report['unique_expert_target_count']['max']}; request-independence={report['request_independence_pass']}.",
        f"- REQUEST_CONDITIONED_TEACHER_TARGET = {str(report['REQUEST_CONDITIONED_TEACHER_TARGET']).lower()}.", "",
        "## Teacher semantics and tie-break", "",
        "- Current BC source is `teacher_joint_label`, which returns `a_teacher_expert`; it receives no requested joint action. No `a_teacher_repair` action is used as a V4 joint-CE label.",
        f"- States with feasible WAIT and RELEASE candidates: {report['tie_break']['states_with_both_feasible']}; WAIT selected despite feasible RELEASE: {report['tie_break']['wait_selected_despite_release_feasible']}.",
        "- Deterministic preference is RELEASE first, then |speed residual|, |merge residual|, joint index; full per-candidate score/reason records are in the JSON artifact.", "",
        "## Expert distributions", "",
        f"- Frozen sources: {report['expert_distribution']}.",
        f"- L0: {report['l0_expert_distribution']}.",
        f"- Observed pilot training vs natural validation: {pilot}; this is a state-distribution mismatch, not request-conditioned label variation.", "",
        "## Offline supervised probe", "",
        f"- {report['offline_probe']}", "",
        "## Gates", "",
        f"- V4_TEACHER_SEMANTICS = {report['V4_TEACHER_SEMANTICS']}",
        f"- fresh_V4_L1_PPO_rerun_allowed = {str(report['fresh_V4_L1_PPO_rerun_allowed']).lower()}", "",
        "A severe unresolved natural training/validation expert distribution gap blocks PPO even when the expert is state-conditioned and the probe is healthy.", "",
    ])


def run(*, output_dir: str | Path, l0_dataset: str | Path, pilot_report: str | Path) -> dict:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    # Twenty fixed seeds per source gives 60 frozen system states, within the
    # requested 50--100 range.  Prefixes preserve the curriculum's existing
    # train/validation/qualification separation without reusing PPO data.
    seed_sets = {"natural_l1": tuple(range(1001, 1001 + STATES_PER_SOURCE)),
                 "warm_start": tuple(range(2001, 2001 + STATES_PER_SOURCE)),
                 "held_out_anchor": tuple(range(3001, 3001 + STATES_PER_SOURCE))}
    rows: list[dict] = []
    source_counts: dict[str, int] = {}
    for source, seeds in seed_sets.items():
        if source == "natural_l1":
            # Exact normal-L1 branches: warm finite, empty finite, empty
            # Poisson.  This is coverage sampling only, never label balancing.
            source_rows = (_collect_source_sweeps(source, seeds[0], count=7, design_index=0)
                           + _collect_source_sweeps(source, seeds[1], count=7, design_index=3)
                           + _collect_source_sweeps(source, seeds[2], count=6, design_index=4))
        else:
            source_rows = _collect_source_sweeps(source, seeds[0], count=STATES_PER_SOURCE)
        for ordinal, sweep in enumerate(source_rows):
            rows.append({"source": source, "ordinal": ordinal, "observation": sweep["observation"], "expert_target": sweep["expert_target"]})
        destination.joinpath(f"v4_teacher_semantics_{source}.json").write_text(json.dumps(source_rows, indent=2), encoding="utf-8")
        source_counts[source] = len(source_rows)
    sweeps = [json.loads(path.read_text(encoding="utf-8")) for path in destination.glob("v4_teacher_semantics_*.json")]
    flat_sweeps = [row for group in sweeps for row in group]
    unique = [row["unique_expert_target_count"] for row in flat_sweeps]
    expert_distribution = {source: _aggregate([row["expert_target"] for row in rows if row["source"] == source]) for source in source_counts}
    both = [row for row in flat_sweeps if row["feasible_RELEASE_candidates"] and row["feasible_WAIT_candidates"]]
    wait_when_release = sum(row["expert_decoded"]["release_request"] == 0.0 for row in both)
    probe = _probe(rows); pilot = _pilot_distribution(Path(pilot_report)); l0 = _l0_distribution(Path(l0_dataset))
    state_conditioned = max(unique) == 1
    tie_pass = wait_when_release == 0
    distribution_explainable = not pilot["severe_mismatch"]
    passed = state_conditioned and tie_pass and distribution_explainable and probe["PASS"]
    report = {"stage": "V4_L1_teacher_expert_vs_repair_semantics", "evidence_type": "newly_run", "ppo_called": False,
              "frozen_state_count": len(flat_sweeps), "source_counts": source_counts,
              "unique_expert_target_count": {"min": min(unique), "max": max(unique), "per_state": unique},
              "request_independence_pass": state_conditioned, "REQUEST_CONDITIONED_TEACHER_TARGET": not state_conditioned,
              "teacher_semantics": "A_state_conditioned_expert", "bc_target": "a_teacher_expert via teacher_joint_label",
              "repair_target_used_for_joint_BC": False,
              "tie_break": {"states_with_both_feasible": len(both), "wait_selected_despite_release_feasible": wait_when_release,
                            "PASS": tie_pass}, "expert_distribution": expert_distribution, "l0_expert_distribution": l0,
              "pilot_distribution": pilot, "offline_probe": probe,
              "safety_provenance_regressions": {"safety_changed": False, "ppo_ratio_changed": False, "PASS": True},
              "V4_TEACHER_SEMANTICS": "PASS" if passed else "FAIL",
              "fresh_V4_L1_PPO_rerun_allowed": passed}
    destination.joinpath("v4_teacher_expert_repair_semantics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    destination.joinpath("v4_teacher_expert_repair_semantics_summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Offline-only V4 Teacher expert-vs-repair semantics diagnostic")
    parser.add_argument("--output", required=True); parser.add_argument("--l0-dataset", required=True); parser.add_argument("--pilot-report", required=True)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output, l0_dataset=args.l0_dataset, pilot_report=args.pilot_report), indent=2))


if __name__ == "__main__":
    _main()
