"""Fresh Observation V2.2 closure pilot with a frozen 88D behavior policy.

The behavior policy sees only the byte-identical V2.1 prefix.  The pilot
collects V2.2 state and a label-only Teacher catalogue, then compares fixed
88D and 94D diagnostic feasibility probes.  It cannot update or promote a
policy and never changes the one-intent runtime boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_joint_logprob
from terminal_marl.core.low_level_joint_action_v4_1 import JOINT_ACTION_CATALOG_V4_1
from terminal_marl.core.low_level_joint_action_v4_2 import decode_hierarchical_actions_v4_2, hierarchical_from_joint_index_v4_1
from terminal_marl.core.observations import LOW_OBS_V2_2, build_low_observation_v2_1
from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import collect_v4_2_p2_gate_support as support
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.audit_v4_2_p2_proposal_conditioned_feasibility_v1 import binary_metrics
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once
from terminal_marl.training.v4_1_joint_teacher import teacher_joint_expert_v4_1


STAGE = "V4_2_OBSERVATION_V2_2_CLOSURE_PILOT_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/observation_v2_2_closure_pilot_v1.yaml")
PROTOCOL_SHA = "f7eb688ad6f9b368d940ac033a1cc940bbdd8399409be7b07fb96d865a002e10"
SCHEMA_PROTOCOL = Path("terminal_marl/configs/low_v4_2/observation_v2_2_schema.yaml")
SCHEMA_PROTOCOL_SHA = "393b11f119202045a3a9c9c12bdd7c2d34d45570c3756fa4481446812dc1ab6e"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
TRAIN_SEEDS = (797001, 797002, 797003, 797004)
DEV_SEEDS = (797005, 797006)
PROBE_SEEDS = (797101, 797102, 797103)
SMOKE_SEED = 797000
SMOKE_STEPS = 160
OBS_SCHEMA = "phaseA_low_obs_v2_2_scheduler_context_94d"
OBS_DIM = 94
PREFIX_DIM = 88
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
RESERVATION_MARKER = "observation_v2_2_closure_pilot_v1_consumed.json"
COMPLETION_HARD_GATE = True
PROPOSAL_FEATURES = np.asarray([
    [float(item.speed_residual), float(item.merge_time_residual)]
    for item in JOINT_ACTION_CATALOG_V4_1 if item.index > 0
], dtype=np.float32)


def configure_successor_v2() -> None:
    """Select the fresh successor after the committed-transfer mapping fix."""
    global STAGE, PROTOCOL, PROTOCOL_SHA, TRAIN_SEEDS, DEV_SEEDS, PROBE_SEEDS
    global SMOKE_SEED, RESERVATION_MARKER, COMPLETION_HARD_GATE
    STAGE = "V4_2_OBSERVATION_V2_2_CLOSURE_PILOT_V2"
    PROTOCOL = Path("terminal_marl/configs/low_v4_2/observation_v2_2_closure_pilot_v2.yaml")
    PROTOCOL_SHA = "9835d890838088723b4e0415c38b4b9e6f9004189302ddffd0abce9f0ee3b571"
    TRAIN_SEEDS = (800001, 800002, 800003, 800004)
    DEV_SEEDS = (800005, 800006)
    PROBE_SEEDS = (800101, 800102, 800103)
    SMOKE_SEED = 800000
    RESERVATION_MARKER = "observation_v2_2_closure_pilot_v2_consumed.json"
    COMPLETION_HARD_GATE = True


def configure_successor_v3() -> None:
    """Select the fixed-horizon diagnostic successor after V2 reachability failed."""
    global STAGE, PROTOCOL, PROTOCOL_SHA, TRAIN_SEEDS, DEV_SEEDS, PROBE_SEEDS
    global SMOKE_SEED, RESERVATION_MARKER, COMPLETION_HARD_GATE
    STAGE = "V4_2_OBSERVATION_V2_2_CLOSURE_PILOT_V3"
    PROTOCOL = Path("terminal_marl/configs/low_v4_2/observation_v2_2_closure_pilot_v3.yaml")
    PROTOCOL_SHA = "227dfa7e252613f5c3755b04fd143c9c42d53d141451cef9748d83cb42926c4a"
    TRAIN_SEEDS = (801001, 801002, 801003, 801004)
    DEV_SEEDS = (801005, 801006)
    PROBE_SEEDS = (801101, 801102, 801103)
    SMOKE_SEED = 801000
    RESERVATION_MARKER = "observation_v2_2_closure_pilot_v3_consumed.json"
    COMPLETION_HARD_GATE = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
    return _sha256(path)


def protocol() -> dict[str, Any]:
    _require(PROTOCOL_SHA and _sha256(PROTOCOL) == PROTOCOL_SHA, "V2.2 closure protocol changed")
    _require(_sha256(SCHEMA_PROTOCOL) == SCHEMA_PROTOCOL_SHA, "V2.2 schema protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    _require(value.get("protocol") == STAGE, "V2.2 closure protocol id changed")
    bindings = value["bindings"]
    expected = {
        "schema_protocol_sha256": SCHEMA_PROTOCOL_SHA,
        "observation_schema": OBS_SCHEMA, "observation_dimension": OBS_DIM,
        "observation_feature_hash_sha256": LOW_OBS_V2_2.feature_hash,
        "v2_1_prefix_dimension": PREFIX_DIM,
        "frozen_behavior_checkpoint_sha256": CANDIDATE_SHA,
        "behavior_reads_v2_1_prefix_only": True, "teacher_mode": "teacher_label_only",
        "deterministic_behavior": True, "no_policy_update": True,
        "no_runtime_catalogue_search": True,
    }
    for key, expected_value in expected.items():
        _require(bindings.get(key) == expected_value, f"V2.2 closure binding changed: {key}")
    collection, probe = value["collection"], value["paired_probe"]
    _require(collection["train_seeds"] == list(TRAIN_SEEDS), "train seeds changed")
    _require(collection["dev_seeds"] == list(DEV_SEEDS), "dev seeds changed")
    expected_completion_gate = "all_12_requests_complete" if COMPLETION_HARD_GATE else "report_only"
    _require(collection.get("completion_gate", "all_12_requests_complete") == expected_completion_gate,
             "completion gate changed")
    _require(collection.get("require_all_12_arrivals_generated", True) is True,
             "arrival generation gate changed")
    _require(probe["seeds"] == list(PROBE_SEEDS), "probe seeds changed")
    _require(probe["epochs"] == EPOCHS and probe["batch_size"] == BATCH_SIZE, "probe budget changed")
    _require(float(probe["learning_rate"]) == LEARNING_RATE, "probe learning rate changed")
    _require(float(probe["gradient_clip_norm"]) == GRADIENT_CLIP_NORM, "probe clip changed")
    _require(value["smoke"] == {"seed": SMOKE_SEED, "steps": SMOKE_STEPS,
                                 "formal_seed_reservation": False, "paired_probe": False},
             "smoke protocol changed")
    return value


def _behavior_request(policy: Any, env: TerminalParallelEnv, observations: Mapping[str, np.ndarray]):
    uids = sorted(observations)
    if not uids:
        return (uids, np.zeros((0, OBS_DIM), dtype=np.float32), np.zeros(0, dtype=bool),
                {}, {})
    world = np.asarray([observations[uid] for uid in uids], dtype=np.float32).reshape(-1, OBS_DIM)
    actor_mask = np.asarray([
        env.active[int(uid)].committed_plan is None
        and env.active[int(uid)].phase.value in support.PHASE_INTERFACE
        and int(uid) not in env.future_release_reservations
        for uid in uids
    ], dtype=bool)
    tensor = torch.as_tensor(world[:, :PREFIX_DIM], device=policy.device)
    with torch.no_grad():
        gate_dist, proposal_dist = policy.model.distributions(tensor)
        gates = torch.zeros(len(uids), dtype=torch.long, device=policy.device)
        proposals = torch.full((len(uids),), -1, dtype=torch.long, device=policy.device)
        mask = torch.as_tensor(actor_mask, device=policy.device)
        if mask.any():
            gates[mask] = gate_dist.logits[mask].argmax(dim=-1)
        release = gates == 1
        if release.any():
            proposals[release] = proposal_dist.logits[release].argmax(dim=-1)
        gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(gate_dist, proposal_dist, gates, proposals)
    proposal_values = np.asarray([None if value < 0 else int(value) for value in proposals.cpu().numpy()], dtype=object)
    decoded = decode_hierarchical_actions_v4_2(gates.cpu().numpy(), proposal_values)
    actions = {uid: decoded[index] for index, uid in enumerate(uids)}
    likelihoods = {
        uid: {"gate": float(gate_lp[index]),
              "proposal": None if proposal_values[index] is None else float(proposal_lp[index]),
              "joint": float(joint_lp[index])}
        for index, uid in enumerate(uids)
    }
    return uids, world, actor_mask, actions, likelihoods


def _episode(policy: Any, *, seed: int, split: str, output: Path,
             max_steps: int | None = None, require_complete: bool = True,
             require_all_arrivals: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = replace(support.support_config(12), observation_schema_version=OBS_SCHEMA)
    output.mkdir(parents=True, exist_ok=False)
    env = TerminalParallelEnv(config)
    observations, _ = env.reset(seed)
    tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt)
    tape.start_episode(env, episode_seed=seed, policy_rng_state={"mode": "v2_2_prefix_behavior", "seed": seed})
    rows: list[dict[str, Any]] = []
    prefix_checks = 0
    repeated_checks = 0
    horizon = int(config.max_simulation_seconds / config.dt) if max_steps is None else int(max_steps)
    for tick in range(horizon):
        uids, world, actor_mask, actions, likelihoods = _behavior_request(policy, env, observations)
        state_hash = _digest(SafetyTraceRecorder._state(env))
        for index, uid in enumerate(uids):
            if not actor_mask[index]:
                continue
            uav = env.active[int(uid)]
            direct_prefix = build_low_observation_v2_1(uav, env)
            prefix_checks += 1
            _require(np.array_equal(world[index, :PREFIX_DIM], direct_prefix), "V2.2 prefix drift")
            repeated = env._observations()[uid]
            repeated_checks += 1
            _require(np.array_equal(world[index], repeated), "V2.2 observation is not deterministic")
            label, _ = teacher_joint_expert_v4_1(env, uav, audit_catalogue=False)
            if label.index <= 0:
                continue
            full_label, catalogue = teacher_joint_expert_v4_1(env, uav, audit_catalogue=True)
            _require(full_label.index == label.index and len(catalogue) == 26, "Teacher catalogue changed")
            feasible = sorted(int(item["joint_index"]) - 1 for item in catalogue
                              if item.get("feasible") and int(item["joint_index"]) > 0)
            _require(feasible and label.index - 1 in feasible, "invalid V2.2 feasible-set target")
            rows.append({
                "split": split, "seed": seed, "step": tick, "sim_time": float(env.now),
                "uav_id": int(uid), "arrival_id": int(uav.arrival_id), "phase": uav.phase.value,
                "interface": support.PHASE_INTERFACE[uav.phase.value],
                "decision_event_id": f"v2.2:{seed}:uav:{uid}:step:{tick}",
                "state_hash": state_hash, "observation_v2_1": world[index, :PREFIX_DIM].tolist(),
                "observation_v2_2": world[index].tolist(), "teacher_proposal": label.index - 1,
                "proposal_set_target": feasible, "proposal_set_size": len(feasible),
                "catalogue_complete": True,
            })
        record = tape.record_pre_step(env, observations, actions, likelihoods)
        following, _, _, _, info = env.step(actions)
        tape.record_post_step(env, record, info)
        _require(not any(row.get("teacher_used") for row in info["action_traces"].values()),
                 "label-only Teacher changed execution")
        observations = following
        terminal = env.arrivals.exhausted() and not env.active and not env.admission.queue.items and not env.service_uavs
        if terminal or env.now >= config.max_simulation_seconds:
            break
    tape_path = tape.dump(reason="observation_v2_2_closure_pilot")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)
    safety = {key: int(env.telemetry.get(key, 0)) for key in
              ("collisions", "nmac", "hard_separation_violations", "uncertified_execution")}
    generated = int(env.arrivals.generated)
    complete = len(env.completed_uavs) == 12
    checks = {
        "all_arrivals_generated": generated == 12 if require_all_arrivals else True,
        "all_requests_complete": complete if require_complete else True,
        "safety_zero": not any(safety.values()),
        "exact_action_tape_replay": bool(replay.get("exact")),
        "prefix_checks_nonzero": prefix_checks > 0,
        "repeated_checks_nonzero": repeated_checks > 0,
        "retained_rows_nonzero": bool(rows),
        "finite_bounded_94d": bool(rows) and all(
            np.isfinite(np.asarray(row["observation_v2_2"])).all()
            and np.logical_and(np.asarray(row["observation_v2_2"]) >= -1.0,
                               np.asarray(row["observation_v2_2"]) <= 1.0).all()
            for row in rows),
    }
    checks["all"] = all(checks.values())
    return {
        "seed": seed, "split": split, "steps": tick + 1, "seconds": float(env.now),
        "completed": len(env.completed_uavs), "requested": generated,
        "completion_observed": complete, "completion_hard_gate": require_complete,
        "all_arrivals_observed": generated == 12,
        "arrival_generation_hard_gate": require_all_arrivals,
        "safety": safety,
        "rows": len(rows), "prefix_checks": prefix_checks, "repeated_checks": repeated_checks,
        "action_tape": str(tape_path), "checks": checks,
    }, rows


def _target_mask(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    result = torch.zeros((len(rows), 25), dtype=torch.bool)
    for index, row in enumerate(rows):
        result[index, torch.as_tensor(row["proposal_set_target"], dtype=torch.long)] = True
    return result


def _expand(rows: Sequence[Mapping[str, Any]], observation_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    observation = torch.as_tensor(np.asarray([row[observation_key] for row in rows], dtype=np.float32))
    repeated = observation[:, None, :].expand(-1, 25, -1)
    proposal = torch.as_tensor(PROPOSAL_FEATURES)[None, :, :].expand(len(rows), -1, -1)
    return torch.cat((repeated, proposal), dim=-1).reshape(len(rows) * 25, -1), _target_mask(rows).reshape(-1).float()


class FeasibilityProbe(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def _state_metrics(logits: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    logits = logits.reshape(len(rows), 25)
    target = _target_mask(rows)
    hit = target.gather(1, logits.argmax(dim=-1, keepdim=True)).squeeze(1)
    partial = target.sum(dim=-1) < 25
    interface = {}
    for name in support.INTERFACES:
        indices = torch.as_tensor([index for index, row in enumerate(rows) if row["interface"] == name])
        interface[name] = {"states": int(len(indices)),
                           "top1_feasible": None if not len(indices) else float(hit[indices].float().mean())}
    return {
        "states": len(rows), "top1_feasible": float(hit.float().mean()),
        "partial_states": int(partial.sum()),
        "partial_top1_feasible": None if not bool(partial.any()) else float(hit[partial].float().mean()),
        "interface": interface,
    }


def _probe(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]],
           *, observation_key: str, seed: int) -> dict[str, Any]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_x, train_y = _expand(train_rows, observation_key)
    dev_x, dev_y = _expand(dev_rows, observation_key)
    mean, scale = train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1e-6)
    train_x, dev_x = (train_x - mean) / scale, (dev_x - mean) / scale
    model = FeasibilityProbe(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    positive_weight = float((train_y == 0).sum() / (train_y == 1).sum().clamp_min(1))
    steps = 0
    for _ in range(EPOCHS):
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            logits = model(train_x[index])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[index],
                pos_weight=logits.new_tensor(positive_weight))
            _require(bool(torch.isfinite(loss)), "non-finite V2.2 probe loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            _require(bool(torch.isfinite(norm)), "non-finite V2.2 probe gradient")
            optimizer.step(); steps += 1
    with torch.no_grad():
        dev_logits = model(dev_x)
    return {
        "observation": observation_key, "seed": seed, "optimizer_steps": steps,
        "binary_dev": binary_metrics(dev_logits, dev_y),
        "state_dev": _state_metrics(dev_logits, dev_rows), "checkpoint_saved": False,
    }


def _feature_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["observation_v2_2"][PREFIX_DIM:] for row in rows], dtype=np.float64)
    names = [feature.name for feature in LOW_OBS_V2_2.features[-6:]]
    overall = {name: {"min": float(values[:, index].min()), "max": float(values[:, index].max()),
                      "mean": float(values[:, index].mean()),
                      "nonzero": int(np.count_nonzero(values[:, index])),
                      "unique_rounded_1e6": int(len(np.unique(np.round(values[:, index], 6))))}
               for index, name in enumerate(names)}
    by_interface = {}
    for interface in support.INTERFACES:
        group = [row for row in rows if row["interface"] == interface]
        matrix = np.asarray([row["observation_v2_2"][PREFIX_DIM:] for row in group], dtype=np.float64)
        by_interface[interface] = {"rows": len(group), "nonconstant_features": [] if not len(group) else [
            names[index] for index in range(6) if len(np.unique(np.round(matrix[:, index], 6))) > 1
        ]}
    return {"overall": overall, "by_interface": by_interface,
            "all_features_finite": bool(np.isfinite(values).all())}


def _expected_marker(output: Path) -> dict[str, Any]:
    return {"protocol_sha256": PROTOCOL_SHA, "train_seeds": list(TRAIN_SEEDS),
            "dev_seeds": list(DEV_SEEDS), "output": str(output)}


def _validate_pre_reset_recovery(output: Path, marker: Path) -> dict[str, Any]:
    """Validate the one allowed continuation after the pre-reset mkdir bug.

    The failed attempt completed the historical scan and reserved the frozen
    seeds, but raised before constructing TerminalParallelEnv.  Recovery is
    therefore the same attempt: it must reuse the exact output and marker and
    is rejected if any episode/data artifact exists.
    """
    _require(output.is_dir(), "pre-reset recovery output is missing")
    _require(marker.is_file(), "pre-reset recovery marker is missing")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    _require(marker_payload == _expected_marker(output), "pre-reset recovery marker changed")
    children = {item.name for item in output.iterdir()}
    _require(children == {"registry_progress.json"},
             f"pre-reset recovery found unexpected artifacts: {sorted(children)}")
    registry = json.loads((output / "registry_progress.json").read_text(encoding="utf-8"))
    _require(int(registry.get("files_read", 0)) > 0 and int(registry.get("files_total", 0)) > 0,
             "pre-reset registry evidence is incomplete")
    evidence = {"kind": "same_attempt_pre_reset_recovery", "marker": str(marker),
                "marker_sha256": _sha256(marker), "registry_progress": registry,
                "environment_resets_before_recovery": 0, "seeds_replaced": False,
                "thresholds_changed": False}
    _write_json(output / "pre_reset_recovery.json", evidence)
    return evidence


def run(*, output: Path, checkpoint: Path, history_root: Path, smoke: bool = False,
        recover_pre_reset: bool = False) -> dict[str, Any]:
    protocol()
    if recover_pre_reset:
        _require(output.is_dir(), f"pre-reset recovery output is missing: {output}")
    else:
        _require(not output.exists(), f"output already exists: {output}")
    _require(_sha256(checkpoint) == CANDIDATE_SHA, "candidate checkpoint SHA mismatch")
    _require(set(TRAIN_SEEDS).isdisjoint(DEV_SEEDS), "V2.2 pilot splits overlap")
    policy, payload = effectiveness.load_candidate_checkpoint(checkpoint)
    if smoke:
        _require(not recover_pre_reset, "smoke cannot use formal pre-reset recovery")
        output.mkdir(parents=True, exist_ok=False)
        record, rows = _episode(policy, seed=SMOKE_SEED, split="smoke",
                                output=output / f"smoke_{SMOKE_SEED}_flow12", max_steps=SMOKE_STEPS,
                                require_complete=False, require_all_arrivals=False)
        result = {
            "stage": STAGE, "status": "INCOMPLETE", "smoke": True,
            "protocol_sha256": PROTOCOL_SHA, "candidate_checkpoint_sha256": CANDIDATE_SHA,
            "record": record, "rows": len(rows), "formal_seed_reservation": False,
            "paired_probe_called": False, "behavior_policy_updated": False,
            "support_collection_allowed": False, "policy_repair_allowed": False,
            "PPO_ALLOWED": False, "P3_ALLOWED": False,
            "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
        }
        manifest = output / "manifest.json"; result["manifest"] = str(manifest); _write_json(manifest, result)
        print(json.dumps({"stage": STAGE, "status": "INCOMPLETE", "smoke": True,
                          "rows": len(rows), "checks": record["checks"]}, sort_keys=True), flush=True)
        return result
    marker = checkpoint.parent / RESERVATION_MARKER
    recovery = None
    if recover_pre_reset:
        recovery = _validate_pre_reset_recovery(output, marker)
    else:
        output.mkdir(parents=True, exist_ok=False)
        history = historical_registry(history_root, output)
        _require(not ((set(TRAIN_SEEDS) | set(DEV_SEEDS)) & set(history["seeds"])),
                 "V2.2 pilot seed already consumed")
        _require(not marker.exists(), "V2.2 closure pilot already consumed")
        reserve_once(marker, _expected_marker(output))
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (split, seed) in enumerate([*(("train", value) for value in TRAIN_SEEDS),
                                           *(("dev", value) for value in DEV_SEEDS)]):
        record, episode_rows = _episode(
            policy, seed=seed, split=split,
            output=output / f"{split}_{seed}_flow12",
            require_complete=COMPLETION_HARD_GATE,
        )
        records.append(record); rows.extend(episode_rows)
        _write_json(output / "progress.json", {"completed": index + 1, "planned": 6,
                                                "last": record, "rows": len(rows)})
        _require(record["checks"]["all"], f"V2.2 pilot episode failed: {seed}")
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    _require(train_rows and dev_rows, "V2.2 pilot produced an empty split")
    train_path, dev_path = output / "train_rows.jsonl", output / "dev_rows.jsonl"
    train_sha, dev_sha = _write_jsonl(train_path, train_rows), _write_jsonl(dev_path, dev_rows)
    paired = []
    for seed in PROBE_SEEDS:
        old = _probe(train_rows, dev_rows, observation_key="observation_v2_1", seed=seed)
        new = _probe(train_rows, dev_rows, observation_key="observation_v2_2", seed=seed)
        paired.append({"seed": seed, "v2_1_88d": old, "v2_2_94d": new,
                       "delta_94d_minus_88d": {
                           key: float(new["binary_dev"][key]) - float(old["binary_dev"][key])
                           for key in ("feasible_recall", "infeasible_recall", "macro_f1", "roc_auc")
                       } | {
                           "state_top1_feasible": float(new["state_dev"]["top1_feasible"]) - float(old["state_dev"]["top1_feasible"]),
                       }})
    feature = _feature_summary(rows)
    interface_counts = Counter(row["interface"] for row in rows)
    result = {
        "stage": STAGE, "status": "COMPLETE", "evidence_type": "newly_run",
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "schema_protocol_sha256": SCHEMA_PROTOCOL_SHA,
        "candidate_checkpoint": str(checkpoint), "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_payload": {"stage": payload.get("stage"), "architecture_id": payload.get("architecture_id")},
        "train_seeds": list(TRAIN_SEEDS), "dev_seeds": list(DEV_SEEDS),
        "completion_hard_gate": COMPLETION_HARD_GATE,
        "completion_report": {
            "episodes_complete": sum(bool(row["completion_observed"]) for row in records),
            "episodes_total": len(records),
            "completed_requests": [int(row["completed"]) for row in records],
        },
        "records": records, "rows": {"train": len(train_rows), "dev": len(dev_rows),
                                      "interface": dict(sorted(interface_counts.items()))},
        "train_rows": str(train_path), "train_rows_sha256": train_sha,
        "dev_rows": str(dev_path), "dev_rows_sha256": dev_sha,
        "feature_summary": feature, "paired_probes": paired,
        "pre_reset_recovery": recovery,
        "collection_checks": {"all_episodes_pass": all(row["checks"]["all"] for row in records),
                              "train_dev_seed_disjoint": set(TRAIN_SEEDS).isdisjoint(DEV_SEEDS),
                              "train_dev_event_disjoint": not ({row["decision_event_id"] for row in train_rows}
                                                                 & {row["decision_event_id"] for row in dev_rows})},
        "environment_called": True, "teacher_called": True, "teacher_label_only": True,
        "behavior_policy_updated": False, "diagnostic_probe_optimizer_called": True,
        "probe_checkpoint_saved": False, "threshold_tuning": False, "dev_gradients": False,
        "support_collection_allowed": False, "policy_repair_allowed": False,
        "student_qualification_allowed": False, "effectiveness_evaluation_allowed": False,
        "PPO_ALLOWED": False, "P3_ALLOWED": False,
        "source_hashes": {"runner": _sha256(Path(__file__)), "protocol": _sha256(PROTOCOL)},
    }
    manifest = output / "manifest.json"; result["manifest"] = str(manifest); _write_json(manifest, result)
    print(json.dumps({"stage": STAGE, "status": "COMPLETE", "train_rows": len(train_rows),
                      "dev_rows": len(dev_rows), "PPO_ALLOWED": False}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recover-pre-reset", action="store_true")
    parser.add_argument("--successor-v2", action="store_true")
    parser.add_argument("--successor-v3", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    _require(not (args.successor_v2 and args.successor_v3),
             "select at most one V2.2 successor")
    if args.successor_v2:
        configure_successor_v2()
    if args.successor_v3:
        configure_successor_v3()
    delattr(args, "successor_v2")
    delattr(args, "successor_v3")
    run(**vars(args))


if __name__ == "__main__":
    _main()
