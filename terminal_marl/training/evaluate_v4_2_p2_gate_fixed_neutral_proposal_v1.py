"""Fresh paired evaluation of a qualified gate with a fixed neutral proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.distributions import Categorical

from terminal_marl.algo.joint_ppo_v4_2 import HierarchicalPolicyBatchV42, hierarchical_joint_logprob
from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE, decode_hierarchical_actions_v4_2
from terminal_marl.training import evaluate_v4_2_p2_gate_effectiveness_v1 as effectiveness
from terminal_marl.training.run_v4_2_ppo_p2_resume_v4 import p2_config
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import reserve_once


STAGE = "V4_2_P2_GATE_FIXED_NEUTRAL_PROPOSAL_EVALUATION_V1"
PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_fixed_neutral_proposal_evaluation_v1.yaml")
PROTOCOL_SHA = "c53e610971dcd2385b063cf646ac17e9237ee8c222708580a1e2e668ba05e125"
CANDIDATE_SHA = effectiveness.REPAIRED_CHECKPOINT_SHA
FORMAL_SEEDS = (803001, 803002, 803003, 803004, 803005, 803006)
SMOKE_SEED = 803000
CONTROLLERS = ("candidate", "gate_fixed_neutral", "rule")
FIXED_PROPOSAL = 12
RESERVATION_MARKER = "p2_gate_fixed_neutral_proposal_evaluation_v1_consumed.json"


def configure_successor_v2() -> None:
    """Select the fresh successor after moving proposal replacement into the model."""
    global STAGE, PROTOCOL, PROTOCOL_SHA, FORMAL_SEEDS, SMOKE_SEED, RESERVATION_MARKER
    STAGE = "V4_2_P2_GATE_FIXED_NEUTRAL_PROPOSAL_EVALUATION_V2"
    PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_gate_fixed_neutral_proposal_evaluation_v2.yaml")
    PROTOCOL_SHA = "6c9c9fb11736c639ec59aca32e79cd56ac78762c599c2c998dc0f464418ab0f9"
    FORMAL_SEEDS = (804001, 804002, 804003, 804004, 804005, 804006)
    SMOKE_SEED = 804000
    RESERVATION_MARKER = "p2_gate_fixed_neutral_proposal_evaluation_v2_consumed.json"


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def protocol() -> dict[str, Any]:
    _require(PROTOCOL_SHA and _sha256(PROTOCOL) == PROTOCOL_SHA, "fixed-proposal protocol changed")
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    _require(value.get("protocol") == STAGE, "fixed-proposal protocol id changed")
    bindings = value["bindings"]
    for key, expected in {
        "observation_schema": effectiveness.OBSERVATION_SCHEMA,
        "observation_dimension": effectiveness.OBSERVATION_DIM,
        "action_schema": effectiveness.ACTION_SCHEMA,
        "fixed_proposal_index": FIXED_PROPOSAL,
        "fixed_speed_residual": 0.0,
        "fixed_merge_time_residual": 0.0,
        "runtime_catalogue_search": False,
        "no_teacher": True,
        "no_labels": True,
        "no_updates": True,
    }.items():
        _require(bindings.get(key) == expected, f"fixed-proposal binding changed: {key}")
    formal = value["formal"]
    _require(formal["seeds"] == list(FORMAL_SEEDS), "formal seeds changed")
    _require(formal["controllers"] == list(CONTROLLERS), "controllers changed")
    _require(value["smoke"]["seed"] == SMOKE_SEED, "smoke seed changed")
    return value


class FixedNeutralProposalModel(nn.Module):
    """Preserve the frozen gate/critic while exposing a degenerate proposal distribution."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def distributions(self, observations: torch.Tensor) -> tuple[Categorical, Categorical]:
        gate, _ = self.base.distributions(observations)
        logits = observations.new_full((len(observations), 25), -1e9)
        logits[:, FIXED_PROPOSAL] = 0.0
        return gate, Categorical(logits=logits)

    def conditional_entropy(self, observations: torch.Tensor) -> torch.Tensor:
        gate, proposal = self.distributions(observations)
        return gate.entropy() + gate.probs[:, RELEASE_GATE] * proposal.entropy()

    def critic(self, agent_sets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.base.critic(agent_sets, mask)


class FixedNeutralProposalPolicy:
    """Use the frozen student's gate and replace only RELEASE proposal selection."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self.model = FixedNeutralProposalModel(base.model)
        self.device = base.device
        self.observation_dim = base.observation_dim

    def act_details(self, observations: np.ndarray, world: np.ndarray, *, deterministic: bool = False) -> HierarchicalPolicyBatchV42:
        _require(deterministic, "fixed-neutral evaluation must be deterministic")
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(world)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            gate_dist, proposal_dist = self.model.distributions(obs)
            gates = gate_dist.probs.argmax(dim=-1)
            proposals = torch.full_like(gates, -1)
            release = gates == RELEASE_GATE
            proposals[release] = FIXED_PROPOSAL
            gate_lp, proposal_lp, joint_lp = hierarchical_joint_logprob(
                gate_dist, proposal_dist, gates, proposals
            )
            entropy = self.model.conditional_entropy(obs)
            value = self.model.critic(
                torch.as_tensor(world[None], dtype=torch.float32, device=self.device), mask
            )[0]
        gate_np, proposal_np = gates.cpu().numpy(), proposals.cpu().numpy()
        proposal_values = np.asarray(
            [None if value == -1 else int(value) for value in proposal_np], dtype=object
        )
        return HierarchicalPolicyBatchV42(
            gate_np,
            tuple(None if value == -1 else int(value) for value in proposal_np),
            decode_hierarchical_actions_v4_2(gate_np, proposal_values),
            gate_lp.cpu().numpy(),
            tuple(None if value == -1 else float(logprob)
                  for value, logprob in zip(proposal_np, proposal_lp.cpu().numpy())),
            joint_lp.cpu().numpy(), entropy.cpu().numpy(), float(value.cpu()),
        )


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _requested_action_signature(action_tape: str | Path) -> tuple[tuple[Any, ...], ...]:
    payload = json.loads(Path(action_tape).read_text(encoding="utf-8"))
    signature = []
    for row in payload["action_tape"]:
        gates = row.get("requested_gate", {})
        proposals = row.get("requested_proposal", {})
        signature.append(tuple(
            (str(uid), int(gates[uid]), None if proposals.get(uid) is None else int(proposals[uid]))
            for uid in sorted(gates, key=lambda value: int(value))
        ))
    return tuple(signature)


def run(*, candidate_checkpoint: Path, output: Path, history_root: Path, smoke: bool = False) -> dict[str, Any]:
    protocol()
    _require(not output.exists(), f"output already exists: {output}")
    _require(_sha256(candidate_checkpoint) == CANDIDATE_SHA, "candidate checkpoint SHA mismatch")
    candidate, payload = effectiveness.load_candidate_checkpoint(candidate_checkpoint)
    fixed = FixedNeutralProposalPolicy(candidate)
    output.mkdir(parents=True, exist_ok=False)
    seeds = (SMOKE_SEED,) if smoke else FORMAL_SEEDS
    reservation = None
    if not smoke:
        registry = historical_registry(history_root, output)
        _require(not (set(FORMAL_SEEDS) & set(registry["seeds"])), "formal seeds already consumed")
        marker = candidate_checkpoint.parent / RESERVATION_MARKER
        _require(not marker.exists(), "fixed-proposal evaluation already consumed")
        reservation = {"protocol_sha256": PROTOCOL_SHA, "seeds": list(FORMAL_SEEDS),
                       "controllers": list(CONTROLLERS), "output": str(output)}
        reserve_once(marker, reservation)
    records = []
    for seed in seeds:
        config = p2_config(12)
        for controller, policy in (("candidate", candidate), ("gate_fixed_neutral", fixed), ("rule", None)):
            folder = output / f"{controller}_{seed}"
            result = (effectiveness._rule_result(config, seed, folder, smoke=smoke)
                      if controller == "rule"
                      else effectiveness._student_result(policy, config, seed, folder, smoke=smoke))
            records.append({"controller": controller, "seed": seed, "scenario": "flow12", "result": result})
            _write(output / "progress.json", {"completed": len(records),
                                                "planned": len(seeds) * len(CONTROLLERS),
                                                "last_controller": controller, "last_seed": seed,
                                                "last_gates": result.get("gates", {})})
            if not smoke:
                _require(bool(result.get("gates", {}).get("all")),
                         f"fixed-proposal episode failed: {controller}/{seed}")
    aggregates = {name: effectiveness.aggregate_records(records, name) for name in CONTROLLERS}
    signatures = {
        (str(row["controller"]), int(row["seed"])):
            _requested_action_signature(row["result"]["summary"]["action_tape"])
        for row in records if row["controller"] != "rule"
    }
    fixed_release_proposals = [
        proposal
        for (controller, _), rows in signatures.items() if controller == "gate_fixed_neutral"
        for step in rows for _, gate, proposal in step if gate == RELEASE_GATE
    ]
    differing_seeds = [
        seed for seed in seeds
        if signatures[("candidate", seed)] != signatures[("gate_fixed_neutral", seed)]
    ]
    implementation_checks = {
        "fixed_release_rows_nonzero": bool(fixed_release_proposals),
        "fixed_release_proposals_all_index_12": bool(fixed_release_proposals)
            and all(value == FIXED_PROPOSAL for value in fixed_release_proposals),
        "candidate_fixed_requested_actions_differ": bool(differing_seeds),
        "differing_seeds": differing_seeds,
    }
    if not smoke:
        _require(implementation_checks["fixed_release_proposals_all_index_12"],
                 "fixed-neutral proposal replacement did not reach rollout")
        _require(implementation_checks["candidate_fixed_requested_actions_differ"],
                 "candidate/fixed requested actions are still identical")
    fixed_agg, candidate_agg = aggregates["gate_fixed_neutral"], aggregates["candidate"]
    deltas = {
        "weighted_wait_seconds": _delta(fixed_agg["weighted_wait_seconds"], candidate_agg["weighted_wait_seconds"]),
        "request_to_touchdown_seconds": _delta(fixed_agg["request_to_touchdown_seconds"], candidate_agg["request_to_touchdown_seconds"]),
        "shield_actor_fraction": _delta(fixed_agg["shield_actor_fraction"], candidate_agg["shield_actor_fraction"]),
    }
    decision = {
        "fixed_controller_all_episode_gates": bool(fixed_agg["all_gates"]),
        "fixed_completion_fraction": float(fixed_agg["completion_fraction"]) >= 1.0,
        "fixed_safety_zero": bool(fixed_agg["safety_all_zero"]),
        "fixed_wait_nonworse": deltas["weighted_wait_seconds"] is not None and deltas["weighted_wait_seconds"] <= 0.0,
        "fixed_request_to_touchdown_nonworse": deltas["request_to_touchdown_seconds"] is not None and deltas["request_to_touchdown_seconds"] <= 0.0,
        "fixed_shield_actor_fraction_nonworse": deltas["shield_actor_fraction"] is not None and deltas["shield_actor_fraction"] <= 0.0,
    }
    decision["all"] = all(decision.values())
    result = {
        "stage": STAGE, "status": "INCOMPLETE" if smoke else "COMPLETE", "smoke": smoke,
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA,
        "candidate_checkpoint": str(candidate_checkpoint), "candidate_checkpoint_sha256": CANDIDATE_SHA,
        "candidate_payload": {"stage": payload.get("stage"), "architecture_id": payload.get("architecture_id")},
        "seeds": list(seeds), "controllers": list(CONTROLLERS), "records": records,
        "aggregates": aggregates, "fixed_minus_candidate": deltas, "decision_gate": decision,
        "implementation_checks": implementation_checks,
        "reservation": reservation, "formal_seed_reservation": not smoke,
        "environment_called": True, "teacher_called": False, "labels_collected": False,
        "policy_updated": False, "checkpoint_saved": False, "runtime_catalogue_search": False,
        "architecture_freeze_allowed": bool(not smoke and decision["all"]),
        "support_collection_allowed": False, "policy_repair_allowed": False,
        "PPO_ALLOWED": False, "P3_ALLOWED": False,
    }
    manifest = output / "manifest.json"; result["manifest"] = str(manifest); _write(manifest, result)
    print(json.dumps({"stage": STAGE, "status": result["status"], "smoke": smoke,
                      "decision_gate": decision["all"]}, sort_keys=True), flush=True)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--successor-v2", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    if args.successor_v2:
        configure_successor_v2()
    delattr(args, "successor_v2")
    run(**vars(args))


if __name__ == "__main__":
    _main()
