"""Bounded gate-only repair fit from the completed terminal-credit probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import collect_p3_proposal_head_probe_v1 as collector
from terminal_marl.training import train_p3_gate_coverage_repair_v2 as coverage_fit


STAGE = "P3_TERMINAL_GATE_REPAIR_V1"
OBSERVATION_DIM = 88
SOURCE_CHECKPOINT_SHA256 = collector.SOURCE_CHECKPOINT_SHA256
FORMAL_ROOT = collector.FORMAL_ROOT
TERMINAL_CREDIT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_terminal_credit_20260928/terminal_credit"
)
ANCHOR_PATH = Path(
    "/home/user/training_outputs/terminal/p3_runtime_rank_repair_20260926/anchors/anchors.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/user/training_outputs/terminal/p3_terminal_gate_repair_20260928/fit"
)
TARGET_STATE_INDICES = (1, 2, 3, 6, 11)
REPEAT_COUNT = 2
MAX_UPDATES = 120
SMOKE_UPDATES = 1
LEARNING_RATE = 1e-4
GRADIENT_CLIP_NORM = 1.0
KL_WEIGHT = 10.0
KL_MEAN_LIMIT = 0.001
KL_MAX_LIMIT = 0.01
ADAPTER_PREFIX = "gate_adapter."


class TerminalGateRepairError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalGateRepairError(message)


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
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require(target.is_file(), f"JSON input missing: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {target}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["seed"]),
        int(row["step"]),
        int(row["uid"]),
        int(row["arrival_id"]),
    )


def _priority_improvement(state: Mapping[str, Any]) -> float:
    comparison = state.get("comparison")
    _require(isinstance(comparison, Mapping), "state comparison is missing")
    label = str(state.get("label", "")).lower()
    _require(label in {"wait", "release"}, "selected state label is not WAIT/RELEASE")
    priority = comparison.get("priority_weighted_wait_seconds")
    _require(isinstance(priority, Mapping), "priority wait comparison is missing")
    wait = priority.get("wait")
    release = priority.get("release")
    _require(wait is not None and release is not None, "priority wait comparison is incomplete")
    improvement = float(release) - float(wait) if label == "wait" else float(wait) - float(release)
    _require(np.isfinite(improvement) and improvement > 0.0,
             f"selected state has no positive {label} priority wait improvement")
    return improvement


def _training_rows(
    selection: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    state_indices: Sequence[int] = TARGET_STATE_INDICES,
) -> list[dict[str, Any]]:
    selected = selection.get("selected_rows")
    states = summary.get("states")
    labels = summary.get("labels", {}).get("states") if isinstance(summary.get("labels"), Mapping) else None
    _require(
        isinstance(selected, list) and isinstance(states, list) and isinstance(labels, list),
        "terminal-credit inputs are incomplete",
    )
    rows_by_identity = {_identity(row): dict(row) for row in selected if isinstance(row, Mapping)}
    state_groups: dict[int, list[Mapping[str, Any]]] = {}
    for state in states:
        if isinstance(state, Mapping):
            state_groups.setdefault(int(state["state_index"]), []).append(state)
    label_map = {
        int(item["state"][0]): item
        for item in labels
        if isinstance(item, Mapping) and isinstance(item.get("state"), (list, tuple))
    }
    result: list[dict[str, Any]] = []
    raw_weights: list[float] = []
    for state_index in state_indices:
        repeat_states = sorted(state_groups.get(int(state_index), []), key=lambda item: int(item["repeat"]))
        _require(len(repeat_states) == REPEAT_COUNT, f"state {state_index} does not have two repeats")
        _require({int(item["repeat"]) for item in repeat_states} == {0, 1},
                 f"state {state_index} repeat ids are not {{0,1}}")
        identities = [
            (int(item["seed"]), int(item["step"]), int(item["uid"]), int(item["arrival_id"]))
            for item in repeat_states
        ]
        _require(len(set(identities)) == 1, f"state {state_index} repeat identities differ")
        label_entry = label_map.get(int(state_index))
        _require(label_entry is not None, f"state label missing: {state_index}")
        label_state = label_entry.get("state")
        _require(
            isinstance(label_state, (list, tuple)) and tuple(map(int, label_state[:4])) ==
            (int(state_index), identities[0][0], identities[0][1], identities[0][2]),
            f"state {state_index} label identity differs",
        )
        repeat_labels = [str(item).lower() for item in label_entry.get("repeat_labels", ())]
        _require(len(repeat_labels) == REPEAT_COUNT and len(set(repeat_labels)) == 1,
                 f"state {state_index} repeats are not direction-consistent")
        label = str(label_entry.get("label", "")).lower()
        _require(label in {"wait", "release"} and repeat_labels[0] == label,
                 f"state {state_index} is not a stable WAIT/RELEASE label")
        _require([str(item.get("label", "")).lower() for item in repeat_states] == [label, label],
                 f"state {state_index} per-repeat labels are not stable")
        seed, step, uid, arrival_id = identities[0]
        row = rows_by_identity.get((
            seed, step, uid, arrival_id,
        ))
        _require(row is not None, f"selection identity missing for state {state_index}")
        observation = np.asarray(row.get("observation"), dtype=np.float32)
        _require(observation.shape == (OBSERVATION_DIM,), f"state {state_index} observation is not 88D")
        _require(bool(np.isfinite(observation).all()), f"state {state_index} observation is not finite")
        improvements = [_priority_improvement(item) for item in repeat_states]
        raw_weight = min(improvements)
        result.append({
            "state_index": int(state_index),
            "seed": seed,
            "step": step,
            "uid": uid,
            "arrival_id": arrival_id,
            "label": label,
            "teacher_gate": 0 if label == "wait" else 1,
            "observation": observation,
            "repeat_improvements_seconds": improvements,
            "raw_weight": float(raw_weight),
        })
        raw_weights.append(raw_weight)
    total = float(sum(raw_weights))
    _require(np.isfinite(total) and total > 0.0, "state improvement weights are not positive")
    for row in result:
        row["weight"] = float(row["raw_weight"]) / total
    return result


def _load_terminal_credit(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base = Path(root)
    return _read_json(base / "selection.json"), _read_json(base / "summary.json")


def _load_anchors(path: str | Path) -> np.ndarray:
    target = Path(path)
    _require(target.is_file(), f"anchor input missing: {target}")
    with np.load(target) as payload:
        _require("observations" in payload, "anchor observations are missing")
        observations = np.asarray(payload["observations"], dtype=np.float32)
    _require(observations.shape == (11589, OBSERVATION_DIM),
             f"anchor observations shape changed: {observations.shape}")
    _require(bool(np.isfinite(observations).all()), "anchor observations are not finite")
    return observations


def _gate_logits(policy: Any, observations: torch.Tensor) -> torch.Tensor:
    gate_distribution, _proposal_distribution = policy.model.distributions(observations)
    return gate_distribution.logits


def _proposal_logits(policy: Any, observations: torch.Tensor) -> torch.Tensor:
    _gate_distribution, proposal_distribution = policy.model.distributions(observations)
    return proposal_distribution.logits


def _anchor_kl(source_logits: torch.Tensor, candidate_logits: torch.Tensor) -> tuple[float, float]:
    values = _anchor_kl_tensor(source_logits, candidate_logits)
    return float(values.mean().detach().cpu()), float(values.max().detach().cpu())


def _anchor_kl_tensor(source_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """Return source||candidate KL values while retaining candidate gradients."""
    source_log_prob = F.log_softmax(source_logits.detach(), dim=-1)
    source_prob = source_log_prob.exp()
    candidate_log_prob = F.log_softmax(candidate_logits, dim=-1)
    return F.kl_div(candidate_log_prob, source_prob, reduction="none").sum(dim=-1)


def _run_updates(
    policy: Any,
    train_observations: torch.Tensor,
    train_labels: torch.Tensor,
    train_weights: torch.Tensor,
    anchor_observations: torch.Tensor,
    *,
    max_updates: int = MAX_UPDATES,
    learning_rate: float = LEARNING_RATE,
    gradient_clip_norm: float = GRADIENT_CLIP_NORM,
    kl_mean_limit: float = KL_MEAN_LIMIT,
    kl_max_limit: float = KL_MAX_LIMIT,
) -> dict[str, Any]:
    trainable, names = coverage_fit._freeze_to_adapter(policy)
    policy.optimizer = torch.optim.Adam(trainable, lr=float(learning_rate))
    source_anchor_logits = _gate_logits(policy, anchor_observations).detach().clone()
    history: list[dict[str, Any]] = []
    accepted_updates = 0
    attempted_updates = 0
    rollback = False
    stop_reason = "max_updates"
    for update in range(1, int(max_updates) + 1):
        before_model = coverage_fit._state_cpu(policy.model)
        before_optimizer = copy.deepcopy(policy.optimizer.state_dict())
        policy.optimizer.zero_grad(set_to_none=True)
        train_logits = _gate_logits(policy, train_observations)
        per_sample_ce = F.cross_entropy(train_logits, train_labels, reduction="none")
        weighted_ce = (per_sample_ce * train_weights).sum()
        train_anchor_logits = _gate_logits(policy, anchor_observations)
        kl_values_before = _anchor_kl_tensor(source_anchor_logits, train_anchor_logits)
        mean_kl_before = float(kl_values_before.mean().detach().cpu())
        max_kl_before = float(kl_values_before.max().detach().cpu())
        loss = weighted_ce + KL_WEIGHT * kl_values_before.mean()
        _require(bool(torch.isfinite(loss).item()), f"non-finite loss at update {update}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, float(gradient_clip_norm))
        _require(bool(torch.isfinite(norm).item()), f"non-finite gradient norm at update {update}")
        policy.optimizer.step()
        attempted_updates += 1
        candidate_anchor_logits = _gate_logits(policy, anchor_observations)
        mean_kl, max_kl = _anchor_kl(source_anchor_logits, candidate_anchor_logits)
        entry = {
            "update": update,
            "weighted_ce": float(weighted_ce.detach().cpu()),
            "mean_kl_before": mean_kl_before,
            "max_kl_before": max_kl_before,
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "loss": float(loss.detach().cpu()),
            "accepted": bool(mean_kl <= kl_mean_limit and max_kl <= kl_max_limit),
            "rolled_back": False,
        }
        if not entry["accepted"]:
            policy.model.load_state_dict(before_model, strict=True)
            policy.optimizer.load_state_dict(before_optimizer)
            entry["rolled_back"] = True
            history.append(entry)
            rollback = True
            stop_reason = "kl_cap"
            break
        accepted_updates += 1
        history.append(entry)
    return {
        "trainable_parameter_names": names,
        "accepted_updates": accepted_updates,
        "attempted_updates": attempted_updates,
        "stop_reason": stop_reason,
        "rollback": rollback,
        "history": history,
        "source_anchor_logits": source_anchor_logits,
    }


def _save_checkpoint(path: Path, policy: Any, metadata: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": copy.deepcopy(dict(metadata)),
        "model": policy.model.state_dict(),
        "report": _jsonable(report),
    }, path)


def _fit(
    *,
    output_dir: Path,
    terminal_credit_root: Path,
    formal_root: Path,
    anchor_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory must be fresh: {output_dir}")
    selection, summary = _load_terminal_credit(terminal_credit_root)
    rows = _training_rows(selection, summary)
    anchors = _load_anchors(anchor_path)
    protocol, configs, _records, source_manifest = collector._load_source_inputs(formal_root)
    source_binding = collector._load_source_policy(protocol, configs["dense24"])
    source_policy = source_binding["policy"]
    source_payload = source_binding["payload"]
    source_sha = str(source_binding["sha256"])
    _require(source_sha == SOURCE_CHECKPOINT_SHA256, "source checkpoint SHA mismatch")
    source_state = coverage_fit._state_cpu(source_policy.model)
    device = next(source_policy.model.parameters()).device
    train_observations = torch.as_tensor(
        np.stack([row["observation"] for row in rows]), dtype=torch.float32, device=device,
    )
    train_labels = torch.as_tensor([row["teacher_gate"] for row in rows], dtype=torch.long, device=device)
    train_weights = torch.as_tensor([row["weight"] for row in rows], dtype=torch.float32, device=device)
    anchor_observations = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    source_proposal_logits = _proposal_logits(source_policy, anchor_observations).detach().clone()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "progress.json", {
        "stage": STAGE, "status": "RUNNING", "accepted_updates": 0,
        "attempted_updates": 0, "smoke": bool(smoke),
    })
    updates = _run_updates(
        source_policy,
        train_observations,
        train_labels,
        train_weights,
        anchor_observations,
        max_updates=SMOKE_UPDATES if smoke else MAX_UPDATES,
    )
    final_state = coverage_fit._state_cpu(source_policy.model)
    final_anchor_mean_kl, final_anchor_max_kl = _anchor_kl(
        updates["source_anchor_logits"], _gate_logits(source_policy, anchor_observations).detach()
    )
    non_adapter_names = [name for name in source_state if not name.startswith(ADAPTER_PREFIX)]
    adapter_names = [name for name in source_state if name.startswith(ADAPTER_PREFIX)]
    non_adapter_unchanged = all(torch.equal(source_state[name], final_state[name]) for name in non_adapter_names)
    adapter_changed = any(not torch.equal(source_state[name], final_state[name]) for name in adapter_names)
    proposal_unchanged = torch.equal(source_proposal_logits, _proposal_logits(source_policy, anchor_observations).detach())
    _require(non_adapter_unchanged, "non-adapter state changed")
    _require(proposal_unchanged, "proposal logits changed")
    metadata = copy.deepcopy(dict(source_payload.get("metadata", {})))
    metadata.update({
        "stage": STAGE,
        "source_checkpoint": str(collector.SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_sha,
        "architecture_id": CALIBRATED_ARCHITECTURE_ID,
        "temperature": float(CALIBRATION_TEMPERATURE),
        "terminal_credit_selection": str(terminal_credit_root / "selection.json"),
        "terminal_credit_summary": str(terminal_credit_root / "summary.json"),
        "source_manifest": str(source_manifest),
        "anchor_path": str(anchor_path),
        "training_seed": 20260928,
        "training_scope": "gate_adapter_only",
        "training_called": True,
        "supervised_training_called": True,
        "teacher_called": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "optimizer_saved": False,
        "smoke": bool(smoke),
    })
    report: dict[str, Any] = {
        "stage": STAGE,
        "status": "COMPLETED",
        "smoke": bool(smoke),
        "source_checkpoint": str(collector.SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_sha,
        "terminal_credit_selection": str(terminal_credit_root / "selection.json"),
        "terminal_credit_summary": str(terminal_credit_root / "summary.json"),
        "anchor_path": str(anchor_path),
        "anchor_observation_count": int(len(anchors)),
        "labels": [
            {key: value for key, value in row.items() if key not in {"observation"}}
            for row in rows
        ],
        "weights": [float(row["weight"]) for row in rows],
        "loss": {"objective": "weighted_CE+10*mean_KL", "kl_weight": KL_WEIGHT},
        "accepted_updates": int(updates["accepted_updates"]),
        "attempted_updates": int(updates["attempted_updates"]),
        "stop_reason": updates["stop_reason"],
        "rollback": bool(updates["rollback"]),
        "history": updates["history"],
        "final_anchor_kl": {
            "mean": final_anchor_mean_kl,
            "max": final_anchor_max_kl,
            "within_cap": bool(final_anchor_mean_kl <= KL_MEAN_LIMIT and final_anchor_max_kl <= KL_MAX_LIMIT),
        },
        "invariants": {
            "trainable_parameter_names": updates["trainable_parameter_names"],
            "non_adapter_state_bitwise_unchanged": bool(non_adapter_unchanged),
            "proposal_logits_bitwise_unchanged": bool(proposal_unchanged),
            "adapter_state_changed": bool(adapter_changed),
            "optimizer_saved": False,
            "checkpoint_roundtrip_model_exact": False,
        },
        "metadata": metadata,
    }
    checkpoint_path = output_dir / "terminal_gate_repair.pt"
    report_path = output_dir / "report.json"
    report["checkpoint"] = str(checkpoint_path)
    _save_checkpoint(checkpoint_path, source_policy, metadata, report)
    restored = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    config = configs["dense24"]
    load_checkpoint(checkpoint_path, restored, config, load_optimizer=False)
    roundtrip_state = coverage_fit._state_cpu(restored.model)
    roundtrip_exact = coverage_fit._state_equal(final_state, roundtrip_state)
    _require(roundtrip_exact, "checkpoint model roundtrip mismatch")
    report["invariants"]["checkpoint_roundtrip_model_exact"] = True
    report["checkpoint"] = str(checkpoint_path)
    report["checkpoint_sha256"] = _sha256(checkpoint_path)
    report["metadata"] = metadata
    _write_json(report_path, report)
    _write_json(output_dir / "progress.json", {
        "stage": STAGE, "status": "COMPLETED", "accepted_updates": updates["accepted_updates"],
        "attempted_updates": updates["attempted_updates"], "checkpoint": str(checkpoint_path),
        "smoke": bool(smoke),
    })
    return report


def run(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    terminal_credit_root: str | Path = TERMINAL_CREDIT_ROOT,
    formal_root: str | Path = FORMAL_ROOT,
    anchor_path: str | Path = ANCHOR_PATH,
    smoke: bool = False,
) -> dict[str, Any]:
    return _fit(
        output_dir=Path(output_dir),
        terminal_credit_root=Path(terminal_credit_root),
        formal_root=Path(formal_root),
        anchor_path=Path(anchor_path),
        smoke=bool(smoke),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--terminal-credit-root", type=Path, default=TERMINAL_CREDIT_ROOT)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--anchor-path", type=Path, default=ANCHOR_PATH)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(
            output_dir=args.output_dir,
            terminal_credit_root=args.terminal_credit_root,
            formal_root=args.formal_root,
            anchor_path=args.anchor_path,
            smoke=args.smoke,
        )
    except (TerminalGateRepairError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"stage": STAGE, "status": "ERROR", "error": str(exc)}, sort_keys=True), flush=True)
        return 1
    print(json.dumps({
        "stage": STAGE,
        "status": report["status"],
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "accepted_updates": report.get("accepted_updates"),
        "attempted_updates": report.get("attempted_updates"),
        "stop_reason": report.get("stop_reason"),
        "smoke": report.get("smoke"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
