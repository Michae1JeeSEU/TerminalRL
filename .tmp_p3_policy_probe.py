"""Read-only actor-distribution probe for P3 diagnostic observation samples.

The input contract is a JSON object with one ``samples`` array.  Each sample
has exactly the fields used by the diagnostic exporter::

    {"run": "run0", "uid": "uav-001", "time": 12.5,
     "observation": [88 finite numbers]}

``run`` may also be the integer 0, 1, or 2.  The script loads the E04 repair
checkpoint and the final checkpoint bound to each sample's run with strict
model-state loading and ``load_optimizer=False``.  It only evaluates actor
distribution tensors under ``torch.no_grad()``; it does not sample actions,
run an environment, or update a policy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    OBSERVATION_DIM,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE
from terminal_marl.training import run_v4_2_p3_controlled_emergency_formal_v5 as formal


RUN_KEYS = ("run0", "run1", "run2")
INPUT_SCHEMA = "p3_policy_probe_samples_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalise_run(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("sample run must be run0/run1/run2 or integer 0/1/2")
    if isinstance(value, int):
        key = f"run{value}"
    else:
        text = str(value).strip().lower()
        key = text if text.startswith("run") else f"run{text}"
    _require(key in RUN_KEYS, f"sample run must be one of {RUN_KEYS}, got {value!r}")
    return key


def _read_samples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "samples JSON must be an object")
    samples = payload.get("samples")
    _require(isinstance(samples, list) and samples, "samples must be a non-empty array")
    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(samples):
        _require(isinstance(raw, Mapping), f"samples[{index}] must be an object")
        for key in ("run", "uid", "time", "observation"):
            _require(key in raw, f"samples[{index}] missing {key!r}")
        run = _normalise_run(raw["run"])
        time_value = float(raw["time"])
        _require(math.isfinite(time_value), f"samples[{index}].time must be finite")
        observation = raw["observation"]
        _require(isinstance(observation, list), f"samples[{index}].observation must be an array")
        _require(len(observation) == OBSERVATION_DIM,
                 f"samples[{index}].observation must have {OBSERVATION_DIM} values")
        values = [float(value) for value in observation]
        _require(all(math.isfinite(value) for value in values),
                 f"samples[{index}].observation must contain finite values")
        normalised.append({
            "run": run,
            "uid": raw["uid"],
            "time": time_value,
            "observation": values,
        })
    return normalised


def _load_config() -> Any:
    """Build the formal dense24 config used for checkpoint schema validation."""
    protocol_path = Path("terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig"))
    _require(isinstance(protocol, Mapping), "formal protocol must be a mapping")
    base_protocol = formal.base._load_yaml_protocol()
    configs = {
        scenario: formal.admission._build_config(scenario, base_protocol)
        for scenario in ("flow6", "flow12", "dense24")
    }
    _require(protocol.get("bindings", {}).get("observation_dimension") == OBSERVATION_DIM,
             "formal protocol observation dimension changed")
    return configs["dense24"]


def _load_policy(path: Path, config: Any) -> TemperatureCalibratedGateResidualAdapterStudentPPOV42:
    _require(path.is_file(), f"checkpoint missing: {path}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        device="cpu",
        temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(path, policy, config, load_optimizer=False)
    _require(payload.get("policy_architecture_migrated") is False,
             f"checkpoint architecture migration occurred: {path}")
    policy.model.eval()
    return policy


def _json_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _probe(policy: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
           observation: list[float]) -> dict[str, Any]:
    values = torch.as_tensor(observation, dtype=torch.float32, device=policy.device).reshape(1, -1)
    model = policy.model
    with torch.no_grad():
        encoded = model.actor_encoder(values)
        base_gate = model.gate_head(encoded)
        adapter = model.gate_adapter(values)
        proposal = model.proposal_head(encoded)
        effective_gate = (base_gate + adapter) / model.gate_temperature
        p_release = torch.softmax(effective_gate, dim=-1)[0, RELEASE_GATE]
        top_count = min(3, int(proposal.shape[-1]))
        top_values, top_indices = torch.topk(proposal[0], k=top_count, dim=-1)
    return {
        "pRelease": _json_float(p_release),
        "base_gate_logits": [_json_float(value) for value in base_gate[0]],
        "adapter_logits": [_json_float(value) for value in adapter[0]],
        "top3_proposal_logits": [
            {"index": int(index.item()), "logit": _json_float(value)}
            for value, index in zip(top_values, top_indices)
        ],
    }


def _parse_final_checkpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        run, separator, path_text = value.partition("=")
        _require(separator == "=" and path_text, "final checkpoint must use RUN=PATH")
        key = _normalise_run(run)
        _require(key not in result, f"duplicate final checkpoint binding for {key}")
        result[key] = Path(path_text)
    _require(set(result) == set(RUN_KEYS),
             f"final checkpoint bindings must contain exactly {RUN_KEYS}")
    return result


def run(*, samples_path: Path, e04_checkpoint: Path,
        final_checkpoints: Mapping[str, Path], output: Path) -> dict[str, Any]:
    samples = _read_samples(samples_path)
    config = _load_config()
    e04_policy = _load_policy(e04_checkpoint, config)
    final_policies = {
        run_key: _load_policy(checkpoint, config)
        for run_key, checkpoint in final_checkpoints.items()
    }
    rows = []
    for sample in samples:
        run_key = sample["run"]
        rows.append({
            "run": run_key,
            "uid": sample["uid"],
            "time": sample["time"],
            "e04": _probe(e04_policy, sample["observation"]),
            "final": _probe(final_policies[run_key], sample["observation"]),
        })
    result = {
        "schema": INPUT_SCHEMA,
        "samples_path": str(samples_path),
        "e04_checkpoint": str(e04_checkpoint),
        "final_checkpoints": {key: str(value) for key, value in final_checkpoints.items()},
        "evaluation": {
            "sampling": False,
            "environment_replay": False,
            "policy_update": False,
            "load_optimizer": False,
            "strict_model_load": True,
            "device": "cpu",
        },
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--e04-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", action="append", required=True,
                        help="run0=PATH, run1=PATH, and run2=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        samples_path=args.samples,
        e04_checkpoint=args.e04_checkpoint,
        final_checkpoints=_parse_final_checkpoints(args.final_checkpoint),
        output=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(main())
