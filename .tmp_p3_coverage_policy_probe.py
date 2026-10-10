"""Pure forward probe for the P3 gate-coverage repair checkpoint.

The probe reuses the twelve finite 88D observations in the existing
``p3_policy_probe_20260925/samples_v2.json`` artifact and places the new
coverage policy beside the existing E04 and formal-final policy outputs.  It
loads model weights strictly with ``load_optimizer=False`` and performs no
environment, Teacher, sampling, or policy-update work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATED_ARCHITECTURE_ID,
    CALIBRATION_TEMPERATURE,
    OBSERVATION_DIM,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.core.low_level_joint_action_v4_2 import RELEASE_GATE
from terminal_marl.training import run_v4_2_p3_controlled_emergency_formal_v5 as formal


RUN_KEYS = ("run0", "run1", "run2")
PROBE_ROOT_NAME = "p3_policy_probe_20260925"
SAMPLES_NAME = "samples_v2.json"
BASELINE_MANIFEST_NAME = "policy_probe.json"
LOCAL_OUTPUT_ROOT = Path(r"D:\7_codespace\training_outputs\terminal")
SERVER_OUTPUT_ROOT = Path("/home/user/training_outputs/terminal")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    _require(isinstance(samples, list) and len(samples) == 12,
             "samples_v2.json must contain exactly twelve samples")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(samples):
        _require(isinstance(raw, Mapping), f"samples[{index}] must be an object")
        for key in ("run", "uid", "time", "observation"):
            _require(key in raw, f"samples[{index}] missing {key!r}")
        run = _normalise_run(raw["run"])
        time_value = float(raw["time"])
        observation = raw["observation"]
        _require(math.isfinite(time_value), f"samples[{index}].time must be finite")
        _require(isinstance(observation, list) and len(observation) == OBSERVATION_DIM,
                 f"samples[{index}].observation must have {OBSERVATION_DIM} values")
        values = [float(value) for value in observation]
        _require(all(math.isfinite(value) for value in values),
                 f"samples[{index}].observation must contain finite values")
        result.append({"run": run, "uid": raw["uid"], "time": time_value,
                       "observation": values})
    return result


def _default_samples_path() -> Path:
    candidates = (
        Path("training_outputs") / "terminal" / PROBE_ROOT_NAME / SAMPLES_NAME,
        SERVER_OUTPUT_ROOT / PROBE_ROOT_NAME / SAMPLES_NAME,
        LOCAL_OUTPUT_ROOT / PROBE_ROOT_NAME / SAMPLES_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _resolve_artifact(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    text = str(value)
    server_prefix = "/home/user/training_outputs/terminal/"
    if text.startswith(server_prefix):
        local = LOCAL_OUTPUT_ROOT / text.removeprefix(server_prefix)
        if local.is_file():
            return local
    local_prefix = "D:\\7_codespace\\training_outputs\\terminal\\"
    if text.startswith(local_prefix):
        server = SERVER_OUTPUT_ROOT / text.removeprefix(local_prefix).replace("\\", "/")
        if server.is_file():
            return server
    return path


def _baseline_manifest(samples_path: Path) -> Path:
    candidates = (
        samples_path.with_name(BASELINE_MANIFEST_NAME),
        LOCAL_OUTPUT_ROOT / PROBE_ROOT_NAME / BASELINE_MANIFEST_NAME,
        SERVER_OUTPUT_ROOT / PROBE_ROOT_NAME / BASELINE_MANIFEST_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("existing policy_probe.json baseline manifest is missing")


def _baseline_paths(samples_path: Path) -> tuple[Path, dict[str, Path], Path]:
    manifest_path = _baseline_manifest(samples_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "policy_probe.json must be an object")
    e04 = _resolve_artifact(payload.get("e04_checkpoint", ""))
    finals_raw = payload.get("final_checkpoints")
    _require(isinstance(finals_raw, Mapping), "policy_probe.json final_checkpoints is missing")
    finals = {key: _resolve_artifact(finals_raw.get(key, "")) for key in RUN_KEYS}
    _require(e04.is_file(), f"E04 checkpoint missing: {e04}")
    _require(all(path.is_file() for path in finals.values()),
             f"one or more final checkpoints are missing: {finals}")
    return e04, finals, manifest_path


def _load_config() -> Any:
    protocol = formal.base._load_yaml_protocol()
    config = formal.admission._build_config("dense24", protocol)
    config = replace(config, v4_2_rejected_release_hold_mode="canonical_wait")
    _require(config.v4_2_rejected_release_hold_mode == "canonical_wait",
             "probe config did not apply canonical_wait")
    _require(config.action_schema_version == "low_action_schema_v4_2_hierarchical",
             "probe requires V4.2 hierarchical actions")
    _require(config.observation_schema_version == "phaseA_low_obs_v2_1_retry_state",
             "probe requires V2.1 observations")
    _require(int(OBSERVATION_DIM) == 88, "probe requires 88D observations")
    return config


def _load_policy(path: Path, config: Any) -> TemperatureCalibratedGateResidualAdapterStudentPPOV42:
    _require(path.is_file(), f"checkpoint missing: {path}")
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM, device="cpu", temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(path, policy, config, load_optimizer=False)
    _require(payload.get("policy_architecture_migrated") is False,
             f"checkpoint architecture migration occurred: {path}")
    _require(getattr(policy.model, "architecture_id", None) == CALIBRATED_ARCHITECTURE_ID,
             f"checkpoint architecture mismatch: {path}")
    _require(len(policy.optimizer.state) == 0,
             f"optimizer state was restored unexpectedly: {path}")
    policy.model.eval()
    return policy


def _forward(policy: TemperatureCalibratedGateResidualAdapterStudentPPOV42,
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
        top_values, top_indices = torch.topk(proposal[0], k=min(3, proposal.shape[-1]), dim=-1)
    proposal_values = [float(value.detach().cpu().item()) for value in proposal[0]]
    return {
        "pRelease": float(p_release.detach().cpu().item()),
        "base_gate_logits": [float(value.detach().cpu().item()) for value in base_gate[0]],
        "adapter_logits": [float(value.detach().cpu().item()) for value in adapter[0]],
        "proposal_logits": proposal_values,
        "top3_proposal_logits": [
            {"index": int(index.item()), "logit": float(value.detach().cpu().item())}
            for value, index in zip(top_values, top_indices)
        ],
    }


def _proposal_check(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    left = [float(value) for value in reference["proposal_logits"]]
    right = [float(value) for value in candidate["proposal_logits"]]
    _require(len(left) == len(right), "proposal logit dimensions differ")
    differences = [abs(a - b) for a, b in zip(left, right)]
    return {
        "exact": bool(left == right),
        "max_abs_diff": max(differences, default=0.0),
    }


def run(*, checkpoint: Path, output: Path, samples_path: Path | None = None) -> dict[str, Any]:
    samples_path = samples_path or _default_samples_path()
    samples_path = _resolve_artifact(samples_path)
    checkpoint = _resolve_artifact(checkpoint)
    samples = _read_samples(samples_path)
    config = _load_config()
    e04_path, final_paths, manifest_path = _baseline_paths(samples_path)
    e04_policy = _load_policy(e04_path, config)
    final_policies = {key: _load_policy(path, config) for key, path in final_paths.items()}
    coverage_policy = _load_policy(checkpoint, config)

    rows: list[dict[str, Any]] = []
    proposal_checks: list[dict[str, Any]] = []
    for sample in samples:
        e04 = _forward(e04_policy, sample["observation"])
        final = _forward(final_policies[sample["run"]], sample["observation"])
        coverage = _forward(coverage_policy, sample["observation"])
        proposal_check = _proposal_check(e04, coverage)
        proposal_checks.append(proposal_check)
        rows.append({
            "run": sample["run"], "uid": sample["uid"], "time": sample["time"],
            "e04": e04, "final": final, "coverage": coverage,
            "coverage_proposal_vs_e04": proposal_check,
        })

    result = {
        "schema": "p3_coverage_policy_probe_v1",
        "samples_path": str(samples_path),
        "baseline_manifest": str(manifest_path),
        "e04_checkpoint": str(e04_path),
        "final_checkpoints": {key: str(value) for key, value in final_paths.items()},
        "coverage_checkpoint": str(checkpoint),
        "coverage_checkpoint_sha256": _sha256(checkpoint),
        "evaluation": {
            "sampling": False,
            "environment_replay": False,
            "teacher": False,
            "policy_update": False,
            "load_optimizer": False,
            "strict_model_load": True,
            "device": "cpu",
            "config_v4_2_rejected_release_hold_mode": "canonical_wait",
        },
        "proposal_invariant": {
            "coverage_vs_e04_all_exact": all(item["exact"] for item in proposal_checks),
            "coverage_vs_e04_max_abs_diff": max(
                (float(item["max_abs_diff"]) for item in proposal_checks), default=0.0,
            ),
        },
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=None,
                        help="defaults to the existing p3_policy_probe samples_v2.json")
    args = parser.parse_args()
    result = run(checkpoint=args.checkpoint, output=args.output, samples_path=args.samples)
    print(json.dumps({
        "schema": result["schema"],
        "coverage_checkpoint": result["coverage_checkpoint"],
        "samples": len(result["samples"]),
        "proposal_invariant": result["proposal_invariant"],
        "output": str(args.output),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
