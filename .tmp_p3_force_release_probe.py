"""Run one bounded forced-gate RELEASE counterfactual diagnostic.

This is an inference-only intervention over the existing diagnostic rollout:
the neural proposal head and safety/energy execution chain remain active, while
eligible actor gates are forced to RELEASE by the existing
``trained_release_only`` controller.  It is not PPO behavior likelihood data,
training, or a new policy.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    OBSERVATION_DIM,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)
from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import (
    diagnose_trained_dense30_v2 as diagnostic,
    evaluate_random_soc_gate_repair_v1 as base,
    run_v4_2_p3_controlled_emergency_admission_v5 as admission,
)


RUNS = {
    0: {
        "seed": 850218,
        "checkpoint": Path(
            "/home/user/training_outputs/terminal/"
            "p3_controlled_emergency_formal_v5_restart1_20260925/"
            "run0/checkpoints/checkpoint_episode_0217.pt"
        ),
    },
    1: {
        "seed": 851032,
        "checkpoint": Path(
            "/home/user/training_outputs/terminal/"
            "p3_controlled_emergency_formal_v5_restart1_20260925/"
            "run1/checkpoints/checkpoint_episode_0031.pt"
        ),
    },
    2: {
        "seed": 852090,
        "checkpoint": Path(
            "/home/user/training_outputs/terminal/"
            "p3_controlled_emergency_formal_v5_restart1_20260925/"
            "run2/checkpoints/checkpoint_episode_0089.pt"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(run_index: int):
    binding = RUNS[run_index]
    checkpoint = binding["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    actual_sha256 = _sha256(checkpoint)
    config = admission._build_config("dense24", base._load_yaml_protocol())
    policy = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        OBSERVATION_DIM,
        device="cpu",
        temperature=CALIBRATION_TEMPERATURE,
    )
    payload = load_checkpoint(checkpoint, policy, config, load_optimizer=False)
    if payload.get("policy_architecture_migrated") is not False:
        raise ValueError("checkpoint architecture migration was not false")
    policy.model.eval()
    checkpoint_info = {
        "path": str(checkpoint),
        "sha256": actual_sha256,
        "load_optimizer": False,
        "strict_model_load": True,
        "controller": "trained_release_only",
        "intervention": "eligible_gate_forced_RELEASE_neural_proposal_preserved",
    }
    return policy, config, checkpoint_info, int(binding["seed"])


def run(*, run_index: int, output: Path, max_steps: int = diagnostic.MAX_STEPS):
    policy, config, checkpoint_info, seed = _load(run_index)
    return diagnostic.run_rollout(
        policy,
        config,
        output=output,
        seed=seed,
        max_steps=max_steps,
        controller="trained_release_only",
        probe_times=(),
        scenario="dense24",
        checkpoint_info=checkpoint_info,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, choices=tuple(RUNS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=diagnostic.MAX_STEPS)
    args = parser.parse_args()
    run(run_index=args.run_index, output=args.output, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
