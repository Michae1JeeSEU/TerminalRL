"""P2 V4 guarded PPO resume runner with transactional target-KL updates.

This runner keeps the historical P2 V3 failure immutable, resumes from the
last accepted checkpoint, and only promotes a checkpoint after all update and
model/optimizer restore gates pass.  Qualification does not update the policy;
entry updates are admission-only and formal training starts again from the
same checkpoint-5 model and Adam state.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.checkpoints import load_checkpoint, save_checkpoint
from terminal_marl.training.run_v4_2_ppo_block2 import (
    replay_geometry,
    source_hashes,
    state_equal,
)
from terminal_marl.training.run_v4_2_curriculum_pilot import config_for
from terminal_marl.training.run_v4_2_i2_dagger_preflight import _policy_tape_logprob_exact
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import _write
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import historical_registry
from terminal_marl.training.run_v4_2_v2_1_i1_v2 import _sha
from terminal_marl.training.run_v4_2_v2_1_ppo import requested_logprob_error
from terminal_marl.training.run_v4_2_v2_1_student_qualification import (
    episode_gates,
    protocol as i3_protocol,
)
from terminal_marl.training.student_rollout_v4_2 import rollout


PROTOCOL = Path("terminal_marl/configs/low_v4_2/ppo_p2_resume_v4.yaml")
# This byte hash is part of the preregistered protocol and is intentionally
# checked before any simulator reset.
PROTOCOL_SHA = "01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf"
FAILED_MANIFEST_SHA = "2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d"


def protocol() -> dict[str, Any]:
    """Load the frozen V4 protocol after checking its exact byte hash."""
    if _sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("P2 V4 protocol changed after preregistration")
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8-sig"))


def p2_config(n: int):
    """Use the P2 natural-flow config with the distance-loop reward disabled."""
    return replace(config_for(n), low_level_progress_reward_per_meter=0.0)


def p2_source_hashes() -> dict[str, str]:
    """Return source hashes bound into every V4 manifest/checkpoint."""
    return {
        **source_hashes(),
        "p2_runner": _sha(Path(__file__)),
        "p2_config": _sha(PROTOCOL),
        "target_kl_kernel": _sha(Path("terminal_marl/algo/student_ppo_v4_2.py")),
    }


def _all_finite(value: object) -> bool:
    """Recursively check numeric leaves without mutating the supplied state."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, Number):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def make_update_gates(
    update: Mapping[str, Any],
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    p: Mapping[str, Any],
) -> dict[str, bool]:
    """Build pure post-update gates from immutable observations.

    The helper deliberately receives state dictionaries rather than a policy,
    so it can be tested with fabricated finite/non-finite states and has no
    side effects.
    """
    finite_update = _all_finite(update)
    finite_model = _all_finite(model_state)
    finite_optimizer = _all_finite(optimizer_state)
    try:
        accepted = update["accepted_updates"]
        attempted = update["attempted_updates"]
        rejected = update["rejected_updates"]
        updates = update["updates"]
        stopped = update["stopped"]
        accepted_gate = accepted == updates and updates >= p["minimum_accepted_updates"]
        accounting_gate = attempted == accepted + rejected
        rejected_gate = 0 <= rejected <= p["maximum_rejected_updates"]
        stopped_gate = bool(stopped) == (rejected > 0)
        target_gate = (
            update["final_rollout_kl"] <= p["target_kl"]
            and update["max_accepted_rollout_kl"] <= p["target_kl"]
        )
        legacy_gate = update["approx_kl"] <= p["maximum_approx_kl"]
    except (KeyError, TypeError, ValueError, OverflowError):
        accepted_gate = accounting_gate = rejected_gate = stopped_gate = False
        target_gate = legacy_gate = False
    return {
        "finite_update": finite_update,
        "finite_model": finite_model,
        "finite_optimizer": finite_optimizer,
        "accepted_updates": bool(accepted_gate),
        "update_accounting": bool(accounting_gate),
        "rejected_updates": bool(rejected_gate),
        "stopped": bool(stopped_gate),
        "target_kl": bool(target_gate),
        "bounded_KL": bool(legacy_gate),
    }


# Stable public alias for small protocol/unit-test callers.
update_gates = make_update_gates


def _validate_failed_manifest_record(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V3 entry-failure semantics without writing it."""
    if report.get("stage") != "entry" or report.get("PASS") is not False:
        raise ValueError("expected V3 entry failure manifest")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("expected exactly one V3 failed entry record")
    record = records[0]
    if not isinstance(record, Mapping) or record.get("seed") != 763001:
        raise ValueError("expected V3 failed seed763001 record")
    gates = record.get("gates")
    if not isinstance(gates, Mapping) or {key for key, value in gates.items() if not value} != {"bounded_KL"}:
        raise ValueError("expected bounded_KL as the only V3 failure gate")
    if not isinstance(record.get("update"), Mapping) or not record["update"]:
        raise ValueError("expected non-empty V3 failed update")
    return dict(report)


def validate_failed_manifest(path: Path, p: Mapping[str, Any]) -> dict[str, Any]:
    """Check the exact V3 manifest SHA and its required semantic shape."""
    expected = p.get("failed_manifest_sha256", FAILED_MANIFEST_SHA)
    if _sha(path) != expected:
        raise ValueError("V3 failed manifest SHA mismatch")
    return _validate_failed_manifest_record(json.loads(path.read_text(encoding="utf-8")))


def restore(checkpoint: Path, p: Mapping[str, Any]) -> StudentPPOV42:
    """Restore checkpoint-5 model and initialized Adam, checking exact state."""
    if checkpoint.name != "checkpoint_5.pt":
        raise ValueError("resume only checkpoint_5.pt")
    if _sha(checkpoint) != p["initial_checkpoint_sha256"]:
        raise ValueError("P2 checkpoint-5 mismatch")
    policy = StudentPPOV42(88, learning_rate=p["learning_rate"])
    payload = load_checkpoint(
        checkpoint,
        policy,
        i3_scenario_config("warm3", i3_protocol()),
        load_optimizer=True,
    )
    if (
        payload.get("stage") != "V4_2_V2_1_PPO"
        or not payload.get("formal")
        or not payload.get("optimizer", {}).get("state")
    ):
        raise ValueError("formal checkpoint with initialized optimizer required")
    if not state_equal(payload["model"], policy.model.state_dict()) or not state_equal(
        payload["optimizer"], policy.optimizer.state_dict()
    ):
        raise ValueError("checkpoint model/optimizer restore mismatch")
    if any(group["lr"] != p["learning_rate"] for group in policy.optimizer.param_groups):
        raise ValueError("checkpoint learning rate differs from continuation protocol")
    return policy


def i3_scenario_config(name: str, p: Mapping[str, Any]):
    """Local alias kept separate from P2's natural-flow config helper."""
    from terminal_marl.training.run_v4_2_v2_1_student_qualification import scenario_config

    return scenario_config(name, p)


def validate_gate(report: Mapping[str, Any], stage: str, p: Mapping[str, Any]) -> None:
    if (
        report.get("stage") != stage
        or not report.get("PASS")
        or not report.get("gates")
        or not all(report["gates"].values())
    ):
        raise PermissionError("P2 prior stage has not passed")
    if (
        report.get("source_hashes") != p2_source_hashes()
        or report.get("protocol_sha256") != _sha(PROTOCOL)
        or report.get("checkpoint_sha256") != p["initial_checkpoint_sha256"]
    ):
        raise ValueError("P2 V4 evidence changed")


def execute(
    stage: str,
    output: Path,
    checkpoint: Path,
    history: Path,
    p: Mapping[str, Any],
    prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage in ("entry", "train"):
        validate_gate(prior or {}, "qualify" if stage == "entry" else "entry", p)
    policy = restore(checkpoint, p)
    if stage == "qualify":
        seeds = list(p["qualification_seeds"])
    elif stage == "entry":
        seeds = list(p["entry_seeds"])
    elif stage == "smoke":
        seeds = [int(p["smoke_seed"])]
    else:
        seeds = list(p["formal_train_seeds"])
    output.mkdir(parents=True, exist_ok=False)
    if set(seeds) & historical_registry(history, output)["seeds"]:
        raise ValueError("fresh seeds previously consumed")
    _write(output / "pre_reset_registry.json", {
        "seeds": seeds,
        "stage": stage,
        "protocol_sha256": _sha(PROTOCOL),
        "initial_checkpoint_sha256": p["initial_checkpoint_sha256"],
    })
    np.random.seed(seeds[0])
    records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        scenario_name = (
            p["entry_scenarios"][index]
            if stage == "entry"
            else p["scenarios"][index % len(p["scenarios"])]
        )
        config = p2_config(int(scenario_name[4:]))
        if stage == "smoke":
            horizon = int(p["smoke_steps"])
        elif stage == "entry":
            horizon = int(p["entry_steps"][index])
        else:
            horizon = int(config.max_simulation_seconds / config.dt)
        steps, summary = rollout(
            policy,
            config,
            seed=seed,
            max_steps=horizon,
            output=output / f"seed{seed}",
            deterministic=stage == "qualify" and index < 3,
            collect_labels=False,
        )
        error = requested_logprob_error(policy, steps)
        summary["replay_exact"], geometry = replay_geometry(config, summary["action_tape"])
        summary["requested_logprob_exact"] = _policy_tape_logprob_exact(
            policy, config, summary["action_tape"]
        )
        gates = {
            "safety": not any(summary["safety"].values()),
            "replay": summary["replay_exact"],
            "requested_logprob": summary["requested_logprob_exact"],
            "initial_ratio": error <= 1e-5,
            "student_only": summary["label_rows"] == 0 and all(not s.teacher_used for s in steps),
            "actor_decisions": summary["counts"]["actor_decisions"] > 0,
            "geometry": geometry["reverse_samples"] == 0 and geometry["max_inner_excursion_m"] <= 0.05,
        }
        if stage in ("qualify", "train"):
            gates.update(episode_gates(summary, config, {**i3_protocol(), "horizon_seconds": config.max_simulation_seconds}))
            gates["geometry_observed"] = geometry["samples"] > 0
        update = None
        checkpoint_path = None
        if stage != "qualify":
            gates["stochastic"] = all(s.stochastic_behavior for s in steps)
            if all(gates.values()):
                keys = (
                    "minibatch_size",
                    "gamma",
                    "gae_lambda",
                    "clip_ratio",
                    "value_coef",
                    "entropy_coef",
                )
                update = policy.update(
                    steps,
                    training_authorized=True,
                    epochs=p["update_epochs"],
                    target_kl=p["target_kl"],
                    **{key: p[key] for key in keys},
                )
                gates.update(make_update_gates(
                    update,
                    policy.model.state_dict(),
                    policy.optimizer.state_dict(),
                    p,
                ))
                # A checkpoint is created only after every pre-save gate passes.
                if all(gates.values()):
                    checkpoint_path = output / f"checkpoint_{index + 1}.pt"
                    try:
                        save_checkpoint(
                            checkpoint_path,
                            policy,
                            config,
                            stage="V4_2_P2_FORMAL_TRAIN" if stage == "train" else "V4_2_P2_ENTRY",
                            formal=stage == "train",
                            ppo_called=True,
                            protocol_sha256=_sha(PROTOCOL),
                            source_hashes=p2_source_hashes(),
                            initial_checkpoint_sha256=p["initial_checkpoint_sha256"],
                        )
                        restored = StudentPPOV42(88, learning_rate=p["learning_rate"])
                        load_checkpoint(checkpoint_path, restored, config, load_optimizer=True)
                        gates["restore"] = state_equal(
                            policy.model.state_dict(), restored.model.state_dict()
                        ) and state_equal(
                            policy.optimizer.state_dict(), restored.optimizer.state_dict()
                        )
                        if gates["restore"]:
                            policy = restored
                        else:
                            # A failed restore must not leave an apparently
                            # promotable checkpoint in the stage directory.
                            checkpoint_path.unlink(missing_ok=True)
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                        gates["restore"] = False
                        checkpoint_path.unlink(missing_ok=True)
                else:
                    checkpoint_path = None
        record = {
            "seed": seed,
            "scenario": scenario_name,
            "summary": summary,
            "geometry": geometry,
            "initial_logprob_error": error,
            "gates": gates,
            "update": update,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        }
        records.append(record)
        _write(output / "progress.json", {
            "stage": stage,
            "completed": len(records),
            "total": len(seeds),
            "records": records,
        })
        if not all(gates.values()):
            break
    gates = {
        "all_episodes": len(records) == len(seeds),
        "all_episode_gates": all(all(r["gates"].values()) for r in records),
    }
    if stage == "qualify":
        gates["both_gates"] = all(
            sum(r["summary"]["counts"][key] for r in records) > 0
            for key in ("WAIT", "RELEASE")
        )
    result = {
        "stage": stage,
        "PASS": all(gates.values()),
        "gates": gates,
        "records": records,
        "PPO_ALLOWED": stage == "entry" and all(gates.values()),
        "protocol_sha256": _sha(PROTOCOL),
        "source_hashes": p2_source_hashes(),
        "checkpoint_sha256": p["initial_checkpoint_sha256"],
        "training_called": stage == "train",
        "formal": stage == "train",
        "prior_stage": prior["stage"] if prior else None,
        "scope": p["scope"],
    }
    _write(output / "manifest.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    p = protocol()
    if _sha(args.failed_manifest) != p["failed_manifest_sha256"]:
        raise ValueError("V3 failed manifest SHA mismatch")
    validate_failed_manifest(args.failed_manifest, p)
    if args.checkpoint.name != "checkpoint_5.pt":
        raise ValueError("resume only checkpoint_5.pt")
    if args.mode == "smoke":
        return execute("smoke", args.output, args.checkpoint, args.history_root, p)
    args.output.mkdir(parents=True, exist_ok=False)
    prior = None
    for stage in ("qualify", "entry", "train"):
        prior = execute(stage, args.output / stage, args.checkpoint, args.history_root, p, prior)
        _write(args.output / "progress.json", {"stage": stage, "PASS": prior["PASS"]})
        if not prior["PASS"]:
            break
    _write(args.output / "manifest.json", prior)
    return prior


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pipeline"], required=True)
    for name in ("output", "checkpoint", "history-root", "failed-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    torch.set_num_threads(1)
    result = run(parser.parse_args())
    print(json.dumps({"stage": result["stage"], "PASS": result["PASS"]}), flush=True)
