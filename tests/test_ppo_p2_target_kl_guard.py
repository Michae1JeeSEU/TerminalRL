"""Protocol and pure-gate tests for the P2 V4 target-KL continuation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from terminal_marl.training import run_v4_2_ppo_p2_resume_v4 as runner


ROOT = Path(__file__).resolve().parents[1]
V4_CONFIG = ROOT / "terminal_marl/configs/low_v4_2/ppo_p2_resume_v4.yaml"
V3_CONFIG = ROOT / "terminal_marl/configs/low_v4_2/ppo_p2_resume_v3.yaml"
HELDOUT = {755001, 755002, 755003, 755004, 755005, 755006}


def _protocol() -> dict:
    return yaml.safe_load(V4_CONFIG.read_text(encoding="utf-8-sig"))


def _finite_update(**overrides) -> dict:
    update = {
        "accepted_updates": 1,
        "attempted_updates": 1,
        "rejected_updates": 0,
        "updates": 1,
        "stopped": False,
        "final_rollout_kl": 0.005,
        "max_accepted_rollout_kl": 0.006,
        "approx_kl": 0.02,
        "loss": 0.1,
        "actor_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 0.3,
        "clip_fraction": 0.0,
        "grad_norm": 1.0,
    }
    update.update(overrides)
    return update


def _finite_states() -> tuple[dict, dict]:
    model = {"weight": torch.ones(2)}
    optimizer = {
        "state": {0: {"step": torch.tensor(1.0), "exp_avg": torch.ones(2)}},
        "param_groups": [{"lr": 3e-5, "params": [0]}],
    }
    return model, optimizer


def test_update_gates_pass_for_finite_accepted_update():
    model, optimizer = _finite_states()
    gates = runner.make_update_gates(_finite_update(), model, optimizer, _protocol())
    assert all(gates.values())


def test_zero_accepted_update_fails_and_is_not_promotable():
    model, optimizer = _finite_states()
    gates = runner.make_update_gates(
        _finite_update(
            accepted_updates=0,
            attempted_updates=0,
            updates=0,
            final_rollout_kl=0.0,
            max_accepted_rollout_kl=0.0,
        ),
        model,
        optimizer,
        _protocol(),
    )
    assert gates["accepted_updates"] is False
    assert not all(gates.values())


def test_update_accounting_mismatch_fails():
    model, optimizer = _finite_states()
    gates = runner.make_update_gates(
        _finite_update(attempted_updates=2), model, optimizer, _protocol()
    )
    assert gates["update_accounting"] is False


def test_target_kl_overshoot_fails_even_when_legacy_kl_is_bounded():
    model, optimizer = _finite_states()
    gates = runner.make_update_gates(
        _finite_update(final_rollout_kl=0.011), model, optimizer, _protocol()
    )
    assert gates["target_kl"] is False
    assert gates["bounded_KL"] is True


@pytest.mark.parametrize(
    ("state_name", "bad_value"),
    [("model", float("nan")), ("optimizer", float("inf"))],
)
def test_nonfinite_model_or_optimizer_fails(state_name, bad_value):
    model, optimizer = _finite_states()
    target = model if state_name == "model" else optimizer
    if state_name == "model":
        target["weight"][0] = bad_value
    else:
        target["state"][0]["exp_avg"][0] = bad_value
    gates = runner.make_update_gates(_finite_update(), model, optimizer, _protocol())
    assert gates[f"finite_{state_name}"] is False
    assert not all(gates.values())


def _v3_seeds(config: dict) -> set[int]:
    return (
        set(config["qualification_seeds"])
        | set(config["entry_seeds"])
        | {int(config["smoke_seed"])}
        | set(range(int(config["seed_start"]), int(config["seed_start"]) + int(config["episodes"])))
    )


def _v4_seeds(config: dict) -> set[int]:
    return (
        set(config["qualification_seeds"])
        | set(config["entry_seeds"])
        | {int(config["smoke_seed"])}
        | set(config["formal_train_seeds"])
    )


def test_phase_seeds_are_fresh_and_heldout_is_reserved():
    v3 = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8-sig"))
    v4 = _protocol()
    phases = [
        set(v4["qualification_seeds"]),
        set(v4["entry_seeds"]),
        {int(v4["smoke_seed"])},
        set(v4["formal_train_seeds"]),
    ]
    assert all(len(seeds) == len(set(seeds)) for seeds in phases)
    assert all(not (left & right) for index, left in enumerate(phases) for right in phases[index + 1 :])
    assert not (_v4_seeds(v4) & _v3_seeds(v3))
    assert not (_v4_seeds(v4) & HELDOUT)
    assert set(v4["evaluation_seeds_reserved"]) == HELDOUT
    assert v4["heldout_evaluation_untouched"] is True


def test_failed_manifest_semantic_validator_checks_entry_kl_failure():
    report = {
        "stage": "entry",
        "PASS": False,
        "records": [
            {
                "seed": 763001,
                "gates": {"safety": True, "bounded_KL": False},
                "update": {"approx_kl": 0.11257042959332467},
            }
        ],
    }
    checked = runner._validate_failed_manifest_record(report)
    assert checked["stage"] == "entry"
    assert checked["PASS"] is False
    assert checked["records"][0]["seed"] == 763001

    historical = (
        ROOT.parent
        / "training_outputs/terminal/p2_resume_v3_20260918/entry/manifest.json"
    )
    if historical.exists():
        validated = runner.validate_failed_manifest(historical, _protocol())
        assert validated["records"][0]["gates"]["bounded_KL"] is False
