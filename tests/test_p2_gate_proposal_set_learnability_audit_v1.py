from __future__ import annotations

import inspect

import torch

from terminal_marl.training import audit_v4_2_p2_gate_proposal_set_learnability_v1 as audit


def test_protocol_freezes_set_probe_and_authorization_boundary():
    value = audit.protocol()
    assert value["bindings"]["support_manifest_sha256"] == audit.SUPPORT_MANIFEST_SHA
    assert value["probe"]["seeds"] == list(audit.PROBE_SEEDS)
    assert value["probe"]["objective"].startswith("0.5_support_set_NLL")
    assert value["evaluation"]["all_three_seeds_required"] is True
    assert value["authorizations"]["always_false"]["PROPOSAL_REPAIR_ALLOWED"] is False
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False


def test_set_nll_rewards_probability_mass_on_any_feasible_proposal():
    target = torch.tensor([[True, False, True, False]])
    good = torch.tensor([[4.0, 0.0, 3.0, 0.0]])
    bad = torch.tensor([[0.0, 4.0, 0.0, 3.0]])
    assert audit.set_nll(good, target) < audit.set_nll(bad, target)
    all_feasible = torch.ones((1, 4), dtype=torch.bool)
    assert abs(float(audit.set_nll(good, all_feasible))) < 1e-6


def test_set_metrics_topk_and_partial_slice_are_exact():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    target = torch.tensor([[False, False, True, False], [True, True, True, True]])
    result = audit.set_metrics(logits, target)
    assert result["rows"] == 2
    assert result["top1_feasible"] == 0.5
    assert result["top3_feasible_hit"] == 1.0
    assert result["partial_rows"] == 1
    assert result["partial_top1_feasible"] == 0.0
    assert result["partial_top3_feasible_hit"] == 1.0
    full_only = audit.set_metrics(logits[:1], torch.ones((1, 4), dtype=torch.bool))
    assert full_only["partial_rows"] == 0
    assert full_only["partial_set_nll"] is None


def test_probe_checks_require_gain_partial_interfaces_and_anchor():
    candidate_dev = {
        "top1_feasible": 0.71, "top3_feasible_hit": 0.87,
        "partial_set_nll": 1.0,
    }
    candidate_d1 = {"top1": 0.99, "top3": 1.0, "nll": 0.02}
    result = {
        "fresh_dev": {
            "top1_feasible": 0.82, "top3_feasible_hit": 0.95,
            "partial_top1_feasible": 0.75, "partial_set_nll": 0.70,
            "interface": {
                "o2m": {"top1_feasible": 0.75},
                "m2i": {"top1_feasible": 0.72},
                "i2f": {"top1_feasible": 0.90},
            },
        },
        "d1_anchor": {"top1": 0.97, "top3": 0.99, "nll": 0.10},
    }
    assert audit.probe_checks(result, candidate_dev, candidate_d1)["all"] is True
    result["fresh_dev"]["interface"]["m2i"]["top1_feasible"] = 0.69
    assert audit.probe_checks(result, candidate_dev, candidate_d1)["all"] is False


def test_audit_has_no_environment_policy_update_or_checkpoint_path():
    source = inspect.getsource(audit)
    assert "TerminalParallelEnv" not in source
    assert '"policy_training_called": False' in source
    assert '"checkpoint_saved": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source
