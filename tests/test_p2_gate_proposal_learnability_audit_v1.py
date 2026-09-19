from __future__ import annotations

import inspect

import torch

from terminal_marl.training import audit_v4_2_p2_gate_proposal_learnability_v1 as audit


def test_protocol_freezes_support_probe_and_authorization_boundary():
    value = audit.protocol()
    assert value["bindings"]["support_manifest_sha256"] == audit.SUPPORT_MANIFEST_SHA
    assert value["probe"]["seeds"] == list(audit.PROBE_SEEDS)
    assert value["probe"]["checkpoint_selection"] == "final_epoch_only"
    assert value["evaluation"]["all_three_seeds_required"] is True
    assert value["authorizations"]["always_false"]["PROPOSAL_REPAIR_ALLOWED"] is False
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False


def test_residual_probe_zero_initialization_preserves_base_logits():
    torch.manual_seed(1)
    probe = audit.ProposalResidualProbe()
    observations = torch.randn(7, audit.OBSERVATION_DIM)
    assert torch.equal(probe(observations), torch.zeros(7, audit.N_PROPOSALS))


def test_topk_metrics_are_exact():
    logits = torch.tensor([
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    targets = torch.tensor([0, 2])
    result = audit.metrics(logits, targets)
    assert result["rows"] == 2
    assert result["top1"] == 0.5
    assert result["top3"] == 1.0
    assert result["nll"] > 0.0


def test_probe_checks_require_fresh_and_anchor_gates():
    result = {
        "fresh_dev": {
            "top1": 0.70, "top3": 0.90, "nll": 1.0,
            "interface": {
                "o2m": {"top1": 0.60},
                "m2i": {"top1": 0.55},
                "i2f": {"top1": 0.80},
            },
        },
        "d1_anchor": {"top1": 0.90, "top3": 0.96, "nll": 0.40},
    }
    candidate = {"top1": 0.92, "top3": 0.97, "nll": 0.35}
    checks = audit.probe_checks(result, candidate)
    assert checks["all"] is True
    result["fresh_dev"]["interface"]["m2i"]["top1"] = 0.49
    assert audit.probe_checks(result, candidate)["all"] is False


def test_audit_has_no_environment_policy_update_or_checkpoint_path():
    source = inspect.getsource(audit)
    assert "TerminalParallelEnv" not in source
    assert '"policy_training_called": False' in source
    assert '"checkpoint_saved": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source

