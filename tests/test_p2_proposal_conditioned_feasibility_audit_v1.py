from __future__ import annotations

import inspect

import torch

from terminal_marl.training import audit_v4_2_p2_proposal_conditioned_feasibility_v1 as audit


def _row(target: tuple[int, ...], interface: str = "o2m") -> dict:
    return {
        "observation_v2_1": [0.0] * 88,
        "proposal_set_target": list(target),
        "teacher_proposal": target[0],
        "interface": interface,
    }


def test_protocol_freezes_action_conditioned_probe_and_no_policy_authority():
    value = audit.protocol()
    assert value["bindings"]["failure_attribution_manifest_sha256"] == audit.FAILURE_MANIFEST_SHA
    assert value["bindings"]["proposal_features"] == ["speed_residual", "merge_time_residual"]
    assert value["probe"]["seeds"] == list(audit.PROBE_SEEDS)
    assert value["evaluation"]["all_three_mlp_seeds_required"] is True
    assert value["authorizations"]["always_false"]["PROPOSAL_REPAIR_ALLOWED"] is False


def test_expand_rows_pairs_each_state_with_all_25_proposals():
    rows = [_row((0, 2)), _row((1,))]
    features, targets = audit.expand_rows(rows)
    assert features.shape == (50, 90)
    assert targets.shape == (50,)
    assert int(targets.sum()) == 3
    assert torch.equal(features[:25, 88:], torch.as_tensor(audit.PROPOSAL_FEATURES))


def test_binary_metrics_are_exact_for_perfect_logits():
    logits = torch.tensor([3.0, -2.0, 4.0, -1.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = audit.binary_metrics(logits, targets)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["feasible_recall"] == 1.0
    assert result["infeasible_recall"] == 1.0
    assert result["roc_auc"] == 1.0


def test_state_ranking_reports_partial_and_interface_hits():
    rows = [_row((0,), "o2m"), _row(tuple(range(25)), "m2i"), _row((2,), "i2f")]
    logits = torch.zeros((3, 25))
    logits[0, 0] = 2.0
    logits[1, 7] = 2.0
    logits[2, 2] = 2.0
    result = audit.state_ranking_metrics(logits, rows)
    assert result["top1_feasible"] == 1.0
    assert result["partial_states"] == 2
    assert result["partial_top1_feasible"] == 1.0
    assert all(value["top1_feasible"] == 1.0 for value in result["interface"].values())


def test_audit_has_no_environment_or_policy_update_path():
    source = inspect.getsource(audit)
    assert "TerminalParallelEnv" not in source
    assert '"policy_training_called": False' in source
    assert '"checkpoint_saved": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source
