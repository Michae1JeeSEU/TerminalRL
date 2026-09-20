import pytest

from terminal_marl.training import evaluate_v4_2_p2_runtime_shortlist_heldout_v1 as heldout


def test_heldout_protocol_binds_fresh_paired_evaluation():
    value = heldout.protocol()
    assert value["bindings"]["runtime_proposal_shortlist_k"] == 3
    assert value["bindings"]["no_updates"]
    assert value["heldout"]["seeds"] == [809001, 809002, 809003, 809004, 809005, 809006]
    assert value["heldout"]["controllers"] == ["trained_k3", "initial_k3", "rule"]
    assert value["hard_gates"]["frozen_proposal_subgraph"]
    assert value["authorization"]["checkpoint_promotable_on_pass"]


def test_heldout_formal_validation_is_fail_closed():
    value = heldout.protocol()
    manifest = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_FORMAL_V1",
        "PASS": True,
        "gates": {"all": True},
        "HELDOUT_EVALUATION_ALLOWED": True,
        "formal_training_called": True,
        "checkpoint_promotable": False,
        "protocol_sha256": value["bindings"]["formal_protocol_sha256"],
        "final_checkpoint_sha256": value["bindings"]["trained_checkpoint_sha256"],
        "candidate_checkpoint_sha256": value["bindings"]["initial_checkpoint_sha256"],
        "heldout_evaluation_seeds_reserved": value["heldout"]["seeds"],
    }
    heldout.validate_formal(manifest, value)
    manifest["checkpoint_promotable"] = True
    with pytest.raises(PermissionError):
        heldout.validate_formal(manifest, value)


def test_heldout_source_hashes_are_repository_relative():
    hashes = heldout.source_hashes()
    assert "terminal_marl/training/evaluate_v4_2_p2_runtime_shortlist_heldout_v1.py" in hashes
    assert "terminal_marl/training/student_rollout_v4_2.py" in hashes
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())
