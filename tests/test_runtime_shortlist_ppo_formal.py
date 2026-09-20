import pytest

from terminal_marl.training import run_v4_2_p2_runtime_shortlist_formal_v1 as formal


def test_formal_protocol_restarts_from_candidate_and_reserves_heldout():
    value = formal.protocol()
    assert value["bindings"]["initialization"] == "original_qualified_candidate"
    assert value["bindings"]["entry_checkpoints_formal_initialization"] == "forbidden"
    assert value["bindings"]["runtime_proposal_shortlist_k"] == 3
    assert value["formal"]["seeds"] == [808001, 808002, 808003, 808004, 808005, 808006]
    assert value["formal"]["heldout_evaluation_seeds_reserved"] == [
        809001, 809002, 809003, 809004, 809005, 809006,
    ]
    assert value["formal"]["reserve_once_before_first_reset"]


def test_formal_entry_validation_is_fail_closed():
    value = formal.protocol()
    admission = {
        "stage": "V4_2_P2_RUNTIME_SHORTLIST_PPO_ENTRY_V1",
        "PASS": True,
        "gates": {"all": True},
        "PPO_ALLOWED": True,
        "formal_training_called": False,
        "entry_checkpoints_eligible_for_formal": False,
        "checkpoint_promotable": False,
        "protocol_sha256": value["bindings"]["entry_protocol_sha256"],
        "candidate_checkpoint_sha256": value["bindings"]["candidate_checkpoint_sha256"],
    }
    formal.validate_entry(admission, value)
    admission["entry_checkpoints_eligible_for_formal"] = True
    with pytest.raises(PermissionError):
        formal.validate_entry(admission, value)


def test_formal_source_hashes_are_repository_relative_and_complete():
    hashes = formal.source_hashes()
    assert "terminal_marl/training/run_v4_2_p2_runtime_shortlist_formal_v1.py" in hashes
    assert "terminal_marl/training/run_v4_2_p2_runtime_shortlist_entry_v1.py" in hashes
    assert "terminal_marl/envs/parallel_env.py" in hashes
    assert all(not key.startswith(("/", "\\")) and ":" not in key for key in hashes)
    assert all(len(value) == 64 for value in hashes.values())
