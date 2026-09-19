from __future__ import annotations

import inspect

from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_failure_v1 as diagnostic


def _row(index: int, target: int, interface: str = "o2m") -> dict:
    observation = [0.0] * 88
    observation[index % 88] = 1.0
    return {
        "observation_v2_1": observation,
        "proposal_target": target,
        "teacher_proposal": target,
        "interface": interface,
        "observation_hash": f"hash-{index}",
    }


def test_protocol_freezes_failed_audit_and_descriptive_boundary():
    value = diagnostic.protocol()
    assert value["bindings"]["audit_manifest_sha256"] == diagnostic.AUDIT_MANIFEST_SHA
    assert value["bindings"]["audit_status"] == "FAIL"
    assert value["standalone_probes"]["seeds"] == list(diagnostic.PROBE_SEEDS)
    assert value["classification"]["all_three_mlp_seeds_required"] is True
    assert value["authorizations"]["always_false"]["PROPOSAL_REPAIR_ALLOWED"] is False


def test_majority_baseline_and_hash_conflicts():
    train = [_row(0, 2), _row(1, 2), _row(2, 3)]
    dev = [_row(3, 2), _row(4, 3)]
    majority = diagnostic.majority_baseline(train, dev)
    assert majority["train_majority_proposal"] == 2
    assert majority["dev_accuracy"] == 0.5
    train[1]["observation_hash"] = train[0]["observation_hash"]
    train[1]["proposal_target"] = 4
    conflicts = diagnostic.exact_hash_conflicts(train)
    assert conflicts["conflicting_hashes"] == 1
    assert conflicts["rows_in_conflicting_hashes"] == 2


def test_nearest_neighbor_reports_target_agreement():
    reference = [_row(0, 1), _row(1, 2)]
    query = [_row(0, 1), _row(1, 2)]
    result = diagnostic.nearest_neighbor(reference, query, statistics_rows=reference)
    assert result["queries"] == 2
    assert result["target_agreement"] == 1.0
    assert result["distance_quantiles"]["q0"] == 0.0


def test_failure_attribution_has_no_policy_repair_path():
    source = inspect.getsource(diagnostic)
    assert "TerminalParallelEnv" not in source
    assert '"policy_training_called": False' in source
    assert '"checkpoint_saved": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source

