from __future__ import annotations

import inspect

import pytest

from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_set_failure_v1 as diagnostic


def _row(index: int, target: tuple[int, ...], *, hash_value: str | None = None) -> dict:
    observation = [0.0] * 88
    observation[index % 88] = 1.0
    return {
        "observation_v2_1": observation,
        "proposal_set_target": list(target),
        "teacher_proposal": target[0],
        "interface": "o2m",
        "observation_hash": hash_value or f"hash-{index}",
    }


def test_protocol_freezes_failed_set_audit_and_no_authority():
    value = diagnostic.protocol()
    assert value["bindings"]["audit_manifest_sha256"] == diagnostic.AUDIT_MANIFEST_SHA
    assert value["bindings"]["audit_status"] == "FAIL"
    assert value["standalone_probes"]["seeds"] == list(diagnostic.PROBE_SEEDS)
    assert value["classification"]["all_three_mlp_seeds_required"] is True
    assert value["authorizations"]["always_false"]["PROPOSAL_REPAIR_ALLOWED"] is False


def test_exact_hash_conflicts_distinguish_disjoint_sets():
    rows = [
        _row(0, (1, 2), hash_value="same"),
        _row(1, (2, 3), hash_value="same"),
        _row(2, (8,), hash_value="same"),
    ]
    result = diagnostic.exact_hash_set_conflicts(rows)
    assert result["unique_hashes"] == 1
    assert result["conflicting_hashes"] == 1
    assert result["disjoint_conflicting_hashes"] == 1
    assert result["rows_in_conflicting_hashes"] == 3


def test_nearest_neighbor_reports_set_overlap():
    reference = [_row(0, (1, 2)), _row(1, (4, 5))]
    query = [_row(0, (2, 3)), _row(1, (5, 6))]
    result = diagnostic.nearest_neighbor_set_overlap(reference, query)
    assert result["queries"] == 2
    assert result["any_intersection"] == 1.0
    assert result["jaccard_mean"] == pytest.approx(1.0 / 3.0)
    assert result["distance_quantiles"]["q0"] == 0.0


def test_failure_attribution_has_no_policy_repair_path():
    source = inspect.getsource(diagnostic)
    assert "TerminalParallelEnv" not in source
    assert '"policy_training_called": False' in source
    assert '"checkpoint_saved": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source
