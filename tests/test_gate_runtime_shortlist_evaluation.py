import json

from terminal_marl.training import evaluate_v4_2_p2_gate_runtime_shortlist_v1 as evaluation


def test_runtime_shortlist_protocol_freezes_bounded_teacher_free_boundary():
    value = evaluation.protocol()
    assert value["formal"]["seeds"] == list(evaluation.FORMAL_SEEDS)
    assert value["formal"]["controllers"] == list(evaluation.CONTROLLERS)
    assert value["bindings"]["one_policy_forward"]
    assert value["bindings"]["max_runtime_proposals"] == 3
    assert not value["bindings"]["full_catalogue_search"]
    assert not value["bindings"]["delayed_candidate_search"]
    assert value["bindings"]["no_teacher"]


def test_tape_shortlist_contract_checks_requested_first_unique_and_bound(tmp_path):
    requested = [1.0, 0.0, 0.0]
    second = [1.0, 0.5, 0.0]
    path = tmp_path / "tape.json"
    path.write_text(json.dumps({"action_tape": [{
        "a_requested": {"1": requested},
        "runtime_proposal_shortlists": {"1": [requested, second]},
        "runtime_shortlist_result": {
            "1": {"used": True, "rank": 2, "candidates_checked": 2},
        },
        "teacher_joint": {"1": None},
    }]}), encoding="utf-8")
    result = evaluation._tape_shortlist_checks(path, expected_k=3)
    assert result["all"]
    assert result["fallback_selected"] == 1
    assert result["max_candidates_checked"] == 2


def test_tape_shortlist_contract_fails_catalogue_expansion_or_teacher_use(tmp_path):
    values = [[1.0, float(index) / 10.0, 0.0] for index in range(4)]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"action_tape": [{
        "a_requested": {"1": values[0]},
        "runtime_proposal_shortlists": {"1": values},
        "runtime_shortlist_result": {
            "1": {"used": False, "rank": None, "candidates_checked": 4},
        },
        "teacher_joint": {"1": 13},
    }]}), encoding="utf-8")
    result = evaluation._tape_shortlist_checks(path, expected_k=3)
    assert not result["all"]
    assert not result["bounded"]
    assert not result["teacher_free"]
