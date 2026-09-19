from __future__ import annotations

import inspect

import pytest

from terminal_marl.training import diagnose_v4_2_p2_gate_proposal_feasible_set_v1 as diagnostic
from terminal_marl.training.student_rollout_v4_2 import rollout


def _row(*, feasible: tuple[int, ...], candidate: int, teacher: int | None,
         accepted: bool, interface: str = "o2m", arrival_id: int = 7) -> dict:
    return {
        "requested_gate": 1,
        "requested_proposal": candidate,
        "teacher_gate": int(teacher is not None),
        "teacher_proposal": teacher,
        "teacher_feasible_proposals": list(feasible),
        "teacher_catalogue_complete": True,
        "candidate_proposal_feasible": candidate in feasible,
        "teacher_proposal_feasible": teacher in feasible if teacher is not None else False,
        "nominal_proposal_12_feasible": 12 in feasible,
        "policy_top1_feasible": candidate in feasible,
        "policy_top3_feasible_hit": bool(feasible),
        "feasible_set_size": len(feasible),
        "execution_accepted": accepted,
        "interface": interface,
        "arrival_id": arrival_id,
    }


def test_protocol_freezes_fresh_catalogue_diagnostic_and_no_authority():
    value = diagnostic.protocol()
    assert value["formal"]["seeds"] == list(diagnostic.FORMAL_SEEDS)
    assert value["bindings"]["catalogue_scope"] == "actor_requested_RELEASE_only"
    assert value["authorizations"]["always_false"]["SET_VALUED_SUPPORT_ALLOWED"] is False
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False


def test_summary_crosses_feasible_membership_with_execution():
    rows = [
        _row(feasible=(12, 13), candidate=12, teacher=12, accepted=True),
        _row(feasible=(12,), candidate=7, teacher=12, accepted=False),
        _row(feasible=(), candidate=7, teacher=None, accepted=False, interface="m2i"),
    ]
    result = diagnostic.summarize(rows)
    assert result["requested_release_events"] == 3
    assert result["teacher_release_events"] == 2
    assert result["empty_feasible_set_events"] == 1
    assert result["candidate_proposal_feasible"] == 1
    assert result["teacher_proposal_feasible"] == 2
    assert result["nominal_proposal_12_feasible"] == 2
    assert result["execution_accepted"] == 1
    assert result["catalogue_complete"] is True


def test_catalogue_rollout_requires_diagnostic_rows(tmp_path):
    with pytest.raises(ValueError, match="catalogue diagnostics require"):
        rollout(
            None, diagnostic.attribution.diagnostic_config(), seed=1, max_steps=1,
            output=tmp_path / "invalid", deterministic=True, collect_labels=True,
            collect_diagnostics=False, collect_catalogue_diagnostics=True,
        )


def test_runner_has_no_training_or_repair_path():
    source = inspect.getsource(diagnostic)
    assert "optimizer.step(" not in source
    assert '"rows_authorized_for_training": False' in source
    assert '"PROPOSAL_REPAIR_ALLOWED": False' in source
    assert '"PPO_ALLOWED": False' in source

