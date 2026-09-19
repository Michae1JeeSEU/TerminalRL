from __future__ import annotations

import inspect

import pytest

from terminal_marl.training import diagnose_v4_2_p2_gate_feasibility_v1 as diagnostic
from terminal_marl.training.student_rollout_v4_2 import rollout


def _row(
    *, interface: str = "m2i", arrival_id: int = 7, requested_gate: int = 1,
    teacher_gate: int = 1, requested_proposal: int | None = 4,
    teacher_proposal: int | None = 5, decision: str | None = "safety_reject",
    blocker: str | None = "PREDICTED_SEPARATION", retry: int = 1,
) -> dict:
    return {
        "interface": interface,
        "arrival_id": arrival_id,
        "requested_gate": requested_gate,
        "teacher_gate": teacher_gate,
        "requested_proposal": requested_proposal,
        "teacher_proposal": teacher_proposal,
        "release_decision": decision,
        "release_primary_blocker": blocker,
        "release_retry_count": retry,
        "release_diagnostic_present": requested_gate == 1,
    }


def test_protocol_and_config_are_frozen_label_only_flow12():
    value = diagnostic.protocol()
    assert value["formal"]["seeds"] == list(diagnostic.FORMAL_SEEDS)
    assert value["formal"]["scenarios"] == ["flow12"] * 6
    assert value["authorizations"]["always_false"]["PPO_ALLOWED"] is False
    config = diagnostic.diagnostic_config()
    assert config.total_requested_uavs == 12
    assert config.teacher_enabled is True
    assert config.teacher_mode == "teacher_label_only"
    assert config.scheduler_mode == "execution"
    assert config.high_level_enabled is False


def test_summary_separates_gate_and_proposal_execution_signals():
    rows = [
        _row(),
        _row(arrival_id=8, requested_proposal=5, teacher_proposal=5,
             decision="approved", blocker=None, retry=0),
        _row(arrival_id=2, requested_gate=0, teacher_gate=1,
             requested_proposal=None, teacher_proposal=3, decision=None,
             blocker=None, retry=0),
        _row(arrival_id=3, requested_gate=1, teacher_gate=0,
             requested_proposal=2, teacher_proposal=None,
             decision="final_capacity", blocker="FINAL_RESOURCE", retry=0),
    ]
    result = diagnostic.summarize_rows(rows)
    assert result["actor_events"] == 4
    assert result["gate_disagreement"] == {"false_WAIT": 1, "false_RELEASE": 1}
    assert result["both_release"] == 2
    assert result["proposal_match"] == 1
    assert result["proposal_mismatch"] == 1
    assert result["release_attempts"] == 3
    assert result["release_accepted"] == 1
    assert result["release_rejected"] == 2
    assert result["rejected_while_teacher_release"] == 1
    assert result["proposal_mismatch_and_rejected_while_teacher_release"] == 1
    assert result["rejected_primary_blockers"] == {
        "FINAL_RESOURCE": 1, "PREDICTED_SEPARATION": 1,
    }
    assert result["release_telemetry_complete"] is True


def test_attribution_preserves_interface_and_arrival_half():
    rows = [
        _row(interface="o2m", arrival_id=1, decision="approved", blocker=None, retry=0),
        _row(interface="m2i", arrival_id=8),
        _row(interface="i2f", arrival_id=11, decision="final_capacity",
             blocker="FINAL_RESOURCE", retry=2),
    ]
    result = diagnostic.attribution(rows)
    assert result["overall"]["actor_events"] == 3
    assert result["by_arrival_half"]["early_0_5"]["actor_events"] == 1
    assert result["by_arrival_half"]["late_6_11"]["actor_events"] == 2
    assert result["by_interface"]["m2i"]["release_rejected"] == 1
    assert result["by_interface"]["i2f"]["rejected_primary_blockers"] == {"FINAL_RESOURCE": 1}


def test_diagnostic_rollout_requires_label_sidecar(tmp_path):
    with pytest.raises(ValueError, match="label-only Teacher rows"):
        rollout(
            None, diagnostic.diagnostic_config(), seed=1, max_steps=1,
            output=tmp_path / "invalid", deterministic=True,
            collect_labels=False, collect_diagnostics=True,
        )


def test_runner_has_no_training_or_promotion_path():
    source = inspect.getsource(diagnostic)
    assert "optimizer.step(" not in source
    assert '"rows_authorized_for_training": False' in source
    assert '"PPO_ALLOWED": False' in source
    assert '"checkpoint_promotable": False' in source

