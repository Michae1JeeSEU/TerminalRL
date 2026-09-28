from __future__ import annotations

import json
from pathlib import Path

from terminal_marl.training import diagnose_p3_landing_outcome_credit as runner


def _row(
    arrival_id: int,
    step: int,
    *,
    p_release: float = 0.5,
    q_k3: float = 0.8,
    certified: list[int] | None = None,
    soc: float = 0.5,
) -> dict:
    return {
        "seed": 873103,
        "split": "train",
        "scenario": "dense24",
        "arrival_id": arrival_id,
        "uid": arrival_id,
        "step": step,
        "soc": soc,
        "source_release_probability": p_release,
        "q_k3": q_k3,
        "certified_proposals": [1] if certified is None else certified,
        "catalogue_certified_proposals": [2],
        "observation": [0.0] * 88,
        "source_gate_logits": [0.0, 0.0],
        "source_proposal_logits": [0.0] * 25,
    }


def test_uniform_probe_bound_keeps_endpoints_and_unique_indices() -> None:
    rows = [_row(12, step) for step in range(100)]
    selected = runner._uniform_rows(rows, limit=40)
    assert len(selected) == 40
    assert selected[0]["step"] == 0
    assert selected[-1]["step"] == 99
    assert len({int(row["step"]) for row in selected}) == 40


def test_final_selection_is_separated_and_uses_runtime_c_only() -> None:
    rows = [
        _row(12, 10, p_release=0.1, q_k3=0.9, certified=[3]),
        _row(12, 30, p_release=0.9, q_k3=0.9, certified=[4]),
        _row(12, 60, p_release=0.2, q_k3=0.8, certified=[5]),
        _row(12, 80, p_release=0.3, q_k3=0.7, certified=[]),
    ]
    selected = runner._select_final_states(rows)
    assert len(selected) == 3
    assert all(row["certified_proposals"] for row in selected)
    assert all(abs(int(left["step"]) - int(right["step"])) >= 10 for left in selected for right in selected if left is not right)
    reasons = {row["final_selection_reason"] for row in selected}
    assert reasons == {"max_withheld", "max_proposal_miss", "lowest_soc"}


def test_local_window_selection_keeps_empty_c_and_non_wait_rows() -> None:
    rows = []
    for step in (102, 103, 104):
        row = _row(12, step, certified=[] if step == 102 else [step - 99])
        row.update({"uid": 42, "scenario": "dense24", "split": "train"})
        row["source_action_metadata"] = {
            "executed_gate": "RELEASE" if step == 103 else "WAIT",
            "requested_gate": 1 if step == 103 else 0,
        }
        rows.append(row)

    selected = runner._select_final_states(rows, local_window=True)

    assert [int(row["step"]) for row in selected] == [102, 103, 104]
    assert selected[0]["certified_proposals"] == []
    assert selected[1]["source_action_metadata"]["executed_gate"] == "RELEASE"
    assert all(str(row["final_selection_reason"]).startswith("local_window_step_") for row in selected)


def test_local_window_selection_requires_all_fixed_steps() -> None:
    rows = [_row(12, step, certified=[]) for step in (102, 104)]
    for row in rows:
        row.update({"uid": 42, "scenario": "dense24", "split": "train"})

    try:
        runner._select_final_states(rows, local_window=True)
    except runner.LandingOutcomeCreditError as exc:
        assert "local window final selection" in str(exc)
    else:
        raise AssertionError("missing local-window step was accepted")


def test_local_window_smoke_is_step_102_and_not_training_input() -> None:
    rows = []
    for step in (102, 103, 104):
        row = _row(12, step, certified=[])
        row.update({"uid": 42, "scenario": "dense24", "split": "train"})
        rows.append(row)

    selected = runner._select_final_states(rows, smoke=True, local_window=True)
    metadata = runner._window_metadata(local_window=True, smoke=True)
    full_metadata = runner._window_metadata(local_window=True, smoke=False)

    assert [int(row["step"]) for row in selected] == [102]
    assert metadata["local_window"] is True
    assert metadata["window_steps"] == [102]
    assert metadata["mode"] == runner.LOCAL_WINDOW_MODE
    assert metadata["training_input_eligible"] is False
    assert full_metadata["window_steps"] == [102, 103, 104]
    assert full_metadata["training_input_eligible"] is True


def test_parser_accepts_local_window_without_changing_legacy_default() -> None:
    parser = runner.build_parser()
    legacy = parser.parse_args(["--output-dir", "out"])
    local = parser.parse_args(["--output-dir", "out", "--local-window"])

    assert legacy.local_window is False
    assert local.local_window is True


def test_runtime_label_replaces_c_and_preserves_catalogue(monkeypatch) -> None:
    source = _row(12, 10, certified=[2])
    branch = {"decision": "approved", "reward": 0.0, "next_value": 0.0, "safety_counter_delta": {}}
    report = {
        "wait": dict(branch),
        "releases": [dict(branch) for _ in range(25)],
        "runtime_singleton_branches": [dict(branch) for _ in range(25)],
        "runtime_certified_proposals": [7, 1, 7],
        "runtime_q_vs_approved_mass_abs_error": 0.0,
        "q_k3_runtime_labels": 0.75,
        "label_scope": "same_prestate_fixed_other_tape_actions_runtime_singleton",
        "branch_rng_restored": True,
        "live_state_digest_before": "a",
        "live_state_digest_after": "a",
        "model_digest_before": "m",
        "model_digest_after": "m",
    }
    monkeypatch.setattr(runner, "_state_digest", lambda _env: "a")
    monkeypatch.setattr(runner, "_model_digest", lambda _policy: "m")
    monkeypatch.setattr(
        runner.collector,
        "_state_row",
        lambda _env, _uid, **_kwargs: {
            **source,
            "certified_proposals": [2],
            "catalogue_certified_proposals": [2],
        },
    )
    monkeypatch.setattr(runner.terminal_credit, "_rng_snapshot", lambda: ("py", "np", "torch"))
    monkeypatch.setattr(runner.terminal_credit, "_restore_rng", lambda _snapshot: None)
    monkeypatch.setattr(runner.terminal_credit, "_rng_equal", lambda _left, _right: True)
    monkeypatch.setattr(runner.runtime_collector.runtime_diagnostic, "_evaluate_branches", lambda *args: report)
    labelled, _ = runner._runtime_row(
        row=source,
        snapshot=object(),
        capture={"actions": {}, "shortlists": {}},
        source_policy={"policy": object()},
        branch_seed=1,
    )
    assert labelled["catalogue_certified_proposals"] == [2]
    assert labelled["certified_proposals"] == [1, 7]
    assert labelled["q_k3"] == 0.75
    json.dumps(runner._jsonable(labelled), allow_nan=False)


def test_source_action_metadata_uses_executed_vector_over_decoded_gate() -> None:
    metadata = runner._source_action_metadata(
        {
            "requested_gate": {"7": 1},
            "requested_proposal": {"7": 4},
            "executed_mapped_action": {"7": [0.0, 0.0, -1.0]},
            "decoded_action": {"7": {"gate": "RELEASE", "proposal": 4}},
        },
        7,
    )
    assert metadata["requested_gate"] == 1
    assert metadata["decoded_action"]["gate"] == "RELEASE"
    assert metadata["executed_gate"] == "WAIT"
    assert metadata["applied_gate"] == "WAIT"


def test_normal_opportunity_rejects_wait_only_and_mixed_sign() -> None:
    wait_only = runner._normal_opportunity([
        {"normal_delta": -1, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": -1, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": -1, "failed_delta": 0, "technical_ok": True},
    ])
    assert wait_only["normal_opportunity"] is False
    mixed = runner._normal_opportunity([
        {"normal_delta": 1, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": -1, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": 1, "failed_delta": 0, "technical_ok": True},
    ])
    assert mixed["normal_opportunity"] is False
    accepted = runner._normal_opportunity([
        {"normal_delta": 0, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": 1, "failed_delta": 0, "technical_ok": True},
        {"normal_delta": 2, "failed_delta": -1, "technical_ok": True},
    ])
    assert accepted["normal_opportunity"] is True
    assert accepted["repeat_count_valid"] is True
    one_repeat = runner._normal_opportunity([
        {"normal_delta": 1, "failed_delta": 0, "technical_ok": True},
    ])
    assert one_repeat["normal_opportunity"] is False
    assert one_repeat["repeat_count_valid"] is False


def test_state_opportunities_can_identify_wait_direction_and_probability_gap() -> None:
    def branch(normal: int, failed: int, wait: float) -> dict:
        return {
            "lifecycle": {
                "outcomes": {"completed": normal, "failed_energy": failed},
                "priority_weighted_wait_seconds": wait,
                "low_soc": {"count": 1, "mean_wait_seconds": wait},
            },
            "target_terminal_row": {"outcome": "completed"},
        }

    reports = [
        {
            "acceptance": True,
            "branches": {
                "wait": branch(10, 0, 10.0),
                "source_release": branch(9, 1, 11.0),
                "runtime_certified_release": branch(9, 1, 11.0),
            },
        }
        for _ in range(3)
    ]
    opportunity = runner._state_opportunities(reports, _row(21, 526, p_release=0.25))
    assert opportunity["gate"]["direction"] == "wait"
    assert opportunity["gate"]["release"]["normal_opportunity"]["normal_opportunity"] is False
    assert opportunity["gate"]["wait"]["normal_opportunity"]["normal_opportunity"] is True
    assert opportunity["gate"]["wait"]["source_policy_gap"] == 0.25
    assert opportunity["gate"]["source_policy_gap"] == 0.25
    assert opportunity["gate"]["training_label_eligible"] is False


def test_probe_selection_serializes_input_structure(tmp_path: Path) -> None:
    path = tmp_path / "probe_selection.json"
    payload = {
        "stage": runner.STAGE,
        "rows": [_row(21, 526)],
        "source_replay": {"exact": True},
        "checks": {"replay_exact": True},
    }
    runner._write_json(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["stage"] == runner.STAGE
    assert loaded["rows"][0]["arrival_id"] == 21
    assert loaded["rows"][0]["observation"]
