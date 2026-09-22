import json
import sys

import pytest

from terminal_marl.training import evaluate_v4_2_p12_current_mdp_v1 as evaluation


def _record(*, complete=True, safety=True, replay=True):
    gates = {
        name: True for name in evaluation.REQUIRED_GATE_NAMES
    }
    gates["completed_equals_requests"] = complete
    gates["safety_zero"] = safety
    gates["exact_replay"] = replay
    return {
        "stage": "P2",
        "controller": "candidate",
        "seed": 1,
        "count": 3,
        "requests": 3,
        "completed": 3 if complete else 2,
        "mean_wait_seconds": 10.0 if complete else None,
        "mean_request_to_touchdown_seconds": 20.0 if complete else None,
        "safety": {"collisions": 0} if safety else {"collisions": 1},
        "action_tape": "tape.json",
        "gates": gates,
    }


def test_protocol_and_current_mdp_config_semantics():
    value = evaluation.protocol()
    assert value["formal"]["p1"]["seeds"] == list(evaluation.P1_FORMAL_SEEDS)
    assert value["formal"]["p1"]["counts"] == list(evaluation.P1_COUNTS)
    assert value["formal"]["p2"]["seeds"] == list(evaluation.P2_FORMAL_SEEDS)
    assert value["formal"]["p2"]["counts"] == list(evaluation.P2_COUNTS)
    config = evaluation.config_for(6)
    assert config.initial_soc_mode == "fixed"
    assert config.initial_soc == 0.8
    assert config.energy_capacity == 1500.0
    assert config.safe_distance == 80.0
    assert config.nmac_distance == 40.0
    assert config.collision_distance == 10.0
    assert config.lz_assignment_mode == "lowest_id"
    assert config.release_arbitration_mode == "legacy_insertion"
    assert config.final_queue_capacity == 1
    assert config.fixed_arrival_schedule == (10.0, 70.0, 130.0, 190.0, 250.0, 310.0)
    assert config.max_simulation_seconds == 1210.0
    assert evaluation.config_for(6, shortlist_k=3).runtime_proposal_shortlist_k == 3


def test_p2_tape_contract_requires_three_proposals_on_release(tmp_path):
    first = [1.0, 0.0, 0.0]
    second = [1.0, 0.2, 0.0]
    third = [1.0, 0.4, 0.0]
    path = tmp_path / "p2_tape.json"
    path.write_text(json.dumps({
        "action_tape": [{
            "requested_gate": {"1": 1},
            "runtime_proposal_shortlists": {"1": [first, second, third]},
            "teacher_joint": {"1": None},
        }]
    }), encoding="utf-8")
    checks = evaluation._tape_execution_checks(path, expected_shortlist_k=3)
    assert checks["actor_decisions"] == 1
    assert checks["release_rows"] == 1
    assert checks["shortlist_rows"] == 1
    assert checks["shortlist_contract"]
    assert checks["teacher_free"]


def test_p2_student_runner_passes_shortlist_three_into_rollout(tmp_path, monkeypatch):
    tape = tmp_path / "runner_tape.json"
    tape.write_text(json.dumps({
        "action_tape": [{
            "requested_gate": {"1": 1},
            "runtime_proposal_shortlists": {"1": [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.0, 0.4, 0.0]]},
            "teacher_joint": {"1": None},
        }]
    }), encoding="utf-8")

    def fake_rollout(policy, config, *, seed, max_steps, output, deterministic, collect_labels):
        assert config.runtime_proposal_shortlist_k == 3
        return [], {
            "action_tape": str(tape),
            "safety": {"collisions": 0},
            "counts": {"shield": 0, "actor_decisions": 1},
            "completed": 3,
            "terminal": True,
            "seconds": 1.0,
            "label_rows": 0,
        }

    monkeypatch.setattr(evaluation, "student_rollout", fake_rollout)
    monkeypatch.setattr(
        evaluation,
        "_replay_lifecycle",
        lambda config, path: ({
            "completed": 3,
            "terminal": True,
            "mean_wait_seconds": 1.0,
            "mean_request_to_touchdown_seconds": 2.0,
            "replay_exact": True,
        }, {"samples": 1, "reverse_samples": 0, "max_inner_excursion_m": 0.0}),
    )
    record = evaluation._run_p2_student(object(), 832001, 3, tmp_path / "out", max_steps=20)
    assert record["execution"]["shortlist_contract"]
    assert record["gates"]["shortlist_contract"]


@pytest.mark.parametrize("field", ["completed_equals_requests", "safety_zero", "exact_replay"])
def test_aggregation_fails_closed_on_incomplete_safety_or_replay(field):
    candidate = _record(
        complete=field != "completed_equals_requests",
        safety=field != "safety_zero",
        replay=field != "exact_replay",
    )
    rule = dict(candidate, controller="rule", seed=1)
    with pytest.raises(evaluation.CurrentMDPInputError):
        evaluation.aggregate_records([candidate, rule], "P2")


def test_smoke_manifest_serializes_without_formal_seed_consumption(tmp_path, monkeypatch):
    candidate = _record()
    candidate["controller"] = "candidate"
    rule = dict(candidate, controller="rule")

    monkeypatch.setattr(evaluation, "_preflight", lambda *_: (object(), object()))
    monkeypatch.setattr(
        evaluation,
        "_run_pilot_episode",
        lambda stage, controller, policy, seed, count, output, *, max_steps: dict(
            candidate if controller == "candidate" else rule,
            stage=stage,
            controller=controller,
            seed=seed,
            count=count,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "_run_p2_student",
        lambda policy, seed, count, output, *, max_steps: dict(
            candidate,
            stage="P2",
            controller="candidate",
            seed=seed,
            count=count,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "_run_rule",
        lambda stage, seed, count, output, *, max_steps: dict(
            rule,
            stage=stage,
            controller="rule",
            seed=seed,
            count=count,
        ),
    )

    output = tmp_path / "smoke"
    report = evaluation.run(
        output=output,
        p1_checkpoint=evaluation.P1_CHECKPOINT_PATH,
        p2_checkpoint=evaluation.P2_CHECKPOINT_PATH,
        history_root=tmp_path,
        smoke=True,
    )
    assert report["status"] == "DIAGNOSTIC_SMOKE"
    assert report["formal_seed_consumption"] is False
    assert report["PPO_ALLOWED"] is False
    assert report["P3_ALLOWED"] is False
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["records"]


def test_formal_safety_gate_persists_record_and_stops_before_next_episode(tmp_path, monkeypatch):
    failed = _record(safety=False)
    failed.update(stage="P1", controller="candidate", seed=834101, count=3)
    calls = []

    monkeypatch.setattr(evaluation, "_preflight", lambda *_: (object(), object()))
    monkeypatch.setattr(
        evaluation,
        "_run_pilot_episode",
        lambda stage, controller, policy, seed, count, output, *, max_steps: (
            calls.append((stage, controller, seed)) or failed
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "_run_rule",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rule episode must not start")),
    )

    output = tmp_path / "formal_fail"
    report = evaluation.run(
        output=output,
        p1_checkpoint=evaluation.P1_CHECKPOINT_PATH,
        p2_checkpoint=evaluation.P2_CHECKPOINT_PATH,
        history_root=tmp_path,
        smoke=False,
    )
    assert report["status"] == "FORMAL_FAIL"
    assert report["record_count"] == 1
    assert report["failure"]["category"] == "safety"
    assert report["failure"]["failed_gates"] == ["safety_zero"]
    assert report["attempted_seeds"] == [834101]
    assert report["consumed_seeds"] == [834101]
    assert len(json.loads((output / "records.json").read_text(encoding="utf-8"))["records"]) == 1
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "FORMAL_FAIL"
    assert progress["record_count"] == 1
    assert calls == [("P1", "candidate", 834101)]


def test_missing_or_invalid_safety_counters_fail_closed():
    assert not evaluation._safety_zero({})
    counters, valid, error = evaluation._safety_counts({"safety": {}})
    assert counters == {}
    assert not valid
    assert error


def test_formal_replay_exception_writes_terminal_failure_manifest(tmp_path, monkeypatch):
    calls = []

    def fail_replay(stage, controller, policy, seed, count, output, *, max_steps):
        calls.append((stage, controller, seed))
        raise RuntimeError("replay tape mismatch")

    monkeypatch.setattr(evaluation, "_preflight", lambda *_: (object(), object()))
    monkeypatch.setattr(evaluation, "_run_pilot_episode", fail_replay)
    monkeypatch.setattr(
        evaluation,
        "_run_rule",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rule episode must not start")),
    )

    output = tmp_path / "replay_fail"
    report = evaluation.run(
        output=output,
        p1_checkpoint=evaluation.P1_CHECKPOINT_PATH,
        p2_checkpoint=evaluation.P2_CHECKPOINT_PATH,
        history_root=tmp_path,
        smoke=False,
    )
    assert report["status"] == "FORMAL_FAIL"
    assert report["record_count"] == 0
    assert report["failure"]["category"] == "replay"
    assert "replay tape mismatch" in report["failure"]["error"]
    assert calls == [("P1", "candidate", 834101)]
    assert json.loads((output / "records.json").read_text(encoding="utf-8"))["records"] == []


def test_cli_returns_nonzero_for_terminal_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "run",
        lambda **kwargs: {"status": "FORMAL_FAIL"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["p12", "--output", str(tmp_path / "out"), "--history-root", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as error:
        evaluation._main()
    assert error.value.code == 1
    counters, valid, error = evaluation._safety_counts({"safety": {"collisions": "zero"}})
    assert counters == {}
    assert not valid
    assert error
