from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.training import evaluate_random_soc_gate_repair_v1 as evaluator


@dataclass
class _Config:
    total_requested_uavs: int = 6
    max_simulation_seconds: float = 1210.0
    dt: float = 1.0
    teacher_enabled: bool = False
    reconcile_late_merge_slots_enabled: bool = False


class _Policy:
    def __init__(self, label: str):
        self.label = label
        self.model = SimpleNamespace()


def _patch_pair(monkeypatch, *, source_gate=True, candidate_gate=True):
    config = _Config()
    captured = {"rollouts": [], "loaders": [], "digest": []}

    monkeypatch.setattr(evaluator, "build_scenario_config", lambda scenario, **_: config)
    monkeypatch.setattr(evaluator, "_config_snapshot", lambda value: {"total": value.total_requested_uavs})

    def source_loader(path, loaded_config):
        captured["loaders"].append(("source", path, loaded_config))
        return _Policy("source"), {"loader": "strict-source"}

    def candidate_loader(path, loaded_config, *, expected_sha256):
        captured["loaders"].append(("candidate", path, loaded_config, expected_sha256))
        return _Policy("candidate"), {"loader": "new-repair"}

    monkeypatch.setattr(evaluator, "load_trained_checkpoint", source_loader)
    monkeypatch.setattr(evaluator, "load_repaired_checkpoint", candidate_loader)
    monkeypatch.setattr(
        evaluator,
        "_policy_weight_digest",
        lambda policy: f"stable-{policy.label}",
    )

    def fake_rollout(policy, loaded_config, **kwargs):
        captured["rollouts"].append((policy.label, loaded_config, kwargs))
        kwargs["output"].mkdir(parents=True)
        tape = kwargs["output"] / "action_tape.json"
        tape.write_text("{}", encoding="utf-8")
        summary = {
            "completed": 4 if policy.label == "source" else 5,
            "outcomes": {"1": "completed", "2": "emergency_landed"}
            if policy.label == "source"
            else {"1": "completed", "2": "completed", "3": "failed_energy"},
            "action_tape": str(tape),
            "label_rows": 0,
            "PPO_ALLOWED": False,
            "safety": {"SafetyInvariantError": 0},
            "safety_error": None,
        }
        return [SimpleNamespace(teacher_used=False)], summary

    monkeypatch.setattr(evaluator, "rollout", fake_rollout)

    def fake_gates(policy, loaded_config, steps, summary, protocol, *, deterministic, scenario):
        return (
            {"raw_gate": source_gate if policy.label == "source" else candidate_gate},
            {"raw_diagnostic": policy.label},
            {"raw_shortlist": policy.label},
            float(summary["completed"]) / loaded_config.total_requested_uavs,
        )

    monkeypatch.setattr(evaluator, "_rollout_gates", fake_gates)
    monkeypatch.setattr(
        evaluator,
        "lifecycle_metrics",
        lambda loaded_config, tape, *, low_soc_threshold: {
            "replay_exact": True,
            "rows": [{"base_priority": 1, "initial_soc": 0.7, "wait_seconds": 2.0}],
        },
    )
    return config, captured


def test_seed_scene_binding_is_exact(tmp_path):
    with pytest.raises(ValueError, match="fixed at 846301"):
        evaluator.run(
            source_checkpoint=tmp_path / "source.pt",
            candidate_checkpoint=tmp_path / "candidate.pt",
            candidate_sha256="a" * 64,
            output=tmp_path / "pair",
            scenario="flow6",
            seed=846302,
        )


def test_source_gate_failure_does_not_hide_candidate_and_completion_uses_outer_rows(
    tmp_path, monkeypatch
):
    config, captured = _patch_pair(monkeypatch, source_gate=False, candidate_gate=True)
    result = evaluator.run(
        source_checkpoint=tmp_path / "source.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        candidate_sha256="b" * 64,
        output=tmp_path / "pair",
        scenario="flow6",
        seed=846301,
    )

    assert [row[0] for row in captured["rollouts"]] == ["source", "candidate"]
    assert result["status"] == "COMPLETED"
    assert result["records"][0]["gates"] == {"raw_gate": False}
    assert result["records"][1]["gates"] == {"raw_gate": True}
    assert result["records"][0]["normal"] == 1
    assert result["records"][0]["emergency"] == 1
    assert result["records"][0]["failed_energy"] == 0
    assert result["records"][0]["unresolved"] == 4
    assert result["records"][1]["failed_energy"] == 1
    assert result["records"][1]["candidate_sha256"] == "b" * 64
    assert result["candidate_sha256"] == "b" * 64
    assert result["selection"]["source_metrics"]["requests"] == config.total_requested_uavs
    assert result["selection"]["source_metrics"]["completion_fraction"] == pytest.approx(4 / 6)
    assert result["selection"]["candidate_metrics"]["completion_fraction"] == pytest.approx(5 / 6)
    assert result["selection"]["gates"]["metrics_defined"] is False

    saved = json.loads((tmp_path / "pair" / "pair_summary.json").read_text(encoding="utf-8"))
    assert saved["P3_QUALIFIED"] is False
    assert saved["PPO_ALLOWED"] is False
    assert saved["P4_ALLOWED"] is False


def test_each_arm_is_deterministic_student_only_and_weights_are_unchanged(tmp_path, monkeypatch):
    _, captured = _patch_pair(monkeypatch)
    monkeypatch.setattr(
        evaluator.newtrainer,
        "train_gate_repair",
        lambda *args, **kwargs: pytest.fail("paired evaluation must not train"),
    )
    result = evaluator.run(
        source_checkpoint=tmp_path / "source.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        candidate_sha256="c" * 64,
        output=tmp_path / "pair",
        scenario="flow6",
        seed=846301,
    )

    assert all(call[2]["deterministic"] is True for call in captured["rollouts"])
    assert all(call[2]["collect_labels"] is False for call in captured["rollouts"])
    assert all(call[1].teacher_enabled is False for call in captured["rollouts"])
    assert [call[2]["max_steps"] for call in captured["rollouts"]] == [1210, 1210]
    assert all(row["label_rows"] == 0 for row in result["records"])
    assert all(row["PPO_ALLOWED"] is False for row in result["records"])
    assert all(row["weights_unchanged"] is True for row in result["records"])
    assert [item[0] for item in captured["loaders"]] == ["source", "candidate"]
    assert captured["loaders"][0][2] is captured["rollouts"][0][1]
    assert captured["loaders"][1][2] is captured["rollouts"][1][1]


def test_custom_checkpoint_loader_is_used_before_strict_label_dispatch(tmp_path, monkeypatch):
    config, captured = _patch_pair(monkeypatch)
    calls = []

    def custom_loader(path, loaded_config):
        calls.append((path, loaded_config))
        return _Policy("custom"), {
            "checkpoint_identity": "canonical_initial",
            "canonical_initial": True,
            "migration": {"allowed": True},
        }

    record, stopped, _ = evaluator._evaluate_arm(
        label="canonical_initial",
        checkpoint=tmp_path / "canonical.pt",
        candidate_sha256=None,
        config=config,
        protocol={"attributes": {"low_soc_analysis_threshold": 0.65}},
        scenario="flow6",
        seed=846301,
        output=tmp_path / "arm",
        checkpoint_loader=custom_loader,
    )

    assert stopped is False
    assert calls == [(tmp_path / "canonical.pt", config)]
    assert captured["loaders"] == []
    assert record["loader"] == {
        "checkpoint_identity": "canonical_initial",
        "canonical_initial": True,
        "migration": {"allowed": True},
    }


def test_safety_invariant_error_stops_later_arm_and_writes_failure_report(tmp_path, monkeypatch):
    config = _Config()
    loader_calls = []
    monkeypatch.setattr(evaluator, "build_scenario_config", lambda scenario, **_: config)
    monkeypatch.setattr(evaluator, "_config_snapshot", lambda value: {})
    monkeypatch.setattr(
        evaluator,
        "load_trained_checkpoint",
        lambda path, loaded_config: (loader_calls.append("source") or _Policy("source"), {}),
    )

    def candidate_loader(*args, **kwargs):
        loader_calls.append("candidate")
        raise AssertionError("candidate arm must not be loaded after SafetyInvariantError")

    monkeypatch.setattr(evaluator, "load_repaired_checkpoint", candidate_loader)
    monkeypatch.setattr(evaluator, "_policy_weight_digest", lambda policy: "same")

    def unsafe_rollout(policy, loaded_config, **kwargs):
        kwargs["output"].mkdir(parents=True)
        raise SafetyInvariantError("bounded safety barrier")

    monkeypatch.setattr(evaluator, "rollout", unsafe_rollout)
    result = evaluator.run(
        source_checkpoint=tmp_path / "source.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        candidate_sha256="d" * 64,
        output=tmp_path / "pair",
        scenario="flow6",
        seed=846301,
    )

    assert loader_calls == ["source"]
    assert result["status"] == "FAILED"
    assert result["records"][0]["safety_invariant_error"] is True
    assert result["records"][1]["status"] == "not_run"
    assert result["failure"]["safety_invariant_error"] is True
    assert (tmp_path / "pair" / "pair_summary.json").is_file()


def test_unexpected_arm_error_also_stops_later_arm(tmp_path, monkeypatch):
    config = _Config()
    loader_calls = []
    monkeypatch.setattr(evaluator, "build_scenario_config", lambda scenario, **_: config)
    monkeypatch.setattr(evaluator, "_config_snapshot", lambda value: {})
    monkeypatch.setattr(
        evaluator,
        "load_trained_checkpoint",
        lambda path, loaded_config: (loader_calls.append("source") or _Policy("source"), {}),
    )
    monkeypatch.setattr(
        evaluator,
        "load_repaired_checkpoint",
        lambda *args, **kwargs: loader_calls.append("candidate"),
    )
    monkeypatch.setattr(evaluator, "_policy_weight_digest", lambda policy: "same")

    def broken_rollout(policy, loaded_config, **kwargs):
        kwargs["output"].mkdir(parents=True)
        raise RuntimeError("replay implementation error")

    monkeypatch.setattr(evaluator, "rollout", broken_rollout)
    result = evaluator.run(
        source_checkpoint=tmp_path / "source.pt",
        candidate_checkpoint=tmp_path / "candidate.pt",
        candidate_sha256="f" * 64,
        output=tmp_path / "pair",
        scenario="flow6",
        seed=846301,
    )

    assert loader_calls == ["source"]
    assert result["status"] == "FAILED"
    assert result["stopped_after_source"] is True
    assert result["records"][1]["status"] == "not_run"
    assert result["failure"]["arm_errors"][0]["type"] == "RuntimeError"


def test_loader_record_omits_checkpoint_payload_state():
    record = evaluator._loader_record({
        "checkpoint_sha256": "sha",
        "stage": "repair",
        "architecture_id": "arch",
        "temperature": 1.0,
        "current_execution_metadata": {"teacher_enabled": False},
        "model": {"large": "state"},
        "report": {"history": [1, 2, 3]},
    })
    assert record == {
        "checkpoint_sha256": "sha",
        "stage": "repair",
        "architecture_id": "arch",
        "temperature": 1.0,
        "current_execution_metadata": {"teacher_enabled": False},
    }


def test_loader_record_keeps_bounded_canonical_provenance():
    record = evaluator._loader_record({
        "checkpoint_identity": "canonical_initial",
        "canonical_initial": True,
        "weights_match": True,
        "source_payload_unchanged": True,
        "migration": {
            "allowed": True,
            "source_unchanged": True,
            "applied_deltas": [
                {"field": "vehicle_profile_version", "source": "v1", "target": "v2"}
            ],
        },
        "payload_bindings": {
            "stage": "canonical",
            "checkpoint_promotable": False,
        },
        "model": {"large": "state"},
    })
    assert record["checkpoint_identity"] == "canonical_initial"
    assert record["canonical_initial"] is True
    assert record["weights_match"] is True
    assert record["source_payload_unchanged"] is True
    assert record["migration"]["applied_deltas"][0]["target"] == "v2"
    assert record["payload_bindings"]["stage"] == "canonical"
    json.dumps(record, allow_nan=False)


def test_output_must_be_fresh(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        evaluator.run(
            source_checkpoint=tmp_path / "source.pt",
            candidate_checkpoint=tmp_path / "candidate.pt",
            candidate_sha256="e" * 64,
            output=output,
            scenario="dense24",
            seed=846303,
        )
