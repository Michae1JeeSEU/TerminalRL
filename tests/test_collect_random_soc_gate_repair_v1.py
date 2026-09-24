from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.student_ppo_v4_2 import StudentPPOV42
from terminal_marl.core.low_level_joint_action_v4_1 import joint_action_v4_1
from terminal_marl.training import collect_random_soc_gate_repair_v1 as collector
from terminal_marl.training import student_rollout_v4_2 as student
from terminal_marl.training.run_v4_2_hierarchical_s0_qualification import _config


def test_selector_catches_instant_actor_and_gate_change_without_stride():
    previous_actor = {}
    previous_gate = {}

    assert student._label_query_uids(
        tick=0, uids=["1"], actor_mask=np.asarray([False]), gates=np.asarray([0]),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == {"1"}
    assert student._label_query_uids(
        tick=1, uids=["1"], actor_mask=np.asarray([True]), gates=np.asarray([0]),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == {"1"}
    assert student._label_query_uids(
        tick=2, uids=["1"], actor_mask=np.asarray([True]), gates=np.asarray([1]),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == {"1"}
    assert student._label_query_uids(
        tick=3, uids=["1"], actor_mask=np.asarray([True]), gates=np.asarray([1]),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == set()
    # Missing for one tick counts as non-actor when the UID reappears.
    assert student._label_query_uids(
        tick=4, uids=[], actor_mask=np.asarray([], dtype=bool), gates=np.asarray([], dtype=int),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == set()
    assert student._label_query_uids(
        tick=5, uids=["1"], actor_mask=np.asarray([True]), gates=np.asarray([1]),
        label_stride_steps=10, previous_actor_mask=previous_actor,
        previous_requested_gates=previous_gate,
    ) == {"1"}


def test_stride_one_keeps_legacy_every_uid_sampling_and_invalid_stride_rejected():
    previous_actor = {}
    previous_gate = {}
    for tick in range(3):
        assert student._label_query_uids(
            tick=tick, uids=["1", "2"], actor_mask=np.asarray([False, True]),
            gates=np.asarray([0, 1]), label_stride_steps=1,
            previous_actor_mask=previous_actor, previous_requested_gates=previous_gate,
        ) == {"1", "2"}
    with pytest.raises(ValueError, match="positive integer"):
        student._validate_label_stride_steps(0)
    with pytest.raises(ValueError, match="positive integer"):
        collector.run(
            checkpoint="unused.pt", output="unused-output", scenario="flow6",
            seed=3, max_steps=1, label_stride=0,
        )


def test_collection_uses_legal_execution_config():
    config = collector.build_collection_configs("flow6")
    assert config.reconcile_late_merge_slots_enabled is False
    assert config.teacher_enabled is False
    assert config.scheduler_mode == "execution"


def test_collector_uses_execution_loader_then_writes_actor_only_rows(tmp_path, monkeypatch):
    captured = {}

    class FakeModel:
        def state_dict(self):
            return {"weight": np.asarray([1.0])}

    policy = SimpleNamespace(model=FakeModel())

    def fake_loader(checkpoint, config):
        captured["loader_config"] = config
        return policy, {"checkpoint_sha256": "checkpoint-sha"}

    def fake_rollout(policy_arg, config, **kwargs):
        captured["rollout_config"] = config
        captured["rollout_kwargs"] = kwargs
        kwargs["output"].mkdir()
        rows = [
            {"uav_id": "1", "actor_mask": True, "observed_soc": 0.7, "base_priority": 2},
            {"uav_id": "2", "actor_mask": False, "observed_soc": 0.8, "base_priority": 1},
        ]
        (kwargs["output"] / "labels.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
        return [], {"terminal": True, "steps": 1, "label_rows": len(rows), "PPO_ALLOWED": False}

    monkeypatch.setattr(collector.diagnostic, "load_trained_checkpoint", fake_loader)
    monkeypatch.setattr(collector.diagnostic, "_policy_weight_digest", lambda _: "stable-digest")
    monkeypatch.setattr(collector.student, "rollout", fake_rollout)
    output = tmp_path / "collection"
    summary = collector.run(
        checkpoint=tmp_path / "frozen.pt", output=output, scenario="flow6",
        seed=3, max_steps=1, label_stride=10,
    )

    assert captured["loader_config"] is captured["rollout_config"]
    assert captured["loader_config"].teacher_enabled is False
    assert captured["rollout_config"].teacher_mode == "teacher_label_only"
    assert captured["rollout_kwargs"]["deterministic"] is True
    assert captured["rollout_kwargs"]["collect_catalogue_diagnostics"] is False
    assert captured["rollout_kwargs"]["label_stride_steps"] == 10
    assert captured["rollout_kwargs"]["state_only_labels"] is True
    assert summary["source_checkpoint_sha256"] == "checkpoint-sha"
    assert summary["all_terminal"] is True
    assert summary["weights_unchanged"] is True
    assert summary["PPO_ALLOWED"] is False
    assert summary["external_state_only_labeler"] is True
    assert summary["teacher_enabled"] is False
    assert summary["labeler_called"] is True
    assert summary["actor_label_rows"] == 1
    assert len((output / "actor_labels.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_collector_uses_strict_repaired_loader_and_summarizes_only_provenance(
    tmp_path, monkeypatch,
):
    captured = {}

    class FakeModel:
        def state_dict(self):
            return {"weight": np.asarray([1.0])}

    policy = SimpleNamespace(model=FakeModel())
    checkpoint = tmp_path / "repaired.pt"
    checkpoint.write_bytes(b"repaired-checkpoint")
    expected_sha = collector._sha256(checkpoint)
    payload = {
        "metadata": {
            "stage": "RANDOM_SOC_GATE_REPAIR_V1",
            "source_checkpoint_sha256": "source-sha",
            "architecture_id": "CALIBRATED_GATE_RESIDUAL_ADAPTER_V4_2",
            "temperature": 2.5,
            "supervised_training_called": True,
            "teacher_called": False,
            "ppo_called": False,
            "PPO_ALLOWED": False,
            "checkpoint_promotable": False,
            "optimizer_restored": False,
            "optimizer_saved": False,
        },
        "model": {"tensor_payload": np.asarray([99.0])},
    }

    def fake_repaired_loader(path, config, *, expected_sha256):
        captured["path"] = path
        captured["config"] = config
        captured["expected_sha256"] = expected_sha256
        return policy, payload

    def fail_execution_loader(*args, **kwargs):
        raise AssertionError("default execution loader must not be used")

    def fake_rollout(policy_arg, config, **kwargs):
        captured["rollout_config"] = config
        kwargs["output"].mkdir()
        row = {"uav_id": "1", "actor_mask": True, "observed_soc": 0.7}
        (kwargs["output"] / "labels.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8",
        )
        return [], {"terminal": True, "steps": 1, "PPO_ALLOWED": False}

    monkeypatch.setattr(collector.repair_trainer, "load_repaired_checkpoint", fake_repaired_loader)
    monkeypatch.setattr(collector.diagnostic, "load_trained_checkpoint", fail_execution_loader)
    monkeypatch.setattr(collector.diagnostic, "_policy_weight_digest", lambda _: "stable-digest")
    monkeypatch.setattr(collector.student, "rollout", fake_rollout)

    output = tmp_path / "collection"
    summary = collector.run(
        checkpoint=checkpoint, output=output, scenario="flow6",
        seed=3, max_steps=1, label_stride=10,
        repaired_checkpoint_sha256=expected_sha,
    )

    assert captured["path"] == checkpoint
    assert captured["config"] is captured["rollout_config"]
    assert captured["expected_sha256"] == expected_sha
    assert summary["collection_loader"] == "load_repaired_checkpoint"
    assert summary["collection_source_stage"] == "RANDOM_SOC_GATE_REPAIR_V1"
    assert summary["collection_source_checkpoint_sha256"] == expected_sha
    assert summary["repaired_checkpoint_sha256"] == expected_sha
    assert summary["checkpoint_info"]["metadata"]["stage"] == "RANDOM_SOC_GATE_REPAIR_V1"
    assert "model" not in summary["checkpoint_info"]
    assert "tensor_payload" not in json.dumps(summary)
    written = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert written["collection_source_checkpoint_sha256"] == expected_sha
    assert "tensor_payload" not in json.dumps(written)


def test_repaired_checkpoint_sha_error_propagates_before_rollout(tmp_path, monkeypatch):
    checkpoint = tmp_path / "repaired.pt"
    checkpoint.write_bytes(b"candidate")
    output = tmp_path / "collection"
    rollout_called = False

    def fail_rollout(*args, **kwargs):
        nonlocal rollout_called
        rollout_called = True
        raise AssertionError("rollout must not run after strict SHA failure")

    monkeypatch.setattr(collector.student, "rollout", fail_rollout)
    with pytest.raises(collector.repair_trainer.RepairInputError, match="SHA256 mismatch"):
        collector.run(
            checkpoint=checkpoint, output=output, scenario="flow6",
            seed=3, max_steps=1, repaired_checkpoint_sha256="0" * 64,
        )
    assert rollout_called is False
    assert not output.exists()


def test_repaired_loader_error_propagates_before_rollout(tmp_path, monkeypatch):
    checkpoint = tmp_path / "repaired.pt"
    checkpoint.write_bytes(b"candidate")
    output = tmp_path / "collection"
    rollout_called = False

    def fail_loader(*args, **kwargs):
        raise RuntimeError("strict repaired loader rejected metadata")

    def fail_rollout(*args, **kwargs):
        nonlocal rollout_called
        rollout_called = True
        raise AssertionError("rollout must not run after loader failure")

    monkeypatch.setattr(collector.repair_trainer, "load_repaired_checkpoint", fail_loader)
    monkeypatch.setattr(collector.student, "rollout", fail_rollout)
    with pytest.raises(RuntimeError, match="strict repaired loader rejected metadata"):
        collector.run(
            checkpoint=checkpoint, output=output, scenario="flow6",
            seed=3, max_steps=1, repaired_checkpoint_sha256="1" * 64,
        )
    assert rollout_called is False
    assert not output.exists()


def test_rollout_rejects_teacher_state_mutation(tmp_path, monkeypatch):
    runtime = replace(
        _config(), observation_schema_version="phaseA_low_obs_v2_1_retry_state",
        experiment_mode="fixed_low_student", teacher_enabled=False,
        warm_start_uavs=1, total_requested_uavs=1, fixed_arrival_schedule=(0.0,),
        max_simulation_seconds=3.0,
    )
    label_config = runtime
    policy = StudentPPOV42(88)

    def mutating_teacher(env, uav, *, audit_catalogue=False, state_only=False):
        assert state_only is True
        env.now += 0.25
        return joint_action_v4_1(0), ()

    monkeypatch.setattr(student, "teacher_joint_expert_v4_1", mutating_teacher)
    with pytest.raises(RuntimeError, match="Teacher label query mutated environment state"):
        student.rollout(
            policy, label_config, seed=3, max_steps=1,
            output=tmp_path / "mutation", deterministic=True,
            collect_labels=True, label_stride_steps=1, state_only_labels=True,
        )
