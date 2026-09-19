import json
from types import SimpleNamespace

import numpy as np
import pytest

from terminal_marl.algo.joint_ppo_v4_2 import JointPPOV42
from terminal_marl.core.config import SimConfig
from terminal_marl.core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import run_v4_2_observation_v2_2_closure_pilot as pilot


def _config():
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version=pilot.OBS_SCHEMA,
        teacher_enabled=True, teacher_mode="teacher_label_only", scheduler_mode="execution",
        experiment_mode="fixed_teacher_heavy", reset_mode="warm_start_rings", warm_start_uavs=3,
        fixed_arrival_schedule=(0.0, 0.0, 0.0), total_requested_uavs=3,
        max_simulation_seconds=2.0,
    )


def _row(value: float, feasible=(0, 1), interface="o2m"):
    return {
        "observation_v2_1": [value] * 88,
        "observation_v2_2": [value] * 88 + [value, 0.0, 0.0, value, value, 0.0],
        "proposal_set_target": list(feasible), "interface": interface,
    }


def test_v2_2_closure_protocol_and_prefix_behavior_are_frozen():
    protocol = pilot.protocol()
    assert protocol["bindings"]["observation_dimension"] == 94
    assert protocol["collection"]["train_seeds"] == list(pilot.TRAIN_SEEDS)
    assert set(pilot.TRAIN_SEEDS).isdisjoint(pilot.DEV_SEEDS)
    env = TerminalParallelEnv(_config()); observations, _ = env.reset(797000)
    policy = JointPPOV42(88)
    uids, world, actor_mask, actions, likelihoods = pilot._behavior_request(policy, env, observations)
    assert world.shape == (len(uids), 94) and actor_mask.shape == (len(uids),)
    assert set(actions) == set(uids) and set(likelihoods) == set(uids)


def test_v2_2_prefix_behavior_handles_empty_active_interval_without_model_call():
    uids, world, actor_mask, actions, likelihoods = pilot._behavior_request(
        None, SimpleNamespace(active={}, future_release_reservations={}), {})
    assert uids == [] and world.shape == (0, 94) and actor_mask.shape == (0,)
    assert actions == {} and likelihoods == {}


def test_v2_2_probe_expansion_and_fixed_one_epoch_smoke(monkeypatch):
    train = [_row(0.0, (0, 1)), _row(0.2, (2, 3), "m2i"), _row(0.4, (4,), "i2f")]
    dev = [_row(0.1, (0, 1)), _row(0.3, (2,), "m2i"), _row(0.5, (4, 5), "i2f")]
    old_x, old_y = pilot._expand(train, "observation_v2_1")
    new_x, new_y = pilot._expand(train, "observation_v2_2")
    assert old_x.shape == (75, 90) and new_x.shape == (75, 96)
    assert np.array_equal(old_y.numpy(), new_y.numpy())
    monkeypatch.setattr(pilot, "EPOCHS", 1)
    result = pilot._probe(train, dev, observation_key="observation_v2_2", seed=797101)
    assert result["optimizer_steps"] == 1 and not result["checkpoint_saved"]
    assert result["state_dev"]["states"] == 3


def test_v2_2_smoke_path_does_not_reserve_formal_seeds(tmp_path, monkeypatch):
    checkpoint = tmp_path / "candidate.pt"; checkpoint.write_bytes(b"candidate")
    monkeypatch.setattr(pilot, "protocol", lambda: {})
    original_sha = pilot._sha256
    monkeypatch.setattr(pilot, "_sha256", lambda path: pilot.CANDIDATE_SHA
                        if str(path) == str(checkpoint) else original_sha(path))
    monkeypatch.setattr(pilot.effectiveness, "load_candidate_checkpoint",
                        lambda path: (object(), {"stage": "test", "architecture_id": "test"}))
    record = {"checks": {"all": True}, "rows": 1}
    monkeypatch.setattr(pilot, "_episode", lambda *args, **kwargs: (record, [_row(0.0)]))
    output = tmp_path / "smoke"
    result = pilot.run(output=output, checkpoint=checkpoint, history_root=tmp_path, smoke=True)
    assert result["status"] == "INCOMPLETE" and result["smoke"]
    assert not result["formal_seed_reservation"] and not result["paired_probe_called"]
    assert (output / "manifest.json").is_file()
    assert not (checkpoint.parent / pilot.RESERVATION_MARKER).exists()


def test_v2_2_pre_reset_recovery_requires_exact_marker_and_registry_only(tmp_path):
    output = tmp_path / "formal"; output.mkdir()
    marker = tmp_path / pilot.RESERVATION_MARKER
    marker.write_text(json.dumps(pilot._expected_marker(output)), encoding="utf-8")
    (output / "registry_progress.json").write_text(
        '{"files_read": 3301, "files_total": 3360}', encoding="utf-8")
    evidence = pilot._validate_pre_reset_recovery(output, marker)
    assert evidence["environment_resets_before_recovery"] == 0
    assert not evidence["seeds_replaced"] and not evidence["thresholds_changed"]
    assert (output / "pre_reset_recovery.json").is_file()


def test_v2_2_run_allows_existing_output_only_for_guarded_recovery(tmp_path, monkeypatch):
    class ReachedRecovery(Exception):
        pass

    output = tmp_path / "formal"; output.mkdir()
    checkpoint = tmp_path / "candidate.pt"; checkpoint.write_bytes(b"candidate")
    monkeypatch.setattr(pilot, "protocol", lambda: {})
    monkeypatch.setattr(pilot, "_sha256", lambda path: pilot.CANDIDATE_SHA
                        if str(path) == str(checkpoint) else "unused")
    monkeypatch.setattr(pilot.effectiveness, "load_candidate_checkpoint",
                        lambda path: (object(), {}))
    monkeypatch.setattr(pilot, "_validate_pre_reset_recovery",
                        lambda *args: (_ for _ in ()).throw(ReachedRecovery()))
    with pytest.raises(ReachedRecovery):
        pilot.run(output=output, checkpoint=checkpoint, history_root=tmp_path,
                  recover_pre_reset=True)


def test_v2_2_successor_protocol_uses_fresh_disjoint_seeds():
    names = ("STAGE", "PROTOCOL", "PROTOCOL_SHA", "TRAIN_SEEDS", "DEV_SEEDS",
             "PROBE_SEEDS", "SMOKE_SEED", "RESERVATION_MARKER")
    original = {name: getattr(pilot, name) for name in names}
    try:
        pilot.configure_successor_v2()
        protocol = pilot.protocol()
        assert protocol["predecessor"]["corrected_mapping_commit"] == "0d3c2e5"
        assert pilot.TRAIN_SEEDS == (800001, 800002, 800003, 800004)
        assert set(pilot.TRAIN_SEEDS).isdisjoint(pilot.DEV_SEEDS)
        assert protocol["predecessor"]["old_rows_reused"] is False
    finally:
        for name, value in original.items():
            setattr(pilot, name, value)
