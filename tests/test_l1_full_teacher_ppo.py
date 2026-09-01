"""Focused coverage for the executable A5-v2 L1 Full-Teacher PPO runner."""
from __future__ import annotations

import json

from terminal_marl.training.train_low_v2_ppo import (
    L1FullTeacherProtocol,
    _episode_design,
    l1_full_teacher_config,
    train_l1_full_teacher,
)


def _teacher_row(corrected: bool) -> dict:
    return {
        "observation": [0.0] * 87, "requested_action": [1.0, 0.0, 0.0],
        "teacher_action": [1.0, 0.0, 0.0] if corrected else None,
        "execution_action": [1.0, 0.0, 0.0], "shield_action": None,
        "teacher_used": corrected, "teacher_kind": "teacher_repair" if corrected else None,
        "teacher_delay_seconds": 0.0 if corrected else None,
        "teacher_witness_plan_id": "test-witness" if corrected else None,
        "teacher_verified": corrected, "correction_magnitude": 0.0,
        "predicted_min_sep_before": None, "d_cpa_before": None, "t_cpa_before": 0.0,
        "predicted_min_sep_after": None, "release_decision": "approved", "reward": 0.0,
        "terminated": False, "truncated": False,
    }


def _write_prerequisites(tmp_path):
    c0 = tmp_path / "c0.json"
    c0.write_text(json.dumps({"frozen_capacity": {"frozen": True, "mu_hat": 0.003,
                                                   "traffic_levels": {"medium": [0.00135, 0.0018]}}}), encoding="utf-8")
    dataset = tmp_path / "l0.jsonl"
    dataset.write_text("".join(json.dumps(_teacher_row(index == 0)) + "\n" for index in range(20)), encoding="utf-8")
    (tmp_path / "seed_partitions.json").write_text(json.dumps({"schema_version": "A5V2-TRAINING-CURRICULUM-v3", "seed_partition": {
        "train": [1001], "validation": [2001, 2002, 2003, 2004], "qualification": [3001], "ood": [4001],
    }}), encoding="utf-8")
    return c0, dataset


def test_l1_config_is_medium_nominal_full_teacher_and_uses_safe_warm_start():
    config = l1_full_teacher_config(medium_rate=0.0015, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=12, horizon_seconds=50.0)
    assert config.scheduler_mode == config.teacher_mode == "teacher_full"
    assert config.teacher_enabled and config.shield_enabled and not config.high_level_enabled
    assert config.initial_soc == 0.80 and config.mission_probabilities == (1.0, 0.0, 0.0, 0.0)
    assert len(config.fixed_arrival_schedule) == 12 and config.fixed_arrival_schedule[:8] == (0.0,) * 8


def test_l1_scenario_matrix_keeps_warm_starts_legal_and_includes_poisson_from_empty():
    designs = [_episode_design(index, medium_rate=0.0015, horizon_seconds=50.0)[0] for index in range(10)]
    assert sum(config.reset_mode == "warm_start_rings" for config in designs) == 6
    assert all(config.arrival_model == "finite_cohort" for config in designs if config.reset_mode == "warm_start_rings")
    assert any(config.arrival_model == "poisson" for config in designs if config.reset_mode == "flow_from_empty")
    finite_empty = next(config for config in designs if config.reset_mode == "flow_from_empty" and config.arrival_model == "finite_cohort")
    assert finite_empty.fixed_arrival_schedule[0] == 0.0


def test_l1_short_run_authorizes_l0_and_writes_checkpoint_and_manifests(tmp_path):
    c0, dataset = _write_prerequisites(tmp_path)
    protocol = L1FullTeacherProtocol(updates=1, episodes_per_update=1, rollout_workers=1, max_steps=3,
                                     validation_every=1, ppo_epochs=1, ppo_minibatch_size=8)
    result = train_l1_full_teacher(c0_report=c0, teacher_dataset_path=dataset, output_dir=tmp_path / "l1", protocol=protocol)
    assert result.mu_hat == 0.003 and result.medium_rate == 0.001575
    assert result.rollout_episodes == 1 and result.rollout_rows > 0
    assert (tmp_path / "l1" / "l1_full_teacher_ppo.pt").is_file()
    manifest = json.loads((tmp_path / "l1" / "l1_full_teacher.manifest.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["teacher"] == "FULL"
    assert manifest["environment"]["warm_start_fraction"] == 0.60
    validation = [json.loads(line) for line in (tmp_path / "l1" / "l1_validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["reset_mode"] == "flow_from_empty" and row["demand_model"] == "poisson" for row in validation)
