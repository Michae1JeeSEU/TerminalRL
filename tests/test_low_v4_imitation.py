from __future__ import annotations

import json

from terminal_marl.training.low_v4_imitation import (
    I0_DEFAULT_SOURCES,
    PPO_ALLOWED,
    _config,
    coverage_report,
    train_i1_joint_bc,
)


def _row(index: int, observation_value: float, source: str) -> dict:
    return {
        "observation": [observation_value] + [0.0] * 86,
        "teacher_joint_index": index,
        "teacher_decoded_action": {}, "source": source, "scenario_class": "S_test",
        "seed": 1, "uav_id": "0", "decision_event_id": f"{source}:{index}:{observation_value}",
        "state_hash": "state", "state_snapshot": {}, "sim_time": 0.0, "phase": "ring_outer",
        "lifecycle": "ring_outer", "raw_decision_tick": 0, "same_uav_target_streak": 1,
    }


def test_v4_i0_coverage_reports_event_sampling_and_joint_classes():
    rows = [_row(0, -1.0, "expert_rollout"), _row(4, -1.0, "safe_warm_start"),
            _row(5, 1.0, "from_empty"), _row(29, 1.0, "fixed_anchor")]
    report = coverage_report(rows, raw_ticks=12)
    assert report["raw_decision_ticks"] == 12
    assert report["unique_decision_events"] == 4
    assert report["temporal_duplication_factor"] == 3.0
    assert report["decision_event_WAIT_RELEASE"] == {"WAIT": 2, "RELEASE": 2}
    assert report["wait_speed_variants"] == [0, 4]
    assert report["release_speed_merge_variants"] == [5, 29]


def test_i1_joint_bc_is_supervised_and_leaves_ppo_locked(tmp_path):
    # The two classes are deliberately separable in the first V2 feature.
    rows = [_row(0, -1.0, "expert_rollout") for _ in range(48)] + [_row(17, 1.0, "targeted_release_opportunity") for _ in range(48)]
    dataset, cohort = tmp_path / "i0.jsonl", tmp_path / "cohort.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cohort.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = tmp_path / "i0.manifest.json"
    manifest.write_text(json.dumps({"I0_STRUCTURED_EXPERT_DATASET": "PASS", "dataset_path": str(dataset),
                                    "fixed_decision_state_cohort_path": str(cohort)}), encoding="utf-8")
    result = train_i1_joint_bc(i0_manifest_path=manifest, output_dir=tmp_path / "i1", epochs=8, batch_size=32,
                               min_ce_improvement=.001, min_class_recall=.01)
    assert result["ppo_called"] is False
    assert result["PPO_ALLOWED"] is False and PPO_ALLOWED is False
    assert result["after"]["joint_ce"] < result["before"]["joint_ce"]
    assert (tmp_path / "i1" / "i1_joint_bc.pt").is_file()


def test_i2_configuration_is_student_authoritative_label_only():
    config = _config(I0_DEFAULT_SOURCES[0])
    assert config.scheduler_mode == "execution"
    assert config.teacher_enabled is True
    assert config.teacher_mode == "teacher_label_only"
    assert config.teacher_release_label_enabled is False
    assert config.shield_enabled is True
