import json

from terminal_marl.training.run_v3_l1_release_intervention_diagnostic import freeze_release_cohort


def _row(requested, teacher, *, step):
    return {"low_action_schema": "V3", "cohort": "warm_n10", "episode": 0, "step": step, "seed": 1001,
            "scenario": {"cohort": "warm_n10"}, "observation_v2": [float(step)] * 4,
            "a_requested": {"release_index": requested, "speed_index": 2, "merge_time_index": 2},
            "a_teacher": {"release_index": teacher, "speed_index": 2, "merge_time_index": 2},
            "teacher_used": requested != teacher, "teacher_correction_heads": ["release"] if requested != teacher else [],
            "teacher_correction_reason": "test", "release_decision": "test"}


def test_fixed_release_cohort_preserves_all_four_release_cases(tmp_path):
    # Exceed the fixed limits so the selection and its ordering are exercised.
    rows = []
    for requested, teacher in ((1, 0), (0, 1), (1, 1), (0, 0)):
        rows.extend(_row(requested, teacher, step=index + requested * 100 + teacher * 200) for index in range(80))
    source = tmp_path / "raw.jsonl"; source.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)), encoding="utf-8")
    artifact = freeze_release_cohort(source, tmp_path / "cohort.json")
    assert artifact["composition"] == {"RELEASE_to_WAIT": 64, "WAIT_to_RELEASE": 32, "correct_RELEASE": 64, "correct_WAIT": 64}
    frozen = json.loads((tmp_path / "cohort.json").read_text(encoding="utf-8"))
    assert len({row["state_hash"] for row in frozen["rows"]}) == 224
    assert all(row["reset_state_hash"] for row in frozen["rows"])
