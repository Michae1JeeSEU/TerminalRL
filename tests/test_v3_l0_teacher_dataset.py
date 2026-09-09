"""Fast regressions for fresh V3 L0 collection semantics."""
from __future__ import annotations

import json

from terminal_marl.core.low_level_action import ACTION_SCHEMA_V3, LowLevelActionV3
from terminal_marl.training.run_v3_l0_teacher_dataset import (
    CatalogExplorationPolicy, V3L0Cohort, _coverage, collect_v3_l0_part,
)
from terminal_marl.training.audit_low_teacher_dataset import audit_teacher_dataset


def test_catalog_exploration_policy_only_emits_v3_actions():
    policy = CatalogExplorationPolicy(42)
    actions = policy.act({"4": None, "9": None})
    assert len(actions) == 2
    for action in actions.values():
        value = LowLevelActionV3.from_normalized(action)
        assert value.indices.shape == (3,)
        assert value.as_dict()["release_category"] in {"WAIT", "RELEASE"}


def test_short_v3_l0_collection_has_requested_and_teacher_categorical_semantics(tmp_path):
    cohort = V3L0Cohort("warm_n5", "warm_start_rings", "finite_cohort", 5, 1, 2)
    rows, evidence = collect_v3_l0_part(cohort, seed=1103, episode=0, tape_dir=tmp_path)
    assert rows and evidence["rows"] == len(rows)
    assert evidence["collisions"] == evidence["nmac"] == evidence["hard_separation_violations"] == 0
    for row in rows:
        assert row["low_action_schema"] == "V3"
        assert row["action_schema_version"] == ACTION_SCHEMA_V3
        assert set(row["a_requested"]) >= {"release_index", "speed_index", "merge_time_index"}
        assert set(row["a_teacher"]) >= {"release_index", "speed_index", "merge_time_index"}
    output = tmp_path / "short_v3_l0.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert audit_teacher_dataset(output, min_corrected_fraction=0.0).passed
    assert _coverage(rows)["teacher_used"] >= 0
