from terminal_marl.training.low_v4_1_imitation import SOURCES, _collect_source, _event_id, _config, i0_coverage_gate
from terminal_marl.envs.parallel_env import TerminalParallelEnv


def test_v4_1_i0_sources_include_required_physical_families():
    names = {row.source for row in SOURCES}
    assert {"expert_rollout", "safe_warm_start", "from_empty", "fixed_anchor",
            "targeted_release_opportunity", "targeted_wait_conflict", "predecessor_successor_conflict",
            "interface_headway_conflict", "resource_constraint"} <= names


def test_v4_1_event_identity_includes_source_seed_reset_uav_and_event():
    spec = SOURCES[0]; env = TerminalParallelEnv(_config(spec)); observations, _ = env.reset(71); uid = next(iter(observations))
    value, lifecycle, _ = _event_id(spec, 71, uid, 0, env, {}, {})
    assert value.startswith("expert_rollout:seed-71:reset-0:uav-") and ":event-" in value and ":sample-" in value
    assert lifecycle in value


def test_v4_1_i0_smoke_rows_are_current_teacher_v4_1_only():
    rows, _ = _collect_source(SOURCES[0], 72, max_rows=2)
    assert len(rows) == 2
    assert {row["action_schema_version"] for row in rows} == {"low_action_schema_v4_1_joint_discrete_26"}
    assert all("reset-0" in row["decision_event_id"] and "state_snapshot" in row for row in rows)
    assert all(row["semantic_sampling_reason"] for row in rows)


def test_v4_1_i0_gate_is_not_catalog_occupancy_gate():
    values = {"WAIT_expert": True, "RELEASE_expert": True, "release_multiple_speed_targets": True,
              "release_multiple_merge_targets": True, "release_multiple_joint_classes": True,
              "simple_release": True, "release_rejection": True, "conflict": True,
              "interface_headway": True, "resource": True}
    assert i0_coverage_gate(values) == (True, ())
