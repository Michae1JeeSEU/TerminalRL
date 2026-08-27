"""Training-free closure tests for the A5-v2 low-level architecture."""
from __future__ import annotations

from collections import OrderedDict
import json

import numpy as np
import pytest

from terminal_marl.core.config import SimConfig
from terminal_marl.core.config import RingRole
from terminal_marl.core.execution_scheduler import ExecutionScheduler
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.observations import LOW_OBS_V2, build_low_observation_v2
from terminal_marl.core.teacher_scheduler import TeacherScheduler
from terminal_marl.core.runtime_traffic_safety import PredictiveSafetyShield
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.checkpoints import checkpoint_metadata, validate_checkpoint_metadata, IncompatibleCheckpointError
from terminal_marl.algo.shared_ppo import RolloutBuffer, SharedPPO, Transition
from terminal_marl.training.build_low_teacher_dataset import TeacherDatasetCohort, build_low_teacher_dataset, build_low_teacher_dataset_matrix
from terminal_marl.training.low_v2_protocol import LOW_V2_TEACHER_CURRICULUM, phase_config, validate_low_v2_config
from terminal_marl.training.train_low_v2_ppo import authorize_low_v2_training, prepare_low_v2_training
from terminal_marl.training.eval_low_v2_student import StudentQualificationRecord, qualify_student_records
from terminal_marl.training.audit_low_teacher_dataset import audit_teacher_dataset
from terminal_marl.training.a5v2_capacity_calibration import CapacityCalibrationProtocol, CapacityTrial, c0_config, freeze_capacity, run_capacity_trial
from terminal_marl.training.low_v2_seed_partitions import LowV2SeedPartition, freeze_seed_partitions
from terminal_marl.envs.parallel_env import TerminalParallelEnv, WorldSnapshot


def _student_env(count: int = 3) -> TerminalParallelEnv:
    config = SimConfig(
        experiment_mode="fixed_low_student", reset_mode="warm_start_rings", warm_start_uavs=count,
        fixed_arrival_schedule=tuple(0.0 for _ in range(count)), total_requested_uavs=count,
        max_simulation_seconds=30.0,
    )
    env = TerminalParallelEnv(config)
    env.reset(7)
    return env


def test_v2_schema_is_named_87d_and_no_neighbor_is_zero_masked():
    env = _student_env(1)
    vector = next(iter(env._observations().values()))
    assert LOW_OBS_V2.dimension == 87
    assert vector.shape == (87,)
    for name in ("same_ring_leader", "same_ring_follower", "target_ring_predecessor", "target_ring_successor", "nearest_predicted_conflict"):
        assert np.array_equal(vector[LOW_OBS_V2.slices[name]], np.zeros(9, np.float32))


def test_v2_topology_observation_is_invariant_to_active_mapping_order():
    env = _student_env(6)
    uid, uav = next(iter(env.active.items()))
    original = build_low_observation_v2(uav, env)
    env.active = OrderedDict(reversed(list(env.active.items())))
    reordered = build_low_observation_v2(env.active[uid], env)
    assert np.array_equal(original, reordered)


def test_execution_scheduler_has_one_requested_candidate_and_no_teacher_search():
    env = _student_env(1)
    scheduler = ExecutionScheduler(env.config, MergeScheduler(env.config))
    assert not hasattr(scheduler, "repair_candidates")
    uav = next(iter(env.active.values()))
    proposal = scheduler.propose_requested(uav, LowLevelIntent(1.0, 0.0, 0.0), WorldSnapshot(env.now, env.active, env.airspace), RingRole.MIDDLE)
    assert scheduler.check_requested(proposal).approved


def test_teacher_is_rejected_for_formal_student_preset_and_available_for_training_preset():
    with pytest.raises(ValueError, match="teacher_enabled"):
        SimConfig(experiment_mode="fixed_low_student", teacher_enabled=True)
    teacher_config = SimConfig(experiment_mode="fixed_teacher_heavy", scheduler_mode="teacher_full", teacher_enabled=True)
    teacher = TeacherScheduler(teacher_config, MergeScheduler(teacher_config))
    assert teacher.candidates(LowLevelIntent(1.0, 0.0, 0.0))[0].kind == "requested"
    witness = teacher.choose_verified(LowLevelIntent(1.0, 0.0, 0.0), lambda candidate: candidate.kind == "speed_repair")
    assert witness is not None and witness.verified and witness.candidate_kind == "speed_repair"


def test_step_emits_requested_execution_shield_provenance():
    env = _student_env(2)
    observations, _, _, _, info = env.step({uid: np.asarray((0.0, 0.0, 0.0), np.float32) for uid in env.agents})
    assert observations
    for trace in info["action_traces"].values():
        assert trace["requested_low_action"] == trace["execution_applied_action"]
        assert trace["teacher_corrected_action"] is None
        assert trace["teacher_used"] is False
        assert trace["teacher_verified"] is False and trace["teacher_witness_plan_id"] is None


def test_predictive_shield_detects_future_conflict_without_rewriting_committed_plan():
    env = _student_env(1)
    holder = next(iter(env.active.values()))
    # Keep an uncommitted holder on the outer ring and introduce a previously
    # checked transfer that will sweep through its next ring-reference state.
    ring = env.airspace.actual_ring(holder.ring_role)
    holder.position = np.asarray((ring.radius, 0.0, ring.height))
    holder.ring_progress = 0.0
    shield = PredictiveSafetyShield(env.config)
    predicted_holder_endpoint = shield._holder_position(holder, env.airspace, 10.0, 0.0)
    plan = TrajectoryPlan("committed-plan", 999, "transfer", [
        TrajectoryPoint(0.0, np.asarray((ring.radius * 2.0, 0.0, ring.height))),
        # Use the same finite-acceleration holder prediction as runtime.
        TrajectoryPoint(10.0, predicted_holder_endpoint),
    ], 0.0, 10.0, metadata={"committed": True})
    committed_points = [point.position.copy() for point in plan.points]
    blocker = type("CommittedTraffic", (), {"uid": 999, "ring_role": None, "committed_plan": plan})()
    env.active[999] = blocker
    risk = shield.assess(env)
    shield.protect(env, {holder.uid: LowLevelIntent(0.0, 0.0, 0.0)})
    assert risk.trigger_reason == "predicted_swept_separation"
    assert all(np.array_equal(before, after.position) for before, after in zip(committed_points, plan.points))


def test_predictive_shield_is_a_safe_noop_without_committed_traffic():
    env = _student_env(1)
    risk = PredictiveSafetyShield(env.config).assess(env)
    assert risk.trigger_reason is None
    assert np.isinf(risk.predicted_min_swept_separation)


def test_checkpoint_schema_rejects_legacy_actor_metadata():
    legacy = SimConfig(experiment_mode="fixed_a5_legacy", observation_schema_version="phaseA_low_obs_v1_41d")
    student = SimConfig(experiment_mode="fixed_low_student")
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_metadata(checkpoint_metadata(legacy), student)


def test_ppo_transition_keeps_requested_action_when_teacher_label_is_present():
    requested = np.asarray((0.2, -0.1, 0.3), np.float32)
    teacher = np.asarray((1.0, 1.0, -1.0), np.float32)
    world = np.zeros((1, LOW_OBS_V2.dimension), np.float32)
    row = Transition("uav-1", world[0], world, requested, 0.0, 0.0, 1.0, True,
                     teacher_action=teacher, applied_action=teacher, teacher_used=True)
    buffer = RolloutBuffer()
    buffer.add(row)
    assert np.array_equal(buffer.by_agent["uav-1"][0].action, requested)
    metrics = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=0.1).update(buffer, epochs=1, minibatch_size=1)
    assert "teacher_bc_loss" in metrics and np.isfinite(metrics["teacher_bc_loss"])


def test_teacher_dataset_records_requested_and_correction_fields_without_training(tmp_path):
    class ZeroPolicy:
        def act(self, observations):
            return {key: np.zeros(3, np.float32) for key in observations}
    config = SimConfig(reset_mode="warm_start_rings", warm_start_uavs=1, fixed_arrival_schedule=(0.0,),
                       total_requested_uavs=1, max_simulation_seconds=10.0)
    output = tmp_path / "teacher.jsonl"
    summary = build_low_teacher_dataset(config, output_path=output, max_steps=1, policy=ZeroPolicy())
    assert summary.rows == 1 and summary.corrected_rows == 0 and output.is_file()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert len(row["observation"]) == 87
    assert row["requested_action"] == [0.0, 0.0, 0.0] and row["teacher_action"] is None
    assert row["teacher_verified"] is False and row["teacher_witness_plan_id"] is None
    assert row["active_agent_count"] == 1 and row["seed"] == 0
    audit = audit_teacher_dataset(output, min_corrected_fraction=0.0)
    assert audit.passed and audit.rows == 1
    assert (tmp_path / "teacher.jsonl.summary.json").is_file()


def test_teacher_dataset_matrix_records_named_cohorts_without_training(tmp_path):
    class ZeroPolicy:
        def act(self, observations):
            return {key: np.zeros(3, np.float32) for key in observations}
    cohort = TeacherDatasetCohort("n1", SimConfig(reset_mode="warm_start_rings", warm_start_uavs=1,
                                  fixed_arrival_schedule=(0.0,), total_requested_uavs=1, max_simulation_seconds=10.0), 1, 1)
    destination = tmp_path / "matrix.jsonl"
    summary = build_low_teacher_dataset_matrix((cohort,), output_path=destination, policy=ZeroPolicy(), min_corrected_fraction=0.0)
    assert summary.rows == 1 and summary.cohorts == ("n1",)
    assert (tmp_path / "matrix.jsonl.manifest.json").is_file()


def test_preflight_freezes_documented_teacher_annealing_without_training(tmp_path):
    preflight = prepare_low_v2_training(manifest_path=tmp_path / "preflight.json")
    assert preflight.observation_dimension == 87 and preflight.action_dimension == 3
    assert preflight.phases == ("teacher_full", "teacher_immediate", "teacher_label_only", "teacher_off")
    assert preflight.configs[-1]["teacher_enabled"] is False
    for phase in LOW_V2_TEACHER_CURRICULUM:
        config = phase_config(SimConfig(), phase)
        validate_low_v2_config(config, for_training=phase.name != "teacher_off")


def test_low_v2_training_authorization_blocks_uncorrected_teacher_dataset(tmp_path):
    dataset = tmp_path / "uncorrected.jsonl"
    dataset.write_text(json.dumps({
        "observation": [0.0] * 87, "requested_action": [0.0] * 3, "teacher_action": None,
        "execution_action": [0.0] * 3, "shield_action": None, "teacher_used": False,
        "teacher_kind": None, "teacher_delay_seconds": None, "teacher_witness_plan_id": None,
        "teacher_verified": False, "correction_magnitude": 0.0, "predicted_min_sep_before": None,
        "d_cpa_before": None, "t_cpa_before": 0.0, "predicted_min_sep_after": None,
        "release_decision": "not_requested", "reward": 0.0, "terminated": False, "truncated": False,
    }) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="training is blocked"):
        authorize_low_v2_training(dataset)


def test_student_qualification_rejects_missing_n_teacher_and_shield_off_configs():
    config = SimConfig(experiment_mode="fixed_low_student")
    records = [StudentQualificationRecord(n, "id" if n <= 20 else "ood", 0, 0, 0, 10, 0, 10, 0)
               for n in (1, 5, 10, 20, 40, 60)]
    report = qualify_student_records(config, records)
    assert report.passed and report.id_interventions_per_100 == report.ood_interventions_per_100 == 0.0
    incomplete = qualify_student_records(config, records[:-1])
    assert not incomplete.passed and incomplete.missing_n == (60,)
    with pytest.raises(ValueError, match="shield_enabled"):
        SimConfig(experiment_mode="fixed_low_student", shield_enabled=False)


def test_safe_warm_start_enforces_the_documented_initial_separation_margin():
    count = 40
    config = SimConfig(experiment_mode="fixed_low_student", reset_mode="warm_start_rings", warm_start_uavs=count,
                       fixed_arrival_schedule=tuple(0.0 for _ in range(count)), total_requested_uavs=count)
    env = TerminalParallelEnv(config)
    env.reset(19)
    positions = [uav.position for uav in env.active.values()]
    distances = [np.linalg.norm(left - right) for index, left in enumerate(positions) for right in positions[index + 1:]]
    assert len(positions) == count
    assert min(distances) >= 1.2 * config.safe_distance - 1e-6
    assert all(uav.merge_reservation_id is None and uav.committed_plan is None for uav in env.active.values())


def test_c0_config_locks_the_nominal_full_teacher_safety_stack():
    config = c0_config(0.03, 3600.0)
    assert config.experiment_mode == "fixed_teacher_heavy"
    assert config.scheduler_mode == config.teacher_mode == "teacher_full"
    assert config.teacher_enabled and config.shield_enabled and not config.high_level_enabled
    assert config.arrival_model == "poisson" and config.initial_soc == pytest.approx(0.85)
    assert config.mission_probabilities == (1.0, 0.0, 0.0, 0.0)


def test_c0_trial_is_measurement_only_and_never_constructs_ppo():
    protocol = CapacityCalibrationProtocol(rates=(1.0,), seeds=(101,), horizon_seconds=30.0, warmup_seconds=5.0)
    trial = run_capacity_trial(1.0, 101, protocol)
    assert trial.requested_measurement > 0
    assert trial.requested_rate_actual > 0.0
    assert trial.collisions == trial.nmac == trial.hard_separation_violations == 0


def test_capacity_freeze_uses_predeclared_ratio_queue_and_safety_gates():
    protocol = CapacityCalibrationProtocol(rates=(0.02, 0.04), seeds=(1,), horizon_seconds=100.0, warmup_seconds=10.0)
    values = dict(seed=1, requested_measurement=100, admitted_measurement=100, completed_measurement=97,
                  completion_to_request_ratio=.97, requested_rate_actual=.02, completed_rate=.02,
                  external_queue_slope_per_second=.0001, external_queue_mean=1., external_queue_max=2,
                  active_mean=2., active_peak=4, ring_occupancy_mean=(1., 1., 0.), lz_utilization_mean=.1,
                  final_utilization_mean=.1, delay_mean=10., delay_p95=20., release_intents=10,
                  teacher_interventions=1, shield_interventions=0, teacher_interventions_per_100_intents=10.,
                  shield_interventions_per_100_intents=0., collisions=0, nmac=0, hard_separation_violations=0,
                  final_overcapacity_steps=0)
    passing = CapacityTrial(rate=.02, **values)
    failing = CapacityTrial(rate=.04, **{**values, "completion_to_request_ratio": .90, "completed_measurement": 90})
    frozen = freeze_capacity((passing, failing), protocol)
    assert frozen["frozen"] and frozen["mu_hat"] == .02
    assert frozen["traffic_levels"]["medium"] == [0.009, 0.012]


def test_seed_partitions_are_disjoint_and_freeze_fail_closed(tmp_path):
    partition = LowV2SeedPartition(train=(1,), validation=(2,), qualification=(3,), ood=(4,))
    destination = tmp_path / "seeds.json"
    assert freeze_seed_partitions(destination, partition) == partition
    assert freeze_seed_partitions(destination, partition) == partition
    with pytest.raises(RuntimeError, match="already frozen"):
        freeze_seed_partitions(destination, LowV2SeedPartition(train=(10,), validation=(20,), qualification=(30,), ood=(40,)))
