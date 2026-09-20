"""Training-free closure tests for the A5-v2 low-level architecture."""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from terminal_marl.core.config import SimConfig
from terminal_marl.core.config import RingRole
from terminal_marl.core.execution_scheduler import ExecutionScheduler
from terminal_marl.core.merge_scheduler import LowLevelIntent, MergeScheduler
from terminal_marl.core.observations import LOW_OBS_V2, build_low_observation_v2
from terminal_marl.core.teacher_scheduler import TeacherScheduler
from terminal_marl.core.runtime_traffic_safety import JointSafetyAssessment, PredictiveSafetyShield, SafetyInvariantError
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.core.trajectory_generator import TrajectoryPlan, TrajectoryPoint
from terminal_marl.core.uav import FlightPhase
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
    assert max(info["prediction_execution_position_error"].values(), default=0.0) < 1e-9


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
    # The transfer is immutable and the bounded holder action set cannot
    # certify a recovery.  S3's fail-closed invariant must block physics,
    # rather than quietly dispatching a nominal residual.
    with pytest.raises(SafetyInvariantError):
        shield.protect(env, {holder.uid: LowLevelIntent(0.0, 0.0, 0.0)})
    assert risk.trigger_reason == "predicted_swept_separation"
    assert all(np.array_equal(before, after.position) for before, after in zip(committed_points, plan.points))


def test_predictive_shield_is_a_safe_noop_without_committed_traffic():
    env = _student_env(1)
    risk = PredictiveSafetyShield(env.config).assess(env)
    assert risk.trigger_reason is None
    assert np.isinf(risk.predicted_min_swept_separation)


def test_predictive_shield_cached_holder_samples_match_scalar_executor_prediction():
    env = _student_env(1)
    holder = next(iter(env.active.values()))
    shield = PredictiveSafetyShield(env.config)
    times = np.asarray((env.now, env.now + .2, env.now + 1.0, env.now + 7.0))
    cached = shield._holder_positions(holder, env.airspace, env.now, times, .5)
    scalar = [shield._holder_position(holder, env.airspace, float(time - env.now), .5) for time in times]
    assert all(np.allclose(left, right, atol=1e-10) for left, right in zip(cached, scalar))


def test_commit_holder_reachable_tube_contains_both_extreme_executor_tracks():
    """The commit guard covers legal residual switching extrema, not 3 traces."""
    env = _student_env(1)
    holder = next(iter(env.active.values()))
    safety = SafetyFilter(env.config)
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    times = np.arange(env.now, env.now + 4.0 + 1e-9, env.config.physics_dt)
    lower, upper = safety.holder_reachable_tube(holder, snapshot, times)
    ring = env.airspace.actual_ring(holder.ring_role)
    for residual, bound in ((-env.config.max_speed_residual_mps, lower),
                            (env.config.max_speed_residual_mps, upper)):
        positions = safety._holder_positions(holder, snapshot, times, residual)
        angles = np.unwrap(np.asarray([np.arctan2(position[1], position[0]) for position in positions]))
        assert np.allclose(angles, bound, atol=1e-10)
        assert np.allclose([np.linalg.norm(position[:2]) for position in positions], ring.radius, atol=1e-10)


def test_joint_shield_projects_multiple_holder_residuals_before_execution(monkeypatch):
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holders = sorted(uid for uid, uav in env.active.items() if uav.ring_role is not None and uav.committed_plan is None)
    assert len(holders) == 2
    requested = {uid: LowLevelIntent(0.0, 0.0, 0.0) for uid in holders}
    calls = []

    def jointly_safe(_env, residuals, *, pairs=None, horizon=None):
        normalized = {uid: round(value / env.config.max_speed_residual_mps, 6) for uid, value in residuals.items()}
        calls.append((normalized, horizon))
        # A component-level complementary correction is safe; nominal zero is
        # deliberately unsafe, proving it is not used as a fail-closed escape.
        safe = normalized[holders[0]] == -1.0 and normalized[holders[1]] == 1.0
        return JointSafetyAssessment(safe, 100.0 if safe else 0.0,
                                     () if safe else (tuple(holders),))

    monkeypatch.setattr(shield, "_joint_assessment", jointly_safe)
    repaired = shield.protect(env, requested)
    assert repaired[holders[0]].speed_residual == -1.0
    assert repaired[holders[1]].speed_residual == 1.0
    assert shield.telemetry["joint_safety_corrections"] == 2
    assert any(candidate == {holders[0]: -1.0, holders[1]: 1.0} for candidate, _ in calls)


def test_safety_trace_keeps_exact_requested_action_provenance(tmp_path):
    env = _student_env(1)
    observations, _ = env.reset(77)
    recorder = SafetyTraceRecorder(tmp_path, dt=env.config.dt)
    recorder.start_episode(env, episode_seed=77, policy_rng_state={"unit": "state"})
    actions = {uid: np.zeros(3, dtype=np.float32) for uid in observations}
    row = recorder.record_pre_step(env, observations, actions, {uid: -0.5 for uid in actions})
    _, _, _, _, info = env.step(actions)
    recorder.record_post_step(env, row, info)
    path = recorder.dump(reason="unit_test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["episode_seed"] == 77
    assert payload["reset_state_hash"]
    assert payload["policy_rng_state_b64"]
    record = payload["records"][0]
    assert len(payload["action_tape"]) == 1
    assert record["observation_hash"] and record["a_requested"] and record["log_prob"]
    assert "a_teacher_shield_applied" in record and "reservations" in record["pre_state"]
    header, tape = SafetyTraceRecorder.load_action_tape(path)
    assert header["reset_state_hash"] == payload["reset_state_hash"]
    assert np.array_equal(tape[0][next(iter(actions))], actions[next(iter(actions))])
    replay = SafetyTraceRecorder.replay_action_tape(_student_env(1), path)
    assert replay["exact"]


def test_safety_trace_retains_full_tape_after_diagnostic_ring_wrap(tmp_path):
    env = _student_env(1)
    observations, _ = env.reset(91)
    # One detailed record is retained, but every requested action must remain
    # available for reset-to-event replay.
    recorder = SafetyTraceRecorder(tmp_path, max_seconds=env.config.dt, dt=env.config.dt)
    recorder.start_episode(env, episode_seed=91, policy_rng_state={"unit": "state"})
    for _ in range(3):
        actions = {uid: np.zeros(3, dtype=np.float32) for uid in observations}
        row = recorder.record_pre_step(env, observations, actions)
        observations, _, _, _, info = env.step(actions)
        recorder.record_post_step(env, row, info)
    path = recorder.dump(reason="late_unit_event")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    assert len(payload["action_tape"]) == 3
    replay = SafetyTraceRecorder.replay_action_tape(_student_env(1), path)
    assert replay["exact"] and replay["steps_replayed"] == 3


def test_safety_trace_marks_fail_closed_safety_invariant_event(tmp_path):
    env = _student_env(1)
    observations, _ = env.reset(93)
    recorder = SafetyTraceRecorder(tmp_path, dt=env.config.dt)
    recorder.start_episode(env, episode_seed=93, policy_rng_state={"unit": "state"})
    actions = {uid: np.zeros(3, dtype=np.float32) for uid in observations}
    row = recorder.record_pre_step(env, observations, actions)
    recorder.record_safety_invariant(row, SafetyInvariantError("blocked", diagnostic={"component": [1, 2]}))
    payload = json.loads(recorder.dumped_path.read_text(encoding="utf-8"))
    assert payload["dump_reason"] == "uncertified_joint_safety_action"
    assert payload["action_tape"][0]["safety_event"]["diagnostic"] == {"component": [1, 2]}


def test_joint_shield_rejects_currently_safe_action_without_next_tick_backup(monkeypatch):
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holders = sorted(uid for uid, uav in env.active.items() if uav.ring_role is not None and uav.committed_plan is None)
    requested = {uid: LowLevelIntent(0.0, 0.0, 0.0) for uid in holders}

    def assessment(_env, residuals, *, pairs=None, horizon=None):
        normalized = tuple(round(residuals[uid] / env.config.max_speed_residual_mps, 6) for uid in holders)
        return JointSafetyAssessment(True, 100.0, ())

    def backup(_env, residuals, _pairs):
        normalized = tuple(round(residuals[uid] / env.config.max_speed_residual_mps, 6) for uid in holders)
        safe = normalized == (-1.0, 1.0)
        return (dict(residuals) if safe else None,
                JointSafetyAssessment(safe, 100.0 if safe else 0.0,
                                      () if safe else (tuple(holders),)))

    monkeypatch.setattr(shield, "_joint_assessment", assessment)
    monkeypatch.setattr(shield, "_next_executor_state", lambda _env, _residuals: env)
    monkeypatch.setattr(shield, "_backup_safe", backup)
    repaired = shield.protect(env, requested)
    assert (repaired[holders[0]].speed_residual, repaired[holders[1]].speed_residual) == (-1.0, 1.0)
    assert shield.telemetry["recursive_feasibility_rejections"] > 0


def test_joint_shield_backup_offers_common_ring_profile_for_cyclic_component():
    env = _student_env(4)
    shield = PredictiveSafetyShield(env.config)
    holders = sorted(uid for uid, uav in env.active.items() if uav.ring_role is not None and uav.committed_plan is None)
    assert len(holders) == 4
    # Make the synthetic component explicitly cyclic on one ring.  The warm
    # start helper otherwise distributes aircraft across rings, which does not
    # exercise a shared ring profile.
    role = env.active[holders[0]].ring_role
    for uid in holders:
        env.active[uid].ring_role = role
    requested = {
        holders[0]: -env.config.max_speed_residual_mps,
        **{uid: env.config.max_speed_residual_mps for uid in holders[1:]},
    }
    # The cyclic neighbour set is intentionally sufficient to give every
    # holder both a predecessor and successor.  The backup catalogue must
    # include an equal target profile instead of applying contradictory local
    # follower/leader overrides around the loop.
    pairs = tuple((holders[index], holders[(index + 1) % len(holders)]) for index in range(len(holders)))
    candidates = shield._ring_profile_backup_candidates(env, requested, pairs)
    assert candidates
    assert any(len({round(candidate[uid], 8) for uid in holders}) == 1 for candidate in candidates)


def test_runtime_repair_tracks_use_executor_equivalent_safety_semantics():
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holders = sorted(uid for uid, uav in env.active.items() if uav.ring_role is not None and uav.committed_plan is None)
    assert len(holders) == 2
    end = env.now + env.config.shield_horizon_seconds
    tracks = {
        uid: shield._repair_plan(env.active[uid], env.airspace, env.now, end, env.config.max_speed_residual_mps)
        for uid in holders
    }
    assert all(plan.metadata["runtime_executor_equivalent"] for plan in tracks.values())
    # The separation verifier must judge the exact shared dynamics track, not
    # re-interpret its sampled chord as a free-form transfer trajectory.
    assert env.safety.verify_predicted_tracks(tracks, start=env.now, end=end).approved


def test_joint_shield_keeps_imminent_transfer_handoff_pair_despite_broad_phase(monkeypatch):
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holder, transfer = (env.active[uid] for uid in sorted(env.active)[:2])
    role = holder.ring_role
    assert role is not None
    end = env.now + 0.5
    transfer.committed_plan = TrajectoryPlan(
        "imminent-handoff", transfer.uid, "transfer", (
            TrajectoryPoint(env.now, transfer.position.copy()),
            TrajectoryPoint(end, holder.position.copy()),
        ), env.now, end, metadata={"target_phase": f"ring_{role.value}", "committed": True},
    )
    transfer.ring_role = None
    # This reproduces the old erroneous cull: an imminent target-ring handoff
    # must remain in the runtime conflict graph even if its current transfer
    # geometry is far from the holder.
    monkeypatch.setattr(shield, "_broad_phase_reject", lambda *_args, **_kwargs: True)
    assert tuple(sorted((holder.uid, transfer.uid))) in shield._joint_pairs(env)


def test_recursive_successor_converts_completed_transfer_to_ring_holder():
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holder, transfer = (env.active[uid] for uid in sorted(env.active)[:2])
    role = holder.ring_role
    assert role is not None
    end = env.now + 0.5 * env.config.dt
    endpoint = holder.position.copy()
    transfer.committed_plan = TrajectoryPlan(
        "successor-handoff", transfer.uid, "transfer", (
            TrajectoryPoint(env.now, transfer.position.copy(), transfer.velocity.copy(), transfer.acceleration.copy()),
            TrajectoryPoint(end, endpoint, transfer.velocity.copy(), transfer.acceleration.copy()),
        ), env.now, end, metadata={"target_phase": f"ring_{role.value}", "committed": True},
    )
    transfer.ring_role = None
    assert shield._has_imminent_ring_handoff(env, within_seconds=env.config.dt)
    expected = env.safety._post_plan_positions(
        transfer.committed_plan, SimpleNamespace(now=env.now, active=env.active, airspace=env.airspace),
        np.asarray((env.now + env.config.dt,)),
    )[0]
    successor = shield._next_executor_state(env, {holder.uid: 0.0})
    predicted = successor.active[transfer.uid]
    assert predicted.committed_plan is None
    assert predicted.ring_role is role
    assert np.allclose(predicted.position, expected)
    assert not np.allclose(predicted.position, endpoint)


def test_recursive_successor_adds_handed_off_holder_to_backup_action_domain():
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    holder, transfer = (env.active[uid] for uid in sorted(env.active)[:2])
    role = holder.ring_role
    assert role is not None
    end = env.now + 0.5 * env.config.dt
    transfer.committed_plan = TrajectoryPlan(
        "backup-domain-handoff", transfer.uid, "transfer", (
            TrajectoryPoint(env.now, transfer.position.copy()),
            TrajectoryPoint(end, holder.position.copy()),
        ), env.now, end, metadata={"target_phase": f"ring_{role.value}", "committed": True},
    )
    transfer.ring_role = None
    successor = shield._next_executor_state(env, {holder.uid: 0.0})
    domain = shield._successor_action_domain(successor, {holder.uid: 0.0})
    assert domain[transfer.uid] == 0.0
    assert set(domain) >= {holder.uid, transfer.uid}


def test_recursive_successor_projection_does_not_mutate_authoritative_uavs():
    env = _student_env(2)
    shield = PredictiveSafetyShield(env.config)
    before = {
        uid: (
            uav.position.copy(), uav.velocity.copy(), uav.acceleration.copy(),
            uav.speed, uav.ring_progress, uav.ring_role, uav.committed_plan,
        )
        for uid, uav in env.active.items()
    }

    successor = shield._next_executor_state(env, {uid: 0.0 for uid in env.active})

    for uid, uav in env.active.items():
        position, velocity, acceleration, speed, progress, role, plan = before[uid]
        assert successor.active[uid] is not uav
        assert np.array_equal(uav.position, position)
        assert np.array_equal(uav.velocity, velocity)
        assert np.array_equal(uav.acceleration, acceleration)
        assert uav.speed == speed
        assert uav.ring_progress == progress
        assert uav.ring_role is role
        assert uav.committed_plan is plan


def test_committed_track_cache_is_step_scoped_and_reuses_exact_array():
    env = _student_env(1)
    uav = next(iter(env.active.values()))
    endpoint = uav.position + np.asarray((10.0, 0.0, -5.0))
    uav.committed_plan = TrajectoryPlan(
        "committed-cache", uav.uid, "final", (
            TrajectoryPoint(env.now, uav.position.copy()),
            TrajectoryPoint(env.now + 2.0, endpoint),
        ), env.now, env.now + 2.0, metadata={"target_phase": "landing"},
    )
    times = np.arange(env.now, env.now + 3.0 + 1e-9, env.config.safety_sampling_dt)
    snapshot = SimpleNamespace(now=env.now, active=env.active, airspace=env.airspace)

    first = env.safety._committed_positions(uav.committed_plan, snapshot, times)
    second = env.safety._committed_positions(uav.committed_plan, snapshot, times.copy())
    assert second is first
    assert not first.flags.writeable

    env.safety.begin_prediction_step()
    third = env.safety._committed_positions(uav.committed_plan, snapshot, times)
    assert third is not first
    assert np.array_equal(third, first)


def test_partial_tick_plan_handoff_matches_post_plan_prediction():
    env = _student_env(1)
    uav = next(iter(env.active.values()))
    role = RingRole.MIDDLE
    ring = env.airspace.actual_ring(role)
    endpoint = np.asarray((ring.radius, 0.0, ring.height))
    endpoint_velocity = np.asarray((0.0, ring.nominal_speed, 0.0))
    uav.phase = FlightPhase.TRANSFER_O2M
    uav.ring_role = None
    end = env.now + 0.5 * env.config.dt
    plan = TrajectoryPlan(
        "partial-tick-handoff", uav.uid, "transfer", (
            TrajectoryPoint(env.now, uav.position.copy(), uav.velocity.copy(), uav.acceleration.copy()),
            TrajectoryPoint(end, endpoint, endpoint_velocity, np.zeros(3)),
        ), env.now, end, metadata={"target_phase": f"ring_{role.value}", "committed": True},
    )
    expected = env.safety._post_plan_positions(
        plan, SimpleNamespace(now=env.now, active=env.active, airspace=env.airspace),
        np.asarray((env.now + env.config.dt,)),
    )[0]
    uav.committed_plan = plan
    uav.trajectory_plan_id = plan.plan_id
    env.safety.checked_trajectory_hashes[plan.plan_id] = plan.trajectory_hash
    observations, _, _, _, info = env.step({str(uav.uid): np.zeros(3, dtype=np.float32)})
    assert observations
    assert np.allclose(env.active[uav.uid].position, expected)
    assert info["prediction_execution_position_error"][str(uav.uid)] < 1e-9


def test_checkpoint_schema_rejects_legacy_actor_metadata():
    legacy = SimConfig(experiment_mode="fixed_a5_legacy", observation_schema_version="phaseA_low_obs_v1_41d")
    student = SimConfig(experiment_mode="fixed_low_student")
    with pytest.raises(IncompatibleCheckpointError):
        validate_checkpoint_metadata(checkpoint_metadata(legacy), student)


def test_ppo_transition_keeps_requested_action_when_teacher_label_is_present():
    requested = np.asarray((1, 2, 3), np.int64)
    teacher = np.asarray((1, 4, 0), np.int64)
    world = np.zeros((1, LOW_OBS_V2.dimension), np.float32)
    row = Transition("uav-1", world[0], world, requested, 0.0, 0.0, 1.0, True,
                     teacher_action=teacher, applied_action=teacher, teacher_used=True)
    buffer = RolloutBuffer()
    buffer.add(row)
    assert np.array_equal(buffer.by_agent["uav-1"][0].action, requested)
    metrics = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=0.1).update(buffer, epochs=1, minibatch_size=1)
    assert "teacher_bc_loss" in metrics and np.isfinite(metrics["teacher_bc_loss"])


def test_ppo_ratio_is_invariant_to_applied_action_and_reports_diagnostics():
    requested = np.asarray((0.2, -0.1, 0.3), np.float32)
    world = np.zeros((1, LOW_OBS_V2.dimension), np.float32)

    def buffer_for(applied: np.ndarray) -> RolloutBuffer:
        buffer = RolloutBuffer()
        buffer.add(Transition("uav-1", world[0], world, requested, -0.2, 0.0, 1.0, True,
                              applied_action=applied, teacher_used=False))
        return buffer

    torch.manual_seed(4)
    first = SharedPPO(LOW_OBS_V2.dimension)
    second = SharedPPO(LOW_OBS_V2.dimension)
    second.model.load_state_dict(copy.deepcopy(first.model.state_dict()))
    np.random.seed(17)
    first_metrics = first.update(buffer_for(np.asarray((1.0, 1.0, 1.0), np.float32)), epochs=1, minibatch_size=1)
    np.random.seed(17)
    second_metrics = second.update(buffer_for(np.asarray((-1.0, -1.0, -1.0), np.float32)), epochs=1, minibatch_size=1)
    assert first_metrics == pytest.approx(second_metrics)
    assert all(torch.allclose(left, right) for left, right in zip(first.model.parameters(), second.model.parameters()))
    assert {"approx_kl", "clip_fraction", "explained_variance", "grad_norm"} <= set(first_metrics)


def test_ppo_d2_snapshot_is_non_mutating_and_keeps_teacher_targets_normalized():
    """D2 can inspect requested/teacher actions without changing PPO state."""
    requested = np.asarray((0.2, -0.1, 0.3), np.float32)
    teacher = np.asarray((1.0, -1.0, 1.0), np.float32)
    world = np.zeros((1, LOW_OBS_V2.dimension), np.float32)
    buffer = RolloutBuffer()
    buffer.add(Transition("uav-1", world[0], world, requested, -0.2, 0.0, 1.0, True,
                          teacher_action=teacher, applied_action=teacher, teacher_used=True))
    buffer.add(Transition("uav-2", world[0], world, -requested, -0.1, 0.0, -1.0, True,
                          teacher_action=-teacher, applied_action=-teacher, teacher_used=True))
    policy = SharedPPO(LOW_OBS_V2.dimension, teacher_bc_weight=.05)
    before = [parameter.detach().clone() for parameter in policy.model.parameters()]
    snapshot = policy.diagnostic_snapshot(buffer, teacher_bc_weight=.05)
    assert snapshot["rows"] == 2 and snapshot["teacher_used_rows"] == 2
    assert set(snapshot["heads"]) == {"release_request", "speed_residual", "merge_time_residual"}
    assert snapshot["heads"]["merge_time_residual"]["teacher_target_saturation_rate"] == 1.0
    # One PPO gradient can legitimately be exactly zero after clipping; the
    # diagnostic excludes undefined cosines rather than fabricating a sign.
    assert snapshot["heads"]["merge_time_residual"]["teacher_used_head_output_gradient_cosine"]["samples"] >= 1
    assert snapshot["gradient_blocks"]["shared_backbone"] == 0.0
    assert all(torch.equal(before_parameter, after_parameter)
               for before_parameter, after_parameter in zip(before, policy.model.parameters()))


def test_low_level_intent_keeps_merge_residual_normalized_until_scheduler_execution():
    intent = LowLevelIntent.from_array((3.0, -3.0, 2.0))
    assert (intent.release_request, intent.speed_residual, intent.merge_time_residual) == (1.0, -1.0, 1.0)
    config = SimConfig()
    assert config.max_merge_time_residual_seconds == 6.0


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
