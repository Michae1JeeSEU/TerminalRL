"""Fixed A4 intervention protocol, with one tiny deferred-delivery smoke test."""
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from pathlib import Path
import json
import time

import pytest

from scripts.stage3 import run_selective_short_continuation_a4 as runner
from scripts.stage3 import run_selective_quality_gap_a4 as a4


@dataclass(frozen=True)
class Plan:
    trajectory_hash: str
    start_time: float = 85.
    pad_id: int | None = None


@dataclass(frozen=True)
class Column:
    aircraft_id: str
    kind: str
    plans: tuple = ()
    objective: tuple = (0, 0, 0, 0, 0)
    @property
    def signature(self): return self.aircraft_id, self.kind, tuple(p.trajectory_hash for p in self.plans)


def fake_problem(case="R4"):
    aid = a4.CASES[case]["aid"]
    keep = Column(aid, "KEEP")
    left = Column(aid, "MOVE", (Plan("left"),), (0, 0, 1, 0, 0))
    right = Column(aid, "MOVE", (Plan("right"),), (0, 0, 2, 0, 0))
    future = Column(aid, "MOVE", (Plan("future", 90.),), (0, 0, -1, 0, 0))
    full = Column(aid, "MOVE", (Plan("full", 85., 0),), (0, 0, -1, 0, 0))
    other = Column("other", "WAIT")
    def key(selected): return tuple(sum(c.objective[i] for c in selected) for i in range(5))
    return dict(snapshot=NS(time=85.), selected=(keep, other),
        catalog=NS(columns={aid: (keep, right, future, full, left), "other": (other,)}, downstream=NS(key=key)))


def passing_context(log=None):
    safe = NS(accepted=True, status="SAFE", reason="", details={}, refinement_required=False)
    def batch(selected):
        if log is not None: log.append(tuple(selected))
        return safe, ()
    return NS(check_batch_conflicts=batch, check_commit_prefix=lambda selected: safe,
        prefix=lambda selected: tuple(c.plans[0] for c in selected if c.plans))


@pytest.mark.parametrize("case", runner.CASES)
def test_fixed_uid_hash_order_only_one_aircraft_changes(case):
    from terminal_marl.core.rotating_slots.selective_replanning.diagnostics import candidate_hash
    problem = fake_problem(case)
    choices, identity = runner.branch_choices(problem, case)
    aid = a4.CASES[case]["aid"]
    expected = min((c for c in problem["catalog"].columns[aid] if c.plans and c.plans[-1].pad_id is None
        and c.plans[0].start_time == 85.), key=candidate_hash)
    assert identity["candidate_hash"] == candidate_hash(expected)
    assert choices["forced_partial"][0] == expected
    assert choices["forced_partial"][1] is problem["selected"][1]
    assert choices["baseline_choice"] == problem["selected"]


def test_missing_fixed_partial_is_not_applicable_without_other_uid_fallback():
    problem = fake_problem()
    aid = a4.CASES["R4"]["aid"]
    problem["catalog"].columns[aid] = (problem["selected"][0],)
    choices, row = runner.branch_choices(problem, "R4")
    assert choices is None and row["status"] == "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="only R4 and R12"):
        runner.branch_choices(problem, "R1")


def test_preparation_requires_independent_safe_certificates_and_strictly_worse_score(monkeypatch):
    problem = fake_problem()
    contexts, selected = [], []
    def independent(problem):
        context = passing_context(selected)
        contexts.append(context)
        return context, NS()
    monkeypatch.setattr(a4, "offline_context", independent)
    prepared = runner.prepare_pair(problem, "R4")
    assert prepared["status"] == "PREPARED" and prepared["forced_strictly_worse"]
    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    assert selected[0] == problem["selected"]
    assert not prepared["physical_execution"] and not prepared["solver_called"]
    assert not prepared["online_performance_evidence"]
    problem["catalog"].downstream.key = lambda selected: (0, 0, 0, 0, 0)
    assert runner.prepare_pair(problem, "R4")["reason"] == "FORCED_CHOICE_NOT_STRICTLY_WORSE"


def test_preparation_unknown_prefix_cannot_make_action_pair(monkeypatch):
    context = passing_context()
    context.check_commit_prefix = lambda selected: NS(accepted=False, status="UNKNOWN", reason="BUDGET", details={})
    monkeypatch.setattr(a4, "offline_context", lambda problem: (context, NS()))
    prepared = runner.prepare_pair(fake_problem(), "R4")
    assert prepared["status"] == "NOT_APPLICABLE"
    assert prepared["reason"] == "FIXED_COMBINATION_NOT_CERTIFIED"


def tiny_problem():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld, STAGE2_POLICY_ID
    from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming, project_visible_snapshot
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider, EvaluationContext, PlanningBudget
    from terminal_marl.core.rotating_slots.selective_replanning import PlanningScope, build_global_frame
    from terminal_marl.core.rotating_slots.trajectory import generate_transfer
    world = RotatingSlotWorld(experiment_policy_id=STAGE2_POLICY_ID)
    world.auto_emergency = False
    aid = a4.CASES["R4"]["aid"]
    world.seed_hold(aid, "middle", 0, .9)
    source = world.build_planning_snapshot()
    future = project_visible_snapshot(source, 20., timing=PlanningTiming.from_lead(0., 20., 1.))
    provider = CandidateProvider(planning_profile="downstream_service_v8")
    frame = build_global_frame(future, provider, EvaluationContext(future))
    scope = PlanningScope(future.version, future.time, (aid,), (aid,), (), actual_k=1, objective_frame_id=frame.identity)
    context = EvaluationContext(future, scope=scope, global_frame=frame)
    plan = generate_transfer(aid, "middle", 0, 0, 20., source_version=0, config=future.config, durations=(40.,)).plan
    assert plan is not None
    columns = {aid: [provider._column(future, future.aircraft[aid], (), kind, context) for kind in ("KEEP", "WAIT")]}
    columns[aid].append(provider._column(future, future.aircraft[aid], (plan,), "MOVE", context))
    catalog = provider._progressive_catalog(future, PlanningBudget(), context, columns,
        {aid: columns[aid][0].reference}, [future.aircraft[aid]])
    return world, source, dict(snapshot=future, catalog=catalog, global_frame=frame, selected=(catalog.columns[aid][0],)), plan


def real_dependencies():
    from terminal_marl.core.rotating_slots import stage2_campaign as campaign
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    return NS(NormalSchedulerCallback=NormalSchedulerCallback, LatencyAwareScheduler=LatencyAwareScheduler,
        episode_metrics=campaign.episode_metrics, save_checkpoint=campaign.save_checkpoint,
        ReplayBundle=ReplayBundle, shared_fifo_admission=campaign.shared_fifo_admission)


def test_adapter_requires_future_port_and_never_direct_physical_submit():
    world, source, problem, plan = tiny_problem()
    d = real_dependencies()
    delegate = runner.normal_callback(d)
    selected = (problem["catalog"].columns[plan.aircraft_id][-1],)
    adapter = runner.FirstDecisionAdapter(delegate, problem, source, selected, "forced_partial")
    with pytest.raises(ValueError, match="requires wrapper FuturePort"):
        adapter(world._scheduler_port)
    assert not world.active and adapter.first_pending


@pytest.mark.parametrize("changed", ("source_energy", "effective_energy", "resource", "timing", "deadline"))
def test_online_intervention_rejects_changed_input_or_late_validation(changed):
    from terminal_marl.core.rotating_slots.planning_latency import _FuturePort
    from terminal_marl.core.rotating_slots.resources import Reservation
    world, source, problem, plan = tiny_problem()
    adapter = runner.FirstDecisionAdapter(runner.normal_callback(real_dependencies()), problem, source,
        (problem["catalog"].columns[plan.aircraft_id][-1],), "forced_partial")
    physical, future = source, problem["snapshot"]
    aid = plan.aircraft_id
    if changed == "source_energy":
        physical = replace(source, aircraft={aid: replace(source.aircraft[aid], energy_j=1.)})
    elif changed == "effective_energy":
        future = replace(future, aircraft={aid: replace(future.aircraft[aid], soc=.8)})
    elif changed == "resource":
        future = replace(future, slots=future.slots+(Reservation(("inner", 1), "other", 0., 100.),))
    elif changed == "timing":
        future = replace(future, planning_timing=replace(future.planning_timing, ready_deadline=18.5))
    deadline = time.perf_counter()+19. if changed != "deadline" else time.perf_counter()-1.
    port = _FuturePort(future, physical, deadline, planning_wall_origin=time.perf_counter())
    result = adapter(port)
    assert not result.certified and port.proposal is None and not world.active
    assert not adapter.provider.previous and not adapter.provider.references.cache
    assert adapter.records[0]["record_type"] == "DIAGNOSTIC_FORCED_FIRST_CALLBACK"


def test_equal_codec_inputs_with_new_generated_versions_are_rebound_and_rechecked():
    from terminal_marl.core.rotating_slots.planning_latency import _FuturePort
    _, source, problem, plan = tiny_problem()
    physical = replace(source, version="new_source_version")
    future = replace(problem["snapshot"], version="new_effective_version")
    adapter = runner.FirstDecisionAdapter(runner.normal_callback(real_dependencies()), problem, source,
        (problem["catalog"].columns[plan.aircraft_id][-1],), "forced_partial")
    port = _FuturePort(future, physical, time.perf_counter()+19., planning_wall_origin=time.perf_counter())
    result = adapter(port)
    assert result.certified and port.proposal == (plan,)
    evidence = adapter.first_evidence
    for item in evidence["input_equivalence"].values():
        assert item["equivalent"] and item["old_version"] != item["new_version"]
        assert item["old_canonical_sha256"] == item["new_canonical_sha256"]
        assert item["excluded_fields"] == ["version"]
    binding = evidence["fresh_binding"]
    assert binding["scope"]["snapshot_id"] == future.version
    assert binding["scope"]["objective_frame_id"] == binding["frame_id"]
    assert binding["frame_id"] != binding["old_frame_id"]
    assert evidence["fresh_validation"]["certified"]


def test_tiny_real_deferred_delivery_and_full_output_boundary(tmp_path, monkeypatch):
    """One aircraft, 20.05 physical seconds; no ordinary CP call or long run."""
    from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
    from terminal_marl.core.rotating_slots import stage2_campaign as campaign
    from terminal_marl.core.rotating_slots.replay import ReplayBundle
    world, source, problem, plan = tiny_problem()
    d = real_dependencies()
    delegate = runner.normal_callback(d)
    delegate.initialize_imported_history(source, events=world.events)
    delegate.provider.previous["warm"] = (plan,)
    delegate.provider.references.cache["warm"] = True
    old_history = deepcopy(delegate.selective_history.state_dict())
    runner.cold_search(delegate)
    assert delegate.selective_history.state_dict() == old_history
    assert not delegate.provider.previous and not delegate.provider.references.cache
    adapter = runner.FirstDecisionAdapter(delegate, problem, source,
        (problem["catalog"].columns[plan.aircraft_id][-1],), "forced_partial")
    calls = []
    monkeypatch.setattr(NormalSchedulerCallback, "__call__", lambda self, port: calls.append(port.time))
    wrapper = d.LatencyAwareScheduler(adapter, lead_seconds=20., preparation_margin=1.,
        search_handoff_reserve_seconds=4., restart_policy="coalesced_restart_v2")
    initial = d.episode_metrics(world)
    holding, restore = runner.holding_observer(world)
    try:
        wrapper(world._scheduler_port)
        assert adapter.first_evidence["certified"] and not world.active, (
            adapter.first_evidence.get("rejection_reason"), adapter.first_evidence.get("fresh_validation"))
        assert wrapper.pending["proposal"] == (plan,)
        assert not delegate.provider.previous and not delegate.provider.validation_cache
        world.advance(20.05, wrapper, settle_only_at_end=True)
    finally:
        restore()
    receipt = runner.first_delivery(adapter, wrapper)
    assert receipt["first_action_delivered"] and receipt["actual_prefix_hashes"] == [plan.trajectory_hash]
    assert world.active[plan.aircraft_id].trajectory_hash == plan.trajectory_hash
    assert calls  # Later callbacks use the ordinary delegate.
    assert holding["middle"]["seconds"] == pytest.approx(20.)
    assert holding["middle"]["energy_j"] > 0. and world.aircraft[plan.aircraft_id].energy_j > 0.
    identity = campaign.checkpoint_identity(dict(source_hash="tiny", base_config_hash=world.config.config_hash,
        overlay_sha256="tiny", planning_profile="downstream_service_v8"), campaign.digest(world.manifest.to_dict()))
    result = runner.emit_outputs(tmp_path, world, adapter, wrapper, {"status": "PREPARED"}, initial, set(), holding,
        d, dict(case="R4", checkpoint_identity=identity, a4_source_hash="tiny", source_commit="tiny", seed=world.manifest.seed))
    assert result["status"] == "CONTINUATION_PARTIAL" and result["absolute_end_time"] == 120.
    terminal = json.loads((tmp_path/"terminal.json").read_text())
    assert terminal["increments"]["energy_j"] > 0. and terminal["physical_ledger_reset"] is False
    assert (tmp_path/"firstcert.json").is_file() and (tmp_path/"actualsubmit.json").is_file()
    assert ReplayBundle.validate(tmp_path/"replay").state_count > 0
    with pytest.raises(FileExistsError):
        a4.write_new(tmp_path/"terminal.json", {})


def test_first_delivery_missing_or_rejected_is_never_valid_action_contrast():
    adapter = NS(records=[dict(actual_submitted_trajectory_hashes=[])], first_evidence={"proposed_prefix_hashes": ["p"]})
    row = runner.first_delivery(adapter, NS(records=[dict(status="REJECTED")]))
    assert not row["first_action_delivered"] and not row["valid_action_contrast"]
    assert row["action_comparison_status"] == "FIRST_ACTION_NOT_DELIVERED"


def test_certified_empty_baseline_wait_requires_real_no_executable_prefix_ack():
    adapter = NS(branch="baseline_choice", selected=(Column("a", "KEEP"), Column("b", "WAIT")),
        records=[dict(actual_delivery_status="NO_EXECUTABLE_PREFIX", actual_submitted_trajectory_hashes=[])],
        first_evidence=dict(certified=True, proposed_prefix_hashes=[]))
    task = dict(status="NO_EXECUTABLE_PREFIX", reason="NO_EXECUTABLE_PREFIX")
    receipt = runner.first_delivery(adapter, NS(records=[task]))
    assert receipt["valid_baseline_wait"] and receipt["valid_action_contrast"]
    assert not receipt["first_action_delivered"] and receipt["action_comparison_status"] == "VALID_BASELINE_WAIT"
    for status, reason in (("REJECTED", "READY_DEADLINE_MISSED"), ("COMPUTING", "")):
        rejected = runner.first_delivery(adapter, NS(records=[dict(status=status, reason=reason)]))
        assert not rejected["valid_baseline_wait"] and not rejected["valid_action_contrast"]
    adapter.branch = "forced_partial"
    assert not runner.first_delivery(adapter, NS(records=[task]))["valid_action_contrast"]


def test_freeze_requires_this_script_and_its_tests(monkeypatch):
    monkeypatch.setattr(runner.base, "verify_sources", lambda source: None)
    with pytest.raises(ValueError, match="script/test freeze"):
        runner.verify_freeze(dict(files={}))
    runner.verify_freeze(dict(files={runner.DRIVER.relative_to(runner.PROJECT).as_posix(): "x",
        "tests/test_selective_short_continuation_a4.py": "y"}))


def test_replay_seed_comes_from_identity_bound_high_flow_manifest():
    from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
    manifest = replace(RotatingSlotWorld().manifest, seed=7325)
    payload = dict(world=a4.codec().encode({"manifest": manifest}),
        identity={"manifest_sha256": runner.base.digest(manifest.to_dict())})
    assert runner.fixed_checkpoint_seed(payload) == 7325
    payload["identity"]["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest/seed differs"):
        runner.fixed_checkpoint_seed(payload)
    wrong = replace(manifest, seed=7324)
    payload = dict(world=a4.codec().encode({"manifest": wrong}),
        identity={"manifest_sha256": runner.base.digest(wrong.to_dict())})
    with pytest.raises(ValueError, match="manifest/seed differs"):
        runner.fixed_checkpoint_seed(payload)


def test_parent_dispatch_has_one_branch_and_shared_300_second_ledger(monkeypatch, tmp_path):
    from contextlib import contextmanager
    output = tmp_path/a4.OUTPUT_NAME
    freeze = tmp_path/"source.json"
    freeze.write_text(json.dumps(dict(source_hash="same", commit="test", files={})))
    ledger = dict(jobs={})
    @contextmanager
    def transaction(*args, **kwargs): yield ledger
    launches = []
    def bounded(output, job_id, command, **kwargs):
        launches.append((job_id, command, kwargs))
        ledger["jobs"][job_id] = {"status": "completed"}
        return {"status": "completed"}
    monkeypatch.setattr(a4, "validated_output", lambda path: Path(path))
    monkeypatch.setattr(runner, "verify_freeze", lambda source: None)
    monkeypatch.setattr(a4, "ledger_transaction", transaction)
    monkeypatch.setattr(a4, "run_bounded_job", bounded)
    runner.main(["--stage", "run", "--case", "R12", "--branch", "forced_partial", "--output", str(output),
        "--source-freeze", str(freeze), "--cpu", "12"])
    job, command, kwargs = launches[0]
    assert job == "short_continuation:R12:forced_partial" and kwargs == {"cap_s": 300.}
    assert command[command.index("--stage")+1] == "run-child"
    assert ledger["a4_source_hash"] == "same" and ledger["jobs"][job]["run_role"] == runner.ROLE


@pytest.mark.parametrize("branch", runner.BRANCHES)
def test_run_branch_stops_at_source_plus_120_and_never_resets_energy(monkeypatch, tmp_path, branch):
    problem = fake_problem()
    choices, selected = runner.branch_choices(problem, "R4")
    prepared = dict(selected, status="PREPARED")
    observed = []
    physical = NS(time=65., version="physical")
    class World:
        time = 65.
        scheduler_period = 5.
        failed = False
        terminated_invalid = False
        aircraft = {}
        _scheduler_port = object()
        def build_planning_snapshot(self): return physical
        def advance(self, target, wrapper, *, settle_only_at_end):
            observed.append((target, settle_only_at_end))
            self.time = target
    world = World()
    class Delegate:
        selective_history = NS(state_dict=lambda: {"executed_history": True})
    delegate = Delegate()
    monkeypatch.setattr(v3 := runner.v3, "restore_cold_with_history", lambda *args: (world, {"history": True}))
    monkeypatch.setattr(v3, "restore_selector_history", lambda *args: None)
    monkeypatch.setattr(runner, "normal_callback", lambda d: delegate)
    monkeypatch.setattr(runner, "cold_search", lambda cb: {"cold": True})
    monkeypatch.setattr(runner, "holding_observer", lambda world: ({}, lambda: None))
    adapter = NS()
    monkeypatch.setattr(runner, "FirstDecisionAdapter", lambda *args: adapter)
    class Wrapper:
        def __init__(self, cb, **kwargs): self.cb, self.settings = cb, kwargs
        def __call__(self, port): observed.append(("first_wrapper_call", world.time))
    d = NS(LatencyAwareScheduler=Wrapper, shared_fifo_admission=object(),
        episode_metrics=lambda w: {"energy_j": 99.})
    captured = {}
    def emit(directory, w, a, wrapper, p, initial, touch, holding, dependencies, binding, error):
        captured.update(end=w.time, initial=initial, settings=wrapper.settings, selected=p, error=error)
        return captured
    monkeypatch.setattr(runner, "emit_outputs", emit)
    runner.run_branch(tmp_path/branch, problem, physical, tmp_path/"checkpoint", prepared,
        {"case": "R4"}, branch, d)
    assert observed[0] == ("first_wrapper_call", 65.)
    assert observed[1:] == [(float(i), True) for i in range(70, 186, 5)]
    assert captured["end"] == 185. and captured["initial"] == {"energy_j": 99.}
    assert captured["settings"] == dict(lead_seconds=20., preparation_margin=1.,
        search_handoff_reserve_seconds=4., restart_policy="coalesced_restart_v2")
