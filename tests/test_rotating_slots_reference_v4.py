"""Fixed-snapshot reference policy contracts; no episode rollout."""
from dataclasses import FrozenInstanceError, replace
from itertools import product
from types import SimpleNamespace
import json

import pytest

from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.stage2_planning import (
    CandidateProvider, EvaluationContext, PlanningBudget, PlanningNode, PredictionColumn,
    CandidateCatalog, PlanningResult, NormalSchedulerCallback, RollingHorizonCPSATScheduler)
from terminal_marl.core.rotating_slots.selective_replanning import RuleSelector, PlanningScope, LocalPlanner
from terminal_marl.core.rotating_slots.selective_replanning.policy import (
    SearchPolicy, REFERENCE_VARIANT, resolve_search_policy)
from terminal_marl.core.rotating_slots.selective_replanning.near_term import (
    summarize_opportunities, cooperative_pad_groups)
from terminal_marl.core.rotating_slots.selective_replanning.relations import Relation, build_relations
from terminal_marl.core.rotating_slots.trajectory import generate_transfer


V2 = "opportunity_cooperative_v2"
V3 = "opportunity_executable_v3"
COMBINATIONS = tuple(product((False, True), repeat=3))


def snapshot(n=13):
    world = RotatingSlotWorld()
    for k in range(n):
        world.seed_hold(str(k), "outer", k, .9)
    return world.build_planning_snapshot()


def history():
    return SimpleNamespace(ages=lambda aid, now: dict(review_age=0., review_unknown=False,
        progress_age=0., progress_unknown=False))


def callback(policy=None, variant=REFERENCE_VARIANT, method="RULE_LOCAL_K8"):
    return NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8",
        selective_method=method, selective_rule_variant=variant, search_policy=policy)


def patch_nodes(monkeypatch, future=(), urgent=()):
    def node(s, a):
        return PlanningNode(a.ring, a.slot, 40. if a.id in future else s.time,
            .249 if a.id in urgent else .9, 0., a.version)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.stage2_planning._node", node)


@pytest.mark.parametrize("switches", COMBINATIONS)
def test_reference_policy_is_explicit_frozen_roundtripped_and_bound(switches):
    policy = SearchPolicy(*switches)
    cb = callback(policy)
    assert cb.search_policy == policy and cb.selective_history.progress_policy == "maneuver_started_v2"
    state = json.loads(json.dumps(cb.state_dict()))
    assert state["selective_replanning"]["search_policy"] == policy.to_dict()
    restored = callback(policy.to_dict())
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    different = replace(policy, pad_anchor_priority=not policy.pad_anchor_priority)
    with pytest.raises(ValueError, match="search policy binding mismatch"):
        callback(different).load_state_dict(state)
    missing = json.loads(json.dumps(state))
    missing["selective_replanning"].pop("search_policy")
    with pytest.raises(ValueError, match="search policy binding mismatch"):
        cb.load_state_dict(missing)
    numeric = json.loads(json.dumps(state))
    numeric["selective_replanning"]["search_policy"]["pad_anchor_priority"] = int(policy.pad_anchor_priority)
    with pytest.raises(ValueError, match="search policy binding mismatch"):
        cb.load_state_dict(numeric)
    with pytest.raises(FrozenInstanceError):
        policy.pad_anchor_priority = True


@pytest.mark.parametrize("variant", ("baseline", "outer_priority_v1", V2, V3))
def test_old_variants_keep_fixed_switches_and_checkpoint_shape(variant):
    fixed = SearchPolicy(True, True, True) if variant == V3 else SearchPolicy()
    cb = callback(variant=variant)
    assert cb.search_policy == fixed
    state = cb.state_dict()
    assert "search_policy" not in state["selective_replanning"]
    callback(variant=variant).load_state_dict(state)
    with pytest.raises(ValueError, match="overrides require"):
        callback(replace(fixed, ordinary_immediate_filter=not fixed.ordinary_immediate_filter), variant)
    with pytest.raises(ValueError, match="overrides require"):
        RuleSelector(variant=variant, search_policy=replace(fixed, target_coverage_order=not fixed.target_coverage_order))


def test_policy_defaults_and_invalid_values():
    assert callback().search_policy == SearchPolicy()
    assert resolve_search_policy(REFERENCE_VARIANT, {"pad_anchor_priority": True}) == SearchPolicy(True, False, False)
    for bad in ({"target_coverage_order": 1}, {"unknown": False}, [], True):
        with pytest.raises(ValueError):
            callback(bad)


def test_default_reference_keeps_v2_selection_and_relation_semantics():
    s = snapshot()
    frame = SimpleNamespace(identity="frame")
    before, old = RuleSelector(variant=V2).select(s, frame, history())
    after, new = RuleSelector(variant=REFERENCE_VARIANT).select(s, frame, history())
    assert before == after
    assert old["fairness_selected_ids"] == new["fairness_selected_ids"]
    assert [g["members"] for g in old["cooperative_groups"]] == [g["members"] for g in new["cooperative_groups"]]
    assert set(new["adjustment_access"]) == set(s.aircraft)
    assert new["search_policy"] == SearchPolicy().to_dict()
    assert build_relations(s, tuple(s.aircraft), variant=V2) == build_relations(s, tuple(s.aircraft), variant=REFERENCE_VARIANT)


@pytest.mark.parametrize("switches", COMBINATIONS)
def test_ordinary_filter_is_independent_and_full_scope_remains_complete(monkeypatch, switches):
    s = replace(snapshot(), time=20., planning_timing=SimpleNamespace(effective_time=20., snapshot_time=0.))
    patch_nodes(monkeypatch, future={"0"})
    h = history()
    h.ages = lambda aid, now: dict(review_age=100. if aid == "0" else 0., review_unknown=False,
        progress_age=0., progress_unknown=False)
    selector = RuleSelector(variant=REFERENCE_VARIANT, search_policy=SearchPolicy(*switches))
    scope, info = selector.select(s, SimpleNamespace(identity="f"), h)
    assert ("0" in scope.replan_ids) == (not switches[2])
    assert not info["adjustment_access"]["0"]["adjustable"]
    assert ("0" in info["fairness_selected_ids"]) == (not switches[2])
    assert scope.actual_k == 8
    full, full_info = selector.select(s, SimpleNamespace(identity="f"), h, method="FULL_H")
    assert set(full.replan_ids) == set(s.aircraft)
    assert not full_info["selected_adjustable_at_effective"]["0"]
    assert full_info["search_policy"] == SearchPolicy(*switches).to_dict()


@pytest.mark.parametrize("filtered", (False, True))
def test_hard_future_urgency_closure_survives_filter(monkeypatch, filtered):
    s = replace(snapshot(), time=20.)
    patch_nodes(monkeypatch, future={"0", "12"}, urgent={"0"})
    scope, info = RuleSelector(variant=REFERENCE_VARIANT,
        search_policy={"ordinary_immediate_filter": filtered}).select(s, SimpleNamespace(identity="f"), history(),
            (Relation("0", "12", "validated_background_blocker", True, True, ("outer", 12)),))
    assert {"0", "12"} <= set(scope.mandatory_ids) <= set(scope.replan_ids)
    assert info["hard_urgent_count"] == 1


def test_future_opportunities_keep_specific_release_epoch_and_pad_start_facts(monkeypatch):
    world = RotatingSlotWorld()
    world.seed_hold("anchor", "inner", 0, .9)
    world.seed_hold("middle", "middle", 1, .9)
    s = replace(world.build_planning_snapshot(), time=20.,
        planning_timing=SimpleNamespace(effective_time=20., snapshot_time=0.))
    patch_nodes(monkeypatch, future={"anchor"})
    rows = summarize_opportunities(s, s.aircraft, include_access=True)
    assert rows["anchor"]["options"]
    assert all(p["start_time"] >= 40. and not p["start_available"] for p in rows["anchor"]["options"])
    linked = [p for p in rows["middle"]["options"] if "anchor" in p["joint_members"]]
    assert linked
    link = linked[0]["release_dependencies"][0]
    assert link["resource"] == ("inner", 0) and link["owner"] == "anchor"
    assert link["owner_start_time"] == 40. and not link["owner_start_available"]
    assert link["estimated_release_time"] > 40. and not link["certified"]
    groups = cooperative_pad_groups(rows, allow_future=True)
    assert groups and not groups[0]["anchor_start_available"]
    assert groups[0]["anchor_start_time"] >= 40.


@pytest.mark.parametrize("switches", COMBINATIONS)
def test_pad_priority_alone_controls_group_path_and_fallback(monkeypatch, switches):
    s = snapshot()
    patch_nodes(monkeypatch)
    calls = []
    def seeds(rows):
        calls.append("v2")
        return (("0", "1"),)
    def pads(rows, **kwargs):
        calls.append("pad")
        return (dict(members=("0", "1"), anchor_id="0", pad_id=0),)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.near_term.cooperative_seed_groups", seeds)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.near_term.cooperative_pad_groups", pads)
    _, info = RuleSelector(variant=REFERENCE_VARIANT, search_policy=SearchPolicy(*switches)).select(
        s, SimpleNamespace(identity="f"), history())
    assert calls == (["pad"] if switches[0] else ["v2"])
    assert ("anchor_id" in info["cooperative_groups"][0]) == switches[0]


@pytest.mark.parametrize("pad_priority", (False, True))
def test_only_pad_priority_adds_anchor_singleton_before_ordinary_queue(monkeypatch, pad_priority):
    world = RotatingSlotWorld()
    for k, aid in enumerate(("anchor", "partner", "ordinary", *["hard" + str(i) for i in range(7)])):
        world.seed_hold(aid, "outer", k, .9)
    s = world.build_planning_snapshot()
    hard = {"hard" + str(i) for i in range(7)}
    patch_nodes(monkeypatch, urgent=hard)
    opportunities = {aid: dict(unknown=False, earliest_arrival=10. if aid == "ordinary" else 100.,
        options=(), joint_members=()) for aid in s.aircraft}
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.near_term.summarize_opportunities",
        lambda *args, **kwargs: opportunities)
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.near_term.cooperative_seed_groups",
        lambda *args: (("anchor", "partner"),))
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.near_term.cooperative_pad_groups",
        lambda *args, **kwargs: (dict(members=("anchor", "partner"), anchor_id="anchor", pad_id=0),))
    scope, _ = RuleSelector(variant=REFERENCE_VARIANT, search_policy={"pad_anchor_priority": pad_priority}).select(
        s, SimpleNamespace(identity="f"), history())
    assert set(scope.replan_ids) == hard | ({"anchor"} if pad_priority else {"ordinary"})


@pytest.mark.parametrize("switches", COMBINATIONS)
@pytest.mark.parametrize("method", ("RULE_LOCAL_K8", "FULL_H"))
def test_target_family_switch_is_independent_under_both_methods(monkeypatch, switches, method):
    s = snapshot(1)
    policy = SearchPolicy(*switches)
    cb = callback(policy, method=method)
    provider = CandidateProvider()
    events = []
    budget = PlanningBudget(max_per_aircraft=32, max_attempts=1024, candidate_seconds=6., soft_seconds=30., candidate_active_seconds=0.)
    budget._selective_executable_v3 = True  # A legacy direct-provider alias cannot override resolved v4 switches.
    monkeypatch.setattr(provider, "_previous_chain", lambda *args: None)
    monkeypatch.setattr(provider, "_column", lambda snapshot, a, plans, kind, context, ref=None, *args:
        PredictionColumn(a.id, kind, plans, (), (0,) * 5, PlanningNode(a.ring, a.slot, 0., .9, 0., 0)))
    monkeypatch.setattr(provider, "_progressive_catalog", lambda snapshot, budget, context, columns, refs, ordered:
        CandidateCatalog({aid: tuple(rows) for aid, rows in columns.items()}, {aid: (0, aid) for aid in columns}, {}, ()))
    def family(name):
        def steps(snapshot, a, node, budget, context, *args):
            events.append(name)
            budget.attempts[a.id] += 1
            if name == "target": args[0]["attempted_targets"].add(0)
            yield ("ATTEMPT", None, None)
        return steps
    monkeypatch.setattr(provider, "_target_coverage_steps", family("target"))
    monkeypatch.setattr(provider, "_progress_steps", family("progress"))
    class Scheduler:
        def plan(self, snapshot, catalog, context, budget, *, incumbent):
            # Stop after one initial publication; no simulation or solver.
            budget.max_attempts = 0
            return PlanningResult(tuple(rows[0] for rows in catalog.columns.values()), (), None, "UNKNOWN", (0,) * 5)
    scope, _ = RuleSelector(variant=REFERENCE_VARIANT, search_policy=cb.search_policy).select(
        s, SimpleNamespace(identity="f"), history(), method=method)
    frame = SimpleNamespace(snapshot_id=s.version, effective_time=s.time, identity="f", policy="fixed",
        order=("0",), upstream_request_ids=(), objective_eligibility={}, upstream_requests=())
    planner = LocalPlanner(provider, Scheduler(), progressive_v2=True, search_policy=cb.search_policy)
    planner.plan(s, scope, frame, budget)
    assert events == (["target"] if switches[1] else ["progress"])
    assert not hasattr(budget, "_selective_target_coverage_order")


@pytest.mark.parametrize("switches", COMBINATIONS)
def test_blocker_filter_alone_controls_future_expansion_with_shared_budget(monkeypatch, switches):
    s = snapshot(2)
    patch_nodes(monkeypatch, future={"1"})
    frame = SimpleNamespace(snapshot_id=s.version, effective_time=s.time, identity="f", policy="fixed",
        order=("0", "1"), upstream_request_ids=(), objective_eligibility={}, upstream_requests=())
    scope = PlanningScope(s.version, s.time, ("0", "1"), ("0",), ("1",), actual_k=1, objective_frame_id="f")
    class Provider:
        validation_cache = {}
        def iter_progressive_columns(self, snapshot, budget, context, *, scope, early_publish):
            for _ in range(3):
                cols = {aid: (PredictionColumn(aid, "KEEP", (), (), (0,) * 5,
                    PlanningNode("outer", int(aid), 0., .9, 0., 0)),) for aid in scope.replan_ids}
                yield CandidateCatalog(cols, {aid: (0, aid) for aid in cols}, {}, (), scope=scope)
    class Scheduler:
        def __init__(self): self.calls = []
        def plan(self, snapshot, catalog, context, budget, *, incumbent):
            self.calls.append((id(budget), tuple(context.scope.replan_ids)))
            context.background_blockers.append(dict(source="0", target="1", snapshot_id=s.version,
                reason="RESOURCE_CONFLICT", resource=("outer", 1)))
            return PlanningResult(tuple(rows[0] for rows in catalog.columns.values()), (), None, "UNKNOWN", (0,) * 5)
    scheduler = Scheduler()
    budget = PlanningBudget(candidate_seconds=6., solver_seconds=8., candidate_active_seconds=0.)
    planner = LocalPlanner(Provider(), scheduler, progressive_v2=True, search_policy=SearchPolicy(*switches))
    planner.plan(s, scope, frame, budget)
    assert len(scheduler.calls) == 2 and len({r[0] for r in scheduler.calls}) == 1
    assert planner.telemetry["scope_expansions"] == (0 if switches[2] else 1)
    assert scheduler.calls[-1][1] == (("0",) if switches[2] else ("0", "1"))
    assert budget.solver_seconds == 8. and budget.candidate_seconds == 6.


def test_future_move_never_enters_execution_prefix_or_authorizes_release():
    world = RotatingSlotWorld()
    world.seed_hold("o", "outer", 1, .9)
    world.seed_hold("m", "middle", 1, .9)
    s = replace(world.build_planning_snapshot(), planning_timing=SimpleNamespace(effective_time=20., snapshot_time=0.))
    context = EvaluationContext(s)
    provider = CandidateProvider()
    outer_plan = generate_transfer("o", "outer", 1, 1, 20., 0, durations=(40.,)).plan
    middle_plan = generate_transfer("m", "middle", 1, 0, 40., 0, durations=(40.,)).plan
    outer = provider._column(s, s.aircraft["o"], (outer_plan,), "MOVE", context)
    middle = provider._column(s, s.aircraft["m"], (middle_plan,), "MOVE", context)
    assert context.prefix((outer, middle)) == (outer_plan,)
    middle_source = next(r for r in middle.resources if r.resource == ("middle", 1))
    assert middle_source.end > 40.
    assert not context.check_commit_prefix((outer, middle)).accepted


@pytest.mark.parametrize("switches", COMBINATIONS)
def test_existing_active_and_soft_plans_stay_fixed_for_future_reference(switches):
    world = RotatingSlotWorld()
    world.seed_hold("a", "outer", 0, .9)
    first = generate_transfer("a", "outer", 0, 0, 0., durations=(40.,)).plan
    assert world.submit((first,)).accepted
    second = generate_transfer("a", "middle", 0, 0, first.end_time + 5., source_version=1, durations=(40.,)).plan
    assert world.submit((second,)).accepted
    s = world.build_planning_snapshot()
    before = world.build_planning_snapshot()
    scope, info = RuleSelector(variant=REFERENCE_VARIANT, search_policy=SearchPolicy(*switches)).select(
        s, SimpleNamespace(identity="f"), history(), method="FULL_H")
    assert scope.replan_ids == ("a",)
    access = info["adjustment_access"]["a"]
    assert not access["adjustable"] and access["node_time"] == second.end_time
    context = EvaluationContext(s)
    column = PredictionColumn("a", "MOVE", (second,), (), (0,) * 5, None)
    assert context.prefix((column,)) == ()
    assert context.all_plans("a", ()) == (first, second)
    assert world.build_planning_snapshot() == before


def test_callback_records_future_selection_separately_from_committed_prefix(monkeypatch):
    class Provider(CandidateProvider):
        def iter_progressive_columns(self, snapshot, budget, context, *, scope=None, early_publish=False):
            aircraft = snapshot.aircraft["a"]
            plan = generate_transfer("a", "outer", 0, 0, 20., durations=(40.,)).plan
            columns = {"a": [self._column(snapshot, aircraft, (), kind, context) for kind in ("KEEP", "WAIT")]}
            columns["a"].append(self._column(snapshot, aircraft, (plan,), "MOVE", context))
            catalog = self._progressive_catalog(snapshot, budget, context, columns,
                {"a": columns["a"][0].reference}, [aircraft])
            yield catalog
            yield catalog
    def choose_future(self, snapshot, catalog, context, budget, *, incumbent=None):
        selected = (catalog.columns["a"][-1],)
        evaluation = context.check_commit_prefix(selected)
        assert evaluation.accepted and context.prefix(selected) == ()
        return PlanningResult(selected, (), evaluation, "FIXED_FUTURE_TEST",
            catalog.downstream.key(selected), certified=True)
    monkeypatch.setattr(RollingHorizonCPSATScheduler, "plan", choose_future)
    world = RotatingSlotWorld()
    world.seed_hold("a", "outer", 0, .9)
    cb = NormalSchedulerCallback("rh_cp_sat", candidate_provider=Provider(planning_profile="downstream_service_v8"),
        planning_profile="downstream_service_v8", selective_method="FULL_H", selective_rule_variant=REFERENCE_VARIANT)
    result = cb(world._scheduler_port)
    row = cb.records[-1]["selective_replanning"]
    assert result.certified and not result.prefix and not world.active
    assert row["search_policy"] == SearchPolicy().to_dict()
    assert row["adjustment_access"]["a"]["adjustable"]
    assert row["round_execution"]["a"] == dict(evidence_phase="CALLBACK_RETURN", kind="MOVE", first_start_time=20.,
        executable_prefix=False, prefix_committed=False, certified=True)


def test_reference_frame_failure_still_records_access_and_policy(monkeypatch):
    from terminal_marl.core.rotating_slots.selective_replanning.frame import FrameIncomplete
    def incomplete(*args): raise FrameIncomplete("FIXED_TEST_UNKNOWN")
    monkeypatch.setattr("terminal_marl.core.rotating_slots.selective_replanning.build_global_frame", incomplete)
    world = RotatingSlotWorld()
    world.seed_hold("a", "outer", 0, .9)
    cb = callback()
    result = cb(world._scheduler_port)
    row = cb.records[-1]["selective_replanning"]
    assert result.status == "GLOBAL_FRAME_INCOMPLETE" and not result.certified
    assert row["search_policy"] == SearchPolicy().to_dict()
    assert set(row["adjustment_access"]) == {"a"} and row["round_execution"] == {}
