"""Normal-stage handoffs and budgets; recorded R10 scalars are not a checkpoint replay."""
from types import SimpleNamespace

import pytest

from terminal_marl.core.rotating_slots.models import ValidationResult
from terminal_marl.core.rotating_slots.planning_latency import LatencyAwareScheduler, PlanningTiming


class Durations:
    def __init__(self, *durations):
        self.durations=iter(durations)
        self.value=0.
        self.open=False

    def __call__(self):
        if self.open: self.value+=next(self.durations,0.)
        self.open=not self.open
        return self.value


class Port:
    def __init__(self, now):
        self.time=now
        self.calls=[]
        self.events=[]
        self.guard=ValidationResult(False,"REJECTED","TRAFFIC_REVALIDATION_REQUIRED")
        self.evaluation=ValidationResult(True,"REVALIDATING",details=dict(attempt=1,pairs_checked=1))
        self.completion=ValidationResult(True,"REVALIDATED")

    def latency_packet_guard(self, token):
        self.calls.append(("guard",token))
        return self.guard

    def revalidate_latency_submission(self, token, max_work_s=None):
        self.calls.append(("revalidate",max_work_s))
        return self.evaluation

    def complete_latency_revalidation(self, token, attempt, promote=False):
        self.calls.append(("complete",attempt,promote))
        return self.completion

    def mark_latency_submission_ready(self, token, handoff_delay=0.):
        self.calls.append(("ready",handoff_delay))
        return ValidationResult(True,"GUARDED",details=dict(ready_at=self.time+handoff_delay))

    def discard_latency_submission(self, token, reason=None):
        self.calls.append(("discard",token))

    def planning_latency_event(self, kind, **details):
        self.events.append((kind,details))


def pending(stage, now=10., timing=None, durations=()):
    timing=timing or PlanningTiming(0.,20.,19.)
    scheduler=LatencyAwareScheduler(SimpleNamespace(),clock=Durations(*durations))
    scheduler.pending=dict(timing=timing,stage=stage,due=now,token="packet",reason=None,
        record_index=0,online_service_s=0.,proposal=(),evidence=[])
    scheduler.records=[dict(status=stage,prepare_work_s=.8,prepare_work_cpu_s=.8,
        dependency_guard_work_s=0.,dependency_guard_work_cpu_s=0.,revalidation_count=0,
        revalidation_work_s=0.,revalidation_work_cpu_s=0.,revalidations=[])]
    return scheduler,Port(now)


def deliver(scheduler, port):
    port.time=scheduler.pending["due"]
    scheduler.planning_event(port)


def test_preparing_dirty_guard_starts_revalidation_without_an_intermediate_ready_delivery():
    scheduler,port=pending("PREPARING",durations=(.2,.3,.1))
    timing=scheduler.pending["timing"]
    scheduler._handoff_service=.4
    scheduler.planning_event(port)
    record=scheduler.records[0]
    row=record["revalidations"][0]
    assert scheduler.pending["stage"]=="REVALIDATING"
    assert [call[0] for call in port.calls]==["guard","revalidate"]
    assert port.calls[1][1]==pytest.approx(8.4)
    assert row["started_time"]==pytest.approx(10.6)
    assert row["due"]==pytest.approx(10.9)
    assert row["queued_service_s"]==pytest.approx(.6)
    assert record["ready_time"] is None and "previous_ready_time" not in row
    assert record["prepare_work_s"]==pytest.approx(1.)
    assert scheduler._handoff_service==0.
    # Certification completion and readiness promotion still inspect fresh state.
    deliver(scheduler,port)
    assert scheduler.pending["stage"]=="READY_SERVICE"
    deliver(scheduler,port)
    assert [call for call in port.calls if call[0]=="complete"]==[
        ("complete",1,False),("complete",1,True)]
    assert len([call for call in port.calls if call[0]=="ready"])==1
    assert scheduler.pending["stage"]=="PREPARED"
    assert scheduler.pending["due"]==timing.effective_time==20.
    assert scheduler.pending["timing"].ready_deadline==19.


def test_prepared_dependency_budget_charges_public_guard_and_handoff_once():
    scheduler,port=pending("PREPARED",now=15.,durations=(.1,.3))
    scheduler.records[0]["ready_time"]=14.
    scheduler.online_service(15.,1.2,1.,"admission")
    scheduler._handoff_service=.2
    scheduler.inspect_dependencies(port)
    record=scheduler.records[0]
    row=record["revalidations"][0]
    budget=record["last_revalidation_budget"]
    assert budget==dict(dispatch_time=15.,queued_service_s=pytest.approx(1.5),
        planned_started_time=pytest.approx(16.5),available_work_s=pytest.approx(2.5))
    assert all(row[key]==value for key,value in budget.items())
    assert row["previous_ready_time"]==14. and record["ready_time"] is None
    assert port.calls[-1]==("revalidate",pytest.approx(2.5))
    assert row["due"]==pytest.approx(16.8)
    assert scheduler._service_until==15. and scheduler._handoff_service==0.
    assert sum(item["wall_s"] for item in scheduler.online_services)==pytest.approx(1.3)
    assert scheduler.pending["online_service_s"]==pytest.approx(1.3)


def test_recorded_r10_queue_scalars_still_reject_without_changing_deadlines():
    # This exercises the recorded timing constraint, not recovery of the R10 world.
    timing=PlanningTiming(167.00476574490312,187.00476574490312,186.00476574490312)
    scheduler,port=pending("PREPARED",now=185.,timing=timing,durations=(0.,8.743302896618843e-5))
    scheduler.records[0]["ready_time"]=183.9886362295365
    queued=186.03481734974775-185.
    scheduler.online_service(185.,queued,queued,"recorded_public_queue")
    port.evaluation=ValidationResult(False,"UNKNOWN","REVALIDATION_WORK_DEADLINE",
        dict(pairs_checked=0,affected_aircraft_ids=["req_000008"]))
    scheduler.inspect_dependencies(port)
    record=scheduler.records[0]
    row=record["revalidations"][0]
    assert port.calls[-1]==("revalidate",0.)
    assert row["planned_started_time"]==pytest.approx(186.03481734974775)
    assert row["planned_started_time"]>timing.ready_deadline
    assert row["available_work_s"]==0. and row["pairs_checked"]==0
    assert row["previous_ready_time"]==183.9886362295365
    assert scheduler.pending["timing"]==timing and record["ready_time"] is None
    deliver(scheduler,port)
    assert scheduler.pending is None and record["status"]=="REJECTED"
    assert record["reason"]=="READY_DEADLINE_MISSED" and row["status"]=="LATE"
    assert not any(call[0] in ("ready","complete") for call in port.calls)


def test_changed_state_at_completion_cannot_reuse_accepted_revalidation():
    scheduler,port=pending("PREPARING",durations=(.2,.3,.1,.2))
    scheduler.planning_event(port)
    port.completion=ValidationResult(False,"REJECTED","TRAFFIC_REVALIDATION_REQUIRED")
    deliver(scheduler,port)
    assert scheduler.pending["stage"]=="READY_SERVICE"
    deliver(scheduler,port)
    assert scheduler.records[0]["revalidations"][0]["status"]=="SUPERSEDED"
    assert scheduler.records[0]["revalidation_count"]==2
    assert scheduler.pending["stage"]=="REVALIDATING"
    assert len([call for call in port.calls if call[0]=="revalidate"])==2
    assert not any(call[0]=="ready" for call in port.calls)
    assert scheduler.records[0]["ready_time"] is None


def test_already_late_dependency_change_records_budget_before_existing_rejection():
    scheduler,port=pending("PREPARED",now=19.5)
    scheduler._service_until=20.
    scheduler.inspect_dependencies(port)
    record=scheduler.records[0]
    assert record["last_revalidation_budget"]==dict(dispatch_time=19.5,queued_service_s=.5,
        planned_started_time=20.,available_work_s=0.)
    assert scheduler.pending is None and record["reason"]=="TRAFFIC_CHANGED_AFTER_DEADLINE"
    assert record["revalidation_count"]==0
    assert not any(call[0]=="revalidate" for call in port.calls)
