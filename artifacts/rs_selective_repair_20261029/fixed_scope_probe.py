"""Explicit saved-state diagnostic only; never part of the frozen source manifest.

The caller must use the frozen repair runner's shared run_bounded_job(...,
cap_s=180). Each command performs one A/B probe or two content-only rebuilds for
one root. No matrix, physical advance, publication, retry or production hash fix.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json
import math
import numbers
import os
from pathlib import Path
import re
import struct
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as repair

SCRIPT=Path(__file__).resolve()
FROZEN_COMMIT="895bc76"
ORIGINAL_SCOPE=("initial_inner_05","initial_inner_06","initial_inner_07","initial_inner_08",
    "initial_inner_09","initial_middle_00","initial_middle_01","initial_middle_10")
EXPECTED_FAIR=("initial_inner_05","initial_inner_06")
FIXED_SCOPES={
    "A":tuple(sorted((set(ORIGINAL_SCOPE)-{"initial_middle_01"})|{"initial_outer_00"})),
    "B":tuple(sorted((set(ORIGINAL_SCOPE)-{"initial_middle_01","initial_inner_07"})|
                     {"initial_outer_00","initial_outer_01"})),
}
FORBIDDEN={"initial_inner_02","initial_inner_04"}
ADDRESS_RE=re.compile(r"\b(?:object|Section) at 0x[0-9a-fA-F]+")


def normalized(value, *, omit_root_fields=()):
    """Lossless JSON evidence for supported immutable snapshot values.

    Only explicitly named fields of this root dataclass are omitted. Nested
    aircraft.version/source_version and every numerical/resource value remain.
    Mapping keys and tuple/list types are preserved, including nonstring keys.
    Unsupported objects fail rather than falling back to an address-bearing str.
    """
    def visit(item, root=False):
        kind=type(item).__module__+"."+type(item).__qualname__
        if item is None or isinstance(item,(bool,str)): return item
        if type(item).__module__.startswith("numpy") and getattr(item,"shape",None)==() and hasattr(item,"tobytes"):
            return {"numpy_scalar_type":kind,"dtype":str(item.dtype),"bytes_hex":item.tobytes().hex()}
        if isinstance(item,numbers.Integral): return {"integer":str(int(item))}
        if isinstance(item,numbers.Real):
            number=float(item)
            return {"float64_bits":struct.pack(">d",number).hex()}
        if isinstance(item,bytes): return {"bytes_hex":item.hex()}
        if isinstance(item,Path): return {"path_type":kind,"value":str(item)}
        if is_dataclass(item) and not isinstance(item,type):
            return {"dataclass_type":kind,"fields":{f.name:visit(getattr(item,f.name))
                for f in fields(item) if not (root and f.name in omit_root_fields)}}
        if isinstance(item,Mapping):
            pairs=[[visit(key),visit(val)] for key,val in item.items()]
            pairs.sort(key=lambda pair:canonical_bytes(pair[0]))
            return {"mapping_type":kind,"entries":pairs}
        if isinstance(item,(set,frozenset)):
            values=[visit(val) for val in item]; values.sort(key=canonical_bytes)
            return {"set_type":kind,"items":values}
        if isinstance(item,(tuple,list)) or isinstance(item,Sequence):
            return {"sequence_type":kind,"items":[visit(val) for val in item]}
        if type(item).__module__.startswith("numpy") and hasattr(item,"tolist"):
            return {"array_type":kind,"dtype":str(item.dtype),"shape":list(item.shape),"values":visit(item.tolist())}
        raise TypeError("unsupported canonical evidence value: "+kind)
    return visit(value,True)


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf8")


def content_evidence(projected, frame):
    payload=dict(projected_physical=normalized(projected,omit_root_fields=("version",)),
        global_evaluation_frame=normalized(frame,omit_root_fields=("snapshot_id",)))
    raw=dict(projected_repr=repr(projected),frame_repr=repr(frame))
    return dict(schema="diagnostic_content_comparison_v1",payload=payload,
        canonical_sha256=sha256(canonical_bytes(payload)).hexdigest(),
        component_sha256={key:sha256(canonical_bytes(value)).hexdigest() for key,value in payload.items()},
        omitted_identity_fields=["projected.version","global_evaluation_frame.snapshot_id"],
        preserved_version_fields="ALL_NESTED_PHYSICAL_AIRCRAFT_AND_TRAJECTORY_VERSIONS",
        raw_repr=raw,raw_repr_has_default_memory_address={key:bool(ADDRESS_RE.search(value)) for key,value in raw.items()},
        production_projected_version=projected.version,production_frame_id=frame.identity,
        authorization_scope="COMPARISON_EVIDENCE_ONLY_NEVER_PRODUCTION_IDENTITY_OR_CACHE_KEY")


def validate_fixed_scope(selected_ids, *, eligible_ids, hard_ids, fair_ids, relations):
    selected=set(selected_ids); eligible=set(eligible_ids)
    base.require(len(selected_ids)==len(selected)==8,"manual diagnostic scope must contain exactly eight distinct IDs")
    base.require(not (selected & FORBIDDEN),"executing inner02/04 cannot be selected")
    base.require(selected<=eligible,"manual scope contains ineligible aircraft")
    base.require(set(hard_ids)<=selected,"manual scope removes hard mandatory aircraft")
    base.require(set(fair_ids)<=selected,"manual scope removes reserved fairness aircraft")
    missing=sorted({r.target for r in relations if r.definite and r.source in selected and r.target in eligible-selected})
    base.require(not missing,"manual scope violates observed definite dependency closure: "+repr(missing))
    return dict(eligible=True,hard_preserved=True,fair_preserved=True,definite_dependency_closure=True,
        hard_ids=sorted(hard_ids),fair_ids=sorted(fair_ids),eligible_ids=sorted(eligible),
        closure_basis="SAME_FROZEN_BUILD_RELATIONS_DEFINITE_EDGES_TARGETING_ELIGIBLE_AIRCRAFT",
        physical_feasibility_claim=False)


@contextmanager
def fixed_selector_adapter(RuleSelector, variant, *, expected_snapshot_id, captures):
    """Process-local, single-callback diagnostic override; always restore class."""
    original=RuleSelector.select
    wanted=FIXED_SCOPES[variant]
    def fixed(selector,snapshot,frame,history,relations=(),*,method="RULE_LOCAL_K8",now=None,
              references=None,opportunity_seed_offset=0):
        base.require(selector.variant==repair.RULE and method=="RULE_LOCAL_K8" and opportunity_seed_offset==0,
            "manual adapter is limited to the frozen v2 K8 offset0 callback")
        base.require(snapshot.version==expected_snapshot_id and snapshot.time==60.,"unexpected callback snapshot")
        scope,telemetry=original(selector,snapshot,frame,history,relations,method=method,now=now,
            references=references,opportunity_seed_offset=opportunity_seed_offset)
        base.require(set(scope.replan_ids)==set(ORIGINAL_SCOPE),"original offset0 initial selected set differs from preregistered saved observation")
        base.require(not scope.mandatory_ids,"saved diagnostic hard set differs from the preregistered empty set")
        fair=telemetry.get("fairness_selected_ids")
        base.require(fair is not None and set(fair)==set(EXPECTED_FAIR),"saved diagnostic fairness reservation differs")
        checks=validate_fixed_scope(wanted,eligible_ids=scope.eligible_ids,hard_ids=scope.mandatory_ids,
            fair_ids=fair,relations=relations)
        old_reasons=dict(scope.selection_reasons)
        scope=replace(scope,replan_ids=wanted,actual_k=len(wanted),
            fixed_background_ids=tuple(sorted(set(snapshot.aircraft)-set(wanted))),
            selection_reasons=tuple((aid,tuple(old_reasons.get(aid,()))+("EXPLICIT_MANUAL_SCOPE_OVERRIDE_DIAGNOSTIC_ONLY",)) for aid in wanted))
        scope.validate(snapshot)
        observation=dict(explicit_manual_scope_override="DIAGNOSTIC_ONLY",variant=variant,
            original_initial_selected_ids=list(ORIGINAL_SCOPE),manual_initial_selected_ids=list(wanted),
            checks=checks,relations=[vars(row) for row in relations],original_selector_measurements=telemetry)
        captures.append(observation)
        return scope,dict(telemetry,explicit_manual_scope_override="DIAGNOSTIC_ONLY",manual_scope_variant=variant,
            manual_original_selected_ids=list(ORIGINAL_SCOPE),manual_initial_selected_ids=list(wanted))
    RuleSelector.select=fixed
    try: yield
    finally: RuleSelector.select=original


def validate_parent(args):
    args.output=repair.validated_output(args.output)
    with repair.ledger_transaction(args.output,write=False) as ledger:
        job=ledger["jobs"].get(args.job_id,{})
        base.require(job.get("status")=="running" and job.get("parent_pid")==os.getppid() and job.get("cap_s")==180.,
            "explicit active shared-ledger parent watchdog with cap180 required")


def checkpoint_path(args):
    path=args.followup_output/"arms"/args.root/"baseline20"/"evaluation"/args.root/"RULE_LOCAL_K8"/"checkpoint_0040.json"
    base.require(path.is_file(),"fixed baseline20 checkpoint0040 missing; no substitute")
    raw=base.load(path)
    base.require(raw["payload"]["time"]==40. and raw["payload"]["identity"]["manifest_sha256"]==base.RAW_HASHES[args.root][0],
        "wrong baseline checkpoint root/time")
    old=base.load(args.followup_output/"arms"/args.root/"baseline20"/"resolved_run_binding.json")
    base.require(old["treatment"]["selector_variant"]=="baseline" and old["treatment"]["lead_seconds"]==20.,"baseline20 binding differs")
    return path


def setup(args):
    validate_parent(args)
    source=base.load(args.source_freeze)
    base.require(source["commit"].startswith(FROZEN_COMMIT),"diagnostic requires the frozen895bc76 source manifest")
    base.verify_sources(source)
    d=base.dependencies(PROJECT);d.launch_threads={key:os.environ.get(key) for key in base.THREADS}
    repair.set_cpu(args,d)
    # Existing helper verifies original lineage and shared frozen source hash.
    bound=repair.binding(args,d)
    return d,bound,checkpoint_path(args)


def rebuild(args,d,checkpoint,directory):
    from terminal_marl.core.rotating_slots.selective_replanning import build_global_frame
    from terminal_marl.core.rotating_slots.stage2_planning import CandidateProvider,EvaluationContext,PlanningBudget,QUALITY_BUDGET_PROFILE
    from terminal_marl.core.rotating_slots.performance.parallel import _clear_worker_accelerators
    reset_started=time.perf_counter();reset=_clear_worker_accelerators();reset_s=time.perf_counter()-reset_started
    restore_started=time.perf_counter();world=base.restore(checkpoint,d,directory);restore_s=time.perf_counter()-restore_started
    physical_before=base.digest(world.to_dict());source=world.build_planning_snapshot()
    timing=d.PlanningTiming.from_lead(source.time,20.,1.)
    method_started=time.perf_counter()
    projected=d.project_visible_snapshot(source,timing.effective_time,timing=timing)
    projection_s=time.perf_counter()-method_started
    frame_started=time.perf_counter()
    frame=build_global_frame(projected,CandidateProvider(planning_profile="downstream_service_v8"),
        EvaluationContext(projected,budget=PlanningBudget.for_profile(QUALITY_BUDGET_PROFILE)))
    frame_s=time.perf_counter()-frame_started
    return world,source,timing,projected,frame,method_started,physical_before,dict(cache_reset=reset,
        cache_reset_s=reset_s,restore_s=restore_s,projection_s=projection_s,frame_s=frame_s)


def probe(args):
    base.require(args.root=="s3r24" and args.variant in FIXED_SCOPES,"only preregistered high-traffic A/B scopes authorized")
    started=time.perf_counter();d,bound,checkpoint=setup(args)
    directory=args.output/"diagnostic_scope_alternatives"/args.variant
    base.require(not directory.exists(),"manual scope already attempted; no retry")
    world,source,timing,projected,frame,method_start,physical_before,costs=rebuild(args,d,checkpoint,directory)
    from terminal_marl.core.rotating_slots.selective_replanning import RuleSelector
    callback=base.scheduler(d,"RULE_LOCAL_K8",frame,selector_variant=repair.RULE,opportunity_seed_offset=0)
    callback.initialize_imported_history(source,events=world.events)
    port=d.FuturePort(projected,source,method_start+19.,search_wall_deadline=method_start+15.,
        planning_wall_origin=method_start,search_handoff_reserve_seconds=4.)
    captures=[];callback_started=time.perf_counter()
    with fixed_selector_adapter(RuleSelector,args.variant,expected_snapshot_id=projected.version,captures=captures): callback(port)
    costs["callback_s"]=time.perf_counter()-callback_started
    preparation=base.counterfactual_preparation(world,port.proposal,timing,ready_wall_deadline=method_start+19.)
    costs["observed_preparation_chain_s"]=time.perf_counter()-method_start
    physical_after=base.digest(world.to_dict())
    base.require(physical_before==physical_after,"manual counterfactual changed physical state or commitments")
    scopes=repair.snapshot_scope_observations(callback.records,FIXED_SCOPES[args.variant])
    base.require(scopes["initial_scope_matches_preselection"] is True,"manual adapter scope not observed as first local solve")
    evidence_started=time.perf_counter();content=content_evidence(projected,frame)
    costs["content_evidence_serialization_s"]=time.perf_counter()-evidence_started
    result=dict(schema="explicit_fixed_scope_probe_v1",status="COMPLETE",root=args.root,variant=args.variant,
        explicit_manual_scope_override="DIAGNOSTIC_ONLY",formal_selection_rule_modified=False,
        source_commit=bound["source_commit"],source_hash=bound["source_hash"],source_freeze_path=str(args.source_freeze),
        source_freeze_sha256=base.file_hash(args.source_freeze),artifact_script_path=str(SCRIPT),artifact_script_sha256=base.file_hash(SCRIPT),
        source_binding=bound,fixed_checkpoint=str(checkpoint),checkpoint_sha256=base.file_hash(checkpoint),
        snapshot_time=source.time,effective_time=timing.effective_time,production_frame_id=frame.identity,
        production_projected_version=projected.version,content_comparison=content,
        physical_before_sha256=physical_before,physical_after_sha256=physical_after,physical_source_unchanged=True,
        physical_execution=False,actual_publication=False,cold_normal_cache=True,preparation=preparation,
        original_offset0_initial_selected_ids=list(ORIGINAL_SCOPE),initial_selected_ids=list(FIXED_SCOPES[args.variant]),
        returned_selected_ids=scopes["returned_selected_ids"],attempted_union=scopes["attempted_union"],scope_observations=scopes,
        manual_selection_observations=captures,scheduler_records=callback.records,
        selection_records=base.selection_envelopes(callback.records,"manual:"+args.variant,bound["origin_input_binding"],args.root),
        costs=costs,whole_job_before_output_s=time.perf_counter()-started,
        fixed_deadlines=dict(search_seconds_from_method_start=15.,ready_seconds_from_method_start=19.,
            includes="PROJECTION_FRAME_CALLBACK_SELECTION_PLANNING_PREPARATION"),
        actual_cpu=sorted(os.sched_getaffinity(0)),parent_job_id=args.job_id,shared_budget_ledger=str(args.output/"repair_budget_ledger.json"),
        unobserved_online_stages=["AFFECTED_TRAFFIC_REVALIDATION","ONLINE_SERVICE_DELAY","ACTUAL_DELIVERY"],
        no_automatic_retry=True,human_replay_status="NOT_APPLICABLE_NO_EXECUTION")
    d.atomic_json(directory/"result.json",result)
    return dict(status="COMPLETE",result_path=str(directory/"result.json"))


def canonical_check(args):
    d,bound,checkpoint=setup(args)
    directory=args.output/"identity_content_checks"/args.root
    base.require(not directory.exists(),"content-only check already attempted; no retry")
    rows=[]
    for repeat in range(2):
        world,source,timing,projected,frame,method_start,physical_before,costs=rebuild(args,d,checkpoint,directory/f"rebuild{repeat}")
        evidence=content_evidence(projected,frame)
        physical_after=base.digest(world.to_dict())
        base.require(physical_before==physical_after,"content rebuild changed physical world")
        row=dict(repeat=repeat,content_comparison=evidence,physical_before_sha256=physical_before,
            physical_after_sha256=physical_after,physical_source_unchanged=True,costs=costs,
            physical_execution=False,actual_publication=False,normal_callback_invoked=False,solver_invoked=False)
        d.atomic_json(directory/f"rebuild{repeat}"/"content.json",row);rows.append(row)
    result=dict(schema="frozen_identity_content_rebuild_v1",status="COMPLETE",root=args.root,
        checkpoint=str(checkpoint),checkpoint_sha256=base.file_hash(checkpoint),source_commit=bound["source_commit"],
        artifact_script_sha256=base.file_hash(SCRIPT),source_freeze_sha256=base.file_hash(args.source_freeze),
        canonical_content_equal=rows[0]["content_comparison"]["canonical_sha256"]==rows[1]["content_comparison"]["canonical_sha256"],
        production_projected_version_equal=rows[0]["content_comparison"]["production_projected_version"]==rows[1]["content_comparison"]["production_projected_version"],
        production_frame_id_equal=rows[0]["content_comparison"]["production_frame_id"]==rows[1]["content_comparison"]["production_frame_id"],
        rows=rows,physical_execution=False,actual_publication=False,normal_callback_invoked=False,solver_invoked=False,
        parent_job_id=args.job_id,shared_budget_ledger=str(args.output/"repair_budget_ledger.json"),
        authorization_scope="CONTENT_COMPARISON_ONLY_PRODUCTION_HASHES_UNMODIFIED")
    d.atomic_json(directory/"result.json",result)
    return dict(status="COMPLETE",result_path=str(directory/"result.json"),canonical_content_equal=result["canonical_content_equal"])


def self_test():
    import unittest
    from dataclasses import dataclass
    from types import SimpleNamespace
    class Checks(unittest.TestCase):
        def test_preregistered_scopes_and_guards(self):
            eligible=set(ORIGINAL_SCOPE)|{"initial_outer_00","initial_outer_01"}
            for selected in FIXED_SCOPES.values():
                checks=validate_fixed_scope(selected,eligible_ids=eligible,hard_ids=[],fair_ids=EXPECTED_FAIR,relations=[])
                self.assertTrue(checks["fair_preserved"])
            with self.assertRaisesRegex(ValueError,"ineligible"):
                validate_fixed_scope(FIXED_SCOPES["A"],eligible_ids=ORIGINAL_SCOPE,hard_ids=[],fair_ids=EXPECTED_FAIR,relations=[])
            with self.assertRaisesRegex(ValueError,"hard mandatory"):
                validate_fixed_scope(FIXED_SCOPES["B"],eligible_ids=eligible,hard_ids=["initial_inner_07"],fair_ids=EXPECTED_FAIR,relations=[])
            missing_fair=tuple((set(FIXED_SCOPES["A"])-{"initial_inner_05"})|{"initial_outer_01"})
            with self.assertRaisesRegex(ValueError,"fairness"):
                validate_fixed_scope(missing_fair,eligible_ids=eligible,hard_ids=[],fair_ids=EXPECTED_FAIR,relations=[])
            executing=tuple((set(FIXED_SCOPES["A"])-{"initial_middle_10"})|{"initial_inner_02"})
            with self.assertRaisesRegex(ValueError,"executing inner02/04"):
                validate_fixed_scope(executing,eligible_ids=eligible|FORBIDDEN,hard_ids=[],fair_ids=EXPECTED_FAIR,relations=[])
            edge=SimpleNamespace(source="initial_middle_00",target="initial_middle_01",definite=True)
            with self.assertRaisesRegex(ValueError,"dependency closure"):
                validate_fixed_scope(FIXED_SCOPES["A"],eligible_ids=eligible,hard_ids=[],fair_ids=EXPECTED_FAIR,relations=[edge])
        def test_canonical_evidence_preserves_nested_versions_and_values(self):
            @dataclass(frozen=True)
            class Snapshot:
                version:str
                aircraft:object
            class Section(Mapping):
                def __init__(self,values):self.values=values
                def __getitem__(self,key):return self.values[key]
                def __iter__(self):return iter(self.values)
                def __len__(self):return len(self.values)
            values={"a":{"version":2,"soc":.7,"resource":("inner",9),"end":math.inf}}
            left=Snapshot("address_hash1",Section(values))
            right=replace(left,version="address_hash2",aircraft=Section(values))
            self.assertNotEqual(repr(left.aircraft),repr(right.aircraft))
            one=normalized(left,omit_root_fields=("version",));two=normalized(right,omit_root_fields=("version",))
            self.assertEqual(canonical_bytes(one),canonical_bytes(two))
            changed=replace(left,aircraft=Section({"a":{"version":3,"soc":.7,"resource":("inner",9),"end":math.inf}}))
            self.assertNotEqual(canonical_bytes(one),canonical_bytes(normalized(changed,omit_root_fields=("version",))))
            json.dumps(one,allow_nan=False)
        def test_shared_budget_math_and_result_scope_schema(self):
            data=dict(settled_child_wall_s=100.,jobs={"active":dict(status="running",started_epoch=10.)})
            self.assertEqual(repair.active_elapsed(data,20.),110.)
            facts=repair.snapshot_scope_observations([dict(selective_replanning=dict(
                calls=[dict(scope={"replan_ids":FIXED_SCOPES["A"]})],scope={"replan_ids":FIXED_SCOPES["A"]},
                attempted_replan_ids=list(FIXED_SCOPES["A"])))],FIXED_SCOPES["A"])
            self.assertTrue(facts["initial_scope_matches_preselection"])
            self.assertEqual(len(facts["initial_selected_ids"]),8)
        def test_selector_adapter_restores_on_validation_error(self):
            class FakeSelector:
                variant=repair.RULE
                def select(self,*args,**kwargs): raise AssertionError("original should not run for wrong snapshot")
            original=FakeSelector.select
            with self.assertRaisesRegex(ValueError,"unexpected callback snapshot"):
                with fixed_selector_adapter(FakeSelector,"A",expected_snapshot_id="expected",captures=[]):
                    FakeSelector().select(SimpleNamespace(version="wrong",time=60.),None,None)
            self.assertIs(FakeSelector.select,original)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    base.require(result.wasSuccessful(),"artifact self-check failed")
    return dict(status="SELF_CHECK_PASS",tests=result.testsRun)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("probe","canonical","self-test"))
    parser.add_argument("--output",type=Path,default=Path.home()/"training_outputs/terminal"/repair.OUTPUT_NAME)
    parser.add_argument("--followup-output",type=Path,default=Path.home()/"training_outputs/terminal/rs_selective_followup_20261028")
    parser.add_argument("--origin-output",type=Path,default=Path.home()/"training_outputs/terminal/rs_selective_reoptimization_a_20261027")
    parser.add_argument("--source-freeze",type=Path);parser.add_argument("--root",choices=tuple(base.ROOTS))
    parser.add_argument("--variant",choices=tuple(FIXED_SCOPES));parser.add_argument("--cpu",type=int)
    parser.add_argument("--job-id")
    args=parser.parse_args(argv)
    if args.action=="self-test": result=self_test()
    else:
        base.require(args.source_freeze and args.root and args.cpu is not None and args.job_id,"explicit freeze/root/cpu/parent job ID required")
        args.stage="snapshot"
        result=probe(args) if args.action=="probe" else canonical_check(args)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
