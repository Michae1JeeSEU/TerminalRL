"""Read frozen-891 receipts only; derive callback facts without running a model/world."""
import argparse
from collections import Counter,defaultdict
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path

BASE=Path("D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027")
COMMIT="891adc1a9209944a9731da23f65788332f8756fe"
ARMS=("heuristic_reduced","learned_reduced")
LAYERS=("F","negative_L","risk","Ehat","tau")
PHASES=("cp_solve","trajectory_validation","prefix_validation","candidate_generation_and_single_vehicle","stage3_features","stage3_inference")
EVENTS={"FIRST_NONEMPTY_CERTIFIED","STRICT_CERTIFIED_IMPROVEMENT","CATALOG_RESCORE","CATALOG_BEST_UPDATE"}


def load(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def require(condition,message):
    if not condition:raise ValueError(message)


def receipt(row):
    if row is None:return None
    return {**{k:row.get(k) for k in ("event","source","wall_elapsed_s","certificate_wall_elapsed_s","catalog_scope","context_scope","objective")},
        "prefix_count":len(row["prefix_hashes"]) if "prefix_hashes" in row else None}


def scoped_difference(before,after,objective_id):
    if not before or not after:return dict(status="MISSING_RECEIPT")
    if not before.get("catalog_scope") or not after.get("catalog_scope"):
        return dict(status="MISSING_CATALOG_SCOPE")
    if not before.get("context_scope") or not after.get("context_scope"):
        return dict(status="MISSING_CONTEXT_SCOPE")
    if (before.get("catalog_scope"),before.get("context_scope"))!=(after.get("catalog_scope"),after.get("context_scope")):
        return dict(status="DIFFERENT_CATALOG_OR_CONTEXT")
    a,b=before.get("objective"),after.get("objective")
    if not objective_id or not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b) or not all(finite(x) for x in a+b):
        return dict(status="MISSING_OBJECTIVE_SEMANTICS")
    names=LAYERS if objective_id=="downstream_service_v8_f_L_risk_Ehat_tau" and len(a)==5 else tuple("component_"+str(i) for i in range(len(a)))
    changed=[dict(layer=names[i],before=x,after=y) for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    return dict(status="COMPARABLE_RECORDED_VECTORS",relation="IMPROVED" if tuple(b)<tuple(a) else "WORSENED" if tuple(b)>tuple(a) else "EQUAL",
        first_different_layer=changed[0]["layer"] if changed else None,changed_layers=changed)


def extract(kind,arm,index,callback,task):
    trace=callback.get("search_handoff") or task.get("search_handoff") or {}
    first=trace.get("first_nonempty_certified")
    events=[event for event in trace.get("events",[]) if event.get("event") in EVENTS]
    returned=trace.get("callback_return") or {}
    final=dict(returned)
    matches=[event for event in events if event.get("objective")==returned.get("objective") and sorted(event.get("prefix_hashes",[]))==sorted(returned.get("prefix_hashes",[]))]
    scopes={(event.get("catalog_scope"),event.get("context_scope")) for event in matches}
    if len(scopes)==1:
        final.update(catalog_scope=matches[-1].get("catalog_scope"),context_scope=matches[-1].get("context_scope"))
    objective_id=callback.get("objective_id")
    elapsed=returned.get("wall_elapsed_s");half=elapsed/2 if finite(elapsed) else None
    previous={};transitions=[];scope_groups=defaultdict(list)
    for event in events:
        key=(event.get("catalog_scope"),event.get("context_scope"))
        difference=scoped_difference(previous.get(key),event,objective_id)
        timed=finite(event.get("wall_elapsed_s")) and half is not None and half<=event["wall_elapsed_s"]<=elapsed
        transitions.append(dict(event=event.get("event"),wall_elapsed_s=event.get("wall_elapsed_s"),in_latter_half=timed,difference=difference))
        previous[key]=event;scope_groups[key].append(event)
    strict=[r for r in transitions if r["event"]=="STRICT_CERTIFIED_IMPROVEMENT" and r["in_latter_half"]]
    verified=[r for r in strict if r["difference"].get("relation")=="IMPROVED"]
    catalog_comparisons=[dict(catalog_scope=key[0],context_scope=key[1],first=receipt(rows[0]),last=receipt(rows[-1]),
        difference=scoped_difference(rows[0],rows[-1],objective_id),last_value_and_prefix_retained_by_return=rows[-1] in matches)
        for key,rows in scope_groups.items()]
    first_group=scope_groups.get(((first or {}).get("catalog_scope"),(first or {}).get("context_scope")),[])
    last_in_first_group=first_group[-1] if first_group else None
    phases={key:(callback.get("timings",{}).get(key) or [None])[0] for key in PHASES}
    solver=callback.get("solver",[])
    catalogs=callback.get("catalog_rounds",[])
    late=[dict(snapshot_time=task["snapshot_time"],original_ready_deadline=task.get("ready_deadline"),**event)
          for event in task.get("revalidations",[]) if event.get("status")=="LATE"]
    readiness=[r["previous_ready_time"] for r in task.get("revalidations",[]) if finite(r.get("previous_ready_time"))]
    if finite(task.get("ready_time")):readiness.append(task["ready_time"])
    row=dict(artifact=kind,arm=arm,callback_index=index,snapshot_time=task["snapshot_time"],objective_id=objective_id,
        first_nonempty_certified=receipt(first),events=[receipt(event) for event in events],final_return=receipt(final),
        final_scope_source="MATCHING_RECORDED_OBJECTIVE_AND_PREFIX_EVENT" if len(scopes)==1 else "AMBIGUOUS_MATCHING_EVENT_SCOPES" if matches else "MISSING_IN_RETURN_RECORD",
        matching_final_receipt_wall_elapsed_s=matches[-1].get("wall_elapsed_s") if matches else None,
        empty_prefix_improvement_exact_time="MISSING_IN_FROZEN_TRACE" if first and not returned.get("prefix_hashes") and not matches else "NOT_APPLICABLE_OR_EVENT_RECORDED",
        first_to_final=scoped_difference(first,final,objective_id),same_catalog_receipt_groups=catalog_comparisons,
        first_catalog_first_to_last=scoped_difference(first,last_in_first_group,objective_id),
        final_value_and_prefix_matches_last_in_first_catalog=last_in_first_group in matches if last_in_first_group else None,
        latter_half=dict(interval_wall_elapsed_s=[half,elapsed],recorded_strict_event_count=len(strict),verified_strict_improvements=verified,
            unverified_strict_event_count=len(strict)-len(verified),lex_layers=sorted({r["difference"]["first_different_layer"] for r in verified}),
            transitions=[r for r in transitions if r["in_latter_half"]]),
        deadlines={key:trace.get(key) for key in ("planning_wall_origin","search_wall_deadline","planning_wall_deadline","reserve_seconds")},
        ready_deadline=task.get("ready_deadline"),preparation_started_time=task.get("preparation_started_time"),
        initial_ready_time=min(readiness) if readiness else None,final_recorded_ready_time=task.get("ready_time"),
        revalidations=task.get("revalidations",[]),status=task.get("status"),reason=task.get("reason"),
        effective_time=task.get("effective_time"),finished_time=task.get("finished_time"),actual_submitted=task.get("submitted"),
        late_revalidation_cases=late,exclusive_phase_wall_s=phases,
        cp=dict(solve_calls=len(solver),status_counts=dict(Counter(r.get("status","MISSING") for r in solver)),
            phase_call_counts=dict(Counter(str(r.get("phase","MISSING")) for r in solver)),
            stop_reasons={key:callback.get(key) for key in ("stop_reason","solver_stop_reason","generation_stop_reason","improvement_stop_reason")},
            truncations=callback.get("truncations"),continuation=callback.get("cp_continuation")),
        rank=dict(batch_cache=callback.get("rank_batch_cache"),per_callback_learned_rank_count=None,scored_group_membership=None,
            availability="Cache aggregates recorded; per-candidate Rank events/group membership are not stored in inspected callback/NormalTask records"),
        catalog_rounds=[dict(round=r.get("round"),columns=r.get("columns"),moving_columns=r.get("moving_columns"),
            cp_wall_s=r.get("cp_wall_s"),same_catalog_improvement=r.get("same_catalog_improvement"),
            aircraft_coverage_count=len(r.get("coverage",{})),full_chain_count=sum(bool(v.get("full_chain")) for v in r.get("coverage",{}).values())) for r in catalogs],
        heavy_checks=trace.get("heavy_checks"))
    return row


def self_test():
    def event(name,time,key,scope="A",prefix=("p",)):
        return dict(event=name,wall_elapsed_s=time,objective=key,catalog_scope=scope,context_scope="ctx",prefix_hashes=list(prefix))
    first=event("FIRST_NONEMPTY_CERTIFIED",2.,[0,0,5,10,12])
    improved=event("STRICT_CERTIFIED_IMPROVEMENT",8.,[0,-1,7,20,15])
    trace=dict(first_nonempty_certified=first,events=[first,improved],callback_return=dict(wall_elapsed_s=10.,objective=improved["objective"],prefix_hashes=["p"]))
    cb=dict(objective_id="downstream_service_v8_f_L_risk_Ehat_tau",search_handoff=trace)
    task=dict(snapshot_time=0.,status="PREPARED",ready_time=12.)
    row=extract("fixture","H",1,cb,task)
    assert row["latter_half"]["lex_layers"]==["negative_L"] and row["first_to_final"]["relation"]=="IMPROVED"
    assert row["latter_half"]["verified_strict_improvements"][0]["difference"]["changed_layers"][-1]["after"]==15
    trace["events"]=[first,event("STRICT_CERTIFIED_IMPROVEMENT",8.,[0,-1,7,20,15],scope="B")]
    assert not extract("fixture","H",1,cb,task)["latter_half"]["verified_strict_improvements"]
    trace["events"]=[first];trace["callback_return"].update(objective=[0,0,5,9,11],prefix_hashes=[])
    row=extract("fixture","H",1,cb,task)
    assert row["empty_prefix_improvement_exact_time"]=="MISSING_IN_FROZEN_TRACE"
    assert row["first_to_final"]["status"]=="MISSING_CATALOG_SCOPE" and row["matching_final_receipt_wall_elapsed_s"] is None


def collect():
    rows=[];inputs={};trace_types={};scopes={}
    for kind in ("fixed_state","closed_loop"):
        root=BASE/kind;scope=load(root/"execution_scope.json")
        require(scope["source_commit"]==COMMIT,"unexpected execution source")
        scopes[kind]={key:scope.get(key) for key in ("source_commit","source_hash","scope_id")}
        inputs[str(root/"execution_scope.json")]=sha256((root/"execution_scope.json").read_bytes()).hexdigest()
        for arm in ARMS:
            directory=root/"evaluation"/arm
            paths=[directory/"scheduler_records.json",directory/"normal_tasks.json"]
            callbacks,tasks=(load(path) for path in paths)
            by_time={task["snapshot_time"]:task for task in tasks}
            require(len(by_time)==len(tasks)==len(callbacks),"callback/task count mismatch")
            for index,callback in enumerate(callbacks,1):
                t=callback["planning_timing"]["snapshot_time"]
                rows.append(extract(kind,arm,index,callback,by_time[t]))
            trace_path=directory/"replay/stage3_trace.jsonl"
            types=Counter()
            if trace_path.is_file():
                paths.append(trace_path)
                for line in trace_path.read_text(encoding="utf8").splitlines():
                    if line.strip():types[json.loads(line).get("record_type","MISSING")]+=1
            trace_types[kind+"/"+arm]=dict(types) if trace_path.is_file() else None
            for path in paths:inputs[str(path)]=sha256(path.read_bytes()).hexdigest()
    groups={}
    for kind in ("fixed_state","closed_loop"):
        for arm in ARMS:
            selected=[r for r in rows if (r["artifact"],r["arm"])==(kind,arm)]
            groups[kind+"/"+arm]=dict(callback_count=len(selected),
                callbacks_with_latter_half_strict_improvement_by_layer=dict(Counter(layer for r in selected for layer in r["latter_half"]["lex_layers"])),
                latter_half_verified_strict_events_by_first_layer=dict(Counter(event["difference"]["first_different_layer"]
                    for r in selected for event in r["latter_half"]["verified_strict_improvements"])),
                final_vs_first_relation_counts=dict(Counter(r["first_to_final"].get("relation",r["first_to_final"]["status"]) for r in selected)),
                first_catalog_first_to_last_relation_counts=dict(Counter(r["first_catalog_first_to_last"].get("relation",r["first_catalog_first_to_last"]["status"]) for r in selected)),
                empty_prefix_exact_time_missing_count=sum(r["empty_prefix_improvement_exact_time"]=="MISSING_IN_FROZEN_TRACE" for r in selected))
    return dict(evidence_type="preexisting_artifact",execution_source_commit=COMMIT,execution_scopes=scopes,callbacks=rows,groups=groups,input_sha256=inputs,
        saved_replay_trace_record_types=trace_types,
        missing_data=dict(candidate_level_full_encoded_features="NOT_PRESENT_IN_INSPECTED_CALLBACK_OR_NORMALTASK_ARTIFACTS",
            conditional_reference_labels="NOT_PRESENT_IN_INSPECTED_CALLBACK_OR_NORMALTASK_ARTIFACTS",scored_comparable_group_membership="NOT_PRESENT_IN_INSPECTED_ARTIFACTS"),
        limits=["No comparison across fixed_state and closed_loop is assumed; differences are per callback and matching catalog/context only.",
            "Recorded objective vectors are compared lexicographically; dimensions are not added and no percentage objective gain is computed.",
            "Final-return catalog/context is resolved only from a matching recorded objective/prefix event; missing empty-prefix improvement time is not filled with return time.",
            "Wall elapsed receipts and perf-clock deadlines are distinct from actual simulation preparation/ready/join times.",
            "callback.timings wall entries are retained as producer phase observations; nested heavy_checks are separate and not added to them."])


def csv_text(rows):
    fields=("artifact","arm","callback_index","snapshot_time","first_wall_s","first_objective","first_prefix_count","return_wall_s","final_objective","final_prefix_count",
        "first_to_final","first_catalog_first_to_last","final_value_and_prefix_matches_last_in_first_catalog","latter_half_lex_layers","missing_empty_prefix_time","preparation_started_time","initial_ready_time","status","finished_time","actual_submitted",*PHASES,"cp_solve_calls","rank_cache_hits","rank_cache_miss_reasons")
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:
        first=row["first_nonempty_certified"] or {};last=row["final_return"] or {};cache=row["rank"]["batch_cache"] or {}
        value={key:row.get(key) for key in fields}
        value.update(first_wall_s=first.get("wall_elapsed_s"),first_objective=json.dumps(first.get("objective")),first_prefix_count=first.get("prefix_count"),
            return_wall_s=last.get("wall_elapsed_s"),final_objective=json.dumps(last.get("objective")),final_prefix_count=last.get("prefix_count"),
            first_to_final=row["first_to_final"].get("relation",row["first_to_final"]["status"]),latter_half_lex_layers="|".join(row["latter_half"]["lex_layers"]),
            first_catalog_first_to_last=row["first_catalog_first_to_last"].get("relation",row["first_catalog_first_to_last"]["status"]),
            missing_empty_prefix_time=row["empty_prefix_improvement_exact_time"],cp_solve_calls=row["cp"]["solve_calls"],
            rank_cache_hits=cache.get("hits"),rank_cache_miss_reasons=json.dumps(cache.get("miss_reasons")))
        value.update(row["exclusive_phase_wall_s"]);writer.writerow(value)
    return stream.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    self_test()
    if args.self_test:print("minimal fixture PASS");return
    result=collect();require(len(result["callbacks"])==13,"expected two fixed-state and eleven closed-loop callbacks")
    directory=Path(__file__).resolve().parent
    jp,cp=(directory/("search_progress_analysis_20261027"+suffix) for suffix in (".json",".csv"))
    text=json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)
    table=csv_text(result["callbacks"])
    # Only these two requested repo-local derived files are written; source records stay read-only.
    with jp.open("w",encoding="utf8") as stream:stream.write(text+"\n")
    with cp.open("w",encoding="utf8",newline="") as stream:stream.write(table)
    saved=load(jp)
    require(len(saved["callbacks"])==len(list(csv.DictReader(cp.open(encoding="utf8"))))==13,"derived readback count mismatch")
    for row in saved["callbacks"]:
        directory=BASE/row["artifact"]/"evaluation"/row["arm"]
        source=load(directory/"scheduler_records.json")[row["callback_index"]-1]
        require(row["final_return"]["objective"]==source["search_handoff"]["callback_return"]["objective"],"objective readback mismatch")
        require(row["exclusive_phase_wall_s"]["cp_solve"]==source["timings"]["cp_solve"][0],"phase readback mismatch")
    require(all(sha256(Path(path).read_bytes()).hexdigest()==expected for path,expected in result["input_sha256"].items()),"input changed during derivation")
    print(json.dumps(dict(status="DERIVED_READONLY",callbacks=13,json=str(jp),csv=str(cp),fixture="PASS",raw_readback="PASS"),sort_keys=True))


if __name__=="__main__":main()
