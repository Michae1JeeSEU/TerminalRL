"""Supplement one completed campaign using saved files only; never fit or simulate."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

ARMS=("heuristic_reduced","learned_reduced")
LAYERS=("f","negative_L","risk","Ehat","tau")
MARKER="<!-- FUTURE_LEARNING_SUPPLEMENT_20261027 -->"
OUTPUTS=("future_learning_supplement.json","future_callback_quality_compute.csv","FUTURE_LEARNING_REPORT.md",
         "training_curve.svg","training_curve.png","FINAL_REPORT.generic.original.md")
REVIEWED_IMAGES={
    "planning_timeline.png":"5a0bc010aea7ce55b42b4aaadedd2334103750e6c58fc2cc668440977e5fb39c",
    "saved_xy_contact.png":"1595685ee16b0e95e0aff94049fbf4f5f80de669e9fe8bec850a6a358f5844f2"}
LAZY_FIELDS=("rank_deferred_groups","rank_deferred_items","rank_scored_groups","rank_scored_descriptors",
             "rank_batch_consumed","rank_cache_hits","learned_rank_calls")


def load(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def file_hash(path): return sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def json_write(path,value):
    with Path(path).open("x",encoding="utf8") as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def lazy_coverage(boundaries,result,expected_callbacks):
    """Each boundary retains only the latest snapshot, not a cumulative total."""
    latest={}
    for value in [*(b.get("costs",{}) for b in boundaries),result.get("costs",{}),result.get("terminal_costs",{})]:
        row=value.get("coverage_summary",{})
        if row.get("snapshot_id"):latest[row["snapshot_id"]]=row
    covered={key:sum(key in row.get("counts",{}) for row in latest.values()) for key in LAZY_FIELDS}
    complete=len(latest)==expected_callbacks and all(n==expected_callbacks for n in covered.values())
    return dict(status="COMPLETE_COUNT_COVERAGE" if complete else "PARTIAL",distinct_snapshot_ids=sorted(latest),
        distinct_count=len(latest),expected_callbacks=expected_callbacks,field_coverage=covered,
        totals={key:sum(row.get("counts",{}).get(key,0) for row in latest.values()) for key in LAZY_FIELDS},
        rule="LATEST_OBSERVATION_PER_SNAPSHOT_ID; NO_SUM_OF_REPEATED_BOUNDARY_COVERAGE",
        identity_limit="Distinct snapshot count is compared with callback count; raw Coverage lacks source time.")


def objective_difference(a,b):
    if not a or not b or len(a)!=len(b):return "UNAVAILABLE",None
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    return ("EQUAL",None) if first is None else ("IMPROVED" if tuple(b)<tuple(a) else "WORSE",LAYERS[first])


def phase_wall(value):
    return value.get("wall_s") if isinstance(value,dict) else value[0] if isinstance(value,(list,tuple)) and value else None


def callback_rows(arm,callbacks,tasks):
    by_time={t["snapshot_time"]:t for t in tasks};rows=[]
    for index,callback in enumerate(callbacks,1):
        source_time=callback.get("planning_timing",{}).get("snapshot_time")
        task=by_time.get(source_time,{})
        trace=callback.get("search_handoff",{});first=trace.get("first_nonempty_certified") or {};returned=trace.get("callback_return") or {}
        relation,layer=objective_difference(first.get("objective"),returned.get("objective"))
        events=[e for e in trace.get("events",[]) if e.get("event")=="STRICT_CERTIFIED_IMPROVEMENT"]
        rows.append(dict(arm=arm,callback=index,snapshot_time=source_time,effective_time=callback.get("time"),
            task_status=task.get("status"),reason=task.get("reason"),
            first_certificate_s=first.get("wall_elapsed_s"),return_s=returned.get("wall_elapsed_s"),
            first_objective=json.dumps(first.get("objective")),return_objective=json.dumps(returned.get("objective")),
            first_nonempty_prefixes=len(first.get("prefix_hashes",[])),return_prefixes=len(returned.get("prefix_hashes",[])),
            first_to_return=relation,first_different_layer=layer,strict_event_count=len(events),
            comparison_scope="RECORDED_CALLBACK_OBJECTIVES; CATALOG_SCOPE_MAY_CHANGE",
            last_strict_event_s=max((e.get("wall_elapsed_s",0) for e in events),default=None),
            cp_s=phase_wall(callback.get("timings",{}).get("cp_solve")),
            ready_deadline=task.get("ready_deadline"),ready_time=task.get("ready_time"),
            revalidation_rounds=len(task.get("revalidations",[])),rank_cache_hits=callback.get("rank_batch_cache",{}).get("hits")))
    return rows


def collect(output):
    card=load(output/"model/model_card.json")
    history=[json.loads(line) for line in (output/"model/training_history.jsonl").read_text().splitlines() if line.strip()]
    require(card.get("status")=="TRAINED_PILOT" and card.get("model_frozen"),"completed frozen pilot required")
    require(file_hash(output/"model/model.pt")==card["model_hash"],"model bytes differ from the unique model card")
    require(file_hash(output/"model/feature_normalizer.json")==card["normalizer_sha256"],"normalizer bytes differ from card")
    require(len(history)==card["epochs_completed"] and any(r["epoch"]==card["best_epoch"] for r in history),"saved history/card mismatch")
    cost=card.get("cost_diagnostics",{})
    training={key:card.get(key) for key in ("status","seed","parameter_count","epochs_completed","best_epoch","validation_rank_loss",
        "train_groups","train_pairs","validation_groups","validation_pairs","model_hash","normalizer_sha256","training_dataset_sha256")}
    training["receipt_wall_s"]=load(output/"train.finished.json")["receipt"]["wall_s"]
    training["cost_diagnostics"]={key:value for key,value in cost.items() if key!="predictions"}
    cells={};rows=[]
    for arm in ARMS:
        directory=output/"evaluation"/arm;result=load(directory/"result.json")
        require(result.get("status")=="COMPLETE" and result.get("completed_boundary_s")==120,"completed 120s pair required")
        callbacks=load(directory/"scheduler_records.json");tasks=load(directory/"normal_tasks.json")
        life=result["normal_lifecycle"];aggregate=result["all_callbacks"]
        misses=Counter()
        for c in callbacks:misses.update(c.get("rank_batch_cache",{}).get("miss_reasons",{}))
        revalidations=[dict(snapshot_time=t["snapshot_time"],ready_deadline=t.get("ready_deadline"),
            **{key:r.get(key) for key in ("previous_ready_time","started_time","completed_time","wall_s","status","reason","affected_aircraft_ids")})
            for t in tasks for r in t.get("revalidations",[])]
        cells[arm]=dict(callback_count=len(callbacks),task_count=len(tasks),
            full_normal_planning_work=life["full_normal_planning_work"],finished_task_work=life["finished_task_full_planning_work"],
            joins=life["committed_nonempty_groups"],finished_nonempty=life["finished_nonempty_groups"],
            no_prefix=life["no_executable_prefix_groups"],pending=life["pending_tasks"],
            ready_deadline_rejections=life["ready_deadline_rejections"],rejection_reasons=life["rejection_reasons"],
            quality=result["quality"],dispatch_wall_s=aggregate["full_dispatch_wall_s"],
            attempts=aggregate["physical_attempts"],queries=aggregate["descriptor_queries"],
            phases=aggregate["exclusive_phase_timings"],rank_learned=aggregate["rank_learned"],
            rank_cache_hits=sum(c.get("rank_batch_cache",{}).get("hits",0) for c in callbacks),rank_cache_misses=dict(misses),
            lazy_coverage=lazy_coverage([load(p) for p in sorted(directory.glob("boundary_*.json"))],result,len(callbacks)),
            revalidations=revalidations)
        rows.extend(callback_rows(arm,callbacks,tasks))
    h,l=(cells[a] for a in ARMS)
    change=(l["full_normal_planning_work"]["total_s"]/h["full_normal_planning_work"]["total_s"]-1)*100
    image_review={name:dict(status="PASSED_PRIMARY_VIEW_IMAGE" if (output/name).is_file() and file_hash(output/name)==expected else "PENDING_OR_FILE_CHANGED",
        sha256=file_hash(output/name) if (output/name).is_file() else None) for name,expected in REVIEWED_IMAGES.items()}
    analysis=load(output/"analysis_summary.json")
    replay={}
    for arm in ARMS:
        saved=analysis["cells"][arm]["replay"]
        replay[arm]={key:saved.get(key) for key in ("file_hashes_match","trajectory_index_matches","checked_pva_samples","pva_within_tolerance","c2_status","start_time_mismatch_count")}
        replay[arm]["c2_checked_count"]=len(saved.get("c2_checks",[]))
        replay[arm]["normal_start_coverage"]={key+("_count" if isinstance(value,list) else ""):len(value) if isinstance(value,list) else value
            for key,value in saved.get("normal_start_coverage",{}).items()}
    value=dict(schema="future_learning_saved_supplement_v1",training=training,arms=cells,
        full_normal_planning_change_percent=change,source=load(output/"execution_scope.json"),replay=replay,
        visual_review=image_review,training_curve_visual_review="PENDING_PRIMARY_VIEW_IMAGE",human_replay_acceptance="HUMAN_PENDING",
        comparison_scope="ONE_DEVELOPMENT_SEEN_H_L_PAIR; NO_ISOLATED_OLD_MODEL_ABLATION",
        primary_conclusion="Modest complete-planning cost benefit, much smaller than candidate reduction; no normal landing throughput gain. Energy is slightly lower with more holding. No causal attribution to this training update or lazy deferral/batch reuse.")
    return value,rows,history


def text_report(value,rows):
    t=value["training"];h,l=(value["arms"][a] for a in ARMS);cost=t["cost_diagnostics"]
    lines=[MARKER,"","## Future-state learning supplement: primary review","",
        "This primary interpretation supersedes the generic pending interpretation and broad READY_DEADLINE wording above. Replay human acceptance remains HUMAN_PENDING. The two existing timeline/contact PNGs have passed primary visual review where their hashes match; the new training curve still requires visual review.","",
        f"One guarded fit completed (seed {t['seed']}, {t['parameter_count']} parameters): {t['epochs_completed']} epochs, selected epoch {t['best_epoch']}, validation rank loss {t['validation_rank_loss']:.9f}. There are {t['train_groups']} eligible training groups (including cost-only groups), {t['validation_groups']} validation rank groups, and {t['train_pairs']}/{t['validation_pairs']} train/validation pairs. Nine validation pairs occupy only two groups. The 10 new train contribution pairs first differ only at Ehat; new validation pairs differ at risk (5) or Ehat (4).", "",
        f"On {cost['count']} saved cost targets, MAE is {cost['mae_log1p_ms']:.9f} versus train-mean baseline {cost['constant_train_mean_mae_log1p_ms']:.9f}, in log1p(milliseconds), not milliseconds. Cost adjustment changes ordering in {cost['ordering_changed_contexts']}/{cost['comparable_order_context_count']} comparable contexts; {cost['penalty_clipped_count']} penalties are clipped. This subset signal does not establish online benefit.","",
        "| Saved measure | H | L |","| --- | ---: | ---: |"]
    for label,a,b in (("All-task normal planning work (s; includes pending partial)",h["full_normal_planning_work"]["total_s"],l["full_normal_planning_work"]["total_s"]),
        ("Finished-task nearest-rank P95 (s)",h["finished_task_work"]["p95_s"],l["finished_task_work"]["p95_s"]),
        ("Dispatch work (s)",h["dispatch_wall_s"],l["dispatch_wall_s"]),("Physical attempts",h["attempts"],l["attempts"]),
        ("Descriptor queries",h["queries"],l["queries"]),("Joined / finished nonempty",f"{h['joins']}/{h['finished_nonempty']}",f"{l['joins']}/{l['finished_nonempty']}")):
        lines.append(f"| {label} | {a} | {b} |")
    lines.extend(["",f"Complete planning work changes by {value['full_normal_planning_change_percent']:.2f}%. {value['primary_conclusion']}",
        "The primary clock is normal_lifecycle.full_normal_planning_work. costs.full_online_wall_s includes setup/checkpoint I/O and is not pure planning work. Final all_callbacks supplies authoritative attempts, queries and phase totals.","",
        "Both H READY_DEADLINE_MISSED tasks were previously ready. At source time 37.587890625, previous readiness was 50.515828962 before deadline 56.587890625; new-traffic revalidation started 56.566670201 and completed 56.593427098 (0.026607508 s), about 0.005536473 s late. At source time 83.5, readiness 97.000200374 preceded deadline 102.5; req_000002 recertified at 99.728308864, then req_000003 revalidation started at 102.654980060, already after the deadline. L at source 95 recertified req_000004 at 109.970703801 before deadline 114 (0.061253889 s work). These records expose late new-traffic/service scheduling rather than an initial CP search failure. No timing protocol change was made.","",
        "Per-callback first certificate, return, objective vectors and status are in future_callback_quality_compute.csv. Late search still improves landing objectives in L at source 0 (negative-L 0 to -5) and 95 (0 to -1). Other callbacks plateau (L50 about 5.52 s; H60 4.27 s; H83.5 4.16 s between first certificate and return). This mixed pattern does not support a uniform CP cut. H25 returns an empty prefix with a better recorded Ehat objective after a nonempty certificate; this is retained as objective selection, not relabeled a budget failure.","",
        "L ranking cache has zero hits: 386 CANDIDATES_CHANGED and 20 NO_BATCH misses. Deduplicated Coverage must cover six snapshot IDs before interpreting lazy counters. This run records 406 scored groups, 7364 scored descriptor inputs, 406 consumed batches and zero deferred groups/items. It gives no observed evidence that lazy deferral or cross-consumption batch reuse reduced scoring work. Historical L had five tasks versus six here, including pending work, so historical comparisons do not isolate this model update.","",
        "Discussion only: repeated ranking/invalidation dependency granularity and the roughly 13.13 s feature-plus-inference work remain relevant; late-new-traffic deadline exposure should be accounted separately; selective observed plateau analysis may be useful. These are reporting observations, not new experiment or code-change authorization.","",
        "Contact-sheet markers follow the saved plot code: circle = HOLD status; downward triangle = FINAL or VERTICAL_DESCENT; square = other non-airborne status; right-pointing triangle = remaining airborne status. H is grey and L blue. The sheet uses saved positions.","",
        "The model, timing, geometry, safety, budgets and runtime protocol were not changed for this supplement. Training curve visual QA is PENDING_PRIMARY_VIEW_IMAGE; replay remains HUMAN_PENDING."])
    return "\n".join(lines)+"\n"


def training_curve(output,history,best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family":"DejaVu Sans","svg.fonttype":"none","font.size":10}):
        fig,axes=plt.subplots(1,2,figsize=(8.2,3.1),layout="constrained")
        for ax,key,title in zip(axes,("train_loss","validation_rank_loss"),("Training total loss","Validation ranking loss")):
            ax.plot([r["epoch"] for r in history],[r[key] for r in history],color="#527C9C",marker="o",ms=3)
            ax.axvline(best_epoch,color="#606060",ls="--",lw=1,label=f"Selected epoch {best_epoch}")
            ax.set(xlabel="Epoch",ylabel="Loss",title=title);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        fig.savefig(output/"training_curve.svg");fig.savefig(output/"training_curve.png",dpi=300);plt.close(fig)


def write_outputs(output,value,rows,history,render=True):
    require((output/"FINAL_REPORT.md").is_file(),"existing archive FINAL_REPORT.md required")
    require(not any((output/name).exists() for name in OUTPUTS),"supplement output already exists; no overwrite")
    original=(output/"FINAL_REPORT.md").read_bytes();require(MARKER.encode() not in original,"supplement already appended")
    (output/"FINAL_REPORT.generic.original.md").write_bytes(original)
    if render:training_curve(output,history,value["training"]["best_epoch"])
    with (output/"future_callback_quality_compute.csv").open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ("arm","callback"));writer.writeheader();writer.writerows(rows)
    text=text_report(value,rows);(output/"FUTURE_LEARNING_REPORT.md").write_text(text,encoding="utf8")
    partial=output/"FINAL_REPORT.md.future.partial"
    with partial.open("xb") as stream:stream.write(original+b"\n\n"+text.encode())
    os.replace(partial,output/"FINAL_REPORT.md")
    value["report_provenance"]=dict(helper_sha256=file_hash(Path(__file__)),generic_original_sha256=sha256(original).hexdigest(),
        final_report_sha256=file_hash(output/"FINAL_REPORT.md"),outputs_sha256={name:file_hash(output/name) for name in OUTPUTS if (output/name).is_file()})
    json_write(output/"future_learning_supplement.json",value)


def self_test(output):
    require(not output.exists(),"self-test output must be fresh and separate from a campaign")
    output.mkdir(parents=True)
    first={"snapshot_id":"one","counts":{k:1 for k in LAZY_FIELDS}}
    latest={"snapshot_id":"one","counts":{k:2 for k in LAZY_FIELDS}}
    coverage=lazy_coverage([{"costs":{"coverage_summary":first}},{"costs":{"coverage_summary":latest}}],{},1)
    require(coverage["totals"]["rank_scored_groups"]==2 and coverage["distinct_count"]==1,"repeated coverage was double counted")
    require(lazy_coverage([{"costs":{"coverage_summary":first}}],{},2)["status"]=="PARTIAL","missing coverage treated as total")
    require(objective_difference([0,0,5],[0,-1,9])==("IMPROVED","negative_L"),"lexicographic comparison failed")
    require(phase_wall([1.25,1.2])==phase_wall({"wall_s":1.25})==1.25,"saved phase schema extraction failed")
    history=[dict(epoch=1,train_loss=.5,validation_rank_loss=.7),dict(epoch=2,train_loss=.4,validation_rank_loss=.65)]
    training_curve(output,history,2)
    require((output/"training_curve.png").stat().st_size>0 and (output/"training_curve.svg").is_file(),"curve exports missing")
    json_write(output/"self_test.json",dict(status="PASSED",checks=["snapshot dedup","partial coverage","objective lexicographic layer","list/dict phase schema","saved-history curve export"],
        scientific_run=False,model_training=False,output=str(output)))
    return dict(status="SELF_TEST_PASSED",output=str(output))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv);output=args.output.resolve()
    if args.self_test:result=self_test(output)
    else:
        value,rows,history=collect(output);write_outputs(output,value,rows,history)
        result=dict(status="SUPPLEMENT_WRITTEN",output=str(output),training_curve_visual_review="PENDING_PRIMARY_VIEW_IMAGE")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
