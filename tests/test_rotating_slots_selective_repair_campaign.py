"""Schema/accounting tests only; no physics rollout or candidate solve."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as repair
from scripts.stage3 import report_selective_reoptimization_repair as report


class RepairCampaignTests(unittest.TestCase):
    def test_fixed_treatment_and_compatibility_passthrough(self):
        row=repair.treatment()
        self.assertEqual((row["nominal_k"],row["max_k"],row["lead_seconds"]),(8,12,20.))
        self.assertEqual(row["opportunity_seed_offset"],0)
        self.assertFalse(row["learning"] or row["dynamic_k"] or row["lead_scan"])
        seen=[]
        class D:
            @staticmethod
            def NormalSchedulerCallback(*args,**kwargs): seen.append(kwargs); return kwargs
        base.scheduler(D,"FULL_H")
        self.assertNotIn("opportunity_seed_offset",seen[-1])
        base.scheduler(D,"FULL_H",selector_variant=repair.RULE,opportunity_seed_offset=2)
        self.assertEqual(seen[-1]["selective_rule_variant"],repair.RULE)
        self.assertEqual(seen[-1]["opportunity_seed_offset"],2)

    def test_active_child_wall_includes_parallel_children(self):
        data=dict(settled_child_wall_s=82.,jobs={"a":dict(status="running",started_epoch=100.),
            "b":dict(status="running",started_epoch=110.),"c":dict(status="completed",started_epoch=0.)})
        self.assertEqual(repair.active_elapsed(data,120.),112.)

    def test_watchdog_readonly_poll_does_not_rewrite_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            with repair.ledger_transaction(tmp) as ledger:
                ledger["jobs"]["a"]=dict(status="running",started_epoch=100.)
            path=Path(tmp)/"repair_budget_ledger.json"; before=path.read_bytes(); mtime=path.stat().st_mtime_ns
            with patch.object(Path,"replace",side_effect=AssertionError("readonly poll wrote ledger")):
                with repair.ledger_transaction(tmp,write=False) as ledger:
                    self.assertEqual(repair.active_elapsed(ledger,120.),repair.PRIOR_AUDIT_S+20.)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_watchdog_stops_during_active_aggregate_budget_and_no_retry(self):
        class Clock:
            now=100.
            def time(self): return self.now
            def sleep(self,n): self.now+=n
        clock=Clock()
        class Process:
            pid=12345
            stopped=False
            def poll(self): return -9 if self.stopped else None
            def wait(self): return -9 if self.stopped else None
        child=Process()
        with tempfile.TemporaryDirectory() as tmp, patch.object(repair,"CAP",repair.PRIOR_AUDIT_S+.2), \
             patch.object(repair.time,"time",clock.time), patch.object(repair.time,"sleep",clock.sleep), \
             patch.object(repair.time,"monotonic",clock.time), \
             patch.object(repair.subprocess,"Popen",return_value=child), \
             patch.object(repair,"_stop",side_effect=lambda p:setattr(p,"stopped",True)):
            result=repair.run_bounded_job(tmp,"unit",["test-only"],cap_s=20.,poll_s=.05)
            self.assertEqual(result["status"],"BUDGET_LIMITED")
            self.assertEqual(result["stop_reason"],"CAMPAIGN_CHILD_WALL_LIMIT")
            self.assertTrue(child.stopped)
            ledger=json.loads((Path(tmp)/"repair_budget_ledger.json").read_text())
            self.assertEqual(ledger["historical_campaign_used_s"],7323.47022066447)
            self.assertLessEqual(ledger["settled_child_wall_s"],repair.CAP)
            with self.assertRaisesRegex(ValueError,"already attempted"):
                repair.run_bounded_job(tmp,"unit",["test-only"],cap_s=20.)

    def test_admission_denominators_are_distinct_and_empty_is_null(self):
        empty=report.admission_facts([],[],{})
        self.assertIsNone(empty["submitted_task_rate"])
        self.assertIsNone(empty["submitted_unique_aircraft_rate"])
        records=[dict(planning_timing={"snapshot_time":0},deferred=True,noop=False,
            selective_replanning={"scope":{"replan_ids":["a","b","c"]}},
            actual_delivery_status="COMMITTED",actual_submitted_trajectory_hashes=["h1","h2"]),
            dict(planning_timing={"snapshot_time":5},deferred=True,noop=False,
            selective_replanning={"scope":{"replan_ids":["a","b"]}})]
        tasks=[dict(snapshot_time=0,status="COMMITTED",finished_time=20,submitted=2),dict(snapshot_time=5,status="PENDING")]
        values=report.admission_facts(tasks,records,{"h1":{"aircraft_id":"a"},"h2":{"aircraft_id":"a"}})
        self.assertEqual(values["submitted_task_rate"],1.)
        self.assertEqual(values["submitted_aircraft_opportunity_rate"],1/5)
        self.assertEqual(values["submitted_unique_aircraft_rate"],1/3)
        self.assertEqual(values["actual_submitted_plans"],2)

    def test_missing_timing_and_unknown_reason_not_invented(self):
        timings=report.timing_facts([{"timings":{"cp_solve":[.2,.1]}}])
        self.assertEqual(timings["exclusive"]["cp_solve"]["wall_s"],.2)
        self.assertIsNone(timings["exclusive"]["trajectory_validation"])
        records=[dict(selective_replanning={"scope":{"eligible_ids":["a","b"],"replan_ids":["a"]},
            "candidate_generation_by_aircraft":{"a":0,"b":0}})]
        facts=report.nonlanding_facts(records,[],{"prediction":{"aircraft_id":"a"}},
            {"a":{"status":"HOLD_OUTER"},"b":{"status":"HOLD_OUTER"}})
        rows={r["aircraft_id"]:r["observation_labels"] for r in facts["rows"]}
        self.assertEqual(rows["a"],{"UNKNOWN":[]})
        self.assertIn("UNSELECTED",rows["b"])
        self.assertNotIn("NO_VALID_CANDIDATE",facts["reason_counts"])
        self.assertNotIn("EXECUTED_NOT_LANDED",facts["reason_counts"])

    def test_touchdown_and_cleared_are_not_executed_not_landed(self):
        facts=report.nonlanding_facts([], [dict(type="maneuver_started",time=1,aircraft_id="a",details={}),
            dict(type="touchdown",time=3,aircraft_id="a")], {}, {"a":{"status":"CLEARED"}})
        self.assertEqual(facts["rows"],[])

    def test_saved_baseline_raw_schema_and_external_audit_override(self):
        candidates=[Path(os.environ.get("RS_REPAIR_SAVED_BASELINE", "__no_override__")),
            Path.home()/"training_outputs/terminal/rs_selective_followup_20261028/arms/s3r23/baseline20/evaluation/s3r23/RULE_LOCAL_K8",
            Path("D:/7_codespace/training_outputs/terminal/rs_selective_followup_20261028/arms/s3r23/baseline20/evaluation/s3r23/RULE_LOCAL_K8")]
        directory=next((p for p in candidates if (p/"scheduler_records.json").is_file()),None)
        if directory is None: self.skipTest("saved baseline raw schema not available on this host")
        records=base.load(directory/"scheduler_records.json")
        choices=[choice for r in records for choice in r.get("aircraft_choices", [])]
        self.assertTrue(choices)
        self.assertTrue(all("move_candidates" in c and "only_keep_wait" in c and "candidate_insufficiency_reasons" in c and "rejection_reasons" in c for c in choices))
        sample=next((c for c in choices if c["move_candidates"]==0),None)
        self.assertIsNotNone(sample)
        aid=sample["aircraft_id"]
        facts=report.nonlanding_facts([dict(aircraft_choices=[sample])],[],{}, {aid:{"status":"HOLD_OUTER"}})
        observed=facts["rows"][0]["observation_labels"]["NO_VALID_CANDIDATE"][0]
        self.assertEqual(observed["candidate_insufficiency_reasons"],sample["candidate_insufficiency_reasons"])
        self.assertIn("NOT_PHYSICAL_INFEASIBILITY",observed["meaning"])
        self.assertEqual(facts["joint_conflict_evidence"],"UNKNOWN_NO_SAVED_OBSERVATION")
        with tempfile.TemporaryDirectory() as tmp:
            audit_path=Path(tmp)/"prior_audit.json"; audit_path.write_text(json.dumps(dict(status="PASS",full_a20_covered=True)))
            cell=report.saved_cell(Path(tmp)/"empty",audit_override=audit_path)
            self.assertEqual(cell["automated_audit"]["status"],"PASS")
            self.assertEqual(cell["automated_audit_evidence_path"],str(audit_path))

    def test_missing_audit_stays_pending_and_lifecycle_includes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"normal_tasks.json").write_text(json.dumps([dict(snapshot_time=0,status="PENDING",normal_work_s=2.,selective_history_work_s=.1)]))
            cell=report.saved_cell(p)
            self.assertEqual(cell["automated_audit"]["status"],"PENDING_NOT_RUN")
            self.assertEqual(cell["human_replay_status"],"HUMAN_PENDING")
            self.assertAlmostEqual(cell["normal_lifecycle"]["full_normal_planning_work"]["total_s"],2.1)
            self.assertIsNone(cell["endpoint_s"])

    def test_duplicate_snapshot_sets_do_not_become_independent_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"snapshots/s3r23"
            for offset in range(3):
                p=root/f"offset{offset}";p.mkdir(parents=True)
                (p/"result.json").write_text(json.dumps(dict(method="RULE_LOCAL_K8",offset=offset,
                    frame_id="same",checkpoint_sha256="same",selected_set_digest="same-set",
                    treatment={"selective_rule_variant":repair.RULE})))
            values=report.snapshot_comparisons(Path(tmp))["s3r23"]
            self.assertEqual(values["unique_local_sets"],1)
            self.assertEqual(values["offset_set_deduplication"][0]["offsets"],[0,1,2])
            self.assertEqual(values["fixed_formal_offset"],0)

    def test_preselection_claim_deduplicates_same_set_even_if_solver_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            options=dict(root="s3r23",frame_id="frame",result_path="first/result.json",method="RULE_LOCAL_K8")
            key,duplicate=repair.claim_snapshot_selection(tmp,selected_ids=["b","a"],offset=0,**options)
            self.assertIsNone(duplicate)
            same,duplicate=repair.claim_snapshot_selection(tmp,selected_ids=["a","b"],offset=1,**options)
            self.assertEqual(key,same)
            self.assertEqual(duplicate["offset"],0)
            self.assertEqual(duplicate["status"],"REGISTERED_SOLVE_MAY_BE_RUNNING")
            self.assertEqual(duplicate["result_path"],"first/result.json")
            different,duplicate=repair.claim_snapshot_selection(tmp,selected_ids=["a"],offset=2,**options)
            self.assertNotEqual(different,key); self.assertIsNone(duplicate)

    def test_snapshot_preselection8_can_return_legally_expanded10(self):
        initial=[f"a{i}" for i in range(8)]; expanded=initial+["b0","b1"]
        records=[dict(selective_replanning=dict(calls=[dict(scope={"replan_ids":initial}),
            dict(scope={"replan_ids":expanded})],scope={"replan_ids":expanded},attempted_replan_ids=expanded))]
        facts=repair.snapshot_scope_observations(records,initial)
        self.assertTrue(facts["initial_scope_matches_preselection"])
        self.assertEqual(facts["comparison_status"],"MATCH")
        self.assertEqual(facts["initial_selected_ids"],initial)
        self.assertEqual(facts["returned_selected_ids"],expanded)
        self.assertEqual(facts["attempted_union"],expanded)
        self.assertEqual(facts["observed_solve_calls"],2)
        missing=repair.snapshot_scope_observations([dict(selective_replanning=dict(status="GLOBAL_FRAME_INCOMPLETE"))],initial)
        self.assertIsNone(missing["initial_scope_matches_preselection"])
        self.assertEqual(missing["comparison_status"],"UNAVAILABLE_NO_INITIAL_CALL_SCOPE")
        self.assertIsNone(missing["returned_selected_ids"])
        self.assertIsNone(missing["attempted_union"])

    def test_joint_conflict_requires_explicit_saved_ids(self):
        evidence=dict(aircraft_ids=["a","b"],reason="RESOURCE_CONFLICT",source="SELECTED_SELECTED_BATCH")
        record=dict(selective_replanning={"joint_conflict_observations":[evidence]},aircraft_choices=[])
        facts=report.nonlanding_facts([record],[],{}, {aid:{"status":"HOLD_OUTER"} for aid in ("a","b","c")})
        rows={r["aircraft_id"]:r["observation_labels"] for r in facts["rows"]}
        self.assertIn("JOINT_CONFLICT",rows["a"]); self.assertIn("JOINT_CONFLICT",rows["b"])
        self.assertEqual(rows["c"],{"UNKNOWN":[]})

    def test_timeline_distinguishes_first_wait_certificate_and_later_nonempty(self):
        wait=dict(call=0,wall_elapsed_s=.2,objective=[0,0,1],prefix_hashes=[])
        nonempty=dict(wall_elapsed_s=1.1,prefix_hashes=["proposal"],event="FIRST_NONEMPTY_CERTIFIED")
        task=dict(snapshot_time=5,status="PREPARED",ready_time=7.)
        record=dict(planning_timing={"snapshot_time":5},selective_replanning={"first_certified":wait},
            search_handoff={"first_nonempty_certified":nonempty})
        row=report.task_timelines([task],[record],[],{})[0]
        self.assertEqual(row["first_certified"],wait)
        self.assertEqual(row["first_certificate"],wait)
        self.assertEqual(row["first_certified_wall_elapsed_s"],.2)
        self.assertEqual(row["first_nonempty_certified"],nonempty)
        self.assertEqual(row["first_nonempty_certified_wall_elapsed_s"],1.1)
        self.assertEqual(row["final_ready_sim_time"],7.)
        self.assertIn("elapsed wall time",row["clocks"])
        self.assertIn("executor simulation time",row["clocks"])

    def test_timeline_missing_first_any_certificate_is_null_for_legacy(self):
        nonempty=dict(wall_elapsed_s=.8,prefix_hashes=["legacy-proposal"])
        task=dict(snapshot_time=0,status="COMMITTED")
        legacy=dict(planning_timing={"snapshot_time":0},search_handoff={"first_nonempty_certified":nonempty})
        row=report.task_timelines([task],[legacy],[],{})[0]
        self.assertIsNone(row["first_certified"])
        self.assertIsNone(row["first_certified_wall_elapsed_s"])
        self.assertIsNone(row["first_certificate"])
        self.assertEqual(row["first_nonempty_certified"],nonempty)
        missing=report.task_timelines([task],[dict(planning_timing={"snapshot_time":0})],[],{})[0]
        self.assertIsNone(missing["first_nonempty_certified"])
        self.assertIsNone(missing["first_nonempty_certified_wall_elapsed_s"])


if __name__ == "__main__": unittest.main()
