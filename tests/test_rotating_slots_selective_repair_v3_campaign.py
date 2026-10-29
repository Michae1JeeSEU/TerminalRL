"""Tiny schema, accounting and serialization checks; never run the solver."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.stage3 import run_selective_reoptimization_a as base
from scripts.stage3 import run_selective_reoptimization_repair as old
from scripts.stage3 import run_selective_reoptimization_repair_v3 as runner


def tagged(values): return {"$":"dict","items":list(values.items())}
def number(value): return {"$":"float","hex":float(value).hex()}


class RepairV3Tests(unittest.TestCase):
    def test_continuation_restores_real_wrapper_without_cold_restore_or_advance(self):
        observed={}
        callback=object();wrapper=object()
        world=SimpleNamespace(time=60.,scheduler_period=5.)
        def scheduler(*args,**kwargs):
            observed["scheduler_options"]=kwargs;return callback
        def make_wrapper(actual_callback,**kwargs):
            self.assertIs(actual_callback,callback);observed["wrapper_options"]=kwargs;return wrapper
        def restore(path,actual_wrapper,identity,**kwargs):
            self.assertIs(actual_wrapper,wrapper);observed["restore"]=(path,identity,kwargs);return world
        d=SimpleNamespace(LatencyAwareScheduler=make_wrapper,restore_checkpoint=restore,shared_fifo_admission=object())
        with patch.object(base,"scheduler",side_effect=scheduler),patch.object(base,"restore",side_effect=AssertionError("ColdRestore forbidden")):
            restored,returned_callback,returned_wrapper=runner.restore_continuation(Path("saved60.json"),{"identity":{"original":"identity"}},d,Path("new"))
        self.assertIs(restored,world);self.assertIs(returned_callback,callback);self.assertIs(returned_wrapper,wrapper)
        self.assertIs(world.decision_callback,wrapper)
        self.assertEqual(observed["wrapper_options"],dict(lead_seconds=20.,preparation_margin=1.,search_handoff_reserve_seconds=4.,restart_policy=old.RESTART))
        self.assertEqual(observed["restore"][2]["stream_copy"],Path("new/restored_states.jsonl"))

    def test_continuation_restore_invariants_cover_pending_prefix_history_and_caches(self):
        class Codec:
            def encode(self,value):return deepcopy(value)
            def decode(self,value):return deepcopy(value)
        cbstate=dict(selective_replanning={"history":{"entries":{"a":{"reviewed":20.}}}},previous={"a":[]},references=[],templates=[],reference_hits=0)
        callback=SimpleNamespace(records=[{"snapshot_time":40.}],provider=SimpleNamespace(validation_cache={}),state_dict=lambda:deepcopy(cbstate))
        pending=dict(stage="PREPARED",due=60.,token="saved",proposal=["stored_proposal"])
        saved=dict(callback=cbstate,pending=pending,records=[{"status":"PREPARED"}],online_services=[{"wall_s":1.}],
            dirty=True,idle_service=(None,0.),handoff_service=0.,service_until=59.,restart_policy=old.RESTART,
            restart_requested=None,last_normal_start_time=40.,last_restart_time=40.)
        wrapper=SimpleNamespace(pending=deepcopy(pending),records=deepcopy(saved["records"]),online_services=deepcopy(saved["online_services"]),state_dict=lambda:deepcopy(saved))
        worldstate=dict(time=60.,records=[{"time":0.},{"time":60.}],active={"a":"active"},soft={},_planning_latency_packets={"saved":{"plans":["stored_proposal"]}})
        world=SimpleNamespace(**deepcopy(worldstate))
        result=runner.continuation_invariants(world,callback,wrapper,{"world":worldstate,"scheduler":saved},Codec())
        self.assertEqual(result["status"],"PASS");self.assertFalse(result["physical_execution"] or result["solver_called"])
        self.assertEqual(result["pending_due"],60.)
        wrapper.pending["due"]=61.
        changed=runner.continuation_invariants(world,callback,wrapper,{"world":worldstate,"scheduler":saved},Codec())
        self.assertEqual(changed["status"],"FAIL");self.assertFalse(changed["checks"]["pending_matches"])

    def test_continuation_costs_keep_prefix_and_new_segment_distinct(self):
        prefix=dict(job_elapsed_wall_s=53.,process_cpu_s=52.,world_advance_wall_s=44.,world_advance_cpu_s=43.)
        costs=runner.continuation_costs(prefix,job_wall_s=100.,process_cpu_s=98.,advance_wall_s=90.,advance_cpu_s=89.)
        self.assertEqual(costs["job_elapsed_wall_s"],153.)
        self.assertEqual(costs["continuation_job_wall_s"],100.)
        self.assertEqual(costs["world_advance_wall_s"],134.)
        self.assertIn("NO_PREFIX_RECREDIT",costs["ledger_cost_scope"])
        self.assertNotIn("original_failed_child_wall_s",costs)
        self.assertEqual(runner.CONTINUATION_BOUNDARIES[0],65)
        self.assertEqual(runner.CONTINUATION_BOUNDARIES[-1],300)
        self.assertEqual(len(runner.CONTINUATION_BOUNDARIES),48)

    def test_new_ledger_separates_history_without_old_prior_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with runner.ledger_transaction(tmp) as ledger:
                self.assertEqual(ledger["settled_child_wall_s"],0.)
                self.assertEqual(ledger["cap_child_wall_s"],5400.)
                self.assertEqual(ledger["previous_repair_settled_child_wall_s"],1477.5308457711415)
                self.assertEqual(ledger["historical_campaign_used_s"],7323.47022066447)
            with self.assertRaisesRegex(ValueError,"limits/history"):
                with old.ledger_transaction(tmp): pass

    def test_receipt_credit_charges_failed_work_and_prevents_duplicate_reference(self):
        receipt=dict(wall_ms=14795,exit_code=1,command="measured test command",start_utc="start",end_utc="end",status="failed")
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"output";path=Path(tmp)/"receipt.json";path.write_text(json.dumps(receipt))
            result=runner.credit_observed_receipt(output,"tests_b",path)
            self.assertEqual(result["child_wall_s"],14.795)
            ledger=json.loads((output/"repair_budget_ledger.json").read_text())
            self.assertEqual(ledger["jobs"]["tests_b"]["original_status"],"failed")
            self.assertEqual(ledger["settled_child_wall_s"],14.795)
            self.assertEqual(ledger["historical_campaign_used_s"],7323.47022066447)
            with self.assertRaisesRegex(ValueError,"cost/job already"):runner.credit_observed_receipt(output,"tests_b",path)
            with self.assertRaisesRegex(ValueError,"receipt already"):runner.credit_observed_receipt(output,"other_job",path)

    def test_v2_diagnostic_only_and_fixed_treatment(self):
        with self.assertRaisesRegex(ValueError,"diagnostic-only"): runner.treatment(old.RULE)
        row=runner.treatment(old.RULE,diagnostic=True)
        self.assertEqual((row["nominal_k"],row["max_k"],row["lead_seconds"],row["heartbeat_seconds"]),(8,12,20.,5.))
        self.assertEqual(runner.treatment(runner.RULE)["restart_policy"],old.RESTART)

    def test_checkpoint_observation_does_not_invent_saved_source(self):
        pending=tagged(dict(timing={"$":"dataclass","fields":[["snapshot_time",number(60.)],["effective_time",number(80.)]]},
                           stage="RESULT",proposal=None))
        payload=dict(time=65.,identity={"manifest_sha256":"manifest"},world=tagged({}),scheduler=tagged(dict(pending=pending)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.json"
            path.write_text(json.dumps(dict(payload=payload,sha256=base.digest(payload))))
            row=runner.checkpoint_observation(path,60.)
            self.assertEqual(row["boundary_minus_historical_source_s"],5.)
            self.assertEqual(row["pending_timing"]["effective_time"],80.)
            self.assertFalse(row["historical_input_recovered"])
            self.assertEqual(row["saved_prepared_packets"],[])

    def test_immediate_probe_plan_keeps_hashes_and_missing_explicit(self):
        a,b=runner.PROBE_IDS
        def row(aid,key,start=80.,node=80.):
            return dict(aircraft_id=aid,kind="MOVE",candidate_hash=key,first_start=start,adjustable_node_time=node,in_catalog=True)
        diagnostic=dict(snapshot_time=80.,candidates=[row(a,"z"),row(a,"a"),row(b,"b"),row(b,"delayed",81.)])
        plan=runner.probe_combinations(diagnostic)
        self.assertEqual(len(plan["cases"]),3)
        self.assertEqual(plan["cases"][-1]["forced_choices"],{a:"a",b:"b"})
        self.assertEqual(runner.probe_combinations(dict(snapshot_time=80.,candidates=[row(a,"a")]))["missing_immediate_move_ids"],[b])

    def test_exclusive_json_serialization_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"new.json"; runner.write_new_json(path,dict(catalog=runner.json_view(MappingProxyType({"a":(1,2)}))))
            before=path.read_bytes()
            self.assertEqual(json.loads(before)["catalog"],{"a":[1,2]})
            with self.assertRaises(FileExistsError): runner.write_new_json(path,{"replaced":True})
            self.assertEqual(path.read_bytes(),before)

    def test_json_view_preserves_infinite_reservation_end_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"reservation.json"
            runner.write_new_json(path,{"end":float("inf")})
            self.assertEqual(json.loads(path.read_text())["end"],{"$nonfinite_float":"+inf"})

    def test_diagnostic_freezes_can_change_without_locking_formal_source(self):
        files={runner.DRIVER.relative_to(runner.PROJECT).as_posix():"hash",
               "tests/test_rotating_slots_selective_repair_v3_campaign.py":"hash"}
        origin=dict(inputs={"s3r24":dict(run_checkpoint="t0",initial_checkpoint_sha256="t0hash",row={"seed":7325})})
        origin["binding_id"]=base.digest(origin)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),origin_output=Path(tmp),source_freeze=Path(tmp)/"freeze.json",
                                 root="s3r24",variant=old.RULE,stage="snapshot")
            real_load=base.load
            def load(path):
                if Path(path)==args.source_freeze:return dict(files=files,source_hash=source_hash,commit="commit")
                if Path(path).name=="repair_budget_ledger.json":return real_load(path)
                return origin
            with patch.object(base,"load",side_effect=load),patch.object(base,"verify_sources"), \
                    patch.object(base,"file_hash",return_value="t0hash"):
                source_hash="first";self.assertEqual(runner.binding(args)["source_hash"],"first")
                source_hash="second";self.assertEqual(runner.binding(args)["source_hash"],"second")
                self.assertFalse((Path(tmp)/"repair_budget_ledger.json").exists())
                args.stage="smoke";args.variant=runner.RULE
                runner.binding(args)
                source_hash="third"
                with self.assertRaisesRegex(ValueError,"formal source differs"):runner.binding(args)

    def test_restores_only_history_and_blockers_to_cold_callback(self):
        class History:
            def load_state_dict(self,state): self.value=state
            def state_dict(self): return self.value
        callback=SimpleNamespace(selective_history=History(),provider=object())
        saved=dict(selective_replanning=dict(method="RULE_LOCAL_K8",history={"entries":{"a":{"reviewed":20.}}},blockers=[{"target":"a"}]))
        result=runner.restore_selector_history(callback,saved)
        self.assertEqual(result["history"]["entries"]["a"]["reviewed"],20.)
        self.assertEqual(result["candidate_cache"],"COLD_NOT_HISTORICAL_WARM_CACHE")
        callback.selective_blockers.clear()
        self.assertTrue(saved["selective_replanning"]["blockers"])

    def test_v3_watchdog_uses_new_policy_and_no_retry(self):
        class Clock:
            now=0.
            def time(self): return self.now
            def sleep(self,n): self.now+=n
        class Process:
            pid=12345;stopped=False
            def poll(self): return -9 if self.stopped else None
            def wait(self): return -9 if self.stopped else None
        child=Process();clock=Clock();policy=dict(runner.POLICY,cap_child_wall_s=.2)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(runner.POLICY,policy,clear=True), \
                patch.object(old.time,"time",clock.time),patch.object(old.time,"monotonic",clock.time), \
                patch.object(old.time,"sleep",clock.sleep),patch.object(old.subprocess,"Popen",return_value=child), \
                patch.object(old,"_stop",side_effect=lambda p:setattr(p,"stopped",True)):
            result=runner.run_bounded_job(tmp,"unit",["test-only"],cap_s=10.)
            self.assertEqual(result["stop_reason"],"CAMPAIGN_CHILD_WALL_LIMIT")
            ledger=json.loads((Path(tmp)/"repair_budget_ledger.json").read_text())
            self.assertLessEqual(ledger["settled_child_wall_s"],.2)
            self.assertEqual(ledger["prior_audits_child_wall_s"],0.)
            with self.assertRaisesRegex(ValueError,"already attempted"):
                runner.run_bounded_job(tmp,"unit",["test-only"],cap_s=10.)


if __name__=="__main__":unittest.main()
