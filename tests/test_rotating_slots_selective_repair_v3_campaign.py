"""Tiny schema, accounting and serialization checks; never run the solver."""
import json
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
