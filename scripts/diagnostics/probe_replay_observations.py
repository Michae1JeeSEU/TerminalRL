"""Bounded read-only archive probe; reconstructed worlds are NOT checkpoints."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from terminal_marl.core.rotating_slots import world as wm
from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.resources import Reservation
from terminal_marl.core.rotating_slots.trajectory import TrajectoryPlan, make_stable_plan
from terminal_marl.core.rotating_slots.qualification import provenance

TIMES = (16.25545875046598, 102.49614261797981, 540., 550.)


def plain(x):
    if hasattr(x, "__dataclass_fields__"): return plain(asdict(x))
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def read_archive(root):
    root = Path(root)
    names = ("replay.json", "config.yaml", "manifest.json", "trajectories.json",
             "events.jsonl", "rejections.jsonl", "states.jsonl")
    hashes = {n: digest(root/n) for n in names}
    index = json.loads((root/"replay.json").read_bytes())
    assert all(index["files"][n] == h for n, h in hashes.items() if n != "replay.json")
    frames = {}
    for n, line in enumerate((root/"states.jsonl").open(encoding="utf-8"), 1):
        f = json.loads(line)
        if f["time"] in TIMES:
            if f["time"] in frames: raise ValueError("duplicate exact frame")
            frames[f["time"]] = (n, f)
    if set(frames) != set(TIMES): raise ValueError("missing exact frame")
    return dict(root=root, hashes=hashes, frames=frames, config=load_config(root/"config.yaml"),
                manifest=json.loads((root/"manifest.json").read_bytes()),
                trajectories=json.loads((root/"trajectories.json").read_bytes()),
                events=[json.loads(l) for l in (root/"events.jsonl").read_text().splitlines()],
                rejections=[json.loads(l) for l in (root/"rejections.jsonl").read_text().splitlines()])


def restore(data, t):
    line, f = data["frames"][t]
    m = data["manifest"]
    manifest = ArrivalManifest(tuple(ArrivalRequest(**r) for r in m["requests"]),
                              **{k: v for k, v in m.items() if k != "requests"})
    w = wm.RotatingSlotWorld(config=data["config"], manifest=manifest)
    w.time, w.failed = t, f["failed"]
    for original in f["aircraft"]:
        a = dict(original); a["emergency"] = wm.EmergencyStatus(**a["emergency"])
        for k in ("p", "v", "a", "jerk"): a[k] = tuple(a[k])
        w.aircraft[a["id"]] = wm.AircraftState(**a)
    w.requests = {r["id"]: ArrivalRequest(**r) for r in f["requests"]}
    w._manifest_index = sum(r.release_time <= t+1e-8 for r in manifest.requests)
    w.trajectories = dict(data["trajectories"])
    plans = {h: TrajectoryPlan.from_dict(p, config=w.config) for h, p in w.trajectories.items()}
    for field in ("slots", "pads"):
        for original in f[field]:
            r = dict(original); r["end"] = math.inf if r["end"] is None else r["end"]
            if isinstance(r["resource"], list): r["resource"] = tuple(r["resource"])
            r["dependencies"] = tuple(r["dependencies"])
            getattr(w, field)._entries.append(Reservation(**r))
    for a in w.aircraft.values():
        if a.status in wm.AIRBORNE and a.status not in wm.HOLD.values():
            w.active[a.id] = plans[a.trajectory_hash]
    active_hashes = {p.trajectory_hash for p in w.active.values()}
    reserved = {r.plan_hash for r in (*w.slots.reservations, *w.pads.reservations) if r.plan_hash}
    for h in reserved-active_hashes:
        p = plans[h]
        if p.start_time <= t: raise ValueError("unclassified calendar plan: "+h)
        w.soft[p.aircraft_id] = p
    for p in (*w.active.values(), *w.soft.values()):
        source = [r.end for r in w.slots.reservations if r.plan_hash == p.trajectory_hash and r.kind == "source"]
        target = [r.start for r in w.slots.reservations if r.plan_hash == p.trajectory_hash and r.kind == "target"]
        if (p.source_ring is not None and len(source) != 1) or (p.target_ring is not None and len(target) != 1):
            raise ValueError("incomplete protection timestamps")
        w._source_clear[p.trajectory_hash] = (source[0] if source else p.start_time, target[0] if target else p.start_time)
        if p.trajectory_hash in active_hashes:
            events = [e for e in data["events"] if e["time"] <= t and e["type"] in ("admitted", "maneuver_started")
                      and e["details"].get("trajectory_hash") == p.trajectory_hash]
            if not events: raise ValueError("missing dependency event")
            w.dependencies[p.trajectory_hash] = tuple(events[-1]["details"].get("dependencies", ()))
    w._initial_settled = True
    w._next_heartbeat = (math.floor(t/w.scheduler_period)+1)*w.scheduler_period
    slots, pads, _ = w._calendar_stage(w.soft, ())
    calendar_match=plain(slots.to_dict()) == f["slots"] and plain(pads.to_dict()) == f["pads"]
    if not calendar_match and not data.get("compare_policy"):
        raise ValueError("reconstructed calendar differs from exact archive frame")
    return w, dict(exact_frame_line=line, time=t, active=sorted(w.active), soft=sorted(w.soft),
                   calendar_read_exact=True, calendar_rebuild_match=calendar_match)


@contextmanager
def admission_trace(w, deadline):
    trace = dict(generation=[], candidates=[], freeflow=[])
    names = ("generate_inbound", "generate_transfer", "generate_final", "validate_plan", "validate_separation")
    originals = {n: getattr(wm, n) for n in names}
    old_screen, old_energy, old_reject = w._freeflow_screen, w._energy, w._reject
    current, screen = [None], [None]
    def tick():
        if time.perf_counter() >= deadline: raise TimeoutError("probe wall budget")
    def generate(name):
        def call(*a, **kw):
            tick(); r = originals[name](*a, **kw)
            entry = dict(operation=name, arguments=plain(a), result=plain(r))
            (trace["generation"] if name == "generate_inbound" else screen[0]["stages"] if screen[0] else []).append(entry)
            return r
        return call
    def validate(p, *a, **kw):
        tick()
        if p.kind == "INBOUND":
            current[0] = dict(slot=p.target_slot, plan=p.to_dict(), separation=[], rejections=[])
            trace["candidates"].append(current[0])
        return originals["validate_plan"](p, *a, **kw)
    def separation(a, b, start, end, **kw):
        tick(); r = originals["validate_separation"](a, b, start, end, **kw)
        if current[0] is not None:
            current[0]["separation"].append(dict(pair=[a.aircraft_id, b.aircraft_id], start=start, end=end, result=plain(r)))
        return r
    def energy(p, start, end):
        tick(); q = old_energy(p, start, end)
        if screen[0] is not None: screen[0]["energy"].append(dict(plan=p.to_dict(), result=plain(q)))
        return q
    def freeflow(p, soc):
        entry = dict(slot=p.target_slot, inbound=p.to_dict(), initial_soc=soc, stages=[], energy=[])
        trace["freeflow"].append(entry); screen[0] = entry
        try:
            r = old_screen(p, soc); entry["result"] = plain(r)
            used = peak = 0.
            for e in entry["energy"]:
                final = e["plan"]["kind"] == "FINAL_NORMAL"
                if not final: used += e["result"]["soc_used"]; peak += e["result"]["peak_seconds"]
                e["soc_after_path"] = soc-used-(e["result"]["soc_used"] if final else 0.)
                e["peak_after_path"] = peak+(e["result"]["peak_seconds"] if final else 0.)
            return r
        finally: screen[0] = None
    def reject(reason, aircraft_id=None, **details):
        if current[0] is not None: current[0]["rejections"].append(dict(reason=reason, aircraft_id=aircraft_id, details=plain(details)))
        return old_reject(reason, aircraft_id, **details)
    for n in names[:3]: setattr(wm, n, generate(n))
    wm.validate_plan, wm.validate_separation = validate, separation
    w._freeflow_screen, w._energy, w._reject = freeflow, energy, reject
    try: yield trace
    finally:
        for n, value in originals.items(): setattr(wm, n, value)
        w._freeflow_screen, w._energy, w._reject = old_screen, old_energy, old_reject


def replay_admission(data, t, key, deadline):
    w, reconstruction = restore(data, t)
    if w.aircraft.pop(key).status != "REROUTED": raise ValueError("target not rerouted")
    with admission_trace(w, deadline) as trace: r = w.admit(key)
    archived = [r["reason"] for r in data["rejections"] if r["time"] == t and r["aircraft_id"] == key]
    reasons = [r["reason"] for r in w.rejections]
    original = [e for e in data["events"] if e["type"] == "rerouted" and e["time"] == t and e["aircraft_id"] == key]
    matched = reasons == archived and len(original) == 1 and r.reason == original[0]["details"]["reason"]
    record = dict(request=key, reconstruction=reconstruction, archived_reasons=archived, actual_reasons=reasons,
                  exact_reason_order_match=matched, result=plain(r), trace=trace)
    if not matched and not data.get("compare_policy"): raise ValueError("admission mismatch: "+json.dumps(record))
    return record


def undo_same_time_normal(w, data, t):
    undone = []
    for e in data["events"]:
        if e["time"] != t or e["type"] != "maneuver_started": continue
        key = e["aircraft_id"]; p = w.active.pop(key)
        if p.kind != "FINAL_NORMAL": raise ValueError("unsupported same-time transition")
        a = w.aircraft[key]
        hold = make_stable_plan(key, p.source_ring, p.source_slot, start_time=t, end_time=t+1000,
                                source_version=a.version, config=w.config)
        s = hold.state(t)
        w.aircraft[key] = replace(a, status=wm.HOLD[p.source_ring], trajectory_hash=hold.trajectory_hash,
                                 p=s.p, v=s.v, a=s.a, jerk=s.jerk)
        w.dependencies.pop(p.trajectory_hash, None)
        undone.append(dict(id=key, plan=p.to_dict()))
    w.slots, w.pads, w._source_clear = w._calendar_stage(w.soft, ())
    return undone


def pad_attempts(data, t, key, deadline):
    rows = []
    for pad in range(6):
        if time.perf_counter() >= deadline: raise TimeoutError("probe wall budget")
        w, reconstruction = restore(data, t)
        undone = undo_same_time_normal(w, data, t) if t == 540. else []
        a = w.aircraft[key]
        blockers = [r.to_dict() for r in w.pads.reservations if r.resource == pad and r.end > t]
        c = wm.generate_final(key, a.ring, a.slot, pad, t, emergency=True, source_version=a.version, config=w.config)
        checks=[]
        original_separation=wm.validate_separation
        def observe_separation(a,b,start,end,**kwargs):
            result=original_separation(a,b,start,end,**kwargs)
            if a.aircraft_id not in ("source_reference","target_reference") and b.aircraft_id not in ("source_reference","target_reference"):
                checks.append(dict(pair=[a.aircraft_id,b.aircraft_id],start=start,end=end,result=plain(result)))
            return result
        wm.validate_separation=observe_separation
        try:
            r = w.submit((c.plan,)) if c.plan is not None and c.status == "FEASIBLE" else None
        finally:
            wm.validate_separation=original_separation
        q=w._energy(c.plan,c.plan.start_time,c.plan.end_time) if c.plan is not None else None
        rows.append(dict(pad=pad, blockers=blockers, generation=plain(c), submit=plain(r),
                         energy=plain(q),predicted_touchdown_soc=a.soc-q.soc_used if q else None,
                         predicted_total_peak_seconds=a.peak_seconds+q.peak_seconds if q else None,
                         continuous_checks=checks,
                         reconstruction=reconstruction, same_time_normal_starts_undone=undone))
    archived = [r["reason"] for r in data["rejections"] if r["time"] == t and r["aircraft_id"] == key]
    reasons = [(r["submit"] or r["generation"])["reason"] for r in rows]
    if archived != reasons and not data.get("compare_policy"): raise ValueError(f"pad reasons mismatch {t}/{key}: {archived}/{reasons}")
    return dict(time=t, aircraft_id=key, archived_reasons=archived, actual_reasons=reasons,
                exact_reason_order_match=archived==reasons, pads=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True); p.add_argument("--output", required=True)
    p.add_argument("--wall-budget", type=float, default=570.)
    p.add_argument("--compare-policy",action="store_true",help="counterfactual archived state queries; never old-rule reproduction or formal evidence")
    args = p.parse_args(argv); output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); deadline = start+args.wall_budget
    record = dict(schema="archive-observation-probe-1", formal_evidence=False, python=sys.version,
                  command=[sys.executable, *sys.argv], status="RUNNING", admissions=[], emergency_pads=[],
                  alternative_reference={"status": "NOT_RUN", "reason": "primary narrowed scope after new user admission-rule authorization; no expanded freeflow search"})
    data = None
    def save(): output.write_text(json.dumps(plain(record), indent=2, allow_nan=False), encoding="utf-8")
    try:
        source_result=json.loads((Path(args.archive).parent/"result.json").read_bytes())
        current=provenance("archive observation probe")
        old_sha=source_result["provenance"]["code_sha256"]
        record["policy_comparison"]={"enabled":args.compare_policy,"legacy_code_sha256":old_sha,
            "current_code_sha256":current["code_sha256"],"new_policy_id":getattr(wm,"POLICY_ID",None),
            "legacy_policy_id":"f67_immediate_admission_departure_pad",
            "scope":"counterfactual exact archived state; no checkpoint/full new-policy episode" if args.compare_policy else "same-source legacy reproduction"}
        if not args.compare_policy and old_sha!=current["code_sha256"]:
            raise ValueError("source code_sha256 differs; old reason reproduction forbidden; use explicit --compare-policy")
        data = read_archive(args.archive)
        data["compare_policy"]=args.compare_policy
        record["archive"] = dict(root=str(data["root"]), input_sha256=data["hashes"])
        for t, key in zip(TIMES[:2], ("req_000001", "req_000008")):
            record["admissions"].append(replay_admission(data, t, key, deadline)); save()
        for t in TIMES[2:]:
            for key in ("req_000009", "req_000010"):
                record["emergency_pads"].append(pad_attempts(data, t, key, deadline)); save()
        record["status"] = "COMPLETED"
    except Exception as exc:
        record.update(status="STOPPED", error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if data is not None:
            record["archive_unchanged"] = {n: digest(data["root"]/n) == h for n,h in data["hashes"].items()}
            if not all(record["archive_unchanged"].values()): record["status"] = "SOURCE_CHANGED"
        record["wall_seconds"] = time.perf_counter()-start; save()
    print(json.dumps({k: record[k] for k in ("status", "wall_seconds")}, ensure_ascii=False))
    return 0 if record["status"] == "COMPLETED" else 1


if __name__ == "__main__": raise SystemExit(main())
