"""Simulation-time review delivery and observed actual maneuver starts."""
from dataclasses import asdict, dataclass
import math


def dependency(snapshot, aid):
    a = snapshot.aircraft[aid]
    e=getattr(a,"emergency",None)
    return (a.version, a.ring, a.slot, a.status,
            getattr(snapshot.active.get(aid), "trajectory_hash", None),
            getattr(snapshot.soft.get(aid), "trajectory_hash", None),
            tuple(getattr(e,key,None) for key in ("eligible","declared","pending","charged","state")))


@dataclass
class AgeEntry:
    origin: float | None = None
    reviewed: float | None = None
    progressed: float | None = None
    last_started_hash: str | None = None
    review_origin_known: bool = True


class PlanningHistory:
    def __init__(self): self.entries = {}; self.pending = {}; self.last_observed_time = None

    def observe(self, snapshot, now=None):
        """Only pass real revealed snapshots, never projected snapshots."""
        now = snapshot.time if now is None else now
        if getattr(snapshot, "planning_timing", None) is not None: raise ValueError("projected state cannot update actual ages")
        self.last_observed_time = now
        for aid, a in snapshot.aircraft.items():
            origin = getattr(a, "admission_time", None)
            if origin is None: origin = getattr(a, "release_time", None)
            if origin is not None and (not math.isfinite(origin) or origin > now): origin = None
            entry = self.entries.setdefault(aid, AgeEntry(origin=origin))
            # The executed energy ledger retains completed maneuver identities,
            # including starts and finishes between normal planning tasks.
            executed={r["trajectory_hash"] for r in snapshot.energy_ledger.get("executed",{}).get(aid,()) if r["end"]<=now and r["end"]>r["start"]}
            for h in executed:
                raw=snapshot.trajectories.get(h)
                if raw is not None and raw.get("kind") in ("TRANSFER","FINAL_NORMAL") and raw["start_time"]<=now:
                    if entry.progressed is None or raw["start_time"]>entry.progressed:
                        entry.progressed=raw["start_time"]; entry.last_started_hash=h
            p = snapshot.active.get(aid)
            if p is not None and p.start_time <= now and p.kind in ("TRANSFER", "FINAL_NORMAL") and p.trajectory_hash != entry.last_started_hash:
                entry.progressed = p.start_time
                entry.last_started_hash = p.trajectory_hash

    def ages(self, aid, now):
        e = self.entries.get(aid, AgeEntry())
        review = e.reviewed if e.reviewed is not None else e.origin if e.review_origin_known else None
        progress = e.progressed if e.progressed is not None else e.origin
        return dict(review_age=None if review is None else max(0., now-review),
                    progress_age=None if progress is None else max(0., now-progress),
                    review_unknown=review is None, progress_unknown=progress is None)

    def queue_reviews(self, key, source, aircraft_ids):
        self.pending[key] = {aid: dependency(source, aid) for aid in aircraft_ids if aid in source.aircraft}

    def initialize_imported(self, snapshot, events=()):
        """Old checkpoints lack delivered per-aircraft review evidence."""
        self.observe(snapshot)
        for entry in self.entries.values(): entry.review_origin_known=False
        for event in events:
            if event.get("type")!="maneuver_started" or event.get("time",snapshot.time+1)>snapshot.time: continue
            details=event.get("details",{})
            if details.get("maneuver_kind") not in ("TRANSFER","FINAL_NORMAL"): continue
            entry=self.entries.get(event.get("aircraft_id"))
            if entry is not None and (entry.progressed is None or event["time"]>entry.progressed):
                entry.progressed=event["time"]; entry.last_started_hash=details.get("trajectory_hash")

    def deliver(self, key, snapshot, now):
        updates = []
        for aid, expected in self.pending.pop(key, {}).items():
            if aid not in snapshot.aircraft: continue
            expected=(*expected[:-1],tuple(expected[-1]))
            actual = dependency(snapshot, aid)
            valid = actual == tuple(expected)
            if not valid:
                version, ring, slot, status, active, soft, emergency = expected
                known = {h for h in (active, soft) if h is not None}
                actual_hashes = {h for h in actual[4:6] if h is not None}
                last = snapshot.trajectories.get(soft or active)
                valid = bool(known and actual[-1]==emergency and actual_hashes <= known and version<=actual[0]<=version+len(known) and
                    (actual_hashes or (last is not None and last["end_time"] <= now and
                     (actual[1], actual[2]) == (last["target_ring"], last["target_slot"]))))
            if valid:
                self.entries.setdefault(aid, AgeEntry()).reviewed = now
                updates.append(aid)
        return tuple(updates)

    def state_dict(self):
        return dict(entries={k:asdict(v) for k,v in self.entries.items()}, pending=self.pending, last_observed_time=self.last_observed_time)

    def load_state_dict(self, state):
        self.entries={k:AgeEntry(**v) for k,v in state.get("entries", {}).items()}
        self.pending={k:{aid:tuple(v) for aid,v in rows.items()} for k,rows in state.get("pending", {}).items()}
        self.last_observed_time=state.get("last_observed_time")
