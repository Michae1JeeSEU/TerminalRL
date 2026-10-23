"""Content-verified FT-RS bundles as bounded, renderer-neutral console frames.

This module queries recorded states and authoritative plans. It never creates
a world, steps a simulation, imports VTK, or interpolates browser poses.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from ..core.rotating_slots.replay import ReplayBundle, _plain
from .rotating_slots_controller import (
    ReplayController, _ACTIVE_MANEUVER, _FROZEN, _GROUNDED, _INVISIBLE,
    _UPSTREAM_POLICIES, _separation_evidence,
)

MAX_FUTURE_POINTS = 24
MAX_TERMINAL_EVENTS = 10000
_TERMINAL_OUTCOMES = {
    "energy_failure": "failed_energy", "aircraft_energy_failure": "failed_energy",
    "energy_mission_failed": "failed_energy", "rerouted": "rerouted",
    "touchdown": "touchdown", "pad_service_completed": "completed",
    "executed_hard_separation_violation": "separation_violation",
}


class _ConsoleReplayController(ReplayController):
    """Reuse immutable same-time queries during one frame conversion only."""
    def begin_frame(self, timestamp):
        self.time = timestamp
        self._frame_aircraft = None
        self._frame_pads = None

    def aircraft_at(self):
        if self._frame_aircraft is None:
            self._frame_aircraft = super().aircraft_at()
        return self._frame_aircraft

    def pads_at(self):
        if self._frame_pads is None:
            self._frame_pads = super().pads_at()
        return self._frame_pads


def _snapshot_plain(value):
    if is_dataclass(value):
        return {field.name: _snapshot_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _snapshot_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot_plain(item) for item in value]
    return value


def _terminal_rows(bundle, controller):
    events = [event for event in bundle.data["events"]
              if str(event["type"]).lower() in _TERMINAL_OUTCOMES]
    if len(events) > MAX_TERMINAL_EVENTS:
        raise ValueError(f"rotating_slots replay exceeds {MAX_TERMINAL_EVENTS} terminal events")
    result = []
    for event in sorted(events, key=lambda row: float(row["time"])):
        timestamp = float(event["time"])
        identity = str(event.get("aircraft_id"))
        uid = controller.uid_by_id.get(identity)
        observed = next((a for a in bundle.data["states"][max(0, bisect_right(controller._sample_times, timestamp)-1)]
                         .get("aircraft", ()) if str(a["id"]) == identity), {})
        details = event.get("details") or {}
        soc = details.get("soc", observed.get("soc"))
        result.append({"uid": uid, "arrival_id": uid if uid is not None else -1,
            "aircraft_id": identity, "outcome": _TERMINAL_OUTCOMES[str(event["type"]).lower()],
            "event_type": str(event["type"]), "reason": str(details.get("reason", "")),
            "time": timestamp, "position": _plain(details.get("position", details.get("p", observed.get("p")))),
            "soc": soc, "soc_source": "recorded_event" if "soc" in details else "recorded_sample" if soc is not None else "unavailable",
            "energy_remaining": soc * controller.config.energy.derived_equivalent_capacity_kWh * 3.6e6 if soc is not None else None,
            "phase": str(observed.get("status", ""))})
    return result


def load_rotating_slots_replay(path, *, max_frames=6000, max_bytes=256*1024*1024):
    """Load all original sample times, rejecting bounds rather than truncating."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    index_path = root / "replay.json"
    if index_path.stat().st_size > max_bytes:
        raise ValueError("rotating_slots replay index exceeds size limit")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    names = index.get("files", {})
    if not isinstance(names, dict) or any(Path(name).name != name for name in names):
        raise ValueError("invalid rotating_slots replay member names")
    total_bytes = sum((root / name).stat().st_size for name in names) + len(index_bytes)
    if total_bytes > max_bytes:
        raise ValueError(f"rotating_slots replay members exceed {max_bytes} bytes")
    archive = ReplayBundle.validate(root)
    if archive.state_count > max_frames:
        raise ValueError(f"rotating_slots replay contains {archive.state_count} frames; maximum is {max_frames}; no frames were dropped")
    bundle = archive.open()
    controller = _ConsoleReplayController(bundle)
    controller.set_trajectories("hidden")
    terminal = _terminal_rows(bundle, controller)
    terminal_times = [row["time"] for row in terminal]
    dispatch = sorted(bundle.data.get("dispatch_timings", ()), key=lambda row: float(row["time"]))
    dispatch_times = [float(row["time"]) for row in dispatch]
    episode_metrics = bundle.data.get("metrics", {})
    separation = _separation_evidence(episode_metrics)
    separation["scope"] = "whole_episode"
    methods = sorted({str(row["method"]) for row in bundle.data.get("decisions", ()) if row.get("method")})
    provenance = bundle.data.get("stage2_provenance", {})
    method = ", ".join(methods) or index.get("method") or provenance.get("method") or "unrecorded"
    scenario = root.parent.parent.name
    frames = []
    serialized_bytes = 2  # JSON frame-array brackets.
    for frame_index, observed in enumerate(bundle.data["states"]):
        controller.begin_frame(float(observed["time"]))
        frame = _snapshot_plain(controller.snapshot(terminal_events=False))
        aircraft = {str(row["id"]): row for row in observed.get("aircraft", ())}
        for uav in frame["uavs"]:
            identity = controller.id_by_uid[uav["uid"]]
            row = aircraft[identity]
            status = str(row.get("status", ""))
            uav.update(aircraft_id=identity, status=status, slot_index=row.get("slot"),
                       battery_power_w=row.get("battery_power_w"))
            plan = controller._plans.get(row.get("trajectory_hash"))
            if (uav["committed"] and status in _ACTIVE_MANEUVER and plan and plan.kind not in ("STABLE", "GROUND")
                    and plan.start_time <= controller.time < plan.end_time):
                uav["trajectory_points"] = [list(plan.state(controller.time + (plan.end_time-controller.time)*i/(MAX_FUTURE_POINTS-1)).p)
                                            for i in range(MAX_FUTURE_POINTS)]
        policy = observed.get("policy_id", observed.get("metrics", {}).get("policy_id", episode_metrics.get("policy_id")))
        upstream = observed.get("metrics", {}).get("upstream_waiting", len(observed.get("upstream", ())))
        latest = bisect_right(dispatch_times, controller.time)-1
        frame.update(console_replay_kind="rotating_slots", interpolation="none", step_count=frame_index,
            active_count=sum(row.get("status") not in _GROUNDED | _FROZEN | _INVISIBLE for row in aircraft.values()),
            external_queue_size=upstream,
            terminal_events=terminal[:bisect_right(terminal_times, controller.time)],
            rotating_slots={"slots": _plain(controller.slots_at()), "pads": _plain(controller.pads_at()),
                "upstream_waiting_count": upstream, "policy_id": policy,
                "latest_dispatch_timing": _plain(dispatch[latest]) if latest >= 0 else None,
                "separation_evidence": separation,
                "admission_model": "capacity-only reroute; unsafe requests wait upstream (external requests, no physical aircraft / energy model)"
                    if policy in _UPSTREAM_POLICIES else "immediate admission or reroute; no upstream waiting queue recorded"
                    if policy is None else "Recorded admission policy: " + str(policy)})
        # Exclude booleans and unavailable/non-finite metrics from numeric telemetry.
        frame["telemetry"] = {key: value for key, value in frame["telemetry"].items()
                              if type(value) in (int, float) and math.isfinite(value)}
        serialized_bytes += len(json.dumps(frame, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) + bool(frames)
        if serialized_bytes > max_bytes:
            raise ValueError(f"converted rotating_slots replay exceeds {max_bytes} bytes; no frames were dropped")
        frames.append(frame)
    title = f"{scenario} / {method}"
    metadata = {"source": "rotating_slots", "replay_kind": "rotating_slots", "interpolation": "none",
        "mode": "replay", "scenario": scenario, "method": method, "methods": methods,
        "method_source": "recorded_decisions" if methods else "recorded_index_or_provenance" if method != "unrecorded" else "unavailable",
        "seed": index.get("seed"), "code_commit": index.get("code_commit"),
        "config_hash": index.get("config_hash"), "config_bytes_sha256": index.get("config_bytes_sha256"),
        "replay_index_sha256": sha256(index_bytes).hexdigest(), "sample_dt_s": index.get("sample_dt_s"),
        "telemetry_dt_s": index.get("telemetry_dt_s"), "time_bounds": list(bundle.time_bounds),
        "frame_count": len(frames), "source_path": str(root.resolve()), "content_verified": True,
        "step_count_semantics": "recorded_frame_index",
        "pose_source": index.get("pose_source"), "telemetry_source": index.get("telemetry_source"),
        "upstream_model": "external unresolved requests; excluded from physical aircraft and battery model",
        "failure_model": "recorded isolated frozen markers; abstract isolation without accident cascade",
        "terminal_event_semantics": "touchdown and pad_service_completed are distinct; completed_count is service clearance",
        "terminal_soc_semantics": "recorded event SOC or preceding recorded state; unavailable is null",
        "separation_evidence_scope": "whole_episode", "prediction_authority": index.get("prediction_authority"),
        "experiment_policy_id": index.get("experiment_policy_id"),
        "scenario_profile": provenance.get("scenario_profile"), "service_profile": provenance.get("service_profile"),
        "emergency_policy_id": provenance.get("emergency_policy_id")}
    envelope_bytes = len(json.dumps({"title": title, "frames": [], "metadata": metadata}, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")) - 2
    if serialized_bytes + envelope_bytes > max_bytes:
        raise ValueError(f"converted rotating_slots replay exceeds {max_bytes} bytes; no frames were dropped")
    return title, frames, metadata
