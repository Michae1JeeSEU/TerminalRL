"""Read-only-parent v7 runtime checks; all output belongs to a validation clone.

Run on lab-server with locally frozen --provenance-stress/diagnostic JSONs.
This script never prepares, branches or runs a formal campaign and never runs Git.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from terminal_marl.core.rotating_slots.config import load_service_profile
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink
from terminal_marl.core.rotating_slots.rescue_rules import RESCUE_PROFILE_V6, RESCUE_PROFILE_V7
from terminal_marl.core.rotating_slots.stage2_campaign import (
    DERIVED_RESCUE_FIELDS, METHODS, Stage2Codec, atomic_json, busy_fixture,
    checkpoint_identity, digest, frozen_manifest, load_json, restore_checkpoint,
    save_checkpoint, verify_source_provenance,
)
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def physical_state(world):
    """Exact codec fingerprint excluding emergency classification metadata."""
    aircraft = {}
    for key, a in world.aircraft.items():
        row = a.to_dict()
        row.pop("emergency")
        aircraft[key] = row
    return dict(time=world.time, aircraft=aircraft,
        active=dict(world.active), soft=dict(world.soft), dependencies=world.dependencies,
        slots=world.slots, pads=world.pads, source_clear=world._source_clear,
        energy=world._targeted_energy, costs=world.costs,
        failure_archive=world.failure_archive)


def fingerprint(value):
    return digest(Stage2Codec().encode(value))


def close(world):
    if world is not None and world.state_sink is not None:
        world.state_sink.close()


def energy_facts(world):
    m = world.metrics()
    return {k: m[k] for k in ("energy_j", "energy_kwh", "E_terminal_J", "cost_emergency")}


def aircraft_facts(world, key):
    a = world.aircraft[key]
    return dict(aircraft_id=key, status=a.status, ring=a.ring, slot=a.slot,
        soc=a.soc, emergency=asdict(a.emergency),
        active=world.active[key].trajectory_hash if key in world.active else None,
        soft=world.soft[key].trajectory_hash if key in world.soft else None)


def migrated_identity(old, provenance, world):
    new = checkpoint_identity(provenance, digest(world.manifest.to_dict()))
    retained = lambda value: {k: v for k, v in value.items()
                              if k not in ("source_hash", "rescue_rule_profile")}
    assert retained(old) == retained(new), "validation migration changed physical/configuration identity"
    assert new["rescue_rule_profile"] == RESCUE_PROFILE_V7
    return new


def roundtrip(directory, world, scheduler, identity):
    """Save and restore only the clone, copying its stream again on restore."""
    before = fingerprint(physical_state(world))
    all_aircraft = fingerprint(world.aircraft)
    scheduler_state = fingerprint(scheduler.state_dict())
    checkpoint = directory / "validation_checkpoint.json"
    save_checkpoint(checkpoint, world, scheduler, identity)
    restored = None
    try:
        restored_scheduler = NormalSchedulerCallback(scheduler.scheduler.method)
        restored = restore_checkpoint(checkpoint, restored_scheduler, identity,
                                      stream_copy=directory / "roundtrip_states.jsonl")
        assert fingerprint(physical_state(restored)) == before
        assert fingerprint(restored.aircraft) == all_aircraft
        assert fingerprint(restored_scheduler.state_dict()) == scheduler_state
        return dict(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                    physical_energy_commitments_preserved=True,
                    emergency_metadata_preserved=True, scheduler_state_preserved=True,
                    physical_state_sha256=before, energy=energy_facts(restored))
    finally:
        close(restored)


def clone_0240(parent, output, method, provenance):
    directory = output / parent.name / method
    directory.mkdir(parents=True)
    path = parent / method / "checkpoint_0240.json"
    old = load_json(path)["payload"]["identity"]
    assert old.get("rescue_rule_profile") == RESCUE_PROFILE_V6
    scheduler = NormalSchedulerCallback(method)
    world = None
    try:
        world = restore_checkpoint(path, scheduler, old, stream_copy=directory / "states.jsonl")
        assert world.time == 240.
        before = fingerprint(physical_state(world))
        world.rescue_rule_profile = RESCUE_PROFILE_V7
        for name in DERIVED_RESCUE_FIELDS:
            world.__dict__.pop(name, None)
        identity = migrated_identity(old, provenance, world)
        assert fingerprint(physical_state(world)) == before
        result = roundtrip(directory, world, scheduler, identity)
        return dict(parent_checkpoint=str(path), parent_identity=old, identity=identity,
                    profile_only_migration_preserved_state=True, **result)
    finally:
        close(world)


def inspect_0420(parent, output):
    directory = output / "stress_energy_slack_0420"
    directory.mkdir()
    path = parent / "energy_slack" / "checkpoint_0420.json"
    identity = load_json(path)["payload"]["identity"]
    assert identity.get("rescue_rule_profile") == RESCUE_PROFILE_V6
    scheduler = NormalSchedulerCallback("energy_slack")
    world = None
    try:
        world = restore_checkpoint(path, scheduler, identity, stream_copy=directory / "states.jsonl")
        assert world.time == 420.
        ids = ("initial_inner_08", "req_000000")
        before = {key: aircraft_facts(world, key) for key in ids}
        assert before[ids[0]]["emergency"]["declared"]
        assert ids[0] in world.soft and world.soft[ids[0]].kind == "FINAL_EMERGENCY"
        physical_before = fingerprint(physical_state(world))
        events_before = len(world.events)
        energy_before = energy_facts(world)
        world.rescue_rule_profile = RESCUE_PROFILE_V7
        for name in DERIVED_RESCUE_FIELDS:
            world.__dict__.pop(name, None)
        world._update_emergency()  # Deliberately no advance, handle or normal callback.
        after = {key: aircraft_facts(world, key) for key in ids}
        events = world.events[events_before:]
        e = world.aircraft[ids[1]].emergency
        assert e.declared and e.declaration_time == 420.
        assert e.rescue_reason == "NORMAL_RESERVE_IMPOSSIBLE"
        assert e.state == "EMERGENCY_WAITING" and not e.charged and e.start_time is None
        assert ids[1] not in world.active and ids[1] not in world.soft
        assert before[ids[0]] == after[ids[0]], "existing reserved rescue metadata changed"
        assert fingerprint(physical_state(world)) == physical_before
        assert energy_facts(world) == energy_before
        trigger = [r for r in events if r["type"] == "emergency_triggered" and r["aircraft_id"] == ids[1]]
        declaration = [r for r in events if r["type"] == "emergency_declared" and r["aircraft_id"] == ids[1]]
        assert len(trigger) == len(declaration) == 1
        assert trigger[0]["details"]["deadline_basis"] == "necessary_normal_energy_bound"
        assert declaration[0]["details"]["opportunity_status"] == "UNKNOWN"
        result = dict(time=world.time, viewer_uids={"11": ids[0], "39": ids[1]},
            before=before, after=after, new_events=events, energy_before=energy_before,
            energy_after=energy_facts(world), physical_state_sha256=physical_before,
            resources_before_after_equal=True, energy_before_after_equal=True,
            existing_reserved_rescue_unchanged=True, proved_need_declared_wait=True)
        atomic_json(directory / "facts.json", result)
        return result
    finally:
        close(world)


def medium_once(output, method, provenance):
    directory = output / "fresh_medium_one_callback" / method
    directory.mkdir(parents=True)
    assert provenance["service_profile"] == "service_half_v1"
    assert provenance["scenario_profile"] == "medium24_rate150_v1"
    assert provenance["rescue_rule_profile"] == RESCUE_PROFILE_V7
    scheduler = NormalSchedulerCallback(method)
    world = None
    sink = BufferedStateJSONLSink(directory / "states.jsonl")
    try:
        world = busy_fixture(frozen_manifest("medium24_rate150_v1"), state_sink=sink,
            config=load_service_profile("service_half_v1"),
            scenario_profile="medium24_rate150_v1", rescue_rule_profile=RESCUE_PROFILE_V7)
        assert world.time == 0. and not scheduler.records
        started = time.perf_counter()
        scheduler(world._scheduler_port)  # Exactly one callback; unchanged default budgets.
        wall = time.perf_counter() - started
        assert world.time == 0. and len(scheduler.records) == 1
        row = scheduler.records[0]
        atomic_json(directory / "callback.json", row)
        identity = checkpoint_identity(provenance, digest(world.manifest.to_dict()))
        saved = roundtrip(directory, world, scheduler, identity)
        keys = ("status", "solver_entered", "skip_reason", "hint_objective", "final_objective",
                "hint_certified", "commit_status", "submitted", "planning_budget", "solver")
        return dict(method=method, time=world.time, callback_count=1, callback_wall_s=wall,
                    callback={k: row.get(k) for k in keys}, **saved)
    finally:
        close(world)
        sink.close()


def main():
    base = Path.home() / "training_outputs" / "terminal"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stress-parent", type=Path,
        default=base / "stage2_rescue_opportunity_v6_stress_480s_20261022")
    parser.add_argument("--diagnostic-parent", type=Path,
        default=base / "stage2_rescue_opportunity_v6_diagnostic_480s_20261022")
    parser.add_argument("--provenance-stress", required=True, type=Path)
    parser.add_argument("--provenance-diagnostic", required=True, type=Path)
    parser.add_argument("--medium-callbacks", action="store_true")
    parser.add_argument("--provenance-medium", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    assert output.is_relative_to(base.resolve()) and output != base.resolve(), "output must be below ~/training_outputs/terminal"
    parents = [args.stress_parent.expanduser().resolve(), args.diagnostic_parent.expanduser().resolve()]
    assert all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in parents)
    assert len(set(parents)) == 2
    provenances = [load_json(args.provenance_stress), load_json(args.provenance_diagnostic)]
    if args.medium_callbacks:
        assert args.provenance_medium is not None, "medium callbacks need locally frozen medium provenance"
        medium = load_json(args.provenance_medium)
    else:
        medium = None
    for provenance in provenances + ([medium] if medium is not None else []):
        verify_source_provenance(provenance)  # Byte hashes only; never source_provenance/Git.
        assert provenance["rescue_rule_profile"] == RESCUE_PROFILE_V7
    files = set()
    for parent in parents:
        for method in METHODS:
            path = parent / method / "checkpoint_0240.json"
            files.add(path)
            metadata = load_json(path)["payload"]["sink"]
            assert metadata is not None, "expected streamed parent checkpoint"
            files.add(Path(metadata["path"]).resolve())
    target = parents[0] / "energy_slack" / "checkpoint_0420.json"
    files.add(target)
    files.add(Path(load_json(target)["payload"]["sink"]["path"]).resolve())
    assert all(not p.is_relative_to(output) for p in files)
    before = {str(p): file_hash(p) for p in sorted(files)}
    output.mkdir(parents=True, exist_ok=False)
    facts = dict(schema="stage2_v7_runtime_validation_v1", status="running",
        interpretation="VALIDATION_CLONES_ONLY_NOT_A_FORMAL_BRANCH_OR_RUN",
        script_sha256=file_hash(__file__), source_hash=provenances[0]["source_hash"],
        source_hashes=[p["source_hash"] for p in provenances], parent_files_before=before)
    atomic_json(output / "facts.json", facts)
    try:
        facts["stress_0420"] = inspect_0420(parents[0], output)
        facts["checkpoint_0240_roundtrips"] = [clone_0240(parent, output, method, provenance)
            for parent, provenance in zip(parents, provenances) for method in METHODS]
        if medium is not None:
            facts["medium_callbacks"] = [medium_once(output, method, medium) for method in METHODS]
        facts["status"] = "completed"
    except Exception as exc:
        facts.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        facts["parent_files_after"] = {str(p): file_hash(p) for p in sorted(files)}
        facts["parents_unchanged"] = before == facts["parent_files_after"]
        if not facts["parents_unchanged"]:
            facts.update(status="failed", parent_mutation_detected=True)
        atomic_json(output / "facts.json", facts)
    assert facts["parents_unchanged"], "parent content changed during validation"
    print(f'status={facts["status"]} facts={output / "facts.json"} parents_unchanged=True')


if __name__ == "__main__":
    main()
