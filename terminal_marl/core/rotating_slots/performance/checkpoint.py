"""Exact, fail-closed checkpoints for the frozen rotating-slot executor.

This module may be loaded externally by an isolated R0 process. The codec takes
its classes from that process's core package, never from another core instance.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import base64
import importlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = "ft_rs_checkpoint_v1_0"
POLICY_ID = "upstream_capacity_only_pad_arrival_v1_20261018"
WORLD_FIELDS = frozenset("""config manifest physics_dt scheduler_period cache_enabled
_record_count _last_record_time time aircraft requests upstream_reasons slots pads
active soft dependencies _source_clear _manifest_index _next_heartbeat _initial_settled
_last_reference_fallback failed failure_time events records rejections trajectories
decision_times auto_emergency cpu cache_hits reference_counts minimum_separation costs
pad_busy_seconds""".split())
CACHE_FIELDS = frozenset(("_q_cache", "_energy_cache", "_reference_templates"))
RUNTIME_FIELDS = frozenset(("state_sink", "decision_callback", "_scheduler_port"))
_MIGRATIONS = {}


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointIdentity:
    policy_id: str
    config_hash: str
    profile_hash: str
    physical_config_bytes_sha256: str
    effective_rules_hash: str
    candidate_policy_hash: str
    numerical_policy_hash: str
    implementation_sha256: str
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or self.policy_id != POLICY_ID:
            raise CheckpointError("unsupported checkpoint schema/policy")
        if any(not isinstance(v, str) or not v for v in asdict(self).values()):
            raise CheckpointError("incomplete checkpoint identity")


@dataclass(frozen=True)
class AdvanceCursor:
    original_target: float
    phase: str
    outer_phase: str
    shim_source_sha256: str = ""

    def __post_init__(self):
        if (not math.isfinite(self.original_target) or self.phase not in ("loop_start", "returned")
                or not isinstance(self.outer_phase, str) or not self.outer_phase):
            raise CheckpointError("invalid advance invocation cursor")


@dataclass(frozen=True)
class RestoredCheckpoint:
    world: object
    driver: object
    cursor: AdvanceCursor


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise CheckpointError("unexpected/missing " + label + " fields")


class StateCodec:
    """Whitelist of exact types; tagged pairs retain key types and order."""
    TYPES = {
        "config": ("Config",), "demand": ("ArrivalRequest", "ArrivalManifest"),
        "models": ("KinematicState", "CandidateResult", "ValidationResult"),
        "resources": ("Reservation",), "trajectory": ("Segment", "TrajectoryPlan"),
        "world": ("EmergencyStatus", "AircraftState"),
        "energy": ("VehicleProfile", "EnergyProfile", "PowerResult", "EnergyResult"),
    }

    def __init__(self, core_package="terminal_marl.core.rotating_slots"):
        self.classes = {}
        for module, names in self.TYPES.items():
            imported = importlib.import_module(core_package + "." + module)
            for name in names:
                self.classes[module + "." + name] = getattr(imported, name)
        self.section = importlib.import_module(core_package + ".config").Section
        self.record_json = importlib.import_module(core_package + ".replay")._json
        resource = importlib.import_module(core_package + ".resources")
        self.calendars = {name: getattr(resource, name) for name in
                          ("ResourceCalendar", "SlotCalendar", "PadCalendar")}

    def encode(self, value):
        t = type(value)
        if value is None or t in (bool, int, str):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                raise CheckpointError("NaN is not checkpoint state")
            return {"$": "float", "hex": float(value).hex()}
        if t in (tuple, list):
            return {"$": t.__name__, "items": [self.encode(x) for x in value]}
        if t in (dict, OrderedDict) or t is self.section:
            return {"$": "section" if t is self.section else t.__name__,
                    "items": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        for name, cls in self.calendars.items():
            if t is cls:
                if set(value.__dict__) != {"_entries"}:
                    raise CheckpointError("unknown calendar state")
                return {"$": "calendar", "type": name, "entries": self.encode(value._entries)}
        for name, cls in self.classes.items():
            if t is cls:
                if hasattr(value, "__dict__") and set(value.__dict__) != {f.name for f in fields(cls)}:
                    raise CheckpointError("unknown dataclass state: " + name)
                return {"$": "dataclass", "type": name,
                        "fields": [[f.name, self.encode(getattr(value, f.name))] for f in fields(cls)]}
        raise CheckpointError("unsupported state type: " + t.__module__ + "." + t.__name__)

    def decode(self, value):
        if value is None or type(value) in (bool, int, str):
            return value
        if not isinstance(value, dict):
            raise CheckpointError("untagged checkpoint value")
        tag = value.get("$")
        if tag == "float":
            _keys(value, ("$", "hex"), "float")
            try:
                result = float.fromhex(value["hex"])
            except (TypeError, ValueError) as exc:
                raise CheckpointError("invalid tagged float") from exc
            if math.isnan(result) or result.hex() != value["hex"]:
                raise CheckpointError("noncanonical tagged float")
            return result
        if tag in ("tuple", "list", "dict", "OrderedDict", "section"):
            _keys(value, ("$", "items"), "container")
            if tag in ("tuple", "list"):
                rows = [self.decode(x) for x in value["items"]]
                return tuple(rows) if tag == "tuple" else rows
            rows = [(self.decode(k), self.decode(v)) for k, v in value["items"]]
            result = OrderedDict(rows) if tag == "OrderedDict" else dict(rows)
            if len(result) != len(rows):
                raise CheckpointError("duplicate checkpoint dictionary key")
            return self.section(result) if tag == "section" else result
        if tag == "calendar":
            _keys(value, ("$", "type", "entries"), "calendar")
            if value["type"] not in self.calendars:
                raise CheckpointError("unknown calendar type")
            entries = self.decode(value["entries"])
            reservation = self.classes["resources.Reservation"]
            if type(entries) is not list or any(type(x) is not reservation for x in entries):
                raise CheckpointError("invalid calendar entries")
            result = self.calendars[value["type"]]()
            result._entries = entries
            return result
        if tag == "dataclass":
            _keys(value, ("$", "type", "fields"), "dataclass")
            cls = self.classes.get(value["type"])
            if cls is None or [name for name, _ in value["fields"]] != [f.name for f in fields(cls)]:
                raise CheckpointError("unknown dataclass type/fields")
            try:
                result = cls(**{k: self.decode(v) for k, v in value["fields"]})
            except (ValueError, TypeError) as exc:
                raise CheckpointError("invalid " + value["type"] + ": " + str(exc)) from exc
            # Constructors may validate but must not normalize saved state.
            if self.encode(result) != value:
                raise CheckpointError("dataclass constructor changed checkpoint state")
            return result
        raise CheckpointError("unknown checkpoint tag")


def register_equivalence_migration(migration_id, source, target):
    """Register an explicit already-qualified implementation pair, never semantics."""
    if not migration_id or migration_id in _MIGRATIONS:
        raise CheckpointError("migration ID missing/already registered")
    a, b = asdict(source), asdict(target)
    a.pop("implementation_sha256"); b.pop("implementation_sha256")
    if a != b:
        raise CheckpointError("migration cannot change semantic identity")
    _MIGRATIONS[migration_id] = (source, target)


def _identity(saved, expected, migration_id=None):
    try:
        actual = CheckpointIdentity(**saved)
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid checkpoint identity") from exc
    if actual != expected:
        if migration_id is None or _MIGRATIONS.get(migration_id) != (actual, expected):
            raise CheckpointError("checkpoint identity mismatch/unregistered migration")
    elif migration_id is not None:
        raise CheckpointError("migration not required for identical implementation")
    return actual


def _prefix_hash(path, length):
    h = sha256()
    with Path(path).open("rb") as stream:
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise CheckpointError("record prefix truncated")
            remaining -= len(block)
            h.update(block)
    return h.hexdigest()


def _capture_sink(sink, codec):
    replay = importlib.import_module(type(sink).__module__)
    if type(sink) is not getattr(replay, "BufferedStateJSONLSink", None):
        raise CheckpointError("unsupported state sink")
    if set(sink.__dict__) != {"path", "_stream", "_pending", "_time", "_offset", "_committed_count"}:
        raise CheckpointError("unknown state sink fields")
    if sink._stream.closed:
        raise CheckpointError("closed state sink")
    sink.flush()
    os.fsync(sink._stream.fileno())
    pending = sink._pending
    return {"path": str(sink.path), "offset": sink._offset,
            "prefix_sha256": _prefix_hash(sink.path, sink._offset),
            "pending_b64": None if pending is None else base64.b64encode(pending).decode("ascii"),
            "pending_sha256": None if pending is None else sha256(pending).hexdigest(),
            "time": codec.encode(sink._time), "committed_count": sink._committed_count}


def _driver_state(callback, core_package, codec):
    if callback is None or callback is False:
        return callback
    cls = importlib.import_module(core_package + ".driver").DeterministicDriver
    if type(callback) is not cls or set(callback.__dict__) != {"calls"}:
        raise CheckpointError("only DeterministicDriver callbacks can be resumed")
    return {"calls": codec.encode(callback.calls)}


def capture_checkpoint(world, *, identity, cursor, driver=None):
    """Capture only at a caller-certified natural cut; never advances the world."""
    core = type(world).__module__.rsplit(".", 1)[0]
    cls = importlib.import_module(core + ".world").RotatingSlotWorld
    if type(world) is not cls or set(world.__dict__) != WORLD_FIELDS | CACHE_FIELDS | RUNTIME_FIELDS:
        raise CheckpointError("unknown/missing world fields or world type")
    if (identity.config_hash != world.config.config_hash or identity.profile_hash != world.config.profile_hash
            or identity.physical_config_bytes_sha256 != world.config.raw_hash):
        raise CheckpointError("world configuration identity mismatch")
    if not isinstance(cursor, AdvanceCursor) or cursor.original_target < world.time - 1e-9:
        raise CheckpointError("cursor precedes checkpoint state")
    codec = StateCodec(core)
    effective = world.decision_callback if driver is None else driver
    default = _driver_state(world.decision_callback, core, codec)
    return {"schema": SCHEMA, "kind": "resumable_world", "identity": asdict(identity),
            "cursor": codec.encode(dict(asdict(cursor))),
            "world": codec.encode({k: v for k, v in world.__dict__.items() if k in WORLD_FIELDS}),
            "derived_caches": codec.encode({k: world.__dict__[k] for k in sorted(CACHE_FIELDS)}),
            "default_callback": default, "effective_callback": _driver_state(effective, core, codec),
            "callback_shared": effective is world.decision_callback,
            "state_stream": None if world.state_sink is None else _capture_sink(world.state_sink, codec)}


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        # POSIX directory fsync makes the rename durable; Windows does not
        # expose directory descriptors through os.open.
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_checkpoint(path, payload):
    atomic_write_json(path, {"payload": payload, "payload_sha256": sha256(_json(payload)).hexdigest()})
    return sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(path, *, expected_identity, migration_id=None):
    try:
        document = json.loads(Path(path).read_bytes())
    except (ValueError, OSError) as exc:
        raise CheckpointError("unreadable complete checkpoint") from exc
    _keys(document, ("payload", "payload_sha256"), "checkpoint envelope")
    if sha256(_json(document["payload"])).hexdigest() != document["payload_sha256"]:
        raise CheckpointError("checkpoint content hash mismatch")
    _validate_payload(document["payload"], expected_identity, migration_id)
    return document["payload"]


def _validate_payload(payload, expected, migration_id):
    _keys(payload, ("schema", "kind", "identity", "cursor", "world", "derived_caches",
                    "default_callback", "effective_callback", "callback_shared", "state_stream"), "checkpoint")
    if payload["schema"] != SCHEMA or payload["kind"] != "resumable_world":
        raise CheckpointError("replay/fixture is not a resumable checkpoint")
    _identity(payload["identity"], expected, migration_id)


def _restore_driver(state, core, codec):
    if state is None or state is False:
        return state
    _keys(state, ("calls",), "driver")
    calls = codec.decode(state["calls"])
    if type(calls) is not list or any(type(t) is not float or not math.isfinite(t) for t in calls):
        raise CheckpointError("invalid DeterministicDriver call state")
    result = importlib.import_module(core + ".driver").DeterministicDriver()
    result.calls = calls
    return result


def _stream_bytes(metadata, codec, records):
    _keys(metadata, ("path", "offset", "prefix_sha256", "pending_b64", "pending_sha256", "time", "committed_count"), "stream")
    if type(metadata["offset"]) is not int or metadata["offset"] < 0:
        raise CheckpointError("invalid stream offset")
    if type(metadata["committed_count"]) is not int or metadata["committed_count"] < 0:
        raise CheckpointError("invalid stream count")
    if _prefix_hash(metadata["path"], metadata["offset"]) != metadata["prefix_sha256"]:
        raise CheckpointError("record prefix hash mismatch")
    try:
        pending = None if metadata["pending_b64"] is None else base64.b64decode(metadata["pending_b64"], validate=True)
        timestamp = codec.decode(metadata["time"])
        if pending is not None:
            if sha256(pending).hexdigest() != metadata["pending_sha256"] or not pending.endswith(b"\n"):
                raise CheckpointError("pending frame hash mismatch")
            frame = json.loads(pending)
            if (frame["time"] != timestamp or len(records) != 1
                    or pending != codec.record_json(records[-1]) + b"\n"):
                raise CheckpointError("pending latest frame/world mismatch")
        elif metadata["pending_sha256"] is not None or timestamp is not None:
            raise CheckpointError("incomplete pending frame")
    except (ValueError, TypeError, KeyError) as exc:
        raise CheckpointError("invalid pending frame") from exc
    # Verify the offset is an actual frame boundary and the saved count agrees
    # with the immutable prefix. A valid hash alone cannot establish either.
    count, previous, consumed = 0, None, 0
    with Path(metadata["path"]).open("rb") as stream:
        while consumed < metadata["offset"]:
            line = stream.readline(metadata["offset"] - consumed)
            if not line.endswith(b"\n"):
                raise CheckpointError("stream offset is not a frame boundary")
            consumed += len(line)
            try:
                frame_time = json.loads(line)["time"]
            except (ValueError, KeyError, TypeError) as exc:
                raise CheckpointError("invalid stream prefix frame") from exc
            if (type(frame_time) not in (int, float) or not math.isfinite(frame_time)
                    or previous is not None and frame_time <= previous):
                raise CheckpointError("stream prefix time order mismatch")
            previous = frame_time
            count += 1
    if count != metadata["committed_count"] or (previous is not None and
            (timestamp is None or timestamp <= previous)):
        raise CheckpointError("stream prefix count/latest time mismatch")
    return pending, timestamp


def restore_checkpoint(payload, *, expected_identity, cache_mode="cold", new_attempt_dir=None,
                       world_class=None, migration_id=None):
    _validate_payload(payload, expected_identity, migration_id)
    if cache_mode not in ("cold", "warm"):
        raise CheckpointError("cache_mode must be cold or warm")
    if world_class is None:
        from ..world import RotatingSlotWorld
        world_class = RotatingSlotWorld
    core = world_class.__module__.rsplit(".", 1)[0]
    if world_class is not importlib.import_module(core + ".world").RotatingSlotWorld:
        raise CheckpointError("unsupported restore world class")
    codec = StateCodec(core)
    state = codec.decode(payload["world"])
    caches = codec.decode(payload["derived_caches"])
    if type(state) is not dict or set(state) != WORLD_FIELDS or set(caches) != CACHE_FIELDS:
        raise CheckpointError("world/cache field inventory mismatch")
    config = state["config"]
    if (config.config_hash != expected_identity.config_hash or config.profile_hash != expected_identity.profile_hash
            or config.raw_hash != expected_identity.physical_config_bytes_sha256):
        raise CheckpointError("restored configuration identity mismatch")
    config_module = importlib.import_module(core + ".config")
    if (config_module.digest(config.to_dict()) != config.config_hash
            or config_module.digest({k: config.to_dict()[k] for k in ("vehicle", "energy", "coordinates")}) != config.profile_hash):
        raise CheckpointError("configuration content hash mismatch")
    plan_class = codec.classes["trajectory.TrajectoryPlan"]
    try:
        if type(state["trajectories"]) is not dict:
            raise CheckpointError("invalid authoritative trajectory archive")
        for key, archive_payload in state["trajectories"].items():
            plan = plan_class.from_dict(archive_payload, config=config)
            if plan.trajectory_hash != key or plan.config_hash != config.config_hash:
                raise CheckpointError("authoritative archive plan identity mismatch")
        for name in ("active", "soft"):
            for aircraft_id, plan in state[name].items():
                if (type(plan) is not plan_class or plan.aircraft_id != aircraft_id
                        or plan.config_hash != config.config_hash or plan.profile_hash != config.profile_hash
                        or state["trajectories"].get(plan.trajectory_hash) != plan.to_dict()):
                    raise CheckpointError("active/soft plan archive identity mismatch")
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError("invalid authoritative trajectory archive: " + str(exc)) from exc
    try:
        cursor = AdvanceCursor(**codec.decode(payload["cursor"]))
    except (TypeError, ValueError) as exc:
        raise CheckpointError("invalid checkpoint cursor") from exc
    if state["time"] > cursor.original_target + 1e-9:
        raise CheckpointError("advance cursor precedes state")
    default = _restore_driver(payload["default_callback"], core, codec)
    effective = _restore_driver(payload["effective_callback"], core, codec)
    if type(payload["callback_shared"]) is not bool:
        raise CheckpointError("invalid callback binding")
    if payload["callback_shared"]:
        if payload["default_callback"] != payload["effective_callback"]:
            raise CheckpointError("shared callback state mismatch")
        effective = default
    # Validate every piece before constructing runtime objects or writing files.
    metadata = payload["state_stream"]
    pending_time = None if metadata is None else _stream_bytes(metadata, codec, state["records"])
    if metadata is not None and new_attempt_dir is None:
        raise CheckpointError("stream restoration requires a NEW attempt directory")
    world = world_class.__new__(world_class)
    world.__dict__.update(state)
    world.__dict__.update(caches if cache_mode == "warm" else
                          {"_q_cache": OrderedDict(), "_energy_cache": OrderedDict(), "_reference_templates": {}})
    world.decision_callback = default
    world._scheduler_port = importlib.import_module(core + ".world").SchedulerPort(world)
    world.state_sink = None
    if metadata is not None:
        root = Path(new_attempt_dir).resolve()
        source = Path(metadata["path"]).resolve()
        if root == source.parent or root in source.parents or source.parent in root.parents:
            raise CheckpointError("restoration must not modify the source attempt")
        root.mkdir(parents=True, exist_ok=False)
        target = root / "states.jsonl"
        sink_class = importlib.import_module(core + ".replay").BufferedStateJSONLSink
        sink = sink_class(target)
        with source.open("rb") as original:
            remaining = metadata["offset"]
            while remaining:
                block = original.read(min(1024 * 1024, remaining))
                if not block:
                    sink.close()
                    raise CheckpointError("source prefix changed during copy")
                sink._stream.write(block); remaining -= len(block)
        sink._offset = metadata["offset"]
        sink._pending, sink._time = pending_time
        sink._committed_count = metadata["committed_count"]
        sink.flush(); os.fsync(sink._stream.fileno())
        if _prefix_hash(target, sink._offset) != metadata["prefix_sha256"]:
            sink.close()
            raise CheckpointError("copied prefix hash mismatch")
        world.state_sink = sink
    return RestoredCheckpoint(world, effective, cursor)


class CheckpointStore:
    """Atomic complete checkpoints: latest two, recent eight corpus cuts, pins."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "checkpoint_index.json"
        self.entries = json.loads(self.index_path.read_bytes())["entries"] if self.index_path.exists() else []

    def save(self, payload, *, pinned=False, corpus=False, milestone=None):
        sequence = max((row["sequence"] for row in self.entries), default=-1) + 1
        name = f"checkpoint-{sequence:08d}.json"
        digest = write_checkpoint(self.directory / name, payload)
        entries = self.entries + [{"sequence": sequence, "file": name, "sha256": digest,
                                   "pinned": bool(pinned or milestone is not None), "corpus": bool(corpus),
                                   "milestone": milestone}]
        keep = {x["sequence"] for x in entries[-2:]}
        keep.update(x["sequence"] for x in [x for x in entries if x["corpus"]][-8:])
        keep.update(x["sequence"] for x in entries if x["pinned"])
        retained = [x for x in entries if x["sequence"] in keep]
        atomic_write_json(self.index_path, {"schema": SCHEMA, "entries": retained})
        self.entries = retained
        for row in entries:
            if row["sequence"] not in keep:
                (self.directory / row["file"]).unlink()
        return self.directory / name
