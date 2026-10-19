"""Content-verified replay archive; recording never imports the legacy world.

Positions at arbitrary times are evaluated by the serialized authoritative plan.
Sampled telemetry is retained as observed, never visually smoothed or regenerated.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from collections.abc import Mapping

from .config import DEFAULT_CONFIG, freeze, load_config


def _plain(value):
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class BufferedStateJSONLSink:
    """One-frame buffer with replacement of the current timestamp.

    ``flush`` may be called during a run. The latest line retains its file
    offset, so a later update at that same time replaces it rather than adding
    a stale or duplicate frame. Earlier timestamps are never revisited.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x+b")
        self._pending = None
        self._time = None
        self._offset = 0
        self._committed_count = 0

    @property
    def frame_count(self):
        return self._committed_count + (self._pending is not None)

    def __call__(self, frame):
        timestamp = float(frame["time"])
        if not math.isfinite(timestamp):
            raise ValueError("record time must be finite")
        if self._stream.closed:
            raise ValueError("recording sink is closed")
        if self._time is not None and timestamp < self._time:
            raise ValueError("record times must not decrease")
        if self._pending is not None and timestamp > self._time:
            self.flush()
            self._offset = self._stream.tell()
            self._committed_count += 1
        self._time = timestamp
        self._pending = _json(frame) + b"\n"

    def flush(self):
        if self._stream.closed:
            return
        if self._pending is not None:
            self._stream.seek(self._offset)
            self._stream.write(self._pending)
            self._stream.truncate()
        self._stream.flush()

    def close(self):
        if not self._stream.closed:
            self.flush()
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_hash(source, target):
    digest = sha256()
    with Path(source).open("rb") as src, Path(target).open("xb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(block)
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayArchive:
    """Verified archive reference without loading the state history into RAM."""
    path: Path
    index: Mapping
    state_count: int
    time_bounds: tuple[float, float]

    def open(self):
        """Opt in to eager history loading for local interactive observation."""
        return ReplayBundle.load(self.path)


@dataclass(frozen=True)
class ReplayBundle:
    path: Path
    index: Mapping
    data: Mapping

    @classmethod
    def write(cls, path, *, world=None, record=None, config_path=DEFAULT_CONFIG,
              code_commit, assertions=(), command="", seed=None, eager=None):
        """Write a new archive from ``world.to_dict()`` or the same plain record.

        Existing archives are never overwritten. Human acceptance always begins
        PENDING; screenshots/observations are added by the separate viewer step.
        With ``states_jsonl`` metadata, state bytes are copied and hashed in
        bounded chunks; the default return is a lightweight ReplayArchive.
        Set ``eager=True`` only when interactive history loading is desired.
        """
        if world is not None and record is not None:
            raise ValueError("provide world or record, not both")
        source = _plain(world.to_dict() if world is not None else record)
        streamed = isinstance(source, dict) and bool(source.get("states_jsonl"))
        if isinstance(source, dict) and source.get("states_streamed") and not streamed:
            raise ValueError("complete streamed states require a states_jsonl path")
        if not isinstance(source, dict) or not (source.get("states") or streamed):
            raise ValueError("replay requires recorded states")
        if not code_commit:
            raise ValueError("code_commit is required")
        config_bytes = Path(config_path).read_bytes()
        config = load_config(config_path)
        if source.get("config_hash", config.config_hash) != config.config_hash:
            raise ValueError("record/config hash mismatch")
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        payloads = {
            "config.yaml": config_bytes,
            "manifest.json": _json(source.get("manifest", {})),
            "trajectories.json": _json(source.get("trajectories", {})),
            "metrics.json": _json(source.get("metrics", {})),
            "assertions.json": _json(assertions),
            "visual_acceptance.json": _json({"researcher_visual_acceptance": "PENDING",
                "observer": None, "observed_at": None, "observations": [],
                "visualization_integration_status": "NOT_OBSERVED"}),
        }
        for name in ("states", "events", "rejections"):
            if name == "states" and streamed:
                continue
            rows = source.get(name, [])
            payloads[name + ".jsonl"] = b"".join(_json(row) + b"\n" for row in rows)
        for name, content in payloads.items():
            (root / name).write_bytes(content)
        hashes = {name: sha256(content).hexdigest() for name, content in payloads.items()}
        if streamed:
            hashes["states.jsonl"] = _copy_hash(source["states_jsonl"], root / "states.jsonl")
        (root / "screenshots").mkdir()
        index = {"schema": "FT-RS-S1-1.0-ReplayBundle", "code_commit": code_commit,
                 "seed": seed, "command": command, "config_hash": config.config_hash,
                 "config_bytes_sha256": sha256(config_bytes).hexdigest(),
                 "trajectory_hashes": sorted(source.get("trajectories", {})),
                 "files": hashes,
                 "pose_source": "TrajectoryPlan.state(t)", "telemetry_source": "recorded samples and exact events",
                 "sample_dt_s": source.get("sample_dt_s", config.timing.physics_sample_dt_s),
                 "telemetry_dt_s": source.get("telemetry_dt_s", source.get("sample_dt_s", config.timing.physics_sample_dt_s)),
                 "screenshots": [], "human_acceptance": "PENDING"}
        if "states_count" in source:
            index["states_count"] = int(source["states_count"])
        (root / "replay.json").write_bytes(_json(index))
        if eager is True or eager is None and not streamed:
            return cls.load(root)
        return cls.validate(root)

    @classmethod
    def validate(cls, path):
        """Verify hashes/config/plans and state ordering with one frame in RAM."""
        root = Path(path)
        index = json.loads((root / "replay.json").read_bytes())
        if index.get("schema") != "FT-RS-S1-1.0-ReplayBundle":
            raise ValueError("unsupported replay schema; legacy replay is not imported")
        required = {"config.yaml", "manifest.json", "trajectories.json", "states.jsonl", "events.jsonl",
                    "metrics.json", "assertions.json", "visual_acceptance.json", "rejections.jsonl"}
        if not required <= index.get("files", {}).keys():
            raise ValueError("replay is missing required members")
        for name, expected in index["files"].items():
            if Path(name).name != name:
                raise ValueError("invalid replay file name")
            if _file_hash(root / name) != expected:
                raise ValueError("replay content hash mismatch: " + name)
        config = load_config(root / "config.yaml")
        if config.config_hash != index["config_hash"] or config.raw_hash != index["config_bytes_sha256"]:
            raise ValueError("configuration provenance mismatch")
        trajectories = json.loads((root / "trajectories.json").read_bytes())
        cls._validate_plans(trajectories, config)
        count = 0
        first = previous = None
        with (root / "states.jsonl").open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                timestamp = float(frame["time"])
                if not math.isfinite(timestamp) or previous is not None and timestamp <= previous:
                    raise ValueError("replay states must be strictly time ordered")
                if first is None:
                    first = timestamp
                previous = timestamp
                count += 1
        if not count:
            raise ValueError("replay requires recorded states")
        if "states_count" in index and count != index["states_count"]:
            raise ValueError("recorded state count mismatch")
        return ReplayArchive(root.resolve(), freeze(index), count, (first, previous))

    @staticmethod
    def _validate_plans(trajectories, config):
        if trajectories:
            from .trajectory import TrajectoryPlan
            for key, parameters in trajectories.items():
                plan = TrajectoryPlan.from_dict(parameters, config=config)
                if plan.config_hash != config.config_hash:
                    raise ValueError("trajectory/config hash mismatch")
                if plan.trajectory_hash != key:
                    raise ValueError("authoritative trajectory hash mismatch")

    @classmethod
    def load(cls, path):
        root = Path(path)
        index = json.loads((root / "replay.json").read_bytes())
        if index.get("schema") != "FT-RS-S1-1.0-ReplayBundle":
            raise ValueError("unsupported replay schema; legacy replay is not imported")
        payloads = {}
        for name, expected in index["files"].items():
            if Path(name).name != name:
                raise ValueError("invalid replay file name")
            content = (root / name).read_bytes()
            if sha256(content).hexdigest() != expected:
                raise ValueError("replay content hash mismatch: " + name)
            payloads[name] = content
        config = load_config(root / "config.yaml")
        if config.config_hash != index["config_hash"] or config.raw_hash != index["config_bytes_sha256"]:
            raise ValueError("configuration provenance mismatch")
        data = {}
        for name, content in payloads.items():
            if name.endswith(".jsonl"):
                data[name[:-6]] = [json.loads(line) for line in content.splitlines() if line]
            elif name.endswith(".json"):
                data[name[:-5]] = json.loads(content)
        states = data["states"]
        if not states or any(not math.isfinite(float(s["time"])) for s in states) or any(
                float(a["time"]) >= float(b["time"]) for a, b in zip(states, states[1:])):
            raise ValueError("replay states must be nonempty and time ordered")
        if "states_count" in index and len(states) != index["states_count"]:
            raise ValueError("recorded state count mismatch")
        # Plan hashes are checked using the authoritative implementation, in
        # addition to the archive file hashes; a renamed plan is not accepted.
        cls._validate_plans(data["trajectories"], config)
        return cls(root.resolve(), freeze(index), freeze(data))

    @property
    def config(self):
        return load_config(self.path / "config.yaml")

    def sampled_state(self, time):
        states = self.data["states"]
        times = [float(row["time"]) for row in states]
        return states[max(0, bisect_right(times, float(time)) - 1)]

    @property
    def time_bounds(self):
        return float(self.data["states"][0]["time"]), float(self.data["states"][-1]["time"])

    def plan(self, trajectory_hash):
        from .trajectory import TrajectoryPlan
        return TrajectoryPlan.from_dict(_plain(self.data["trajectories"][trajectory_hash]), config=self.config)

    def curves(self, aircraft_id):
        """Return logged values only; missing power is explicit, not recomputed."""
        rows = []
        for frame in self.data["states"]:
            for aircraft in frame.get("aircraft", ()):
                if str(aircraft["id"]) != str(aircraft_id):
                    continue
                norm = lambda value: sum(float(x) ** 2 for x in value) ** .5
                rows.append({"time": frame["time"], "speed": norm(aircraft.get("v", (0, 0, 0))),
                    "acceleration": norm(aircraft.get("a", (0, 0, 0))),
                    "jerk": norm(aircraft.get("jerk", (0, 0, 0))),
                    "battery_power": aircraft.get("battery_power_w"), "SOC": aircraft["soc"]})
        return tuple(freeze(row) for row in rows)
