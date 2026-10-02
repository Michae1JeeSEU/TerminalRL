"""Local web console backend for terminal-world visualization.

The module intentionally keeps the HTTP/static path free of training and
renderer imports.  A rollout, when requested, runs in a spawned child
process.  That process emits plain ``WorldRenderSnapshot`` dictionaries to a
bounded queue; the parent collector owns the replay buffer and serves it over
small, non-blocking HTTP requests.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hmac
import json
import math
import multiprocessing as mp
import re
import secrets
from pathlib import Path
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser


VERSION = "1"
DEFAULT_PORT = 8787
DEFAULT_SCENARIO = "fixed_rule_warm_flow"
DEFAULT_MODE = "fixed_rule"
DEFAULT_SEED = 0
DEFAULT_MAX_SIMULATION_SECONDS = 600.0
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_START_PAUSED = True
MAX_BODY_BYTES = 64 * 1024
MAX_REPLAY_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 6000
MAX_FRAME_BATCH = 200
FRAME_QUEUE_SIZE = 128
PROCESS_JOIN_SECONDS = 0.75
MAX_MAX_SIMULATION_SECONDS = 7 * 24 * 60 * 60
MAX_PLAYBACK_SPEED = 100.0
WEB_ROOT = Path(__file__).with_name("web")
STATIC_FILES = frozenset({"index.html", "console.css", "console.js", "scene.js"})
STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "console.css": "text/css; charset=utf-8",
    "console.js": "text/javascript; charset=utf-8",
    "scene.js": "text/javascript; charset=utf-8",
}
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

SCENARIOS = (
    {"id": "default", "label": "Default"},
    {"id": "fixed_rule_warm_flow", "label": "Warm flow"},
    {"id": "fixed_rule_dense30_baseline", "label": "Dense 30 · baseline"},
    {"id": "fixed_rule_dense30_coordinated", "label": "Dense 30 · coordinated"},
    {"id": "warm3", "label": "Warm 3"},
    {"id": "flow3", "label": "Flow 3"},
    {"id": "flow6", "label": "Flow 6"},
    {"id": "flow8", "label": "Flow 8"},
    {"id": "flow12", "label": "Flow 12"},
    {"id": "p2_dense24", "label": "P2 dense 24"},
    {"id": "dense24", "label": "P3 dense 24"},
)
MODES = (
    {"id": "fixed_rule", "label": "Fixed rule"},
    {"id": "fixed_low_rl", "label": "Fixed geometry · learned low"},
    {"id": "dynamic_rule", "label": "Dynamic rule"},
    {"id": "learned_high_rule_low", "label": "Learned high · rule low"},
    {"id": "learned_high_frozen_low", "label": "Learned high · frozen low"},
    {"id": "checkpoint", "label": "Checkpoint"},
)
PROFILES = (
    {"id": "checkpoint12", "label": "Checkpoint 12"},
    {"id": "p2_initial_k3", "label": "P2 initial K=3"},
    {"id": "p2_trained_k3", "label": "P2 trained K=3"},
    {"id": "p3_task_neutral", "label": "P3 task-neutral"},
)
SCENARIO_IDS = frozenset(item["id"] for item in SCENARIOS)
MODE_IDS = frozenset(item["id"] for item in MODES)
PROFILE_IDS = frozenset(item["id"] for item in PROFILES)


class ConsoleStopRequested(RuntimeError):
    """Raised inside a rollout when the console requests an immediate stop."""


class ConsoleInputError(ValueError):
    """Raised for malformed or out-of-contract console inputs."""


def _close_ipc_queue(queue: Any) -> None:
    try:
        queue.cancel_join_thread()
    except (AttributeError, AssertionError, OSError, ValueError):
        pass
    try:
        queue.close()
    except (AttributeError, AssertionError, OSError, ValueError):
        pass


def _reject_json_constant(value: str) -> None:
    raise ConsoleInputError(f"non-finite JSON constant is not allowed: {value}")


def _json_loads(raw: bytes | str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_json_constant)
    except ConsoleInputError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConsoleInputError(f"invalid JSON: {error}") from error


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert ordinary Python/NumPy-like scalar containers safely."""
    if depth > 32:
        raise ConsoleInputError("metadata nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConsoleInputError("non-finite value is not serializable")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    # Child results can contain NumPy scalars without importing NumPy here.
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item(), depth=depth + 1)
    raise ConsoleInputError(f"unsupported value type: {type(value).__name__}")


def _sanitize_live_snapshot(value: Any, *, field: str | None = None) -> Any:
    """Make diagnostic-only non-finite telemetry JSON-safe.

    A freshly reset environment reports ``Infinity`` for not-yet-observed
    separation metrics.  Those values are unavailable telemetry, while a
    non-finite position or trajectory would make the scene invalid and is
    therefore surfaced as an error.
    """
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if field == "telemetry":
            return None
        raise ConsoleInputError("non-finite live snapshot geometry")
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_live_snapshot(item, field=str(key) if str(key) == "telemetry" else field)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_live_snapshot(item, field=field) for item in value]
    return value


def _dump_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _finite_number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConsoleInputError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConsoleInputError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ConsoleInputError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ConsoleInputError(f"{name} must be <= {maximum}")
    return number


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsoleInputError(f"{name} must be an integer")
    if value <= 0:
        raise ConsoleInputError(f"{name} must be positive")
    if maximum is not None and value > maximum:
        raise ConsoleInputError(f"{name} must be <= {maximum}")
    return value


def _nonnegative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConsoleInputError(f"{name} must be an integer")
    if value < 0:
        raise ConsoleInputError(f"{name} must be non-negative")
    if maximum is not None and value > maximum:
        raise ConsoleInputError(f"{name} must be <= {maximum}")
    return value


def _validate_point3(value: Any, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConsoleInputError(f"{name} must be a 3D point")
    for index, item in enumerate(value):
        _finite_number(item, f"{name}[{index}]")


def _validate_frame_geometry(frame: Mapping[str, Any]) -> None:
    rings = frame.get("rings")
    if rings is not None:
        for index, ring in enumerate(rings):
            if not isinstance(ring, Mapping):
                raise ConsoleInputError(f"frame.rings[{index}] must be an object")
            for name in ("radius", "height", "target_radius", "target_height"):
                if name in ring:
                    _finite_number(ring[name], f"frame.rings[{index}].{name}")
            for name in ("capacity", "occupancy"):
                if name in ring:
                    _nonnegative_int(ring[name], f"frame.rings[{index}].{name}")
    gates = frame.get("gates")
    if gates is not None:
        for index, gate in enumerate(gates):
            if not isinstance(gate, Mapping):
                raise ConsoleInputError(f"frame.gates[{index}] must be an object")
            for name in ("spawn_position", "gate_position"):
                if name not in gate:
                    raise ConsoleInputError(f"frame.gates[{index}] requires {name}")
                _validate_point3(gate[name], f"frame.gates[{index}].{name}")
    landing_zones = frame.get("landing_zones")
    if landing_zones is not None:
        for index, zone in enumerate(landing_zones):
            if not isinstance(zone, Mapping):
                raise ConsoleInputError(f"frame.landing_zones[{index}] must be an object")
            if "position" not in zone:
                raise ConsoleInputError(f"frame.landing_zones[{index}] requires position")
            _validate_point3(zone["position"], f"frame.landing_zones[{index}].position")
    uavs = frame.get("uavs")
    if uavs is not None:
        for index, uav in enumerate(uavs):
            if not isinstance(uav, Mapping):
                raise ConsoleInputError(f"frame.uavs[{index}] must be an object")
            if "uid" not in uav or "position" not in uav:
                raise ConsoleInputError(f"frame.uavs[{index}] requires uid and position")
            _nonnegative_int(uav["uid"], f"frame.uavs[{index}].uid")
            _validate_point3(uav["position"], f"frame.uavs[{index}].position")
            if "velocity" in uav:
                _validate_point3(uav["velocity"], f"frame.uavs[{index}].velocity")
            if "trajectory_points" in uav:
                if not isinstance(uav["trajectory_points"], list):
                    raise ConsoleInputError(f"frame.uavs[{index}].trajectory_points must be an array")
                for point_index, point in enumerate(uav["trajectory_points"]):
                    _validate_point3(point, f"frame.uavs[{index}].trajectory_points[{point_index}]")
    events = frame.get("terminal_events")
    if events is not None:
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise ConsoleInputError(f"frame.terminal_events[{index}] must be an object")
            if event.get("position") is not None:
                _validate_point3(event["position"], f"frame.terminal_events[{index}].position")


def _optional_text(value: Any, name: str, *, maximum: int = 4096) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ConsoleInputError(f"{name} must be a non-empty short string")
    return value


def _validate_frame(frame: Any, previous_time: float | None) -> float:
    if not isinstance(frame, Mapping):
        raise ConsoleInputError("each frame must be an object")
    if "time" not in frame or "step_count" not in frame:
        raise ConsoleInputError("each frame requires time and step_count")
    current_time = _finite_number(frame["time"], "frame.time", minimum=0.0)
    _nonnegative_int(frame["step_count"], "frame.step_count")
    if previous_time is not None and current_time < previous_time:
        raise ConsoleInputError("frame times must be non-decreasing")
    for name in ("rings", "gates", "landing_zones", "uavs", "terminal_events", "recent_conflict_uids"):
        if name in frame and not isinstance(frame[name], list):
            raise ConsoleInputError(f"frame.{name} must be an array")
    if "telemetry" in frame and not isinstance(frame["telemetry"], Mapping):
        raise ConsoleInputError("frame.telemetry must be an object")
    if "transition_active" in frame and not isinstance(frame["transition_active"], bool):
        raise ConsoleInputError("frame.transition_active must be boolean")
    if "transition_progress" in frame:
        _finite_number(frame["transition_progress"], "frame.transition_progress", minimum=0.0, maximum=1.0)
    for name in ("external_queue_size", "service_count", "completed_count"):
        if name in frame:
            _nonnegative_int(frame[name], f"frame.{name}")
    _validate_frame_geometry(frame)
    _jsonable(frame)
    return current_time


def _validate_frames(frames: Any) -> list[dict[str, Any]]:
    if not isinstance(frames, list):
        raise ConsoleInputError("replay frames must be an array")
    if not frames:
        raise ConsoleInputError("replay contains no frames")
    if len(frames) > MAX_FRAMES:
        raise ConsoleInputError(f"replay contains more than {MAX_FRAMES} frames")
    result: list[dict[str, Any]] = []
    previous_time: float | None = None
    for frame in frames:
        current_time = _validate_frame(frame, previous_time)
        result.append(dict(_jsonable(frame)))
        previous_time = current_time
    return result


def _extract_replay_object(raw: bytes, path: Path) -> dict[str, Any]:
    if len(raw) > MAX_REPLAY_BYTES:
        raise ConsoleInputError(f"replay exceeds {MAX_REPLAY_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConsoleInputError(f"replay is not UTF-8: {error}") from error
    if path.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        value = _json_loads(text)
        if not isinstance(value, Mapping):
            raise ConsoleInputError("replay JSON must be an object")
        return dict(value)
    # Existing snapshot_playback HTML embeds a JSON object in `const data=`.
    match = re.search(r"(?:(?:const|let|var)\s+data|window\.__TERMINAL_REPLAY__)\s*=\s*", text)
    if match is None:
        raise ConsoleInputError("HTML replay does not contain const data")
    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    try:
        value, _ = decoder.raw_decode(text[match.end():])
    except (TypeError, ValueError, json.JSONDecodeError, ConsoleInputError) as error:
        raise ConsoleInputError(f"embedded replay JSON is invalid: {error}") from error
    if not isinstance(value, Mapping):
        raise ConsoleInputError("embedded replay data must be an object")
    return dict(value)


def load_replay(path: str | Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    replay_path = Path(path)
    try:
        size = replay_path.stat().st_size
    except OSError as error:
        raise ConsoleInputError(f"replay cannot be read: {replay_path}") from error
    if size > MAX_REPLAY_BYTES:
        raise ConsoleInputError(f"replay exceeds {MAX_REPLAY_BYTES} bytes")
    try:
        raw = replay_path.read_bytes()
    except OSError as error:
        raise ConsoleInputError(f"replay cannot be read: {replay_path}") from error
    payload = _extract_replay_object(raw, replay_path)
    frames = _validate_frames(payload.get("frames"))
    title = payload.get("title", replay_path.stem)
    if not isinstance(title, str) or not title:
        raise ConsoleInputError("replay title must be a non-empty string")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConsoleInputError("replay metadata must be an object")
    return title, frames, dict(_jsonable(metadata))


def _scenario_is_allowed(mode: str, scenario: str, profile: str | None) -> None:
    if scenario not in SCENARIO_IDS:
        raise ConsoleInputError(f"unsupported scenario: {scenario}")
    if mode == "checkpoint":
        if profile == "p3_task_neutral":
            allowed = {"flow6", "flow12", "dense24"}
        elif profile in {"p2_initial_k3", "p2_trained_k3"}:
            allowed = {"flow6", "flow8", "flow12", "p2_dense24"}
        else:
            allowed = {"warm3", "flow3", "flow6", "flow8", "flow12"}
        if scenario not in allowed:
            raise ConsoleInputError(f"scenario {scenario!r} is not valid for profile {profile!r}")
        return
    if scenario in {"fixed_rule_warm_flow", "fixed_rule_dense30_baseline", "fixed_rule_dense30_coordinated"} and mode != "fixed_rule":
        raise ConsoleInputError(f"{scenario} is available only with fixed_rule")
    if scenario not in {"default", "fixed_rule_warm_flow", "fixed_rule_dense30_baseline", "fixed_rule_dense30_coordinated"}:
        raise ConsoleInputError(f"scenario {scenario!r} is not valid for mode {mode!r}")


def validate_start_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None or not isinstance(payload, Mapping):
        raise ConsoleInputError("start body must be an object")
    mode = payload.get("mode", DEFAULT_MODE)
    if mode not in MODE_IDS:
        raise ConsoleInputError(f"unsupported mode: {mode}")
    profile = payload.get("profile")
    if profile is None and mode == "checkpoint":
        profile = "checkpoint12"
    if profile is not None and profile not in PROFILE_IDS:
        raise ConsoleInputError(f"unsupported profile: {profile}")
    scenario_default = DEFAULT_SCENARIO if mode == DEFAULT_MODE else "default"
    scenario = payload.get("scenario", scenario_default)
    if not isinstance(scenario, str):
        raise ConsoleInputError("scenario must be a string")
    _scenario_is_allowed(mode, scenario, profile)
    seed = payload.get("seed", DEFAULT_SEED)
    seed = _nonnegative_int(seed, "seed", maximum=2**31 - 1)
    max_seconds = _finite_number(
        payload.get("max_simulation_seconds", DEFAULT_MAX_SIMULATION_SECONDS),
        "max_simulation_seconds", minimum=0.001, maximum=MAX_MAX_SIMULATION_SECONDS,
    )
    max_steps_value = payload.get("max_steps")
    max_steps = None if max_steps_value is None else _positive_int(max_steps_value, "max_steps", maximum=10_000_000)
    playback_speed = _finite_number(
        payload.get("playback_speed", DEFAULT_PLAYBACK_SPEED),
        "playback_speed", minimum=0.0, maximum=MAX_PLAYBACK_SPEED,
    )
    start_paused = payload.get("start_paused", DEFAULT_START_PAUSED)
    if not isinstance(start_paused, bool):
        raise ConsoleInputError("start_paused must be boolean")
    low_checkpoint = _optional_text(payload.get("low_checkpoint"), "low_checkpoint")
    high_checkpoint = _optional_text(payload.get("high_checkpoint"), "high_checkpoint")
    if mode in {"fixed_low_rl", "learned_high_frozen_low"} and low_checkpoint is None:
        raise ConsoleInputError("low_checkpoint is required for this mode")
    if mode in {"learned_high_rule_low", "learned_high_frozen_low"} and high_checkpoint is None:
        raise ConsoleInputError("high_checkpoint is required for this mode")
    if mode == "checkpoint" and low_checkpoint is None:
        raise ConsoleInputError("low_checkpoint is required for checkpoint mode")
    deterministic = payload.get("deterministic")
    if profile == "p3_task_neutral":
        if deterministic is None:
            deterministic = False
        if not isinstance(deterministic, bool):
            raise ConsoleInputError("deterministic must be boolean")
    else:
        # Existing profiles retain their deterministic evaluation semantics.
        deterministic = True
    return {
        "mode": mode,
        "scenario": scenario,
        "seed": seed,
        "max_simulation_seconds": max_seconds,
        "max_steps": max_steps,
        "playback_speed": playback_speed,
        "low_checkpoint": low_checkpoint,
        "high_checkpoint": high_checkpoint,
        "start_paused": start_paused,
        "profile": profile,
        "deterministic": deterministic,
    }


def _control_value(payload: Mapping[str, Any]) -> tuple[str, Any]:
    action = payload.get("action")
    if action not in {"pause", "resume", "step", "stop", "speed"}:
        raise ConsoleInputError("action must be pause, resume, step, stop, or speed")
    value = payload.get("value")
    if action == "speed":
        value = _finite_number(value, "value", minimum=0.0, maximum=MAX_PLAYBACK_SPEED)
    return str(action), value


class ConsoleStepObserver:
    """Child-process observer with explicit run/step permits."""

    def __init__(
        self,
        env: Any,
        config: Any,
        run_config: Mapping[str, Any],
        frame_queue: Any,
        status_queue: Any,
        stop_event: Any,
        run_event: Any,
        step_semaphore: Any,
        speed_value: Any,
    ) -> None:
        self._env = env
        self._run_config = dict(run_config)
        self.config = config
        self._frame_queue = frame_queue
        self._status_queue = status_queue
        self._stop_event = stop_event
        self._run_event = run_event
        self._step_semaphore = step_semaphore
        self._speed_value = speed_value
        self._closed = False
        self._error: Exception | None = None
        self._last_step = -1
        self._ready_waited = False
        requested_limit = getattr(env, "_console_step_limit", None)
        if requested_limit is None:
            requested_limit = self._run_config.get("max_steps")
        if requested_limit is None:
            self._base_step_limit: int | None = None
        else:
            interval = float(getattr(getattr(env, "config", None), "high_level_decision_interval_seconds", 1.0))
            dt = float(getattr(getattr(env, "config", None), "dt", 1.0))
            multiplier = max(1, round(interval / dt)) if bool(getattr(getattr(env, "config", None), "high_level_enabled", False)) else 1
            self._base_step_limit = int(requested_limit) * multiplier
        self._emit(env)
        self._status({"kind": "ready"})

    @property
    def paused(self) -> bool:
        return False

    @property
    def playback_speed(self) -> float:
        return 0.0

    @property
    def stop_requested(self) -> bool:
        return self._closed or bool(self._stop_event.is_set())

    @property
    def error(self) -> Exception | None:
        return self._error

    def wait_ready(self, timeout: float = 10.0) -> bool:
        del timeout
        if self._ready_waited:
            return self._error is None
        self._ready_waited = True
        self._wait_for_permission(self._env, pace=False)
        return self._error is None

    def _status(self, payload: Mapping[str, Any]) -> None:
        try:
            self._status_queue.put(dict(payload), timeout=0.2)
        except Exception:
            # The parent may have closed the session while the child is
            # unwinding.  The stop event remains authoritative in that case.
            if not self.stop_requested:
                raise

    def _emit(self, env: Any, *, final: bool = False) -> None:
        from .snapshot import build_world_render_snapshot
        from .snapshot_playback import plain

        snapshot = _sanitize_live_snapshot(plain(build_world_render_snapshot(env, trajectory_max_points=30)))
        message = {"kind": "frame", "frame": snapshot, "final": bool(final)}
        while not self._closed:
            if self._stop_event.is_set():
                raise ConsoleStopRequested("console stop requested")
            try:
                self._frame_queue.put(message, timeout=0.1)
                self._last_step = int(snapshot.get("step_count", self._last_step))
                return
            except Exception:
                if self._stop_event.is_set():
                    raise ConsoleStopRequested("console stop requested")
        raise ConsoleStopRequested("observer closed")

    def _pace(self, env: Any) -> None:
        speed = float(self._speed_value.value)
        if speed <= 0.0:
            return
        dt = float(getattr(getattr(env, "config", None), "dt", 1.0))
        deadline = time.monotonic() + dt / speed
        while True:
            if self._stop_event.is_set():
                raise ConsoleStopRequested("console stop requested")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self._stop_event.wait(min(0.05, remaining))

    @staticmethod
    def _environment_finished(env: Any) -> bool:
        arrivals = getattr(env, "arrivals", None)
        if arrivals is not None and callable(getattr(arrivals, "exhausted", None)):
            queue = getattr(getattr(env, "admission", None), "queue", None)
            queue_items = getattr(queue, "items", ())
            if arrivals.exhausted() and not getattr(env, "active", {}) and not queue_items and not getattr(env, "service_uavs", {}):
                return True
        return float(getattr(env, "now", 0.0)) >= float(getattr(getattr(env, "config", None), "max_simulation_seconds", math.inf))

    def _at_stop_boundary(self, env: Any) -> bool:
        return (
            self._environment_finished(env)
            or (self._base_step_limit is not None and int(getattr(env, "step_count", 0)) >= self._base_step_limit)
        )

    def _wait_for_permission(self, env: Any, *, pace: bool) -> None:
        while not self._closed:
            if self._stop_event.is_set():
                raise ConsoleStopRequested("console stop requested")
            if self._run_event.is_set():
                if pace:
                    self._pace(env)
                    if not self._run_event.is_set():
                        continue
                return
            if self._step_semaphore.acquire(timeout=0.05):
                return
        raise ConsoleStopRequested("observer closed")

    def __call__(self, env: Any) -> None:
        self._env = env
        self._emit(env)
        if not self._at_stop_boundary(env):
            self._wait_for_permission(env, pace=True)

    def finish(self, env: Any, *, wait_for_close: bool = True) -> bool:
        del wait_for_close
        if self._closed or self._stop_event.is_set():
            return False
        if int(getattr(env, "step_count", -1)) == self._last_step:
            return True
        self._emit(env, final=True)
        return True

    def close(self) -> None:
        self._closed = True


def _simulation_worker(
    config: dict[str, Any],
    frame_queue: Any,
    status_queue: Any,
    stop_event: Any,
    run_event: Any,
    step_semaphore: Any,
    speed_value: Any,
) -> None:
    """Spawn target; all project/training imports stay inside this function."""
    observer_factory = lambda env, visual_config: ConsoleStepObserver(
        env, visual_config, config, frame_queue, status_queue, stop_event, run_event, step_semaphore, speed_value
    )
    try:
        if config["mode"] == "checkpoint":
            from .console_policy import run_checkpoint

            result = run_checkpoint(config=config, observer_factory=observer_factory)
            status_queue.put({"kind": "completed", "metadata": _jsonable(_sanitize_live_snapshot(result))})
            return
        from .rollout_viewer import run_rollout

        env = run_rollout(
            mode=config["mode"],
            seed=config["seed"],
            gui=False,
            frame_stride=1,
            playback_speed=0.0,
            show_trajectories=False,
            show_labels=False,
            low_checkpoint=config.get("low_checkpoint"),
            high_checkpoint=config.get("high_checkpoint"),
            max_simulation_seconds=config["max_simulation_seconds"],
            max_steps=config.get("max_steps"),
            scenario=config["scenario"],
            observer_factory=observer_factory,
        )
        result = {
            "time": float(env.now),
            "completed": len(env.completed_uavs),
            "outcomes": dict(env.ledger.outcomes),
            "telemetry": dict(env.telemetry),
        }
        status_queue.put({"kind": "completed", "metadata": _jsonable(_sanitize_live_snapshot(result))})
    except ConsoleStopRequested:
        try:
            status_queue.put({"kind": "stopped"}, timeout=0.2)
        except Exception:
            pass
    except Exception as error:  # pragma: no cover - exercised in spawned child
        payload = {
            "kind": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=12)[-8000:],
        }
        try:
            status_queue.put(payload, timeout=0.2)
        except Exception:
            pass


@dataclass
class _RunHandles:
    generation: int
    process: Any
    frame_queue: Any
    status_queue: Any
    stop_event: Any
    run_event: Any
    step_semaphore: Any
    speed_value: Any
    collector: threading.Thread | None = None
    collector_done: threading.Event | None = None


class ConsoleSession:
    """Own one replay buffer and at most one spawned simulation."""

    def __init__(
        self,
        *,
        token: str | None = None,
        context: Any | None = None,
        worker_target: Callable[..., Any] | None = None,
        replay_path: str | Path | None = None,
    ) -> None:
        self.token = token or secrets.token_urlsafe(24)
        self.version = VERSION
        self._context = context or mp.get_context("spawn")
        self._worker_target = worker_target or _simulation_worker
        self._lock = threading.RLock()
        self._closed = False
        self._run: _RunHandles | None = None
        self._frames: deque[dict[str, Any]] = deque(maxlen=MAX_FRAMES)
        self._base_cursor = 0
        self._dropped_frames = 0
        self._generation = 0
        self._status = "idle"
        self._error: str | None = None
        self._config: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._title = "Terminal · Airspace Lab"
        if replay_path is not None:
            self._load_replay(replay_path)

    def session_payload(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "version": self.version,
            "scenarios": [dict(item) for item in SCENARIOS],
            "modes": [dict(item) for item in MODES],
            "profiles": [dict(item) for item in PROFILES],
        }

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def _load_replay(self, path: str | Path) -> None:
        title, frames, metadata = load_replay(path)
        with self._lock:
            self._frames.clear()
            self._frames.extend(frames)
            self._base_cursor = 0
            self._dropped_frames = 0
            self._status = "replay"
            self._title = title
            self._metadata = {"source": "replay", **metadata}
            self._config = {"mode": "replay", "scenario": "imported", "replay_path": str(path)}

    def _stop_locked(self, *, explicit: bool = True) -> None:
        run = self._run
        if run is None:
            if explicit and self._status not in {"completed", "error", "replay"}:
                self._status = "stopped"
            return
        run.stop_event.set()
        run.run_event.set()
        try:
            run.step_semaphore.release()
        except Exception:
            pass
        process = run.process
        try:
            process.join(timeout=PROCESS_JOIN_SECONDS)
        except Exception:
            pass
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=PROCESS_JOIN_SECONDS)
        except Exception:
            pass
        self._run = None
        if explicit and self._status not in {"completed", "error", "replay"}:
            self._status = "stopped"

    def start(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        config = validate_start_payload(payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("console session is closed")
            self._stop_locked(explicit=False)
            self._generation += 1
            self._frames.clear()
            self._base_cursor = 0
            self._dropped_frames = 0
            self._error = None
            self._metadata = {
                "profile": config.get("profile"),
                "checkpoint_path": config.get("low_checkpoint"),
                "seed": config.get("seed"),
                "deterministic": config.get("deterministic"),
            }
            self._config = dict(config)
            self._title = f"Terminal · {config['mode']} · {config['scenario']}"
            context = self._context
            frame_queue = context.Queue(maxsize=FRAME_QUEUE_SIZE)
            status_queue = context.Queue(maxsize=8)
            stop_event = context.Event()
            run_event = context.Event()
            step_semaphore = context.Semaphore(0)
            speed_value = context.Value("d", float(config["playback_speed"]))
            if not config["start_paused"]:
                run_event.set()
            process = context.Process(
                target=self._worker_target,
                args=(config, frame_queue, status_queue, stop_event, run_event, step_semaphore, speed_value),
                name="terminal-console-simulation",
            )
            generation = self._generation
            self._status = "starting"
            run = _RunHandles(
                generation, process, frame_queue, status_queue, stop_event,
                run_event, step_semaphore, speed_value,
            )
            self._run = run
            try:
                process.start()
            except Exception as error:
                self._run = None
                self._status = "error"
                self._error = f"simulation process failed to start: {error}"
                _close_ipc_queue(frame_queue)
                _close_ipc_queue(status_queue)
                raise RuntimeError(self._error) from error
            collector = threading.Thread(
                target=self._collect,
                args=(run,),
                daemon=True,
                name=f"terminal-console-collector-{generation}",
            )
            run.collector = collector
            run.collector_done = threading.Event()
            collector.start()
            return self._state_locked()

    def _collect(self, run: _RunHandles) -> None:
        try:
            while True:
                try:
                    item = run.frame_queue.get(timeout=0.05)
                except Exception:
                    item = None
                if isinstance(item, Mapping) and item.get("kind") == "frame":
                    self._append_frame(run, item.get("frame"), validate_previous=True)
                self._drain_status(run)
                try:
                    alive = bool(run.process.is_alive())
                except Exception:
                    alive = False
                if not alive:
                    while True:
                        try:
                            item = run.frame_queue.get_nowait()
                        except Exception:
                            break
                        if isinstance(item, Mapping) and item.get("kind") == "frame":
                            self._append_frame(run, item.get("frame"), validate_previous=False)
                    self._drain_status(run)
                    with self._lock:
                        if self._generation == run.generation and self._status in {"starting", "running", "paused"}:
                            if run.stop_event.is_set():
                                self._status = "stopped"
                            else:
                                code = getattr(run.process, "exitcode", None)
                                self._error = self._error or f"simulation process exited unexpectedly (exitcode={code})"
                                self._status = "error"
                    break
        finally:
            _close_ipc_queue(run.frame_queue)
            _close_ipc_queue(run.status_queue)
            if run.collector_done is not None:
                run.collector_done.set()

    def _append_frame(self, run: _RunHandles, frame: Any, *, validate_previous: bool) -> None:
        try:
            previous = None
            if validate_previous:
                with self._lock:
                    if self._generation == run.generation and self._frames:
                        previous = float(self._frames[-1].get("time", 0.0))
            _validate_frame(frame, previous)
            item = dict(_jsonable(frame))
        except Exception as error:
            with self._lock:
                if self._generation == run.generation:
                    self._error = f"invalid simulation frame: {error}"
                    self._status = "error"
                    run.stop_event.set()
            return
        with self._lock:
            if self._generation != run.generation:
                return
            if len(self._frames) >= MAX_FRAMES:
                self._base_cursor += 1
                self._dropped_frames += 1
            self._frames.append(item)

    def _drain_status(self, run: _RunHandles) -> None:
        while True:
            try:
                status = run.status_queue.get_nowait()
            except Exception:
                return
            self._apply_status(run, status)

    def _apply_status(self, run: _RunHandles, status: Any) -> None:
        if not isinstance(status, Mapping):
            return
        kind = status.get("kind")
        with self._lock:
            if self._generation != run.generation:
                return
            if kind == "ready":
                self._status = "paused" if not run.run_event.is_set() else "running"
            elif kind == "completed":
                self._status = "completed"
                metadata = status.get("metadata", {})
                if isinstance(metadata, Mapping):
                    try:
                        self._metadata.update(dict(_jsonable(metadata)))
                    except Exception as error:
                        self._metadata = {"error": f"invalid worker metadata: {error}"}
            elif kind == "stopped":
                self._status = "stopped"
            elif kind == "error":
                error = str(status.get("error", "simulation worker failed"))
                detail = str(status.get("traceback", ""))
                self._error = f"{error}\n{detail}".strip()
                self._status = "error"

    def control(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None or not isinstance(payload, Mapping):
            raise ConsoleInputError("control body must be an object")
        action, value = _control_value(payload)
        with self._lock:
            run = self._run
            if action == "stop":
                self._stop_locked(explicit=True)
                return self._state_locked()
            if run is None or self._status in {"replay", "idle", "completed", "error", "stopped"}:
                raise ConsoleInputError("no live simulation is available")
            if action == "pause":
                run.run_event.clear()
                self._status = "paused"
            elif action == "resume":
                run.run_event.set()
                self._status = "running"
            elif action == "step":
                run.run_event.clear()
                run.step_semaphore.release()
                self._status = "paused"
            elif action == "speed":
                run.speed_value.value = float(value)
                self._config["playback_speed"] = float(value)
            return self._state_locked()

    def _state_locked(self) -> dict[str, Any]:
        process_alive = False
        if self._run is not None:
            try:
                process_alive = bool(self._run.process.is_alive())
            except Exception:
                process_alive = False
        return {
            "generation": self._generation,
            "status": self._status,
            "error": self._error,
            "config": dict(self._config),
            "metadata": dict(self._metadata),
            "process_alive": process_alive,
            "base_cursor": self._base_cursor,
            "next_cursor": self._base_cursor + len(self._frames),
            "dropped_frames": self._dropped_frames,
        }

    def frames_payload(self, after: int = 0, generation: int | None = None) -> dict[str, Any]:
        with self._lock:
            current_generation = self._generation
            base = self._base_cursor
            next_cursor = base + len(self._frames)
            if generation != current_generation:
                start = base
            else:
                start = max(base, min(next_cursor, after))
            offset = start - base
            frames = list(self._frames)[offset:offset + MAX_FRAME_BATCH]
            return {
                "generation": current_generation,
                "status": self._status,
                "error": self._error,
                "config": dict(self._config),
                "metadata": dict(self._metadata),
                "base_cursor": base,
                "next_cursor": start + len(frames),
                "frames": frames,
                "dropped_frames": self._dropped_frames,
                "process_alive": self._process_alive_locked(),
            }

    def _process_alive_locked(self) -> bool:
        if self._run is None:
            return False
        try:
            return bool(self._run.process.is_alive())
        except Exception:
            return False

    def export_payload(self) -> dict[str, Any]:
        with self._lock:
            frames = list(self._frames)
            metadata = {
                "config": dict(self._config),
                "generation": self._generation,
                "truncated": bool(self._dropped_frames),
                "dropped_frames": self._dropped_frames,
                "status": self._status,
                "run_metadata": dict(self._metadata),
            }
            return {"title": self._title, "frames": frames, "metadata": metadata}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_locked(explicit=False)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_locked()


def _host_is_loopback(host_header: str | None) -> bool:
    if not host_header:
        return False
    value = host_header.strip()
    if value.startswith("["):
        end = value.find("]")
        hostname = value[1:end] if end > 0 else ""
    else:
        hostname = value.rsplit(":", 1)[0] if value.count(":") == 1 else value
    return hostname.lower() in LOOPBACK_HOSTS


def _origin_is_loopback(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and (parsed.hostname or "").lower() in LOOPBACK_HOSTS


class ConsoleRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def session(self) -> ConsoleSession:
        return self.server.session  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _authorized(self, *, token: bool = False) -> bool:
        if not _host_is_loopback(self.headers.get("Host")):
            self._json_error(HTTPStatus.FORBIDDEN, "invalid_host", "Host must be loopback")
            return False
        if not _origin_is_loopback(self.headers.get("Origin")):
            self._json_error(HTTPStatus.FORBIDDEN, "invalid_origin", "Origin must be loopback")
            return False
        if token and not hmac.compare_digest(self.headers.get("X-Terminal-Token", ""), self.session.token):
            self._json_error(HTTPStatus.FORBIDDEN, "invalid_token", "X-Terminal-Token is invalid")
            return False
        return True

    def _json_error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json({"error": code, "message": message}, status)

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            body = _dump_json(value)
        except Exception as error:
            body = _dump_json({"error": "serialization_error", "message": str(error)})
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        if origin and _origin_is_loopback(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._json_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type_required", "application/json is required")
            return None
        length_text = self.headers.get("Content-Length")
        try:
            length = int(length_text) if length_text is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._json_error(HTTPStatus.LENGTH_REQUIRED, "content_length_required", "Content-Length is required")
            return None
        if length > MAX_BODY_BYTES:
            self._json_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body_too_large", "request body is too large")
            return None
        raw = self.rfile.read(length)
        try:
            payload = _json_loads(raw)
        except ConsoleInputError as error:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_json", str(error))
            return None
        if not isinstance(payload, dict):
            self._json_error(HTTPStatus.BAD_REQUEST, "object_required", "JSON body must be an object")
            return None
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/session":
            self._send_json(self.session.session_payload())
            return
        if parsed.path == "/api/frames":
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                after = _nonnegative_int(int(query.get("after", ["0"])[0]), "after")
                generation_value = query.get("generation", [None])[0]
                generation = None if generation_value in (None, "") else _nonnegative_int(int(generation_value), "generation")
            except (ValueError, ConsoleInputError) as error:
                self._json_error(HTTPStatus.BAD_REQUEST, "invalid_cursor", str(error))
                return
            self._send_json(self.session.frames_payload(after=after, generation=generation))
            return
        if parsed.path == "/api/export":
            self._send_json(self.session.export_payload())
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized(token=True):
            return
        parsed = urlsplit(self.path)
        if parsed.path not in {"/api/start", "/api/control"}:
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "endpoint not found")
            return
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            result = self.session.start(payload) if parsed.path == "/api/start" else self.session.control(payload)
        except ConsoleInputError as error:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid_request", str(error))
        except RuntimeError as error:
            self._json_error(HTTPStatus.CONFLICT, "run_error", str(error))
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, "server_error", str(error))
        else:
            self._send_json(result)

    def _serve_static(self, path: str) -> None:
        requested = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
        if requested not in STATIC_FILES or "/" in requested or "\\" in requested or ".." in requested:
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return
        root = Path(self.server.web_root).resolve()  # type: ignore[attr-defined]
        candidate = (root / requested).resolve()
        if candidate.parent != root or not candidate.is_file():
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self._json_error(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", STATIC_TYPES[requested])
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ConsoleHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], session: ConsoleSession, web_root: str | Path = WEB_ROOT) -> None:
        self.session = session
        self.web_root = Path(web_root)
        super().__init__(address, ConsoleRequestHandler)


def build_server(
    *,
    port: int = DEFAULT_PORT,
    session: ConsoleSession | None = None,
    web_root: str | Path = WEB_ROOT,
) -> ConsoleHTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or port < 0 or port > 65535:
        raise ValueError("port must be an integer in [0, 65535]")
    active_session = session or ConsoleSession()
    return ConsoleHTTPServer(("127.0.0.1", port), active_session, web_root)


def create_server(**kwargs: Any) -> ConsoleHTTPServer:
    return build_server(**kwargs)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Terminal Airspace Lab web console")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-autostart", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mode", choices=tuple(sorted(MODE_IDS)), default=DEFAULT_MODE)
    parser.add_argument("--scenario", choices=tuple(sorted(SCENARIO_IDS)), default=DEFAULT_SCENARIO)
    parser.add_argument("--profile", choices=tuple(sorted(PROFILE_IDS)))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-simulation-seconds", type=float, default=DEFAULT_MAX_SIMULATION_SECONDS)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--playback-speed", type=float, default=DEFAULT_PLAYBACK_SPEED)
    parser.add_argument("--low-checkpoint")
    parser.add_argument("--high-checkpoint")
    parser.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=DEFAULT_START_PAUSED)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    try:
        session = ConsoleSession(replay_path=args.replay)
        server = build_server(port=args.port, session=session)
    except (ConsoleInputError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 2
    try:
        if args.replay is None and not args.no_autostart:
            payload = {
                "mode": args.mode,
                "scenario": args.scenario,
                "profile": args.profile,
                "seed": args.seed,
                "max_simulation_seconds": args.max_simulation_seconds,
                "max_steps": args.max_steps,
                "playback_speed": args.playback_speed,
                "low_checkpoint": args.low_checkpoint,
                "high_checkpoint": args.high_checkpoint,
                "start_paused": args.start_paused,
                "deterministic": args.deterministic,
            }
            session.start(payload)
        host, bound_port = server.server_address[:2]
        print(f"Terminal console listening at http://{host}:{bound_port}/", flush=True)
        if not args.no_browser:
            webbrowser.open(f"http://{host}:{bound_port}/")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    finally:
        session.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
