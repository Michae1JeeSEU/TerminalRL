"""Passive safe cuts inserted into the exact source of R0 ``advance``."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import ast
import difflib
import inspect
import math
import textwrap
import time

from .checkpoint import AdvanceCursor, CheckpointError


class SuspendAtCut(Exception):
    """Explicit nonphysical suspension; callers retain the last complete cut."""


@dataclass(frozen=True)
class ObservedAdvance:
    function: object
    original_source: str
    generated_source: str
    patch: str
    original_source_sha256: str
    generated_source_sha256: str

    def run(self, world, target, driver=None, hook=None, *, outer_phase="episode"):
        return self.function(world, target, driver, _performance_hook=hook,
                             _performance_outer_phase=outer_phase)

    def resume(self, restored, hook=None):
        cursor = restored.cursor
        if cursor.shim_source_sha256 != self.generated_source_sha256:
            raise CheckpointError("observation shim source mismatch")
        if cursor.phase == "returned":
            return restored.world.snapshot()
        return self.run(restored.world, cursor.original_target, restored.driver, hook,
                        outer_phase=cursor.outer_phase)


def make_observed_advance(world_class):
    """Insert two passive callbacks; retain every original algorithm source line.

    No core module is imported here. The generated function retains the exact
    globals of the supplied class, including an isolated R0's core instances.
    """
    original = textwrap.dedent(inspect.getsource(world_class.advance))
    tree = ast.parse(original)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "advance":
        raise CheckpointError("unexpected advance source")
    loops = [node for node in function.body if isinstance(node, ast.While)]
    if len(loops) != 1 or not isinstance(function.body[-1], ast.Return):
        raise CheckpointError("unrecognized advance control flow")
    loop = loops[0]
    last = loop.body[-1]
    expected = ast.parse("self._record()").body[0]
    if ast.dump(last, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise CheckpointError("advance no longer ends its loop with _record")
    if (len(function.args.args) != 3 or function.args.vararg or function.args.kwarg
            or function.args.kwonlyargs):
        raise CheckpointError("unexpected advance signature")
    lines = original.splitlines(keepends=True)
    # Insertion offsets are derived from the AST, but original statements are
    # copied as text so expression order and floating-point expressions survive.
    signature = lines[0].rstrip("\r\n")
    if not signature.endswith("):"):
        raise CheckpointError("multiline advance signature unsupported")
    lines[0] = signature[:-2] + ", *, _performance_hook=None, _performance_outer_phase='episode'):\n"
    after_loop_record = last.end_lineno
    before_return = function.body[-1].lineno - 1
    generated = []
    # Hash does not include itself: the literal placeholder is replaced in the
    # exec globals, preserving a stable generated-source identity.
    for i, line in enumerate(lines):
        if i == before_return:
            generated.extend([
                "    if _performance_hook is not None:\n",
                "        _performance_hook(self, _performance_cursor(until, 'returned', _performance_outer_phase, _performance_source_hash), callback)\n",
            ])
        generated.append(line)
        if i + 1 == after_loop_record:
            generated.extend([
                "        if _performance_hook is not None:\n",
                "            _performance_hook(self, _performance_cursor(until, 'loop_start', _performance_outer_phase, _performance_source_hash), callback)\n",
            ])
    source = "".join(generated)
    source_hash = sha256(source.encode()).hexdigest()
    namespace = dict(world_class.advance.__globals__)
    namespace.update(_performance_cursor=AdvanceCursor, _performance_source_hash=source_hash)
    exec(compile(source, "<R0+OBS:" + source_hash + ">", "exec"), namespace)
    patch = "".join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                                       fromfile="R0/advance", tofile="R0+OBS/advance"))
    return ObservedAdvance(namespace["advance"], original, source, patch,
                           sha256(original.encode()).hexdigest(), source_hash)


class CutRequests:
    """Save requests observed only at original safe cuts (30sim/300wall)."""
    def __init__(self, on_capture, *, sim_interval=30., wall_interval=300.,
                 clock=time.perf_counter, start_sim=0., requested_sim=()):
        if sim_interval <= 0 or wall_interval <= 0:
            raise ValueError("checkpoint intervals must be positive")
        self.on_capture = on_capture
        self.sim_interval = float(sim_interval)
        self.wall_interval = float(wall_interval)
        self.clock = clock
        self.last_sim = float(start_sim)
        self.last_wall = clock()
        self.requested_sim = list(sorted(float(x) for x in requested_sim))
        if any(not math.isfinite(x) for x in self.requested_sim):
            raise ValueError("nonfinite requested cut")
        self.captures = []

    def __call__(self, world, cursor, driver):
        now = self.clock()
        reasons, requests = [], []
        if world.time >= self.last_sim + self.sim_interval:
            reasons.append("simulation_interval")
            requests.append(self.last_sim + self.sim_interval)
        if now - self.last_wall >= self.wall_interval:
            reasons.append("wall_interval")
        due = [t for t in self.requested_sim if t <= world.time]
        if due:
            reasons.append("requested_milestone")
            requests.extend(due)
        if not reasons:
            return
        evidence = {"requested_sim_times": requests, "actual_sim_time": world.time,
                    "simulation_deferral_s": [world.time - t for t in requests],
                    "wall_since_last_capture_s": now - self.last_wall, "reasons": reasons}
        # Bookkeeping is passive and updated before a capture callback may
        # request suspension. The callback owns persisting this observer state.
        # Wall-only and milestone saves must not postpone the independent
        # simulation request cadence used for corpus retention.
        if "simulation_interval" in reasons:
            self.last_sim = world.time
        self.last_wall = now
        self.requested_sim = [t for t in self.requested_sim if t > world.time]
        self.captures.append(evidence)
        self.on_capture(world, cursor, driver, evidence)
