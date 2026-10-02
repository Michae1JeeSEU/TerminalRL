"""Run the original Yang--Wei multi-aircraft MCTS in a bounded headless loop.

The files under :mod:`upstream` are kept byte-for-byte as acquired.  The only
source compatibility adjustment made at import time is the string guard in
the two node modules.  The original code compares ``numpy.ndarray`` values to
the string ``"random"``; on current NumPy that comparison can raise while
expanding a node.  ``_load_upstream`` compiles the source after replacing that
one exact condition in memory.  It never writes a modified upstream file.

Each command-line invocation runs one case in a fresh Python process.  The
case modules use top-level imports with shared names (``Config``, ``common``,
and the node modules), so a fresh process is also the isolation boundary used
by the wrapper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
UPSTREAM = ROOT / "upstream"
MCTS_DIR = UPSTREAM / "MCTS"
SIM_DIR = UPSTREAM / "Simulators"
SOURCE_SHA = "2c83aeab807daee6bab0561199116210d44839da"
PLAN_PERIOD = 5


@dataclass(frozen=True)
class CaseSpec:
    name: str
    agent_module: str
    config_module: str
    env_module: str
    node_module: str
    start_counter: int
    threshold_multiplier: int
    natural_after_empty_steps: int
    two_stage: bool = False


CASES: Dict[str, CaseSpec] = {
    "crossing": CaseSpec(
        name="crossing",
        agent_module="Agent_vertiport",
        config_module="config_vertiport",
        env_module="MultiAircraftVertiportEnv",
        node_module="nodes_multi",
        start_counter=1,
        threshold_multiplier=3,
        natural_after_empty_steps=100,
    ),
    "hex": CaseSpec(
        name="hex",
        agent_module="Agent_vertiHexSecGatePlus",
        config_module="config_hex_sec",
        env_module="MultiAircraftVertiHexSecGatePlusEnv",
        node_module="nodesHexSecGatePlus",
        start_counter=0,
        threshold_multiplier=2,
        natural_after_empty_steps=100,
    ),
    "hex_two_stage": CaseSpec(
        name="hex_two_stage",
        agent_module="Agent_vertiHexSecGatePlusTwoStage",
        config_module="config_hex_sec",
        env_module="MultiAircraftVertiHexSecGatePlusTwoStageEnv",
        node_module="nodesHexSecGatePlus",
        start_counter=0,
        threshold_multiplier=5,
        natural_after_empty_steps=10,
        two_stage=True,
    ),
}


@dataclass
class RunOptions:
    """Explicit runtime controls.

    ``None`` budget fields preserve the values from the acquired Config class.
    Setting them is a diagnostic workload override and is recorded in the
    metadata; it is not a claim about the paper's experimental protocol.
    """

    case: str
    seed: int = 2
    max_steps: int = 120
    output: Path = field(default_factory=lambda: Path("."))
    cohort_limit: Optional[int] = None
    full_simulations: Optional[int] = None
    lite_simulations: Optional[int] = None
    search_depth: Optional[int] = None
    lite_search_depth: Optional[int] = None
    simulate_frame: Optional[int] = None
    debug: bool = False


@dataclass
class RunResult:
    case: str
    seed: int
    steps: int
    complete: bool
    truncated: bool
    natural_termination: bool
    generated: int
    source_id_tracker: int
    active: int
    goals: int
    conflicts: int
    nmacs: int
    raw_conflicts: int
    raw_nmacs: int
    conflict_pairs: float
    nmac_pairs: float
    total_timesteps: int
    mcts_decisions: int
    walltime_seconds: float
    output_dir: str
    source_sha: str = SOURCE_SHA


class RuntimeLoadError(RuntimeError):
    """Raised when an acquired source module cannot be loaded headlessly."""


def _clear_upstream_modules() -> None:
    """Remove top-level names used by the source before a fresh load."""

    names = {
        "common",
        "search_multi",
        "nodes_multi",
        "nodesHexSecGatePlus",
        "config_multi",
        "config_vertiport",
        "config_hex_sec",
        "MultiAircraftVertiportEnv",
        "MultiAircraftVertiHexSecGatePlusEnv",
        "MultiAircraftVertiHexSecGatePlusTwoStageEnv",
        "Agent_vertiport",
        "Agent_vertiHexSecGatePlus",
        "Agent_vertiHexSecGatePlusTwoStage",
    }
    for name in names:
        sys.modules.pop(name, None)


def _patched_node_source(path: Path) -> Tuple[str, int]:
    source = path.read_text(encoding="utf-8")
    old = "if self.state.init_action == 'random':"
    new = "if isinstance(self.state.init_action, str) and self.state.init_action == 'random':"
    count = source.count(old)
    if count != 1:
        raise RuntimeLoadError(
            f"expected exactly one compatibility condition in {path.name}, found {count}"
        )
    return source.replace(old, new), count


def _exec_module_from_text(name: str, path: Path, source: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_upstream(spec: CaseSpec) -> Dict[str, Any]:
    """Load one source case and return its classes/configuration.

    This function is intentionally process-local.  The original repository
    relies on top-level imports and reuses module names between cases.
    """

    if not UPSTREAM.is_dir():
        raise RuntimeLoadError(f"upstream directory missing: {UPSTREAM}")
    _clear_upstream_modules()
    # Keep MCTS ahead of Simulators for ``common`` and ``search_multi`` while
    # allowing the simulator's top-level config/env imports to resolve.
    for directory in (str(MCTS_DIR), str(SIM_DIR)):
        if directory not in sys.path:
            sys.path.insert(0, directory)

    config = importlib.import_module(spec.config_module)
    common = importlib.import_module("common")
    node_path = MCTS_DIR / f"{spec.node_module}.py"
    patched_source, patch_count = _patched_node_source(node_path)
    nodes = _exec_module_from_text(spec.node_module, node_path, patched_source)
    search = importlib.import_module("search_multi")
    env_module = importlib.import_module(spec.env_module)

    return {
        "Config": config.Config,
        "MultiAircraftNode": nodes.MultiAircraftNode,
        "MultiAircraftState": nodes.MultiAircraftState,
        "MCTS": search.MCTS,
        "Env": env_module.MultiAircraftEnv,
        "patches": {str(node_path.relative_to(ROOT)): patch_count},
        "common": common,
    }


def _source_file_hashes() -> Dict[str, str]:
    files: Dict[str, str] = {}
    for path in sorted(UPSTREAM.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(UPSTREAM)).replace(os.sep, "/")] = digest
    return files


def _assert_external_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    raise ValueError(
        f"output must be outside the workspace repository {WORKSPACE_ROOT}; received {resolved}"
    )


def _number(value: Any) -> Any:
    """Convert NumPy scalar values without importing NumPy in the wrapper."""

    try:
        return value.item()
    except AttributeError:
        return value


def _active_count(env: Any) -> int:
    aircraft_dict = getattr(env, "aircraft_dict", None)
    return int(getattr(aircraft_dict, "num_aircraft", 0))


def _safe_counter(env: Any, name: str) -> int:
    return int(_number(getattr(env, name, 0)))


def _record_decision(
    trace: List[Dict[str, Any]],
    *,
    step: int,
    phase: str,
    sector: Optional[int],
    aircraft_id: int,
    action: Any,
    distance: Any,
    budget: int,
    depth: int,
) -> None:
    trace.append(
        {
            "step": step,
            "phase": phase,
            "sector": "" if sector is None else sector,
            "aircraft_id": aircraft_id,
            "action": int(_number(action)),
            "distance": float(_number(distance)),
            "search_budget": budget,
            "search_depth": depth,
        }
    )


def _mcts_action(
    modules: Dict[str, Any],
    observation: Any,
    index: int,
    init_action: Any,
    *,
    sector_id: Optional[int] = None,
    goal_exit_id: Optional[int] = None,
    full: bool,
    config: Any,
    options: RunOptions,
) -> Any:
    state_kwargs: Dict[str, Any] = {
        "state": observation,
        "index": index,
        "init_action": init_action,
    }
    if sector_id is not None:
        state_kwargs["sector_id"] = sector_id
        state_kwargs["goal_exit_id"] = goal_exit_id
    state = modules["MultiAircraftState"](**state_kwargs)
    root = modules["MultiAircraftNode"](state=state)
    mcts = modules["MCTS"](root)
    if full:
        simulations = options.full_simulations
        depth = options.search_depth
        if simulations is None:
            simulations = int(config.no_simulations)
        if depth is None:
            depth = int(config.search_depth)
    else:
        simulations = options.lite_simulations
        depth = options.lite_search_depth
        if simulations is None:
            simulations = int(getattr(config, "no_simulations_lite", config.no_simulations))
        if depth is None:
            depth = int(getattr(config, "search_depth_lite", config.search_depth))
    best_node = mcts.best_action(int(simulations), int(depth))
    return best_node.state.prev_action[index], int(simulations), int(depth)


def _plan_crossing(
    modules: Dict[str, Any],
    env: Any,
    last_observation: Tuple[Any, List[int]],
    info: Any,
    step: int,
    options: RunOptions,
    trace: List[Dict[str, Any]],
    config: Any,
) -> Dict[int, Any]:
    observation, id_list = last_observation
    num_existing = observation.shape[0]
    action = __import__("numpy").ones(num_existing, dtype=__import__("numpy").int32)
    action_by_id: Dict[int, Any] = {}
    for index in range(num_existing):
        distance = info[index]
        full = distance < 3 * config.minimum_separation
        selected, budget, depth = _mcts_action(
            modules,
            observation,
            index,
            action,
            full=full,
            config=config,
            options=options,
        )
        action[index] = selected
        action_by_id[id_list[index]] = selected
        _record_decision(
            trace,
            step=step,
            phase="crossing",
            sector=None,
            aircraft_id=id_list[index],
            action=selected,
            distance=distance,
            budget=budget,
            depth=depth,
        )
    return action_by_id


def _plan_hex(
    modules: Dict[str, Any],
    env: Any,
    last_observation: Dict[int, Tuple[Any, List[int], List[int]]],
    info: Any,
    step: int,
    options: RunOptions,
    trace: List[Dict[str, Any]],
    config: Any,
) -> Dict[int, Any]:
    action_by_id: Dict[int, Any] = {}
    for sector_id in range(7):
        observation, id_list, goal_exit_ids = last_observation[sector_id]
        action = __import__("numpy").ones(observation.shape[0], dtype=__import__("numpy").int32)
        for index in range(len(id_list)):
            distance = info[id_list[index]]
            full = distance < 2 * config.minimum_separation
            selected, budget, depth = _mcts_action(
                modules,
                observation,
                index,
                action,
                sector_id=sector_id,
                goal_exit_id=goal_exit_ids[index],
                full=full,
                config=config,
                options=options,
            )
            action[index] = selected
            action_by_id[id_list[index]] = selected
            _record_decision(
                trace,
                step=step,
                phase="sector",
                sector=sector_id,
                aircraft_id=id_list[index],
                action=selected,
                distance=distance,
                budget=budget,
                depth=depth,
            )
    return action_by_id


def _plan_two_stage(
    modules: Dict[str, Any],
    env: Any,
    last_observation: Dict[int, Tuple[Any, ...]],
    info: Any,
    step: int,
    options: RunOptions,
    trace: List[Dict[str, Any]],
    config: Any,
) -> Dict[int, Any]:
    import numpy as np

    action_by_id: Dict[int, Any] = {}
    for sector_id in range(7):
        (
            ob_high_in,
            id_high,
            goal_exit_high,
            ob_high_out,
            ob_in,
            id_list,
            goal_exit_ids,
            ob_out,
        ) = last_observation[sector_id]

        ob_high = np.concatenate([ob_high_in, ob_high_out])
        action_high = np.ones(ob_high.shape[0], dtype=np.int32)
        for index in range(len(id_high)):
            distance = info[id_high[index]]
            full = distance < 5 * config.minimum_separation
            selected, budget, depth = _mcts_action(
                modules,
                ob_high,
                index,
                action_high,
                sector_id=sector_id,
                goal_exit_id=goal_exit_high[index],
                full=full,
                config=config,
                options=options,
            )
            action_high[index] = selected
            action_by_id[id_high[index]] = selected
            _record_decision(
                trace,
                step=step,
                phase="high",
                sector=sector_id,
                aircraft_id=id_high[index],
                action=selected,
                distance=distance,
                budget=budget,
                depth=depth,
            )

        ob = np.concatenate([ob_in, ob_high_in, ob_high_out, ob_out])
        num_considered = len(id_list)
        if ob_in.shape[0] != num_considered:
            raise ValueError("error dimension")
        action = np.ones(ob.shape[0], dtype=np.int32)
        action[num_considered : num_considered + action_high.shape[0]] = action_high
        for index in range(num_considered):
            distance = info[id_list[index]]
            full = distance < 5 * config.minimum_separation
            selected, budget, depth = _mcts_action(
                modules,
                ob,
                index,
                action,
                sector_id=sector_id,
                goal_exit_id=goal_exit_ids[index],
                full=full,
                config=config,
                options=options,
            )
            action[index] = selected
            action_by_id[id_list[index]] = selected
            _record_decision(
                trace,
                step=step,
                phase="low",
                sector=sector_id,
                aircraft_id=id_list[index],
                action=selected,
                distance=distance,
                budget=budget,
                depth=depth,
            )
    return action_by_id


def _apply_config_override(config: Any, options: RunOptions) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if options.simulate_frame is not None:
        config.simulate_frame = int(options.simulate_frame)
        overrides["simulate_frame"] = int(options.simulate_frame)
    return overrides


def _write_trace(path: Path, trace: Iterable[Dict[str, Any]]) -> None:
    rows = list(trace)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step",
            "phase",
            "sector",
            "aircraft_id",
            "action",
            "distance",
            "search_budget",
            "search_depth",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(options: RunOptions) -> RunResult:
    """Execute one bounded case and write JSON/CSV artifacts."""

    if options.case not in CASES:
        raise ValueError(f"unknown case {options.case!r}; choose from {sorted(CASES)}")
    if options.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if options.cohort_limit is not None and options.cohort_limit <= 0:
        raise ValueError("cohort_limit must be positive when supplied")
    output = _assert_external_output(Path(options.output))
    spec = CASES[options.case]

    started = time.perf_counter()
    # Set both RNGs before loading or constructing the source environment.
    random.seed(options.seed)
    import numpy as np

    np.random.seed(options.seed)
    modules = _load_upstream(spec)
    config = modules["Config"]
    original_config = {
        "full_simulations": int(getattr(config, "no_simulations", -1)),
        "lite_simulations": int(getattr(config, "no_simulations_lite", -1)),
        "search_depth": int(getattr(config, "search_depth", -1)),
        "lite_search_depth": int(getattr(config, "search_depth_lite", -1)),
        "simulate_frame": int(getattr(config, "simulate_frame", -1)),
    }
    config_overrides = _apply_config_override(config, options)
    env = modules["Env"](options.seed, debug=options.debug)
    last_observation = env.reset()
    initial_cohort = _active_count(env)
    info: Any = None
    trace: List[Dict[str, Any]] = []
    steps = 0
    natural = False
    done = False
    near_end = False

    while not done and steps < options.max_steps:
        # ``steps`` counts completed transitions.  The source crossing agent
        # starts its episode counter at one, while both sector agents start at
        # zero.  Use that source counter for both cadence and trace labels.
        episode_clock = steps + spec.start_counter
        if episode_clock % PLAN_PERIOD == 0:
            if info is None:
                action_by_id: Dict[int, Any] = {}
            elif spec.two_stage:
                action_by_id = _plan_two_stage(
                    modules, env, last_observation, info, episode_clock, options, trace, config
                )
            elif options.case == "crossing":
                action_by_id = _plan_crossing(
                    modules, env, last_observation, info, episode_clock, options, trace, config
                )
            else:
                action_by_id = _plan_hex(
                    modules, env, last_observation, info, episode_clock, options, trace, config
                )
        elif steps == 0:
            action_by_id = {}

        if options.case == "crossing":
            observation, _reward, done, info = env.step(action_by_id)
        else:
            observation, _reward, done, info = env.step(action_by_id, near_end)
        last_observation = observation
        steps += 1

        active = _active_count(env)
        # The source sets near_end *after* a step and passes it to the next
        # transition.  A supplied cohort limit replaces the source's 10,000
        # aircraft cap and is recorded as an altered workload.
        if options.case != "crossing":
            if options.cohort_limit is not None:
                if int(env.id_tracker) >= options.cohort_limit:
                    near_end = True
            elif int(env.id_tracker) - 1 >= 10000:
                near_end = True
        # This mirrors the stopping conditions in each original Agent file.
        episode_clock = steps + spec.start_counter
        if episode_clock > spec.natural_after_empty_steps and active == 0:
            natural = True
            break

    truncated = not natural and not done and steps >= options.max_steps
    elapsed = time.perf_counter() - started
    source_id_tracker = int(getattr(env, "id_tracker", 0))
    generated = initial_cohort + source_id_tracker if options.case == "crossing" else source_id_tracker
    raw_conflicts = _safe_counter(env, "conflicts")
    raw_nmacs = _safe_counter(env, "NMACs")
    result = RunResult(
        case=options.case,
        seed=options.seed,
        steps=steps,
        complete=bool(natural or done),
        truncated=bool(truncated),
        natural_termination=bool(natural),
        generated=generated,
        source_id_tracker=source_id_tracker,
        active=_active_count(env),
        goals=_safe_counter(env, "goals"),
        conflicts=raw_conflicts,
        nmacs=raw_nmacs,
        raw_conflicts=raw_conflicts,
        raw_nmacs=raw_nmacs,
        conflict_pairs=raw_conflicts / 2.0,
        nmac_pairs=raw_nmacs / 2.0,
        total_timesteps=_safe_counter(env, "total_timesteps"),
        mcts_decisions=len(trace),
        walltime_seconds=elapsed,
        output_dir=str(output),
    )
    metadata: Dict[str, Any] = asdict(result)
    metadata.update(
        {
            "source_repository": "https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance",
            "source_commit": SOURCE_SHA,
            "source_file_sha256": _source_file_hashes(),
            "case_spec": asdict(spec),
            "headless": True,
            "render": False,
            "plan_period": PLAN_PERIOD,
            "original_config": original_config,
            "overrides": {
                "cohort_limit": options.cohort_limit,
                "full_simulations": options.full_simulations,
                "lite_simulations": options.lite_simulations,
                "search_depth": options.search_depth,
                "lite_search_depth": options.lite_search_depth,
                **config_overrides,
            },
            "compatibility_transform": modules["patches"],
            "trace_rows": len(trace),
            "source_reported_conflicts": raw_conflicts / 2.0,
            "source_reported_nmacs": raw_nmacs / 2.0,
            "workload_note": "bounded execution; truncated or finite-cohort runs are not paper results",
            "counter_semantics": {
                "raw_conflicts": "env.conflicts; directional conflict-entry detections in the source simulator",
                "raw_nmacs": "env.NMACs; affected-aircraft NMAC detections in the source simulator",
                "source_reported_conflicts": "raw_conflicts / 2, matching original Agent reporting",
                "source_reported_nmacs": "raw_nmacs / 2, matching original Agent reporting",
                "conflict_pairs": "raw_conflicts / 2 convenience field; not a unique physical-pair count",
                "nmac_pairs": "raw_nmacs / 2 convenience field; not a unique physical-pair count",
            },
        }
    )
    _write_trace(output / "actions.csv", trace)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-limit", type=int, default=None)
    parser.add_argument("--full-simulations", type=int, default=None)
    parser.add_argument("--lite-simulations", type=int, default=None)
    parser.add_argument("--search-depth", type=int, default=None)
    parser.add_argument("--lite-search-depth", type=int, default=None)
    parser.add_argument("--simulate-frame", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    options = RunOptions(
        case=args.case,
        seed=args.seed,
        max_steps=args.max_steps,
        output=args.output,
        cohort_limit=args.cohort_limit,
        full_simulations=args.full_simulations,
        lite_simulations=args.lite_simulations,
        search_depth=args.search_depth,
        lite_search_depth=args.lite_search_depth,
        simulate_frame=args.simulate_frame,
        debug=args.debug,
    )
    result = run_case(options)
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
