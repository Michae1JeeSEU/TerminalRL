"""Command-line runner for the transparent Saxena scheduling port."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:  # package execution
    from .model import InputError, Parameters, ScheduleModel, build_flight_set, fixture, load_requests
except ImportError:  # direct ``python run.py`` from this directory
    from model import InputError, Parameters, ScheduleModel, build_flight_set, fixture, load_requests


SOURCE_SHA = "87a8e5ba7393d622768a7695770a4fa4e09c26bd"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_node_times(path: Path, result: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "node", "time"])
        for record_id in sorted(result.node_times):
            for node, value in result.node_times[record_id].items():
                writer.writerow([record_id, node, f"{value:.12g}"])


def run_once(
    *,
    requests: list[dict[str, Any]],
    output: str | Path,
    seed: int,
    fixture_name: str | None,
    formulation: str,
    time_limit: float | None,
    mip_rel_gap: float | None,
    preprocess: bool,
    gate_capacity: int = 3,
) -> Any:
    if time_limit is not None and time_limit <= 0:
        raise InputError("time_limit must be positive")
    if mip_rel_gap is not None and mip_rel_gap < 0:
        raise InputError("mip_rel_gap must be nonnegative")
    output_path = Path(output).expanduser().resolve()
    if output_path == PROJECT_ROOT or PROJECT_ROOT in output_path.parents:
        raise InputError(f"output must be outside the project tree: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    params = Parameters(gate_capacity=gate_capacity)
    flight_set = build_flight_set(requests, params=params, preprocess=preprocess)
    model = ScheduleModel(flight_set, formulation=formulation)
    result = model.solve(time_limit=time_limit, mip_rel_gap=mip_rel_gap)

    input_record = {
        "source_snapshot": SOURCE_SHA,
        "evidence_type": "newly_run",
        "native_matlab_run": False,
        "seed": int(seed),
        "fixture": fixture_name,
        "formulation": formulation,
        "time_limit": time_limit,
        "mip_rel_gap": mip_rel_gap,
        "preprocess_arrival_requests": preprocess,
        "raw_requests": requests,
        "parameters": flight_set.params.to_json(),
        "flight_set": flight_set.to_json(),
    }
    _json_dump(output_path / "inputs.json", input_record)
    _json_dump(output_path / "result.json", result.to_json())
    _json_dump(output_path / "constraint_checks.json", {"checks": result.constraint_checks, "summary": result.summary})
    _write_node_times(output_path / "node_times.csv", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fixture", choices=["arrival", "departure", "tat", "mixed", "gate_capacity3"], default="arrival")
    source.add_argument("--input", type=Path, help="JSON file containing a requests list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26, help="metadata only for explicit fixtures")
    parser.add_argument("--formulation", choices=["paper", "upstream-legacy"], default="paper")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=None)
    parser.add_argument("--gate-capacity", type=int, choices=[1, 3], default=3)
    parser.add_argument("--no-preprocess", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        fixture_name = None if args.input is not None else args.fixture
        requests = load_requests(args.input) if args.input is not None else fixture(args.fixture)
        result = run_once(
            requests=requests,
            output=args.output,
            seed=args.seed,
            fixture_name=fixture_name,
            formulation=args.formulation,
            time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap,
            preprocess=not args.no_preprocess,
            gate_capacity=args.gate_capacity,
        )
    except (InputError, OSError, json.JSONDecodeError) as exc:
        print(f"input/error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "objective": result.objective,
                "mip_gap": result.mip_gap,
                "formulation": result.formulation,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if result.status in {"optimal", "time_limit"} and result.node_times else 2 if result.status == "infeasible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
