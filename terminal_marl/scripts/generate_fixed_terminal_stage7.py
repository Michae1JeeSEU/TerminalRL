"""Generate newly instrumented finite-reference traces on frozen scenario inputs."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.core.fixed_terminal.search_data import generate_scenario_data
from terminal_marl.configs.fixed_terminal_stage7 import scenario_jobs, SNAPSHOT_TIMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "pilot"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-stage4", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-horizon", type=float, help="pilot-only short integration check")
    parser.add_argument("--smoke-times", help="pilot-only comma-separated snapshot ticks")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args(argv)
    if args.list_jobs:
        print(json.dumps(scenario_jobs(args.split)))
        return 0
    if args.seed is None or args.output is None:
        parser.error("--seed and --output required")
    if not args.smoke and (args.smoke_horizon is not None or args.smoke_times is not None):
        parser.error("custom horizon/times require --smoke")
    config = replace(DEFAULT_CONFIG, horizon=args.smoke_horizon) if args.smoke_horizon is not None else DEFAULT_CONFIG
    times = tuple(float(t) for t in args.smoke_times.split(",")) if args.smoke_times is not None else SNAPSHOT_TIMES
    records = generate_scenario_data(args.output, args.split, args.seed, args.load, args.soc,
                                    source_stage4=args.source_stage4, smoke=args.smoke, config=config, times=times)
    print(json.dumps({"status": "completed", "contexts": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
