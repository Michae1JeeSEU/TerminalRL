"""Run a stage-1 feasibility exerciser and persist auditable small JSON artifacts.

Example: python -m terminal_marl.scripts.evaluate_fixed_terminal_stage1
    --load low --soc high --seed 101 --output tmp_stage1/low-high-101
Use --matrix for all nine frozen load/SOC cells at the specified seed.
"""
import argparse
import json
from pathlib import Path
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG, LOAD_PROFILES, SOC_PROFILES
from terminal_marl.core.fixed_terminal.scenarios import generate_scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv


def run(output, load="low", soc="high", seed=101, search_limit=256, config=DEFAULT_CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot defaults and request draws before the first simulated outcome.
    env = FixedTerminalEnv(config, generate_scenario(config, seed, load, soc))
    snapshot = env.report()
    (output/"frozen_input.json").write_text(json.dumps({key: snapshot[key] for key in ("schema", "config", "scenario", "capacity", "assumptions")}, indent=2, allow_nan=False), encoding="utf-8")
    while not env.done:
        env.exercise_feasibility(search_limit)
        env.step()
    result = env.report()
    result["run_parameters"] = {"search_limit": search_limit, "seed": seed, "load": load, "soc": soc}
    (output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (output/"requests.jsonl").open("w", encoding="utf-8") as stream:
        for row in result["requests"]:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    with (output/"events.jsonl").open("w", encoding="utf-8") as stream:
        for row in env.events:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", choices=LOAD_PROFILES, default="low")
    parser.add_argument("--soc", choices=SOC_PROFILES, default="high")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--search-limit", type=int, default=256,
                        help="positive budget; 0 means exhaustive; unknown cannot authorize emergency")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.search_limit < 0:
        parser.error("--search-limit must be nonnegative")
    cells = [(load, soc) for load in LOAD_PROFILES for soc in SOC_PROFILES] if args.matrix else [(args.load, args.soc)]
    for load, soc in cells:
        destination = args.output/f"{load}-{soc}-{args.seed}" if args.matrix else args.output
        result = run(destination, load, soc, args.seed, args.search_limit or None)
        print(json.dumps({"output": str(destination), "summary": result["summary"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
