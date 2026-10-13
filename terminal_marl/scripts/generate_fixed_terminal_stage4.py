"""Generate one registered scenario's four pre-decision observed label sets."""
import argparse
import hashlib
import json
from pathlib import Path
from terminal_marl.configs.fixed_terminal_stage4 import SPLIT_SEEDS, jobs
from terminal_marl.core.fixed_terminal.learning_data import generate_scenario_data, write_json


def frozen_models(directory, *, allow_smoke=False):
    if directory is None:
        raise ValueError("frozen model directory required before test generation/evaluation")
    directory = Path(directory)
    training = json.loads((directory/"training.json").read_text(encoding="utf-8"))
    if training["status"] != "completed" or training["protocol"]["test_read"]:
        raise ValueError("invalid completed training protocol")
    if training["protocol"]["smoke_only"] and not allow_smoke:
        raise ValueError("pilot smoke checkpoint cannot be used for formal test")
    return {"path": str(directory), "files": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                               for name in ("value.pt", "rank.pt", "training.json", "protocol.json")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_SEEDS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--load", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--soc", choices=("high", "medium", "low"), default="medium")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path, help="required frozen checkpoints before generating test")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args(argv)
    if args.list_jobs:
        print(json.dumps(jobs(args.split)))
        return 0
    if args.seed is None or args.output is None:
        parser.error("--seed and --output required")
    provenance = frozen_models(args.models) if args.split == "test" else None
    records = generate_scenario_data(args.output, args.split, args.seed, args.load, args.soc)
    if provenance:
        write_json(args.output/"frozen_models.json", provenance)
    print(json.dumps({"status": "completed", "contexts": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
