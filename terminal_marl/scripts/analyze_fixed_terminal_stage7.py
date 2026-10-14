"""Describe finite-reference contribution labels and actual search operations."""
import argparse
import json
from pathlib import Path
from terminal_marl.core.fixed_terminal.search_data import analyze_contexts
from terminal_marl.core.fixed_terminal.learning_data import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_contexts(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"bottleneck.json", report)
    print(json.dumps({"status": "completed", "contexts": report["totals"]["contexts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
