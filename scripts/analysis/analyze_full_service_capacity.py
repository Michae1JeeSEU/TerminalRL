"""Analyze completed full-world capacity jobs without treating partial runs as complete."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures", action="store_true", help="Plot only verified completed observations")
    args = parser.parse_args(argv)
    from terminal_marl.analysis.full_capacity_report import analyze_campaign, write_report
    report = write_report(analyze_campaign(args.campaign), args.output, figures=args.figures)
    print(json.dumps(dict(coverage=report["coverage"], output=str(args.output.resolve())), sort_keys=True))
    return 0 if report["coverage"]["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
