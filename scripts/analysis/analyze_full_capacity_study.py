"""Analyze an explicit same-version study grid across existing campaigns."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, action="append", type=Path)
    parser.add_argument("--spec", required=True, type=Path, help="Explicit analysis-only expected-grid JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    from terminal_marl.analysis.full_capacity_study import analyze_study, write_report
    try:
        report = analyze_study(args.campaign, json.loads(args.spec.read_text(encoding="utf-8")))
        write_report(report, args.output, figures=args.figures)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps(dict(error=str(exc), output=str(args.output.resolve())), sort_keys=True))
        return 1
    print(json.dumps(dict(coverage=report["coverage"], output=str(args.output.resolve())), sort_keys=True))
    return 0 if report["coverage"]["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
