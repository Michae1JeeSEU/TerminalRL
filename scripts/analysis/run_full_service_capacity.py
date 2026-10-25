"""Run an explicit paired full-physics capacity experiment specification."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="JSON experiment specification")
    parser.add_argument("--output", required=True, type=Path, help="Separate immutable campaign directory")
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--cpus", help="Optional comma-separated CPU affinity identifiers, e.g. 4,6,8")
    args = parser.parse_args(argv)
    from terminal_marl.analysis.full_service_capacity import run_campaign
    cpus = [int(cpu.strip()) for cpu in args.cpus.split(",")] if args.cpus else None
    rows = run_campaign(json.loads(args.spec.read_text(encoding="utf-8")), args.output, workers=args.workers, cpus=cpus)
    print(json.dumps(dict(jobs=len(rows), completed=sum(row["status"] == "COMPLETED" for row in rows),
                          output=str(args.output.resolve())), sort_keys=True))
    return 0 if all(row["status"] == "COMPLETED" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
