"""Run pre-specified reduced_model capacity experiments; --output is required."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from terminal_marl.analysis.service_capacity import (
    ASSUMPTIONS, DEFAULT_RATES, DEFAULT_SEEDS, MODEL_KIND, constants_table, run_tasks, scenarios,
)


def _atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(output, rows, failed, counts, *, workers, elapsed_s, source_commit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_commit, source_kind = resolve_source_commit(source_commit)
    payload = {"model_kind": MODEL_KIND,
               "source_commit": source_commit, "source_commit_source": source_kind,
               "assumptions": ASSUMPTIONS, "constants": constants_table(), "task_counts": counts,
               "failed_tasks": failed, "execution": {"workers": workers, "elapsed_s": elapsed_s}, "rows": rows}
    stream = io.StringIO(newline="")
    fields = ["scenario", "seed", "offered_rate_per_h", "warmup_s", "cohort_duration_s", "arrivals", "normal",
              "reject_buffer", "reject_deadline", "empirical_offered_rate_per_h", "empirical_normal_rate_per_h",
              "normal_fraction", "end_waiting_backlog", "end_in_service", "calendar_drain_end_s", "waiting_buffer",
              "holding_power_w", "all_served_fixed_mix_upper_per_h", "normal_wait_p50_s", "normal_wait_p95_s"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        summary = {key: row[key] for key in fields if key in row}
        summary["all_served_fixed_mix_upper_per_h"] = row["pad_moments"]["all_served_fixed_mix_upper_per_h"]
        summary["normal_wait_p50_s"] = row["normal_wait_quantiles_s"]["p50"]
        summary["normal_wait_p95_s"] = row["normal_wait_quantiles_s"]["p95"]
        writer.writerow(summary)
    _atomic_text(output / "summary.csv", stream.getvalue())
    _atomic_text(output / "results.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def resolve_source_commit(explicit=None):
    if explicit:
        return explicit, "explicit_argument"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        return commit, "local_git"
    except (OSError, subprocess.CalledProcessError):
        return None, "unavailable"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-commit", help="Explicit commit for synchronized server trees without .git")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(DEFAULT_RATES))
    parser.add_argument("--warmup-s", type=float, default=3600.)
    parser.add_argument("--duration-s", type=float, default=14400.)
    parser.add_argument("--scenario", nargs="+", choices=list(scenarios()), default=list(scenarios()))
    args = parser.parse_args(argv)
    if args.workers < 1 or args.warmup_s < 0 or args.duration_s <= 0 or any(rate <= 0 for rate in args.rates):
        parser.error("positive workers/duration/rates and nonnegative warmup required")
    tasks = [(name, seed, rate, args.warmup_s, args.duration_s)
             for name in args.scenario for seed in args.seeds for rate in args.rates]
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    commit, commit_source = resolve_source_commit(args.source_commit)
    initial = {"model_kind": MODEL_KIND, "source_commit": commit, "source_commit_source": commit_source,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "workers": args.workers, "scenarios": args.scenario, "seeds": args.seeds,
               "rates_per_h": args.rates, "warmup_s": args.warmup_s, "duration_s": args.duration_s,
               "constants": constants_table()}
    def progress(counts):
        payload = {**initial, **counts, "elapsed_s": time.monotonic() - started}
        _atomic_text(args.output / "progress.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
    progress({"expected": len(tasks), "completed": 0, "failed": 0})
    rows, failed, counts = run_tasks(tasks, args.workers, progress_callback=progress)
    write_outputs(args.output, rows, failed, counts, workers=args.workers, elapsed_s=time.monotonic() - started, source_commit=args.source_commit)
    print(json.dumps({"model_kind": MODEL_KIND, "output": str(args.output.resolve()), "task_counts": counts}))
    for failure in failed:
        print(json.dumps(failure), file=sys.stderr)
    return 1 if failed or counts["missing"] or counts["duplicate"] or counts["duplicate_requested"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
