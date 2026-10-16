"""Run actual FT-RS-S1-1.0 fixtures and save every case's evidence."""
import argparse
import json
import os
import subprocess
import sys
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
from ..core.rotating_slots.qualification import run_catalogue, merge_reports, audit_replay


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--cases", help="comma-separated normative case IDs; omitted runs all94")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--code-commit", help="localGit commitidentity forserverruns; no serverGit commands")
    p.add_argument("--merge", nargs="+", metavar="RESULTS_JSON", help="strictly merge existingexecutedreports")
    p.add_argument("--audit", metavar="REPLAY_BUNDLE", help="readonlybounded independentrecordaudit")
    p.add_argument("--diagnostic-report", help="G09 consumes completedformaldiagnostic benchmarkevidence")
    args = p.parse_args(argv)
    if args.code_commit: os.environ["TERMINAL_CODE_COMMIT"] = args.code_commit
    if args.diagnostic_report: os.environ["TERMINAL_DIAGNOSTIC_REPORT"] = args.diagnostic_report
    if args.merge and args.audit: p.error("--merge and--audit are mutuallyexclusive")
    if args.audit:
        result = audit_replay(args.audit, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 1
    command = subprocess.list2cmdline([sys.executable, "-m", "terminal_marl.scripts.run_stage1_tests", *(sys.argv[1:] if argv is None else argv)])
    result = merge_reports(args.merge, args.output) if args.merge else run_catalogue(args.output, cases=args.cases.split(",") if args.cases else None, workers=args.workers, command=command)
    print(json.dumps({k: result[k] for k in ("automatic_status", "visualization_integration_status", "researcher_visual_acceptance", "stage1_status", "counts", "coverage")}, ensure_ascii=False, indent=2))
    return 1 if result["coverage"]["failures"] else 0


if __name__ == "__main__": raise SystemExit(main())
