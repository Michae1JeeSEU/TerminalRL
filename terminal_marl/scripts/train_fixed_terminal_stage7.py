"""Fit the single frozen expand classifier; formal validation is report-only."""
import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
import json
from pathlib import Path
from terminal_marl.core.fixed_terminal.search_data import load_contexts
from terminal_marl.core.fixed_terminal.search_models import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    train = load_contexts(args.train, "pilot" if args.smoke else "train", training=True)
    val = load_contexts(args.val, "pilot" if args.smoke else "val", training=True)
    manifest = train_model(train, val, args.output, smoke=args.smoke)
    print(json.dumps({"status": manifest["status"], "output": str(args.output), "model_class": manifest["model_class"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
