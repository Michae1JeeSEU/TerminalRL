"""Train four frozen stage5 scorers using stage4 train/val only."""
import argparse
import json
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

from terminal_marl.core.fixed_terminal.validation_models import train_validation_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", type=Path, required=True,
                        help="root containing data/train and data/val; test is never traversed")
    parser.add_argument("--output", type=Path, required=True, help="new model directory; no overwrite")
    parser.add_argument("--smoke", action="store_true",
                        help="explicit pilot5001 fixtures only;3epochs/trees, never formal models")
    args = parser.parse_args(argv)
    print(json.dumps(train_validation_models(args.stage4_root, args.output, smoke=args.smoke), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
