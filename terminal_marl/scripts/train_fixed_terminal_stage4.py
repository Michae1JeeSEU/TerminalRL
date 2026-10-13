"""Train both frozen small CPU MLPs using train and validation scenarios only."""
import argparse
import json
from pathlib import Path
from terminal_marl.core.fixed_terminal.learning_data import load_contexts
from terminal_marl.core.fixed_terminal.learning_models import train_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="pilot-only abbreviated serialization test, never formal checkpoints")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(argv)
    train = load_contexts(args.train, "pilot" if args.smoke else "train", training=True)
    val = load_contexts(args.val, "pilot" if args.smoke else "val", training=True)
    print(json.dumps(train_models(train, val, args.output, epochs=args.epochs, smoke=args.smoke), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
