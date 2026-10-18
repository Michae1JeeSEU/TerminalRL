"""Standalone six-input corpus freeze; no benchmark execution or qualification."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "terminal_marl/core/rotating_slots"))
from performance.corpus import main

if __name__ == "__main__":
    raise SystemExit(main())
