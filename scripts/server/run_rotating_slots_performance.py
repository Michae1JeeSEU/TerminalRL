"""Stdlib-only bootstrap for isolated FT-RS-S1-PERF-1.0 execution."""
from pathlib import Path
import sys

# Select helper code without importing terminal_marl (which imports numerical
# modules in some historical package versions). runner.main binds one core root.
HELPER_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HELPER_ROOT / "terminal_marl/core/rotating_slots"))
from performance.runner import main


if __name__ == "__main__":
    raise SystemExit(main(helper_root=HELPER_ROOT))
