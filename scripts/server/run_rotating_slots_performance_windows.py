"""Stdlib-only, fresh-process bootstrap for a single serial common window."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HELPER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HELPER_ROOT / "terminal_marl/core/rotating_slots"))
from performance.windows import main


if __name__ == "__main__":
    raise SystemExit(main(helper_root=HELPER_ROOT))
