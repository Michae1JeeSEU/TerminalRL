"""Stdlib bootstrap: collector never imports core before its fresh kernel child."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

HELPER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HELPER_ROOT / "terminal_marl/core/rotating_slots"))
from performance.paired import main


if __name__ == "__main__":
    raise SystemExit(main(helper_root=HELPER_ROOT))
