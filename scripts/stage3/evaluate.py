"""One frozen evaluation; the runner owns heldout gating and hard deadlines."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from terminal_marl.core.rotating_slots.stage3_learning.campaign import main

if __name__ == "__main__": raise SystemExit(main(["evaluate",*sys.argv[1:]]))
