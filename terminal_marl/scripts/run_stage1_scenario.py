"""Generate a named actual scenario and an existing-platform replay bundle."""
import argparse
import json
import os
import subprocess
import sys
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
from pathlib import Path
from ..core.rotating_slots.qualification import Evidence, provenance, write_json


def main(argv=None):
    from ..core.rotating_slots.fixtures import SCENARIOS, execute_scenario
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--output", required=True)
    p.add_argument("--code-commit")
    args = p.parse_args(argv)
    if args.code_commit: os.environ["TERMINAL_CODE_COMMIT"] = args.code_commit
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    command = subprocess.list2cmdline([sys.executable, "-m", "terminal_marl.scripts.run_stage1_scenario", *(sys.argv[1:] if argv is None else argv)])
    e = Evidence(args.scenario, root, provenance(command))
    execute_scenario(args.scenario, e)
    e.save_replays()
    write_json(root/"fixture.json", {"actions": e.actions, "observations": e.observations, "assertions": e.assertions, "provenance": e.provenance})
    instructions = {"scenario": args.scenario, "replay_paths": e.replay_paths,
        "researcher_visual_acceptance": "PENDING", "required_views": ["top_down", "side", "3d"],
        "commands": [f'python -m terminal_marl.scripts.open_stage1_replay "{path}" --screenshots "{path}/screenshots"' for path in e.replay_paths]}
    write_json(root/"viewer_instructions.json", instructions)
    print(json.dumps(instructions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
