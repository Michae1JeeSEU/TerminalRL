"""Parallel seed expansion for A6.3 rule-policy short screens only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from .a63_dynamic_screen import run_pair, write_pair


def main() -> None:
    tasks = [(scenario, seed) for scenario in ("high", "bursty") for seed in (1, 2)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_pair, scenario, seed): (scenario, seed) for scenario, seed in tasks}
        for future in as_completed(futures):
            result = future.result()
            write_pair(result, Path.home() / "training_outputs" / "terminal" / "event_gated_stage15_diagnostics" / "a63")
            print(json.dumps({"scenario": result["scenario"], "seed": result["seed"],
                              "protected_safety_passed": result["protected_safety_passed"],
                              **result["delta_dynamic_minus_fixed"]}), flush=True)
            if not result["protected_safety_passed"]:
                raise SystemExit(2)


if __name__ == "__main__":
    main()
