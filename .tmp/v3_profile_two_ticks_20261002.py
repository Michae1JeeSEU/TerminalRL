from __future__ import annotations

import cProfile
import hashlib
import io
import json
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terminal_marl.scripts import evaluate_p3_foundation_repair_20261002 as evaluator


OUTPUT = Path(
    "/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/"
    "profile_legacy88_two_ticks"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_stats(profile: cProfile.Profile, path: Path, sort_key: str, limit: int) -> None:
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(sort_key).print_stats(limit)
    path.write_text(stream.getvalue(), encoding="utf-8")


def main() -> int:
    result_path = OUTPUT / "result.json"
    profile_path = OUTPUT / "profile.pstats"
    cumulative_path = OUTPUT / "top35_cumulative.txt"
    tottime_path = OUTPUT / "top25_tottime.txt"
    targets = (result_path, profile_path, cumulative_path, tottime_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing artifacts: {existing}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    base = evaluator._base_config()
    spec = evaluator._arm_specs()[0]
    profile = cProfile.Profile()
    result = profile.runcall(evaluator._run_arm, spec, base, max_steps=2)

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    profile.dump_stats(str(profile_path))
    _write_stats(profile, cumulative_path, "cumulative", 35)
    _write_stats(profile, tottime_path, "tottime", 25)

    summary = {
        "source_sha256": _sha256(Path(evaluator.__file__)),
        "result_status": result.get("status"),
        "result_censored": result.get("censored"),
        "validity": result.get("validity"),
        "safety": result.get("safety"),
        "elapsed_wall_seconds": result.get("elapsed_wall_seconds"),
        "profile_dir": str(OUTPUT),
    }
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0 if result.get("error") is None and result.get("lifecycle_error") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
