"""Render fixed-time views from the formal A4 replay bundles.

This artifact-local helper reads saved replay bundles and uses the production
replay viewer in off-screen mode. It does not run the physical simulator and
does not modify any replay bundle or production source.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from terminal_marl.visualization.rotating_slots_replay import (  # noqa: E402
    ReplayController,
    RotatingSlotsReplayViewer,
)


OUTPUT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path(
    r"D:\7_codespace\training_outputs\terminal\rs_selective_quality_a4_20261030_review"
)
ARMS = (
    ("s3r23", "FULL_H"),
    ("s3r23", "RULE_LOCAL_K8"),
    ("s3r24", "FULL_H"),
    ("s3r24", "RULE_LOCAL_K8"),
)
TIMES = (20.0, 170.0, 260.0, 320.0)
VIEWS = ("top_down", "side", "3d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(replay_root: Path) -> dict:
    replay_index_path = replay_root / "replay.json"
    replay_index = json.loads(replay_index_path.read_bytes())
    names = ["replay.json", *sorted(replay_index["files"])]
    files = []
    for name in names:
        path = replay_root / name
        files.append(
            {
                "relative_path": name,
                "absolute_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(
        [(row["relative_path"], row["sha256"]) for row in files],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    acceptance = json.loads((replay_root / "visual_acceptance.json").read_bytes())
    return {
        "replay_root": str(replay_root.resolve()),
        "replay_json_sha256": next(
            row["sha256"] for row in files if row["relative_path"] == "replay.json"
        ),
        "input_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "replay_index_human_acceptance_observed": replay_index.get(
            "human_acceptance"
        ),
        "human_acceptance_observed": acceptance.get("researcher_visual_acceptance"),
        "visualization_integration_status_observed": acceptance.get(
            "visualization_integration_status"
        ),
    }


def time_label(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"t{text}"


def arm_replay_root(scenario: str, method: str) -> Path:
    return (
        INPUT_ROOT
        / "arms"
        / scenario
        / method
        / "evaluation"
        / scenario
        / method
        / "replay"
    )


def render_arm(scenario: str, method: str) -> dict:
    replay_root = arm_replay_root(scenario, method)
    inventory = source_inventory(replay_root)
    controller = ReplayController(replay_root)
    viewer = None
    frames = []
    try:
        viewer = RotatingSlotsReplayViewer(controller, off_screen=True)
        for requested_time in TIMES:
            target = OUTPUT_ROOT / scenario / method / time_label(requested_time)
            target.mkdir(parents=True, exist_ok=True)
            snapshot = viewer.draw(controller.seek(requested_time))
            actual_time = float(snapshot.time)
            if actual_time != requested_time:
                raise RuntimeError(
                    f"seek time mismatch for {scenario}/{method}: "
                    f"requested={requested_time} actual={actual_time}"
                )
            paths = viewer.screenshots(target)
            by_view = {Path(path).stem: Path(path) for path in paths}
            if set(by_view) != set(VIEWS):
                raise RuntimeError(
                    f"unexpected screenshot views for {scenario}/{method} "
                    f"{requested_time}: {sorted(by_view)}"
                )
            images = []
            for view in VIEWS:
                path = by_view[view]
                if not path.is_file() or path.stat().st_size <= 0:
                    raise RuntimeError(f"missing or empty screenshot: {path}")
                images.append(
                    {
                        "view": view,
                        "path": str(path.resolve()),
                        "relative_path": path.resolve()
                        .relative_to(OUTPUT_ROOT.resolve())
                        .as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            frames.append(
                {
                    "requested_time_s": requested_time,
                    "actual_time_s": actual_time,
                    "images": images,
                }
            )
    finally:
        if viewer is not None:
            viewer.close()
    return {
        "scenario": scenario,
        "method": method,
        "replay_root": str(replay_root.resolve()),
        "time_bounds_s": list(controller.bundle.time_bounds),
        "sample_dt_s": float(controller.bundle.index["sample_dt_s"]),
        "input": inventory,
        "frames": frames,
    }


def main() -> None:
    renderer_source = Path(inspect.getfile(ReplayController)).resolve()
    renderer_source_sha256_before = sha256_file(renderer_source)
    results = [render_arm(scenario, method) for scenario, method in ARMS]
    renderer_source_sha256_after = sha256_file(renderer_source)
    if renderer_source_sha256_before != renderer_source_sha256_after:
        raise RuntimeError(
            "production renderer source SHA changed during rendering: "
            f"before={renderer_source_sha256_before} "
            f"after={renderer_source_sha256_after}"
        )
    manifest = {
        "artifact": "rs_select_quality_a4_20261030_visual_review_formal",
        "human_visual_acceptance_status": "HUMAN_PENDING",
        "renderer": {
            "module": "terminal_marl.visualization.rotating_slots_replay",
            "controller": "ReplayController",
            "viewer": "RotatingSlotsReplayViewer",
            "off_screen": True,
            "python": sys.executable,
            "pyvista_off_screen_env": os.environ.get("PYVISTA_OFF_SCREEN"),
            "source_path": str(renderer_source),
            "source_sha256_before": renderer_source_sha256_before,
            "source_sha256_after": renderer_source_sha256_after,
        },
        "input_root": str(INPUT_ROOT.resolve()),
        "arms": [
            {"scenario": scenario, "method": method} for scenario, method in ARMS
        ],
        "requested_times_s": list(TIMES),
        "requested_views": list(VIEWS),
        "results": results,
        "frame_count": len(results) * len(TIMES),
        "image_count": len(results) * len(TIMES) * len(VIEWS),
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "frames": len(results) * len(TIMES),
                "images": len(results) * len(TIMES) * len(VIEWS),
            }
        )
    )


if __name__ == "__main__":
    main()
