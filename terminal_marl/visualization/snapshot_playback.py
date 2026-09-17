"""Portable playback of the existing renderer-neutral viewer snapshots."""
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from .snapshot import build_world_render_snapshot


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class SnapshotRecorder:
    def __init__(self, stride=2):
        self.stride, self.frames = stride, []

    def capture(self, env):
        snapshot = plain(build_world_render_snapshot(env, trajectory_max_points=30))
        if not self.frames or self.frames[-1]["step_count"] != snapshot["step_count"]:
            self.frames.append(snapshot)

    def __call__(self, env):
        if env.step_count % self.stride == 0:
            self.capture(env)


def write_playback(path: Path, frames, title):
    if not frames:
        raise ValueError("no viewer frames captured")
    template = Path(__file__).with_name("snapshot_playback.html").read_text(encoding="utf-8")
    data = json.dumps({"frames": frames, "title": title}, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(template.replace("__SNAPSHOT_DATA__", data), encoding="utf-8")
