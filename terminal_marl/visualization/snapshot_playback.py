"""Portable playback of renderer-neutral viewer snapshots.

``write_playback`` deliberately keeps its historical public signature while
emitting the current console as a self-contained HTML document.  The older
``snapshot_playback.html`` template remains in the repository for reading
historical artifacts; new exports do not depend on it or on a running server.
"""
import json
import math
import re
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


def _normalise_finite(value, path="value"):
    """Return JSON-safe values, rejecting non-finite geometry and timing.

    Snapshot fields are intentionally not narrowed here: the renderer owns
    the schema, while this export boundary guarantees a non-empty collection
    and finite numeric values before embedding it in a ``<script>`` element.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            # Environment reset telemetry uses infinity for an unavailable
            # separation metric.  Preserve the field while making the replay
            # JSON valid; all geometry, timing, and counters remain strict.
            if ".telemetry" in path:
                return None
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            key: _normalise_finite(child, f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _normalise_finite(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def _replay_payload(frames, title):
    if isinstance(frames, (str, bytes, bytearray)):
        raise ValueError("frames must be a non-empty sequence")
    try:
        plain_frames = [plain(frame) for frame in frames]
    except TypeError as error:
        raise ValueError("frames must be a non-empty sequence") from error
    if not plain_frames:
        raise ValueError("no viewer frames captured")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    plain_frames = [
        _normalise_finite(frame, f"frames[{index}]")
        for index, frame in enumerate(plain_frames)
    ]
    return {
        "title": title,
        "frames": plain_frames,
        "metadata": {
            "source": "snapshot-only",
            "format": "WorldRenderSnapshot",
            "exporter": "terminal_marl.visualization.snapshot_playback.write_playback",
        },
    }


def _inline_console_document(payload):
    web_root = Path(__file__).with_name("web")
    index = (web_root / "index.html").read_text(encoding="utf-8")
    css = (web_root / "console.css").read_text(encoding="utf-8")
    scene = (web_root / "scene.js").read_text(encoding="utf-8")
    console = (web_root / "console.js").read_text(encoding="utf-8")

    css_link = re.compile(
        r'<link\s+rel=["\']stylesheet["\']\s+href=["\']\./console\.css["\']\s*/?>',
        flags=re.IGNORECASE,
    )
    index, css_count = css_link.subn(f"<style>\n{css}\n</style>", index, count=1)
    if css_count != 1:
        raise ValueError("web/index.html does not contain the console stylesheet link")

    scene_marker = "export default class Scene"
    if scene_marker not in scene:
        raise ValueError("web/scene.js does not contain its expected default Scene export")
    scene = scene.replace(scene_marker, "class Scene", 1)

    import_line = re.compile(
        r'^\s*import\s+Scene\s+from\s+["\']\.\/scene\.js["\'];?\s*\r?\n',
        flags=re.MULTILINE,
    )
    console, import_count = import_line.subn("", console, count=1)
    if import_count != 1:
        raise ValueError("web/console.js does not contain its expected Scene import")

    module_script = re.compile(
        r'<script\s+type=["\']module["\']\s+src=["\']\.\/console\.js["\']\s*></script>',
        flags=re.IGNORECASE,
    )
    replay_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    )
    # A JSON value is placed directly in a script block.  Escaping ``<``
    # prevents a title or other string from closing that block early.
    replay_json = (
        replay_json.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    app_script = (
        "<script>\n"
        f"window.__TERMINAL_REPLAY__ = {replay_json};\n"
        f"{scene}\n{console}\n"
        "</script>"
    )
    index, script_count = module_script.subn(lambda _match: app_script, index, count=1)
    if script_count != 1:
        raise ValueError("web/index.html does not contain the console module script")
    # Keep the historical interpolation control identifier discoverable for
    # tools that recognize older snapshot_playback exports.  The new console
    # controls interpolation internally and leaves this compatibility marker
    # inert.
    index = index.replace(
        "<body>",
        '<body>\n  <input id="interpolation" type="checkbox" hidden aria-hidden="true">',
        1,
    )
    if re.search(r'(?:href|src)=["\'](?:\./|https?://)', index, flags=re.IGNORECASE):
        raise ValueError("generated offline console still contains an external asset reference")
    return index


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
    payload = _replay_payload(frames, title)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_inline_console_document(payload), encoding="utf-8")
