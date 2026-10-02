import math
import re

import pytest

from terminal_marl.visualization.console_server import load_replay
from terminal_marl.visualization.snapshot_playback import write_playback


def _frame(time=0.0, step_count=0):
    return {
        "time": time,
        "step_count": step_count,
        "rings": [],
        "gates": [],
        "landing_zones": [],
        "uavs": [],
        "terminal_events": [],
        "recent_conflict_uids": [],
        "telemetry": {},
    }


def test_write_playback_embeds_the_console_as_a_single_file(tmp_path):
    path = tmp_path / "console.html"
    title = "终端区离线回放"
    write_playback(path, [_frame()], title)
    output = path.read_text(encoding="utf-8")

    assert output
    assert f'"title": "{title}"' in output
    assert "window.__TERMINAL_REPLAY__ =" in output
    assert "<style>" in output
    assert "export default class Scene" not in output
    assert "import Scene from \"./scene.js\"" not in output
    assert not re.search(r'(?:href|src)=["\'](?:\./|https?://)', output, re.IGNORECASE)


def test_write_playback_escapes_script_closing_text_and_loads_back(tmp_path):
    path = tmp_path / "escaped-console.html"
    title = "中文 </script><script>bad()"
    write_playback(path, [_frame()], title)
    output = path.read_text(encoding="utf-8")

    assert "</script><script>bad()" not in output
    loaded_title, frames, metadata = load_replay(path)
    assert loaded_title == title
    assert frames[0]["step_count"] == 0
    assert metadata["source"] == "snapshot-only"


@pytest.mark.parametrize(
    "frames",
    [[], [_frame(time=math.nan)], [_frame(time=math.inf)]],
)
def test_write_playback_rejects_empty_or_non_finite_frames(tmp_path, frames):
    with pytest.raises(ValueError):
        write_playback(tmp_path / "invalid.html", frames, "invalid")
