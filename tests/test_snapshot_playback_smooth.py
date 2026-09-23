import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from terminal_marl.visualization.snapshot_playback import write_playback


TEMPLATE = Path("terminal_marl/visualization/snapshot_playback.html")


def _script_source():
    source = TEMPLATE.read_text(encoding="utf-8")
    match = re.search(r"<script>(.*?)</script>", source, flags=re.DOTALL)
    assert match is not None
    return source, match.group(1)


def _run_node_behavior(script, frames):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser playback behavior checks")

    payload = {"title": "behavior fixture", "frames": frames}
    script = script.replace(
        "const data=__SNAPSHOT_DATA__;",
        f"const data={json.dumps(payload, separators=(',', ':'))};",
    )
    prelude = textwrap.dedent(
        """
        const assert = require('node:assert/strict');
        class MockElement {
          constructor(id) {
            this.id = id;
            this.value = '';
            this.checked = true;
            this.textContent = '';
            this.style = {};
            this.children = [];
            this.replaceCount = 0;
            this.appendCount = 0;
            this.widthAssignments = 0;
            this.heightAssignments = 0;
            this._width = 0;
            this._height = 0;
          }
          set width(value) { this._width = value; this.widthAssignments += 1; }
          get width() { return this._width; }
          set height(value) { this._height = value; this.heightAssignments += 1; }
          get height() { return this._height; }
          getBoundingClientRect() { return {width: 400, height: 300}; }
          replaceChildren(...children) {
            this.children = children;
            this.replaceCount += 1;
          }
          append(child) { this.children.push(child); this.appendCount += 1; }
        }
        const elements = new Map();
        for (const id of ['scene', 'timeline', 'interpolation', 'play', 'reset', 'speed', 'angle', 'trails', 'title', 'hud', 'agents']) {
          elements.set(id, new MockElement(id));
        }
        elements.get('speed').value = '10';
        elements.get('angle').value = '35';
        elements.get('trails').checked = true;
        elements.get('interpolation').checked = true;
        const canvasContext = {
          setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {},
          stroke() {}, fillText() {}, fillRect() {}, arc() {}, fill() {},
          strokeStyle: '', lineWidth: 0, fillStyle: ''
        };
        elements.get('scene').getContext = () => canvasContext;
        globalThis.document = {
          getElementById(id) { return elements.get(id); },
          createElement() { return new MockElement('created'); }
        };
        globalThis.window = {devicePixelRatio: 2};
        globalThis.requestAnimationFrame = () => 0;
        """
    )
    postlude = textwrap.dedent(
        """
        assert.equal(displayAlpha(), 0);
        sim = 5;
        assert.equal(displayAlpha(), 0.5);
        const first = frames[0].uavs[0];
        assert.deepEqual(displayPosition(first, frames[1], displayAlpha()), [5, 0, 100]);
        const phaseChanged = frames[0].uavs[1];
        assert.deepEqual(displayPosition(phaseChanged, frames[1], displayAlpha()), phaseChanged.position);
        const missing = frames[0].uavs[2];
        assert.deepEqual(displayPosition(missing, frames[1], displayAlpha()), missing.position);
        interpolationToggle.checked = false;
        assert.equal(displayAlpha(), 0);
        interpolationToggle.checked = true;

        const scene = document.getElementById('scene');
        assert.equal(scene.widthAssignments, 1);
        assert.equal(scene.heightAssignments, 1);
        draw();
        draw();
        assert.equal(scene.widthAssignments, 1);
        assert.equal(scene.heightAssignments, 1);

        const agentsElement = document.getElementById('agents');
        const detailsBeforeRepeat = agentsElement.replaceCount;
        draw();
        assert.equal(agentsElement.replaceCount, detailsBeforeRepeat);
        setSourceFrame(1, true);
        assert.equal(agentsElement.replaceCount, detailsBeforeRepeat + 1);
        assert.match(document.getElementById('hud').textContent, /仿真时间 10 s/);
        draw();
        assert.equal(agentsElement.replaceCount, detailsBeforeRepeat + 1);

        setSourceFrame(0, true);
        sim = frames[0].time;
        playing = false;
        last = 0;
        playButton.onclick();
        tick(1000);
        tick(2000);
        assert.equal(index, frames.length - 1);
        assert.equal(sim, frames[frames.length - 1].time);
        assert.equal(playing, false);
        assert.equal(playButton.textContent, '播放');
        """
    )
    result = subprocess.run(
        [node, "-"],
        input=f"{prelude}\n{script}\n{postlude}",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Node playback behavior check failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_playback_behavior_updates_canvas_hud_and_interpolates_positions():
    _, script = _script_source()
    frames = [
        {
            "step_count": 0,
            "time": 0.0,
            "external_queue_size": 1,
            "service_count": 0,
            "completed_count": 0,
            "rings": [{"radius": 20, "height": 10, "role": "outer", "occupancy": 1, "capacity": 4}],
            "gates": [{"spawn_position": [0, 0, 0], "gate_position": [1, 0, 0], "gate_id": 1}],
            "landing_zones": [{"position": [2, 0, 0], "status": "free"}],
            "uavs": [
                {"uid": 1, "phase": "arrival", "position": [0, 0, 100], "speed": 2.0, "soc": 0.8, "committed": False, "last_release_decision": "wait", "trajectory_points": []},
                {"uid": 2, "phase": "wait", "position": [0, 1, 100], "speed": 2.0, "soc": 0.7, "committed": False, "last_release_decision": "wait", "trajectory_points": []},
                {"uid": 3, "phase": "wait", "position": [0, 2, 100], "speed": 2.0, "soc": 0.6, "committed": False, "last_release_decision": "wait", "trajectory_points": []},
            ],
        },
        {
            "step_count": 1,
            "time": 10.0,
            "external_queue_size": 0,
            "service_count": 1,
            "completed_count": 0,
            "rings": [{"radius": 20, "height": 10, "role": "outer", "occupancy": 2, "capacity": 4}],
            "gates": [{"spawn_position": [0, 0, 0], "gate_position": [1, 0, 0], "gate_id": 1}],
            "landing_zones": [{"position": [2, 0, 0], "status": "occupied"}],
            "uavs": [
                {"uid": 1, "phase": "arrival", "position": [10, 0, 100], "speed": 2.0, "soc": 0.7, "committed": True, "last_release_decision": "release", "trajectory_points": []},
                {"uid": 2, "phase": "final", "position": [10, 1, 100], "speed": 2.0, "soc": 0.6, "committed": True, "last_release_decision": "release", "trajectory_points": []},
            ],
        },
    ]
    _run_node_behavior(script, frames)


def test_write_playback_embeds_template_with_smooth_controls(tmp_path):
    frames = [
        {
            "step_count": 0,
            "time": 0.0,
            "uavs": [],
        }
    ]
    path = tmp_path / "viewer.html"
    write_playback(path, frames, "smooth playback")
    output = path.read_text(encoding="utf-8")

    assert "__SNAPSHOT_DATA__" not in output
    assert 'id="interpolation"' in output
    assert "WorldRenderSnapshot" in output
    assert '"title": "smooth playback"' in output
