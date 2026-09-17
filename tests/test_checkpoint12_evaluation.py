import json
import pytest
from terminal_marl.training.evaluate_v4_2_ppo_checkpoint12 import PROTOCOL, PROTOCOL_SHA, _sha, run
from terminal_marl.training.run_v4_2_v2_1_student_qualification import scenario_config, protocol
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.visualization.snapshot_playback import SnapshotRecorder, write_playback


def test_protocol_hash_and_wrong_training_evidence(tmp_path):
    assert _sha(PROTOCOL) == PROTOCOL_SHA
    path = tmp_path / "manifest.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="training evidence"):
        run(tmp_path / "out", path, path, tmp_path)


def test_existing_snapshots_serialize_without_mutation_and_escape_html(tmp_path):
    env = TerminalParallelEnv(scenario_config("warm3", protocol()))
    env.reset(1)
    recorder = SnapshotRecorder()
    recorder.capture(env)
    recorder.capture(env)
    assert len(recorder.frames) == 1
    assert len(recorder.frames[0]["uavs"]) == 3
    json.dumps(recorder.frames)
    path = tmp_path / "viewer.html"
    write_playback(path, recorder.frames, '</script><script>bad()</script>')
    assert '__SNAPSHOT_DATA__' not in path.read_text(encoding='utf-8')
    assert '</script><script>bad()' not in path.read_text(encoding='utf-8')
    assert env.step_count == 0
