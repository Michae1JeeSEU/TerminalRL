"""Small replay projections preserve archive verification and display facts."""
from hashlib import sha256
from pathlib import Path
import json

import pytest

from terminal_marl.core.rotating_slots.config import load_config
from terminal_marl.core.rotating_slots import replay as replay_module
from terminal_marl.core.rotating_slots.replay import ReplayBundle, _plain
from terminal_marl.core.rotating_slots.trajectory import generate_transfer
from terminal_marl.visualization import rotating_slots_controller as controller_module
from terminal_marl.visualization.rotating_slots_controller import ReplayController


OVERLAY_TYPES = {"WorkDescriptor", "FactualResult", "Rank"}


@pytest.fixture
def archive(tmp_path):
    plan = generate_transfer("actual", "outer", 0, 1, 0., durations=(40.,)).plan
    assert plan is not None
    states = []
    for time in (0., 1., 2.):
        state = plan.state(time)
        states.append(dict(time=time, aircraft=[dict(id="actual", status="TRANSFER_O_M",
            soc=.9, p=state.p, v=state.v, a=state.a, jerk=state.jerk,
            ring="outer", slot=0, trajectory_hash=plan.trajectory_hash)],
            slots=[], pads=[], metrics={}))
    trace = []
    for time in (0., 2.):
        identity = f"proposal-{time}"
        trace.extend([
            dict(record_type="WorkDescriptor", as_of=time, descriptor_id=identity,
                 snapshot_id=f"snapshot-{time}", extra={"source": ["recorded", time]}),
            dict(record_type="FeaturesBeforeQuery", as_of=time, feature_id=f"features-{time}",
                 features={"query": [time, 7.], "entities": [[1., 2.], [3., 4.]]}),
            dict(record_type="Rank", as_of=time, ordered_ids=[identity, "other"],
                 scores={identity: .8}),
            dict(record_type="FactualResult", as_of=time, descriptor_id=identity,
                 outcome="SINGLE_VALID", reason=None, features_before_query_ref=f"features-{time}"),
            dict(record_type="Coverage", as_of=time, attempted=1),
        ])
    return ReplayBundle.write(tmp_path / "replay", code_commit="projection-unit-fixture",
        record=dict(config_hash=load_config().config_hash, states=states,
                    trajectories={plan.trajectory_hash: plan.to_dict()},
                    manifest={"requests": []}, events=[], metrics={}, rejections=[],
                    stage3_trace=trace), eager=False)


def test_projection_streams_trace_and_retains_complete_selected_rows(archive, monkeypatch):
    trace_path = archive.path / "stage3_trace.jsonl"
    before = {path.name: sha256(path.read_bytes()).hexdigest()
              for path in archive.path.iterdir() if path.is_file()}
    read_bytes = Path.read_bytes

    def forbid_trace_read_bytes(path):
        assert path != trace_path, "trace must be streamed rather than read into one bytes object"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbid_trace_read_bytes)
    full = ReplayBundle.load(archive.path)
    projected = ReplayBundle.load(archive.path, stage3_trace_types=OVERLAY_TYPES)
    assert any(row["record_type"] == "FeaturesBeforeQuery" for row in full.data["stage3_trace"])
    assert set(row["record_type"] for row in projected.data["stage3_trace"]) == OVERLAY_TYPES
    assert projected.data["stage3_trace"] == tuple(
        row for row in full.data["stage3_trace"] if row["record_type"] in OVERLAY_TYPES)
    assert _plain(projected.index) == _plain(full.index)
    assert before == {path.name: sha256(read_bytes(path)).hexdigest()
                      for path in archive.path.iterdir() if path.is_file()}


def test_omitted_feature_tampering_still_rejected_by_complete_hash(archive):
    trace_path = archive.path / "stage3_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_bytes().splitlines()]
    feature = next(row for row in rows if row["record_type"] == "FeaturesBeforeQuery")
    feature["features"]["query"][0] = 99.
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="replay content hash mismatch: stage3_trace.jsonl"):
        ReplayBundle.load(archive.path, stage3_trace_types=OVERLAY_TYPES)


def test_trace_change_between_precheck_and_parse_is_rejected(archive, monkeypatch):
    trace_path = archive.path / "stage3_trace.jsonl"
    original = trace_path.read_bytes()
    changed = original.replace(b'"query":[0.0,7.0]', b'"query":[99.0,7.0]', 1)
    assert changed != original
    file_hash = replay_module._file_hash
    prechecked = []

    def replace_after_precheck(path):
        digest = file_hash(path)
        if path == trace_path:
            prechecked.append(digest)
            trace_path.write_bytes(changed)
        return digest

    monkeypatch.setattr(replay_module, "_file_hash", replace_after_precheck)
    with pytest.raises(ValueError, match="replay content hash mismatch: stage3_trace.jsonl"):
        ReplayBundle.load(archive.path, stage3_trace_types=OVERLAY_TYPES)
    assert prechecked == [sha256(original).hexdigest()]
    assert trace_path.read_bytes() == changed


@pytest.mark.parametrize("time, count", [(0., 1), (1., 1), (2., 2)])
def test_controller_projection_matches_full_overlay_hud_and_commits(archive, time, count, monkeypatch):
    full = ReplayBundle.load(archive.path)
    from_bundle = ReplayController(full)
    from_path = ReplayController(archive.path)
    assert from_bundle.bundle is full
    assert set(row["record_type"] for row in from_path.bundle.data["stage3_trace"]) == OVERLAY_TYPES
    original_plain = controller_module._plain

    def forbid_unused_conversion(value):
        if hasattr(value, "get"):
            assert value.get("record_type") not in {"FeaturesBeforeQuery", "Coverage"}
        return original_plain(value)

    monkeypatch.setattr(controller_module, "_plain", forbid_unused_conversion)
    from_bundle.seek(time)
    from_path.seek(time)
    expected = from_bundle.overlays()
    actual = from_path.overlays()
    assert actual == expected
    for name in ("stage3_proposals", "stage3_attempts", "stage3_ranking"):
        assert len(actual[name]) == count
    assert len(actual["current_commits"]) == 1
    assert actual["current_commits"][0]["id"] == "actual"
    assert actual["current_commits"][0]["p"] == from_path._plans[
        actual["current_commits"][0]["trajectory_hash"]].state(time).p
    assert from_path.hud_lines() == from_bundle.hud_lines()
    assert from_path.hud_lines(details=True) == from_bundle.hud_lines(details=True)
    assert any(f"proposals={count}; attempts={count}; commits=1" in line
               for line in from_path.hud_lines())
    assert any("Last proposal order:" in line for line in from_path.hud_lines())
    assert any("Last attempt" in line for line in from_path.hud_lines())
