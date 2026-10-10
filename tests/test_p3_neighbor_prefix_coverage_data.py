import json
from pathlib import Path

import pytest

from terminal_marl.training import p3_neighbor_prefix_coverage_data as data


def _bundle(seed=893511, *, interface="o2m", entries=None, source_group="old"):
    entries = entries or [(10, 0, 2, True)]
    row_by_id = {}
    selected_by_id = {}
    annotation_by_id = {}
    label_by_id = {}
    label_files = {}
    label_sha256 = {}
    for index, (tick, uid, joint, feasible) in enumerate(entries):
        decision_id = f"did:{seed}:{tick}:{uid}:{index}"
        row = {
            "seed": seed,
            "partition": "train",
            "tick": tick,
            "uid": str(uid),
            "decision_id": decision_id,
            "actor_mask": True,
            "teacher_gate": True,
            "teacher_proposal": joint - 1,
            "context": {"interface": interface},
            "ego": [0.0] * 105,
            "traffic": [],
            "traffic_mask": [],
        }
        label = {
            "status": "OK",
            "decision_id": decision_id,
            "teacher_gate": True,
            "teacher_proposal": joint - 1,
            "teacher_joint_index": joint,
            "teacher_audit": [{
                "rank": 0,
                "joint_index": joint,
                "feasible": feasible,
                "reason": "certified" if feasible else "nominal_rejected",
            }],
        }
        row_by_id[decision_id] = row
        selected_by_id[decision_id] = {"row": {**row, "pre_state_hash": f"hash:{decision_id}"},
                                       "snapshot": "snapshot.pkl", "snapshot_sha256": "snapshot-sha"}
        annotation_by_id[decision_id] = {"status": "OK", "decision_id": decision_id,
                                         "label_file": f"labels/{index}.json"}
        label_by_id[decision_id] = label
        label_files[decision_id] = f"labels/{index}.json"
        label_sha256[decision_id] = f"label-sha:{index}"
    return {
        "seed": seed,
        "scenario": "fixture",
        "source_group": source_group,
        "directory": "fixture",
        "episode": "fixture/seed",
        "rows_path": "fixture/context.jsonl",
        "selected_path": "fixture/selected.json",
        "annotation_path": "fixture/annotation_result.json",
        "rows_sha256": "rows",
        "selected_sha256": "selected",
        "annotation_sha256": "annotation",
        "label_files": label_files,
        "label_sha256": label_sha256,
        "row_by_id": row_by_id,
        "selected_by_id": selected_by_id,
        "annotation_by_id": annotation_by_id,
        "label_by_id": label_by_id,
        "rows_count": len(row_by_id),
        "selected_count": len(row_by_id),
        "annotation_count": len(row_by_id),
        "label_count": len(row_by_id),
    }


def _reuse_for(choice, *, source="old"):
    candidate = {
        "joint": choice["joint"],
        "original_feasible": choice["original_feasible"],
        "original_reason": choice["original_reason"],
        "destinations": [],
    }
    return {
        (choice["decision_id"], choice["joint"]): {
            "source": source,
            "row": {"decision_id": choice["decision_id"], "candidates": [candidate]},
            "candidate": candidate,
        }
    }


def test_lower_median_selection_is_deterministic_and_unknown_suffix_rejected():
    bundle = _bundle(entries=[(30, 3, 2, True), (10, 1, 2, True), (20, 2, 2, True)])
    selected, counts = data.select_candidates([bundle], {})
    assert len(selected) == 1
    assert selected[0]["tick"] == 20
    assert selected[0]["uid"] == 2
    assert counts["total"]["candidate_requests"] == 1
    bad = _bundle(entries=[(10, 1, 26, True)])
    bad["label_by_id"][next(iter(bad["label_by_id"]))]["teacher_audit"][0]["joint_index"] = 26
    with pytest.raises(ValueError, match="outside 1..25"):
        data.select_candidates([bad], {})
    wait_prefix = _bundle(entries=[(11, 1, 2, True)])
    wait_id = next(iter(wait_prefix["label_by_id"]))
    wait_prefix["label_by_id"][wait_id]["teacher_audit"].insert(
        0, {"rank": 0, "joint_index": 0, "feasible": False, "reason": "wait"}
    )
    selected, _ = data.select_candidates([wait_prefix], {})
    assert len(selected) == 1 and selected[0]["joint"] == 2


def test_reuse_is_exact_candidate_key_and_smoke_skips_default_joints():
    bundles = [
        _bundle(entries=[(1, 1, 2, True), (4, 4, 4, True)], interface="o2m"),
        _bundle(entries=[(2, 2, 3, True)], interface="m2i"),
        _bundle(entries=[(3, 3, 13, True), (4, 4, 2, True)], interface="i2f"),
    ]
    selected, counts = data.select_candidates(bundles, {})
    assert counts["selected_candidate_groups"] == 5
    reuse = _reuse_for(next(item for item in selected if item["interface"] == "o2m"))
    selected, counts = data.select_candidates(bundles, reuse)
    reused = [item for item in selected if item["reuse"]]
    assert len(reused) == 1 and reused[0]["action"] == "reuse_existing_witness"
    smoke = data.select_smoke_candidates(selected)
    assert {item["interface"]: item["joint"] for item in smoke} == {
        "o2m": 4, "m2i": 3, "i2f": 2,
    }
    with pytest.raises(ValueError, match="no new non-default candidate"):
        data.select_smoke_candidates([item for item in selected if item["joint"] != 2])


def test_selection_tracks_ring_final_budget_and_no_duplicate_smoke_keys():
    ring = _bundle(seed=893511, interface="o2m", entries=[(1, 1, 2, True)])
    final = _bundle(seed=893512, interface="i2f", entries=[(1, 1, 2, True)])
    selected, counts = data.select_candidates([ring, final], {})
    assert counts["per_parent"]["893511"]["plan_upper_bound"] == 1
    assert counts["per_parent"]["893512"]["plan_upper_bound"] == 6
    assert counts["total"]["plan_upper_bound"] == 7
    with pytest.raises(ValueError, match="duplicated"):
        data.select_smoke_candidates([
            dict(selected[0], interface="o2m"),
            dict(selected[0], interface="m2i"),
            dict(selected[1], interface="i2f"),
        ])


def test_merge_parent_results_requires_all_complete_and_merges_whole_parent_sets(tmp_path):
    selected = [
        {"decision_id": "did-b", "parent_seed": 893512},
        {"decision_id": "did-a", "parent_seed": 893511},
    ]
    parents = []
    for seed, did in ((893511, "did-a"), (893512, "did-b")):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        (directory / "witnesses.jsonl").write_text(json.dumps({"decision_id": did}) + "\n")
        parents.append({"parent_seed": seed, "output": str(directory), "status": "COMPLETE",
                        "states": 1, "candidate_requests": 1, "plan_calls": 1})
    result = data.merge_parent_results(tmp_path, parents, selected)
    assert result["status"] == "COMPLETE"
    rows = [json.loads(line) for line in (tmp_path / "witnesses.jsonl").read_text().splitlines()]
    assert [row["decision_id"] for row in rows] == ["did-a", "did-b"]

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = [{"parent_seed": 893511, "output": str(tmp_path / "missing"),
               "status": "INCOMPLETE", "error": "fixture failure"}]
    result = data.merge_parent_results(failed_root, failed, selected[:1])
    assert result["status"] == "INCOMPLETE"
    assert not (failed_root / "witnesses.jsonl").exists()


def test_prepare_only_protocol_has_all_nine_inputs_and_zero_runtime_counters(tmp_path, monkeypatch):
    bundle = _bundle()
    inputs = [dict(bundle, seed=seed) for seed in data.ALL_TRAIN_SEEDS]
    monkeypatch.setattr(data, "load_validated_inputs", lambda *args: inputs)
    monkeypatch.setattr(data, "load_witness_reuse", lambda *args: ({}, {"streams": []}))
    result = data.collect([], [], [], "old", "supp", tmp_path / "prepared", prepare_only=True)
    assert result["status"] == "PREPARED"
    protocol = json.loads((tmp_path / "prepared" / "protocol.json").read_text())
    assert protocol["protocol"] == data.PREFIX_PROTOCOL
    assert protocol["input_seeds"] == list(data.ALL_TRAIN_SEEDS)
    assert protocol["resets"] == protocol["steps"] == protocol["teacher_queries"] == 0
    assert protocol["Q_accessed"] is False and protocol["original_labels_unchanged"] is True
    assert result["snapshot_reads"] == 0 and result["plan_calls"] == 0


def test_runtime_bundle_loads_all_labels_and_checks_label_hash(tmp_path):
    bundle = _bundle(entries=[(1, 1, 2, True), (2, 2, 3, False)])
    episode = tmp_path / "episode"
    episode.mkdir()
    rows_path = episode / "context.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in bundle["row_by_id"].values()))
    selected_path = episode / "selected.json"
    selected_path.write_text(json.dumps(list(bundle["selected_by_id"].values())))
    label_files = {}
    label_sha256 = {}
    for index, (did, label) in enumerate(bundle["label_by_id"].items()):
        label_path = episode / f"label_{index}.json"
        label_path.write_text(json.dumps(label))
        label_files[did] = label_path.name
        label_sha256[did] = data._sha(label_path)
    item = dict(bundle, episode=str(episode), rows_path=str(rows_path), selected_path=str(selected_path),
                label_files=label_files, label_sha256=label_sha256, rows_sha256=data._sha(rows_path),
                selected_sha256=data._sha(selected_path))
    runtime = data._runtime_bundle(item, [{"decision_id": next(iter(label_files))}])
    assert set(runtime["labels"]) == set(label_files)
    first_label = episode / next(iter(label_files.values()))
    first_label.write_text(first_label.read_text() + "\n")
    with pytest.raises(ValueError, match="runtime label changed"):
        data._runtime_bundle(item, [])


def test_run_items_reuses_manifest_candidate_but_writes_new_prefix_only(tmp_path, monkeypatch):
    selected = [
        {"parent_seed": 893511, "decision_id": "did:1", "joint": 2, "reuse": True, "audit_rank": 0},
        {"parent_seed": 893511, "decision_id": "did:1", "joint": 3, "reuse": False, "audit_rank": 1},
    ]
    reused_candidate = {"joint": 2, "destinations": [{"feasible": False}]}
    reuse_records = [{"joint": 2, "row": {"decision_id": "did:1", "candidates": [reused_candidate]},
                      "candidate": reused_candidate, "source": "old"}]
    runtime = {
        "row_by_id": {"did:1": {"decision_id": "did:1"}},
        "selected_by_id": {"did:1": {"row": {"pre_state_hash": "pre"}}},
        "labels": {"did:1": {}},
    }
    monkeypatch.setattr(data, "_runtime_bundle", lambda item, state: runtime)
    requested = []

    def fake_witness(episode, entry, row, label, *, remaining_calls, requested_joints):
        requested.append(list(requested_joints))
        return {"decision_id": "did:1", "candidates": [{"joint": 3, "destinations": []}], "plan_calls": 0}

    monkeypatch.setattr(data.base, "witness_state", fake_witness)
    result = data._run_items(
        tmp_path / "prefix",
        selected,
        {"893511": {"episode": "fixture"}},
        {"did:1": reuse_records},
        wall_seconds=10,
        max_candidate_requests=3,
        max_plan_calls=18,
    )
    assert result["status"] == "COMPLETE"
    assert result["states"] == 1 and result["candidate_requests"] == 1
    assert result["new_candidate_requests"] == 1 and result["reused_candidate_requests"] == 1
    assert requested == [[3]]
    output = [json.loads(line) for line in (tmp_path / "prefix" / "witnesses.jsonl").read_text().splitlines()]
    assert [candidate["joint"] for candidate in output[0]["candidates"]] == [3]


def test_incomplete_root_result_replaces_running_marker(tmp_path):
    output = tmp_path / "root"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"status": "RUNNING"}))
    result = data._incomplete_root_result(output, [], "parent failure")
    assert result["status"] == "INCOMPLETE"
    assert json.loads((output / "result.json").read_text())["status"] == "INCOMPLETE"
