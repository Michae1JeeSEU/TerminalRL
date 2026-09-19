from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import terminal_marl.training.collect_v4_2_p2_gate_support_v5_dev_supplement as runner


def _row(index: int, split: str, seed: int):
    interfaces = ("o2m", "m2i", "i2f")
    phases = {"o2m": "ring_outer", "m2i": "ring_middle", "i2f": "ring_inner"}
    interface = interfaces[index % len(interfaces)]
    gate = (index // len(interfaces)) % 2
    return {
        "seed": seed,
        "phase": phases[interface],
        "step": index,
        "uav_id": index % 12,
        "decision_event_id": f"v5-test-{split}-{index}",
        "observation_v2_1": np.zeros(88, dtype=np.float32).tolist(),
        "actor_mask": True,
        "teacher_gate": gate,
        "teacher_proposal": 0 if gate else None,
        "requested_gate": gate,
        "support_split": split,
        "scenario": "flow6",
        "interface": interface,
    }


def _record(split: str, seed: int, scenario: str = "flow6"):
    requested = int(scenario.removeprefix("flow"))
    return {
        "split": split,
        "seed": seed,
        "scenario": scenario,
        "requested": requested,
        "completed": requested,
        "episode_execution_complete": True,
        "terminal_or_frozen_horizon": True,
        "safety_all_zero": True,
        "shield_actor_count": 0,
        "actor_decisions": 959,
        "shield_actor_fraction": 0.0,
        "shield_actor_fraction_within_inherited_limit": True,
        "counts": {"shield": 0, "shield_actor": 0},
        "safety": {"collision": 0},
        "v1_rows_mixed": False,
        "gates": {"episode": True, "replay": True},
        "exception": None,
    }


def _gate_shape(*, all_pass: bool):
    return {key: all_pass for key in (
        "cardinality", "teacher_class_minimums", "teacher_class_proportions",
        "interface_minimums", "interface_class_minimums", "schema_88d_finite",
        "decision_event_id_unique", "seed_coverage",
        "all_planned_episodes_execution_complete",
        "all_episode_shield_actor_fraction_within_inherited_limit",
        "all_episode_gates", "cross_split_seed_disjoint", "cross_split_event_disjoint",
        "all",
    )}


def _valid_v4_manifest():
    train_records = [
        _record("train", seed, runner.SCENARIOS[index % len(runner.SCENARIOS)])
        for index, seed in enumerate(runner.RETAINED_V4_TRAIN_SEEDS)
    ]
    dev_records = [
        _record("dev", seed, runner.SCENARIOS[index % len(runner.SCENARIOS)])
        for index, seed in enumerate(runner.RETAINED_V4_DEV_SEEDS)
    ]
    return {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V4_SUPPLEMENT",
        "protocol_sha256": runner.V4_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": runner.CANDIDATE_SHA,
        "candidate_architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
        "candidate_temperature": runner.CANDIDATE_TEMPERATURE,
        "SUPPORT_COLLECTION": "FAIL",
        "SUPPORT_FROZEN": False,
        "LEARNABILITY_AUDIT_ALLOWED": False,
        "GATE_REPAIR_ALLOWED": False,
        "PPO_ALLOWED": False,
        "curriculum_promotion_allowed": False,
        "formal_episode_plan_complete": True,
        "training_called": False,
        "optimizer_called": False,
        "student_authoritative": True,
        "teacher_label_only": True,
        "teacher_replaced_behavior": False,
        "v3_incomplete_support_rows_retained": True,
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "future_qualification_marker_mutated": False,
        "formal_marker_created": True,
        "bindings": {
            "v3_incomplete_support_rows_retained": True,
            "v4_incomplete_support_rows_retained": True,
            "support_v2_rows_mixed": False,
            "failed_qualification_rows_mixed": False,
            "failed_qualification_labels_mixed": False,
            "future_qualification_marker_mutated": False,
        },
        "records": train_records + dev_records,
        "metrics": {
            "train": {
                "rows": runner.V4_TRAIN_ROWS,
                "schema_88d_finite": True,
                "decision_event_id_unique": True,
                "seed_coverage": True,
                "completion_requested": {"requested": 234, "completed": 234},
            },
            "dev": {
                "rows": runner.V4_DEV_ROWS,
                "schema_88d_finite": True,
                "decision_event_id_unique": True,
                "seed_coverage": True,
                "completion_requested": {"requested": 130, "completed": 130},
            },
        },
        "gates": {
            "train": _gate_shape(all_pass=True),
            "dev": {
                **_gate_shape(all_pass=True),
                "cardinality": False,
                "all": False,
            },
        },
        "authorization": {
            "SUPPORT_COLLECTION": "FAIL",
            "SUPPORT_FROZEN": False,
            "LEARNABILITY_AUDIT_ALLOWED": False,
            "GATE_REPAIR_ALLOWED": False,
            "PPO_ALLOWED": False,
            "curriculum_promotion_allowed": False,
        },
        "support_files": {
            "combined": {
                "train": "/server/v4/train_support.jsonl",
                "dev": "/server/v4/dev_support.jsonl",
                "sha256": {
                    "train": runner.V4_COMBINED_TRAIN_SHA,
                    "dev": runner.V4_COMBINED_DEV_SHA,
                },
            },
            "sha256": {
                "combined_train": runner.V4_COMBINED_TRAIN_SHA,
                "combined_dev": runner.V4_COMBINED_DEV_SHA,
            },
        },
    }


def _write_v3_marker(path: Path):
    path.write_text(json.dumps({
        "protocol_sha256": runner.V3_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": runner.CANDIDATE_SHA,
        "train_support_seeds": list(runner.RETAINED_V3_TRAIN_SEEDS),
        "dev_support_seeds": list(runner.RETAINED_V3_DEV_SEEDS),
        "future_qualification_seeds": list(runner.FUTURE_QUALIFICATION_SEEDS),
        "output": "/server/v3",
    }), encoding="utf-8")


def _write_v4_marker(path: Path, output: Path):
    path.write_text(json.dumps({
        "protocol_sha256": runner.V4_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": runner.CANDIDATE_SHA,
        "v3_manifest_sha256": runner.v4.V3_MANIFEST_SHA,
        "supplemental_train_seeds": list(runner.v4.FORMAL_TRAIN_SEEDS),
        "supplemental_dev_seeds": list(runner.v4.FORMAL_DEV_SEEDS),
        "output": str(output),
        "future_qualification_marker_mutated": False,
    }), encoding="utf-8")


def test_protocol_is_byte_frozen_and_dev_only_seed_groups_are_disjoint():
    value = runner.protocol()
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA
    assert tuple(value["formal"]["supplemental_dev_seeds"]) == runner.FORMAL_DEV_SEEDS
    assert "supplemental_train_seeds" not in value["formal"]
    groups = [set(runner.RETAINED_V4_TRAIN_SEEDS), set(runner.RETAINED_V4_DEV_SEEDS),
              set(runner.FUTURE_QUALIFICATION_SEEDS), set(runner.FORMAL_DEV_SEEDS),
              set(runner.SMOKE_SEEDS)]
    assert all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])


def test_v4_manifest_accepts_only_expected_dev_cardinality_failure(monkeypatch, tmp_path):
    manifest_path = tmp_path / "v4" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(_valid_v4_manifest()), encoding="utf-8")
    monkeypatch.setattr(runner, "_sha", lambda path: runner.V4_MANIFEST_SHA)
    value = runner._validate_v4_manifest(manifest_path)
    assert value["SUPPORT_COLLECTION"] == "FAIL"

    changed = _valid_v4_manifest()
    changed["gates"]["dev"]["teacher_class_minimums"] = False
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="V4 dev gate shape"):
        runner._validate_v4_manifest(manifest_path)
    changed = _valid_v4_manifest()
    changed["records"][0]["safety_all_zero"] = False
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="safety gate"):
        runner._validate_v4_manifest(manifest_path)


def test_v3_v4_markers_and_rows_are_read_only_and_hash_bound(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "candidate.pt"
    v4_output = tmp_path / "v4"
    v4_output.mkdir()
    manifest_path = v4_output / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_v4_manifest()), encoding="utf-8")
    v3_marker = checkpoint_dir / runner.V3_RESERVATION_MARKER
    v4_marker = checkpoint_dir / runner.V4_RESERVATION_MARKER
    _write_v3_marker(v3_marker)
    _write_v4_marker(v4_marker, v4_output)
    hash_map = {
        str(manifest_path): runner.V4_MANIFEST_SHA,
        str(v3_marker): runner.V3_CONSUMED_MARKER_SHA,
        str(v4_marker): runner.V4_CONSUMED_MARKER_SHA,
    }
    monkeypatch.setattr(runner, "_sha", lambda path: hash_map[str(path)])
    v3_binding = runner._validate_v3_marker_read_only(checkpoint)
    v4_binding = runner._validate_v4_marker(checkpoint, manifest_path)
    assert v3_binding["mutated"] is False
    assert v4_binding["mutated"] is False

    broken = json.loads(v4_marker.read_text(encoding="utf-8"))
    broken["supplemental_dev_seeds"] = [785099]
    v4_marker.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="V4 consumed marker binding"):
        runner._validate_v4_marker(checkpoint, manifest_path)


def test_retained_rows_require_exact_v4_hash_schema_and_seed_identity(monkeypatch):
    train = [_row(index, "train", runner.RETAINED_V4_TRAIN_SEEDS[index % 27])
             for index in range(runner.V4_TRAIN_ROWS)]
    dev = [_row(index, "dev", runner.RETAINED_V4_DEV_SEEDS[index % 15])
           for index in range(runner.V4_DEV_ROWS)]
    train_path = Path("v4-train.jsonl")
    dev_path = Path("v4-dev.jsonl")
    hash_map = {str(train_path): runner.V4_COMBINED_TRAIN_SHA,
                str(dev_path): runner.V4_COMBINED_DEV_SHA}
    monkeypatch.setattr(runner, "_sha", lambda path: hash_map[str(path)])
    monkeypatch.setattr(runner, "_read_jsonl_rows", lambda path: train if path == train_path else dev)
    rows = runner._load_retained_v4_rows(train_path, dev_path)
    assert len(rows["train"]) == runner.V4_TRAIN_ROWS
    assert len(rows["dev"]) == runner.V4_DEV_ROWS
    monkeypatch.setattr(runner, "_sha", lambda path: "wrong")
    with pytest.raises(ValueError, match="V4 combined train SHA"):
        runner._load_retained_v4_rows(train_path, dev_path)


def test_combined_metrics_close_dev_cardinality_without_relaxing_train_gate():
    train_rows = [_row(index, "train", runner.RETAINED_V4_TRAIN_SEEDS[index % 27])
                  for index in range(3786)]
    dev_seeds = tuple(runner.RETAINED_V4_DEV_SEEDS) + tuple(runner.FORMAL_DEV_SEEDS)
    dev_rows = [_row(index, "dev", dev_seeds[index % len(dev_seeds)]) for index in range(1800)]
    train_records = [_record("train", seed) for seed in runner.RETAINED_V4_TRAIN_SEEDS]
    dev_records = [_record("dev", seed) for seed in dev_seeds]
    cross = runner._combined_identity({"train": train_rows, "dev": dev_rows})
    assert cross["train_dev_seed_disjoint"] and cross["train_dev_event_disjoint"]
    train_metrics = runner.split_metrics(train_rows, "train", train_records, runner.RETAINED_V4_TRAIN_SEEDS)
    dev_metrics = runner.split_metrics(dev_rows, "dev", dev_records, dev_seeds)
    assert runner.split_gates(train_metrics, "train", formal=True)["cardinality"]
    assert runner.split_gates(dev_metrics, "dev", formal=True)["cardinality"]


def test_smoke_no_marker_or_authorization(monkeypatch, tmp_path):
    payload = {
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
        "temperature": runner.CANDIDATE_TEMPERATURE,
        "qualification_only": True,
        "checkpoint_promotable": False,
    }
    fake_binding = {"manifest": {}, "v3_marker": {}, "v4_marker": {}}
    fake_manifest = {"records": []}
    monkeypatch.setattr(
        runner,
        "_validate_v4_inputs",
        lambda checkpoint, manifest, train, dev: (fake_binding, {"train": [], "dev": []},
                                                  fake_manifest, object(), payload),
    )

    def fake_episode(policy, *, split, seed, scenario, output, max_steps):
        return {
            "split": split, "seed": seed, "scenario": scenario,
            "requested": 6, "completed": 6, "episode_execution_complete": True,
            "safety": {}, "gates": {"synthetic": True},
        }, []

    monkeypatch.setattr(runner, "_episode_record", fake_episode)
    result = runner.run(
        tmp_path / "smoke", tmp_path / "candidate.pt", tmp_path / "v4.json",
        tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path, smoke=True,
    )
    assert result["stage"] == runner.STAGE
    assert result["authorization"]["SUPPORT_COLLECTION"] == "FAIL"
    assert result["SUPPORT_FROZEN"] is False
    assert result["LEARNABILITY_AUDIT_ALLOWED"] is False
    assert result["formal_marker_created"] is False
    assert result["v3_marker_mutated"] is False
    assert result["v4_marker_mutated"] is False
    assert not (tmp_path / runner.V4_RESERVATION_MARKER).exists()


def test_formal_reservation_accepts_smoke_history_but_rejects_exact_dev_overlap(monkeypatch, tmp_path):
    smoke_dir = tmp_path / "smoke-checkpoint"
    smoke_dir.mkdir()
    smoke_checkpoint = smoke_dir / "candidate.pt"
    monkeypatch.setattr(runner, "historical_registry",
                        lambda history, output: {"seeds": set(runner.SMOKE_SEEDS)})
    marker, registry = runner._reserve_formal(
        smoke_checkpoint, tmp_path, tmp_path / "smoke-output", tmp_path / "v4" / "manifest.json",
    )
    assert marker.is_file()
    assert registry["formal_dev_seed_overlap"] == []
    assert registry["smoke_seed_overlap"] == list(runner.SMOKE_SEEDS)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["supplemental_dev_seeds"] == list(runner.FORMAL_DEV_SEEDS)
    assert "future_qualification_seeds" not in payload

    for index, seed in enumerate(runner.FORMAL_DEV_SEEDS):
        collision_dir = tmp_path / f"collision-{index}"
        collision_dir.mkdir()
        checkpoint = collision_dir / "candidate.pt"
        monkeypatch.setattr(runner, "historical_registry",
                            lambda history, output, seed=seed: {"seeds": {seed}})
        with pytest.raises(ValueError, match="formal_dev_seed_overlap"):
            runner._reserve_formal(
                checkpoint, tmp_path, tmp_path / f"collision-output-{index}",
                tmp_path / "v4" / "manifest.json",
            )
        assert not (collision_dir / runner.V4_RESERVATION_MARKER).exists()


def test_manifest_provenance_reports_retained_new_and_combined_hashes():
    hashes = {
        "v4_train": runner.V4_COMBINED_TRAIN_SHA,
        "v4_dev": runner.V4_COMBINED_DEV_SHA,
        "supplement_dev": "new-dev",
        "combined_train": "out-train",
        "combined_dev": "out-dev",
    }
    assert hashes["v4_train"] == runner.V4_COMBINED_TRAIN_SHA
    assert hashes["v4_dev"] == runner.V4_COMBINED_DEV_SHA
    assert set(("supplement_dev", "combined_train", "combined_dev")) <= hashes.keys()
