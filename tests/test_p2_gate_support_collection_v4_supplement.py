from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import terminal_marl.training.collect_v4_2_p2_gate_support_v4_supplement as runner


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
        "decision_event_id": f"supplement-test-{split}-{index}",
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


def _v3_gate_shape():
    return {
        "cardinality": False,
        "teacher_class_minimums": False,
        "teacher_class_proportions": True,
        "interface_minimums": False,
        "interface_class_minimums": False,
        "schema_88d_finite": True,
        "decision_event_id_unique": True,
        "seed_coverage": True,
        "all_planned_episodes_execution_complete": True,
        "all_episode_shield_actor_fraction_within_inherited_limit": True,
        "all_episode_gates": True,
        "cross_split_seed_disjoint": True,
        "cross_split_event_disjoint": True,
        "all": False,
    }


def _valid_v3_manifest():
    train_records = [
        _record("train", seed, runner.SCENARIOS[index % len(runner.SCENARIOS)])
        for index, seed in enumerate(runner.RETAINED_V3_TRAIN_SEEDS)
    ]
    dev_records = [
        _record("dev", seed, runner.SCENARIOS[index % len(runner.SCENARIOS)])
        for index, seed in enumerate(runner.RETAINED_V3_DEV_SEEDS)
    ]
    train_metrics = {
        "rows": runner.V3_TRAIN_ROWS,
        "schema_88d_finite": True,
        "decision_event_id_unique": True,
        "seed_coverage": True,
        "completion_requested": {"requested": 78, "completed": 78},
    }
    dev_metrics = {
        "rows": runner.V3_DEV_ROWS,
        "schema_88d_finite": True,
        "decision_event_id_unique": True,
        "seed_coverage": True,
        "completion_requested": {"requested": 52, "completed": 52},
    }
    return {
        "stage": "V4_2_P2_GATE_SUPPORT_COLLECTION_V3",
        "protocol_sha256": runner.V3_PROTOCOL_SHA,
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
        "support_v2_rows_mixed": False,
        "failed_qualification_rows_mixed": False,
        "failed_qualification_labels_mixed": False,
        "future_qualification_rollouts": 0,
        "future_qualification_consumed_marker": False,
        "formal_marker_created": True,
        "formal_marker": "/server/v3/p2_gate_support_collection_v3_consumed.json",
        "records": train_records + dev_records,
        "metrics": {"train": train_metrics, "dev": dev_metrics},
        "gates": {"train": _v3_gate_shape(), "dev": _v3_gate_shape()},
        "authorization": {
            "SUPPORT_COLLECTION": "FAIL",
            "SUPPORT_FROZEN": False,
            "LEARNABILITY_AUDIT_ALLOWED": False,
            "GATE_REPAIR_ALLOWED": False,
            "PPO_ALLOWED": False,
            "curriculum_promotion_allowed": False,
        },
        "support_files": {
            "train": "/server/v3/train_support.jsonl",
            "dev": "/server/v3/dev_support.jsonl",
            "sha256": {"train": runner.V3_TRAIN_SUPPORT_SHA, "dev": runner.V3_DEV_SUPPORT_SHA},
        },
    }


def _write_marker(path: Path, *, output: Path):
    path.write_text(json.dumps({
        "protocol_sha256": runner.V3_PROTOCOL_SHA,
        "candidate_checkpoint_sha256": runner.CANDIDATE_SHA,
        "train_support_seeds": list(runner.RETAINED_V3_TRAIN_SEEDS),
        "dev_support_seeds": list(runner.RETAINED_V3_DEV_SEEDS),
        "future_qualification_seeds": list(runner.FUTURE_QUALIFICATION_SEEDS),
        "output": str(output),
    }), encoding="utf-8")


def test_protocol_freezes_v4_supplement_and_seed_disjointness():
    value = runner.protocol()
    assert runner._sha(runner.PROTOCOL) == runner.PROTOCOL_SHA
    assert tuple(value["formal"]["supplemental_train_seeds"]) == runner.FORMAL_TRAIN_SEEDS
    assert tuple(value["formal"]["supplemental_dev_seeds"]) == runner.FORMAL_DEV_SEEDS
    groups = [
        set(runner.RETAINED_V3_TRAIN_SEEDS), set(runner.RETAINED_V3_DEV_SEEDS),
        set(runner.FUTURE_QUALIFICATION_SEEDS), set(runner.FORMAL_TRAIN_SEEDS),
        set(runner.FORMAL_DEV_SEEDS), set(runner.SMOKE_SEEDS),
    ]
    assert all(not (left & right) for index, left in enumerate(groups) for right in groups[index + 1:])
    assert value["collection"]["combined_row_order"] == "retained_v3_rows_then_supplement_rows"


def test_v3_manifest_and_marker_validation_is_strict(monkeypatch, tmp_path):
    manifest_path = tmp_path / "v3" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(_valid_v3_manifest()), encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "candidate.pt"
    marker = checkpoint_dir / runner.V3_RESERVATION_MARKER
    _write_marker(marker, output=manifest_path.parent)
    hash_map = {
        str(manifest_path): runner.V3_MANIFEST_SHA,
        str(marker): runner.V3_CONSUMED_MARKER_SHA,
    }
    monkeypatch.setattr(runner, "_sha", lambda path: hash_map[str(path)])
    value = runner._validate_v3_manifest(manifest_path)
    assert value["SUPPORT_COLLECTION"] == "FAIL"
    marker_binding = runner._validate_v3_marker(checkpoint, manifest_path)
    assert marker_binding["mutated"] is False

    changed = _valid_v3_manifest()
    changed["gates"]["train"]["teacher_class_proportions"] = False
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="gate failure shape"):
        runner._validate_v3_manifest(manifest_path)

    changed = _valid_v3_manifest()
    changed["records"][0]["safety_all_zero"] = False
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="safety gate"):
        runner._validate_v3_manifest(manifest_path)

    _write_marker(marker, output=tmp_path / "wrong-output")
    with pytest.raises(ValueError, match="marker binding"):
        runner._validate_v3_marker(checkpoint, manifest_path)


def test_retained_rows_require_exact_hash_schema_and_seed_identity(monkeypatch):
    train = [_row(index, "train", runner.RETAINED_V3_TRAIN_SEEDS[index % 9])
             for index in range(runner.V3_TRAIN_ROWS)]
    dev = [_row(index, "dev", runner.RETAINED_V3_DEV_SEEDS[index % 6])
           for index in range(runner.V3_DEV_ROWS)]
    paths = {"train": Path("train.jsonl"), "dev": Path("dev.jsonl")}
    hash_map = {
        str(paths["train"]): runner.V3_TRAIN_SUPPORT_SHA,
        str(paths["dev"]): runner.V3_DEV_SUPPORT_SHA,
    }
    monkeypatch.setattr(runner, "_sha", lambda path: hash_map[str(path)])
    monkeypatch.setattr(runner, "_read_jsonl_rows", lambda path: train if path == paths["train"] else dev)
    rows = runner._load_retained_v3_rows(paths["train"], paths["dev"])
    assert len(rows["train"]) == runner.V3_TRAIN_ROWS
    assert len(rows["dev"]) == runner.V3_DEV_ROWS

    monkeypatch.setattr(runner, "_sha", lambda path: "wrong")
    with pytest.raises(ValueError, match="V3 train support SHA"):
        runner._load_retained_v3_rows(paths["train"], paths["dev"])


def test_combined_metrics_keep_unchanged_support_thresholds():
    train_seeds = tuple(runner.RETAINED_V3_TRAIN_SEEDS) + tuple(runner.FORMAL_TRAIN_SEEDS)
    dev_seeds = tuple(runner.RETAINED_V3_DEV_SEEDS) + tuple(runner.FORMAL_DEV_SEEDS)
    train_rows = [_row(index, "train", train_seeds[index % len(train_seeds)]) for index in range(3000)]
    dev_rows = [_row(index, "dev", dev_seeds[index % len(dev_seeds)]) for index in range(1800)]
    train_records = [_record("train", seed) for seed in train_seeds]
    dev_records = [_record("dev", seed) for seed in dev_seeds]
    cross = runner._combined_identity({"train": train_rows, "dev": dev_rows})
    assert cross == {"train_dev_seed_disjoint": True, "train_dev_event_disjoint": True}
    train_metrics = runner.split_metrics(train_rows, "train", train_records, train_seeds)
    dev_metrics = runner.split_metrics(dev_rows, "dev", dev_records, dev_seeds)
    assert runner.split_gates(train_metrics, "train", formal=True)["cardinality"]
    assert runner.split_gates(dev_metrics, "dev", formal=True)["cardinality"]


def test_smoke_never_reserves_or_authorizes_and_reports_provenance(monkeypatch, tmp_path):
    payload = {
        "stage": "V4_2_P2_GATE_TEMPERATURE_CALIBRATION_V3",
        "architecture_id": runner.CANDIDATE_ARCHITECTURE_ID,
        "temperature": runner.CANDIDATE_TEMPERATURE,
        "qualification_only": True,
        "checkpoint_promotable": False,
    }
    fake_binding = {"manifest": {}, "marker": {"mutated": False}}
    fake_manifest = {"records": []}
    monkeypatch.setattr(
        runner,
        "_validate_v3_inputs",
        lambda checkpoint, manifest, train, dev: (fake_binding, {"train": [], "dev": []},
                                                  fake_manifest, object(), payload),
    )

    def fake_episode(policy, *, split, seed, scenario, output, max_steps):
        return {
            "split": split,
            "seed": seed,
            "scenario": scenario,
            "requested": 6,
            "completed": 6,
            "episode_execution_complete": True,
            "safety": {},
            "gates": {"synthetic": True},
        }, []

    monkeypatch.setattr(runner, "_episode_record", fake_episode)
    result = runner.run(
        tmp_path / "smoke",
        tmp_path / "candidate.pt",
        tmp_path / "v3_manifest.json",
        tmp_path / "train_support.jsonl",
        tmp_path / "dev_support.jsonl",
        tmp_path,
        smoke=True,
    )
    assert result["stage"] == runner.STAGE
    assert result["authorization"]["SUPPORT_COLLECTION"] == "FAIL"
    assert result["SUPPORT_FROZEN"] is False
    assert result["LEARNABILITY_AUDIT_ALLOWED"] is False
    assert result["formal_marker_created"] is False
    assert result["future_qualification_marker_mutated"] is False
    assert result["v3_incomplete_support_rows_retained"] is True
    assert result["support_v2_rows_mixed"] is False
    assert result["failed_qualification_rows_mixed"] is False
    assert not (tmp_path / runner.RESERVATION_MARKER).exists()


def test_formal_reservation_only_uses_supplement_seeds_and_rejects_collision(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "candidate.pt"
    monkeypatch.setattr(runner, "historical_registry", lambda history, output: {"seeds": set()})
    marker, registry = runner._reserve_formal(
        checkpoint, tmp_path, tmp_path / "supplement", tmp_path / "v3" / "manifest.json",
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["supplemental_train_seeds"] == list(runner.FORMAL_TRAIN_SEEDS)
    assert payload["supplemental_dev_seeds"] == list(runner.FORMAL_DEV_SEEDS)
    assert "future_qualification_seeds" not in payload
    assert payload["future_qualification_marker_mutated"] is False
    assert registry["future_qualification_marker_mutated"] is False
    with pytest.raises(ValueError, match="consumed marker"):
        runner._reserve_formal(
            checkpoint, tmp_path, tmp_path / "again", tmp_path / "v3" / "manifest.json",
        )

    collision_dir = tmp_path / "collision"
    collision_dir.mkdir()
    collision_checkpoint = collision_dir / "candidate.pt"
    monkeypatch.setattr(
        runner, "historical_registry",
        lambda history, output: {"seeds": {next(iter(runner.FORMAL_TRAIN_SEEDS))}},
    )
    with pytest.raises(ValueError, match="audited seed range"):
        runner._reserve_formal(
            collision_checkpoint, tmp_path, tmp_path / "collision-output",
            tmp_path / "v3" / "manifest.json",
        )
    assert not (collision_dir / runner.RESERVATION_MARKER).exists()


def test_manifest_provenance_declares_four_output_hash_slots():
    result = {
        "support_files": {
            "sha256": {
                "v3_train": runner.V3_TRAIN_SUPPORT_SHA,
                "v3_dev": runner.V3_DEV_SUPPORT_SHA,
                "supplement_train": "supplement-train",
                "supplement_dev": "supplement-dev",
                "combined_train": "combined-train",
                "combined_dev": "combined-dev",
            }
        },
        "v3_incomplete_support_rows_retained": True,
        "future_qualification_marker_mutated": False,
    }
    hashes = result["support_files"]["sha256"]
    assert hashes["v3_train"] == runner.V3_TRAIN_SUPPORT_SHA
    assert hashes["v3_dev"] == runner.V3_DEV_SUPPORT_SHA
    assert all(key in hashes for key in ("supplement_train", "supplement_dev", "combined_train", "combined_dev"))
    assert result["v3_incomplete_support_rows_retained"] is True
    assert result["future_qualification_marker_mutated"] is False
