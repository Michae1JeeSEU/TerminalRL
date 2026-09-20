import copy
import hashlib
import json
from pathlib import Path

import pytest

from terminal_marl.training import (
    audit_v4_2_p3_mixed_priority_soc_training_v2_formal_v1 as audit,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    checkpoint = tmp_path / "checkpoint_8.pt"
    checkpoint.write_bytes(b"checkpoint-8-fixture")
    rows = [
        {
            "uid": 0,
            "base_priority": 1,
            "initial_soc": 0.55,
            "landing_soc": 0.52,
            "wait_seconds": 1.0,
            "request_to_touchdown_seconds": 10.0,
        },
        {
            "uid": 1,
            "base_priority": 3,
            "initial_soc": 0.72,
            "landing_soc": 0.69,
            "wait_seconds": 2.0,
            "request_to_touchdown_seconds": 11.0,
        },
        {
            "uid": 2,
            "base_priority": 5,
            "initial_soc": 0.90,
            "landing_soc": 0.86,
            "wait_seconds": 3.0,
            "request_to_touchdown_seconds": 12.0,
        },
    ]
    records = [
        {
            "seed": 821001 + index,
            "scenario": "dense24" if index % 2 == 0 else "flow6",
            "gates": {"episode_integrity": True},
            "attributes": {
                "completed": len(rows),
                "replay_exact": True,
                "rows": copy.deepcopy(rows),
            },
            "checkpoint": str(checkpoint),
        }
        for index in range(8)
    ]
    formal_gates = {
        "all_episodes": True,
        "all_episode_gates": True,
        "priority_coverage": False,
        "soc_in_bounds": False,
        "soc_range_observed": False,
    }
    formal = {
        "stage": audit.FORMAL_STAGE,
        "PASS": False,
        "gates": formal_gates,
        "records": records,
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": audit.CHECKPOINT_8_SHA,
    }
    root = {
        "stage": audit.STAGE,
        "PASS": False,
        "HELDOUT_EVALUATION_ALLOWED": False,
        "heldout_untouched": True,
        "P4_ALLOWED": False,
        "checkpoint_promotable": False,
        "formal": {
            "PASS": False,
            "gates": formal_gates,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": audit.CHECKPOINT_8_SHA,
        },
    }
    formal_path = tmp_path / "formal_manifest.json"
    root_path = tmp_path / "training_manifest.json"
    formal_path.write_text(json.dumps(formal), encoding="utf-8")
    root_path.write_text(json.dumps(root), encoding="utf-8")
    return root_path, formal_path, checkpoint, root, formal


def _install_fixture_hashes(monkeypatch, root_path: Path, formal_path: Path, checkpoint: Path):
    real_sha = audit._sha256
    expected = {
        root_path: audit.TRAINING_ROOT_MANIFEST_SHA,
        formal_path: audit.FORMAL_MANIFEST_SHA,
        checkpoint: audit.CHECKPOINT_8_SHA,
        audit.PROTOCOL: audit.PROTOCOL_SHA,
        audit.TRAINING_PROTOCOL: audit.TRAINING_PROTOCOL_SHA,
        audit.SCHEDULER_SOURCE: audit.SCHEDULER_SOURCE_SHA,
    }

    def fake_sha(path):
        source = Path(path)
        return expected.get(source, real_sha(source))

    monkeypatch.setattr(audit, "_sha256", fake_sha)


def test_formal_audit_happy_path_recomputes_rows_and_writes_separate_manifest(
    tmp_path, monkeypatch,
):
    root_path, formal_path, checkpoint, root_before, formal_before = _fixture(tmp_path)
    _install_fixture_hashes(monkeypatch, root_path, formal_path, checkpoint)
    output = tmp_path / "audit"

    result = audit.run(
        training_root_manifest=root_path,
        formal_manifest=formal_path,
        checkpoint_8=checkpoint,
        output=output,
    )

    assert result["PASS"] is True
    assert result["HELDOUT_EVALUATION_ALLOWED"] is True
    assert result["heldout_untouched"] is True
    assert result["P4_ALLOWED"] is False
    assert result["checkpoint_promotable"] is False
    assert result["corrected_coverage"]["rows_count"] == 24
    assert result["corrected_coverage"]["priorities_observed"] == [1, 3, 5]
    assert result["corrected_coverage"]["initial_soc_min"] == 0.55
    assert result["corrected_coverage"]["initial_soc_max"] == 0.90
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["PASS"] is True
    assert json.loads(root_path.read_text(encoding="utf-8")) == root_before
    assert json.loads(formal_path.read_text(encoding="utf-8")) == formal_before


def test_formal_audit_rejects_manifest_sha_mismatch(tmp_path):
    root_path, formal_path, checkpoint, _, _ = _fixture(tmp_path)
    root_path.write_text(root_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(audit.AuditInputError, match="training root manifest SHA mismatch"):
        audit.run(
            training_root_manifest=root_path,
            formal_manifest=formal_path,
            checkpoint_8=checkpoint,
            output=tmp_path / "audit",
        )


def test_formal_audit_rejects_extra_failed_formal_gate(tmp_path, monkeypatch):
    root_path, formal_path, checkpoint, _, formal = _fixture(tmp_path)
    formal["gates"]["unexpected_gate"] = False
    formal_path.write_text(json.dumps(formal), encoding="utf-8")
    _install_fixture_hashes(monkeypatch, root_path, formal_path, checkpoint)

    with pytest.raises(audit.AuditInputError, match="unexpected gate"):
        audit.run(
            training_root_manifest=root_path,
            formal_manifest=formal_path,
            checkpoint_8=checkpoint,
            output=tmp_path / "audit",
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda row: row.pop("base_priority"), "missing base_priority"),
        (lambda row: row.__setitem__("initial_soc", float("nan")), "initial SOC is not finite"),
    ],
)
def test_formal_audit_rejects_missing_or_nonfinite_rows(
    tmp_path, monkeypatch, mutation, message,
):
    root_path, formal_path, checkpoint, _, formal = _fixture(tmp_path)
    mutation(formal["records"][0]["attributes"]["rows"][0])
    formal_path.write_text(json.dumps(formal, allow_nan=True), encoding="utf-8")
    _install_fixture_hashes(monkeypatch, root_path, formal_path, checkpoint)

    with pytest.raises(audit.AuditInputError, match=message):
        audit.run(
            training_root_manifest=root_path,
            formal_manifest=formal_path,
            checkpoint_8=checkpoint,
            output=tmp_path / "audit",
        )


def test_formal_audit_reports_coverage_shortfall_without_authorizing_heldout(
    tmp_path, monkeypatch,
):
    root_path, formal_path, checkpoint, _, formal = _fixture(tmp_path)
    for record in formal["records"]:
        for row in record["attributes"]["rows"]:
            row["base_priority"] = 1
            row["initial_soc"] = 0.70
    formal_path.write_text(json.dumps(formal), encoding="utf-8")
    _install_fixture_hashes(monkeypatch, root_path, formal_path, checkpoint)

    result = audit.run(
        training_root_manifest=root_path,
        formal_manifest=formal_path,
        checkpoint_8=checkpoint,
        output=tmp_path / "audit",
    )

    assert result["PASS"] is False
    assert result["HELDOUT_EVALUATION_ALLOWED"] is False
    assert result["corrected_coverage"]["priorities_observed"] == [1]
    assert result["corrected_coverage"]["priority_coverage"] is False
    assert result["corrected_coverage"]["soc_range_observed"] is False


def test_formal_audit_input_hashes_are_unchanged(tmp_path, monkeypatch):
    root_path, formal_path, checkpoint, _, _ = _fixture(tmp_path)
    _install_fixture_hashes(monkeypatch, root_path, formal_path, checkpoint)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root_path, formal_path, checkpoint)
    }

    audit.run(
        training_root_manifest=root_path,
        formal_manifest=formal_path,
        checkpoint_8=checkpoint,
        output=tmp_path / "audit",
    )

    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root_path, formal_path, checkpoint)
    }
    assert after == before
