"""Write the reviewed P3 formal-admission decision after the final YAML hash is known."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path("/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925")
ENTRY_MANIFEST = ROOT / "entry/manifest.json"
PILOT_MANIFEST = ROOT / "pilot/manifest.json"
ADMISSION_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_controlled_emergency_admission_v5.yaml"
)
FORMAL_PROTOCOL = Path(
    "terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml"
)
DECISION = ROOT / "formal_admission_decision.json"
CHECKPOINT_SHA256 = (
    "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
)


class DecisionInputError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionInputError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def validate_entry(entry: Mapping[str, Any]) -> None:
    require(entry.get("stage") == "V5_ENTRY", "entry stage mismatch")
    require(entry.get("PASS") is True, "entry PASS is false")
    require(entry.get("PPO_ENTRY_ALLOWED") is True, "entry PPO authorization missing")
    require(entry.get("bounded_pilot_training_authorized") is True,
            "entry bounded-pilot authorization missing")
    require(entry.get("selected_checkpoint_sha256") == CHECKPOINT_SHA256,
            "entry checkpoint SHA mismatch")
    require(entry.get("selected_checkpoint_kind") == "repair",
            "entry checkpoint kind mismatch")
    require(entry.get("entry_updates_discarded") is True, "entry update discard binding changed")
    require(entry.get("fresh_optimizer") is True, "entry optimizer freshness binding changed")
    for key in ("P3_QUALIFIED", "P4_ALLOWED", "PPO_ALLOWED", "checkpoint_promotable"):
        require(entry.get(key) is False, f"entry nonpromotion flag changed: {key}")
    for key in ("actual_config_snapshots", "optimizer_boundary", "schema_binding", "protocol_sha256"):
        require(entry.get(key) is not None, f"entry binding missing: {key}")


def validate_pilot(pilot: Mapping[str, Any]) -> None:
    require(pilot.get("stage") == "V5_PILOT", "pilot stage mismatch")
    require(pilot.get("PASS") is True, "pilot PASS is false")
    require(pilot.get("PILOT_TECHNICAL_PASS") is True,
            "pilot technical pass is false")
    require(pilot.get("selected_checkpoint_sha256") == CHECKPOINT_SHA256,
            "pilot checkpoint SHA mismatch")
    require(pilot.get("selected_checkpoint_kind") == "repair",
            "pilot checkpoint kind mismatch")
    require(pilot.get("FORMAL_P3_ALLOWED") is False,
            "raw pilot formal authorization changed")
    require(pilot.get("P3_QUALIFIED") is False and pilot.get("P4_ALLOWED") is False,
            "raw pilot qualification flags changed")
    records = pilot.get("records")
    require(isinstance(records, list) and len(records) == 6,
            "pilot record count is not six")
    for index, record in enumerate(records):
        require(isinstance(record, Mapping), f"pilot record {index} is invalid")
        require(record.get("hard_gate_pass") is True,
                f"pilot hard gate failed at record {index}")
        gates = record.get("update_gates")
        require(isinstance(gates, Mapping) and gates and all(bool(value) for value in gates.values()),
                f"pilot update gates failed at record {index}")
        require(record.get("checkpoint_restore_exact") is True,
                f"pilot checkpoint restore failed at record {index}")
        require(record.get("ppo_called") is True,
                f"pilot PPO update missing at record {index}")


def build_decision(*, expected_formal_sha256: str) -> dict[str, Any]:
    require(ROOT.is_dir(), f"admission output root missing: {ROOT}")
    require(not DECISION.exists(), f"decision output already exists: {DECISION}")
    entry = read_json(ENTRY_MANIFEST)
    pilot = read_json(PILOT_MANIFEST)
    validate_entry(entry)
    validate_pilot(pilot)
    require(ADMISSION_PROTOCOL.is_file(), f"admission protocol missing: {ADMISSION_PROTOCOL}")
    require(FORMAL_PROTOCOL.is_file(), f"formal protocol missing: {FORMAL_PROTOCOL}")
    actual_formal_sha256 = sha256(FORMAL_PROTOCOL)
    require(actual_formal_sha256 == expected_formal_sha256,
            "formal protocol SHA differs from caller-supplied final SHA")
    pilot_sha256 = sha256(PILOT_MANIFEST)
    comparison = pilot.get("performance_comparison") or {}
    dense = (comparison.get("pooled_emergency") or {}).get("dense") or {}
    require(dense.get("before_emergency") == 4 and dense.get("after_emergency") == 3,
            "pilot dense emergency comparison mismatch")
    return {
        "stage": "P3_FORMAL_ADMISSION_DECISION_V5",
        "FORMAL_P3_ALLOWED": True,
        "PPO_ENTRY_ALLOWED": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "PPO_ALLOWED": False,
        "checkpoint_promotable": False,
        "initial_checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_checkpoint_kind": "repair",
        "pilot_manifest_path": str(PILOT_MANIFEST),
        "pilot_manifest_sha256": pilot_sha256,
        "admission_protocol_sha256": sha256(ADMISSION_PROTOCOL),
        "formal_protocol_sha256": actual_formal_sha256,
        "actual_config_snapshots": entry["actual_config_snapshots"],
        "optimizer_boundary": entry["optimizer_boundary"],
        "schema_binding": entry["schema_binding"],
        "protocol_sha256": entry["protocol_sha256"],
        "primary_review_reason": (
            "Pilot technical PASS accepted for registered formal P3 execution: "
            "dense emergency count 4 to 3; low-SOC wait 177.697 to 133.661; "
            "priority-weighted wait 123.016 to 125.778, with the observed "
            "priority-wait increase accepted as a reportable learning result."
        ),
        "nonqualification": (
            "This decision authorizes the registered formal run after strict "
            "preflight; it does not grant P3 qualification or P4 authorization."
        ),
        "formal_not_started": True,
        "pilot_technical_pass": True,
        "pilot_records": 6,
        "source_entry_manifest_path": str(ENTRY_MANIFEST),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-sha256", required=True)
    args = parser.parse_args(argv)
    decision = build_decision(expected_formal_sha256=str(args.formal_sha256).lower())
    DECISION.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(DECISION), "formal_protocol_sha256": decision["formal_protocol_sha256"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
