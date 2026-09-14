"""Read-only V4.1 I0/I0-R to V4.2 target-view derivation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..core.low_level_joint_action_v4_1 import ACTION_SCHEMA_V4_1
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_joint_index_v4_1


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive(source: Path, destination: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    derived: list[dict] = []
    for row in rows:
        index = int(row["teacher_joint_index"]); target = hierarchical_from_joint_index_v4_1(index)
        copied = dict(row)
        copied.update({"derived_target_schema": ACTION_SCHEMA_V4_2, "parent_action_schema_version": ACTION_SCHEMA_V4_1,
                       "teacher_semantic_joint_index_v4_1": index, "gate_target": target.gate,
                       "proposal_target": target.proposal})
        derived.append(copied)
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in derived), encoding="utf-8")
    unchanged = all(all(left[key] == right[key] for key in ("observation", "teacher_joint_index", "teacher_decoded_action", "state_hash", "state_snapshot", "source", "seed"))
                    for left, right in zip(rows, derived))
    return {"path": str(destination), "rows": len(rows), "sha256": _digest(destination), "unchanged_semantics": unchanged,
            "wait_targets": sum(row["gate_target"] == 0 for row in derived), "release_targets": sum(row["gate_target"] == 1 for row in derived)}


def derive(i0_manifest: str | Path, output_dir: str | Path) -> dict:
    parent_path = Path(i0_manifest); parent = json.loads(parent_path.read_text(encoding="utf-8")); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    train = _derive(Path(parent["dataset_path"]), output / "i0_v4_2_hierarchical_target_view.jsonl")
    cohort = _derive(Path(parent["fixed_decision_state_cohort_path"]), output / "frozen_cohort_v4_2_hierarchical_target_view.jsonl")
    result = {"stage": "V4_2_DERIVED_I0_TARGET_VIEW", "derived_target_schema": ACTION_SCHEMA_V4_2,
              "parent_i0_manifest": str(parent_path), "parent_dataset_sha256": parent["dataset_sha256"],
              "parent_frozen_cohort_sha256": parent["fixed_cohort_sha256"], "parent_action_schema": ACTION_SCHEMA_V4_1,
              "derived_dataset": train, "derived_frozen_cohort": cohort, "PPO_ALLOWED": False,
              "lineage_PASS": bool(train["unchanged_semantics"] and cohort["unchanged_semantics"])}
    (output / "v4_2_derived_i0_target_view.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--i0-manifest", required=True); parser.add_argument("--output-dir", required=True)
    print(json.dumps(derive(parser.parse_args().i0_manifest, parser.parse_args().output_dir), indent=2))
