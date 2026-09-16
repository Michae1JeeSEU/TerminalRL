"""Prospective R4 collection with fixed episodes and unchanged physical semantics."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .run_v4_2_v2_1_formal_i0 import SourceSpec, _config, _sha
from .run_v4_2_v2_1_formal_i0_r import _aliasing, _provenance
from .run_v4_2_v2_1_formal_i0_r3 import (
    STRATA as R3_STRATA, FixedCollectionSchedule, R3Stratum, _collect_partition,
    _coverage, _verify_protocol, _write,
)
from .run_v4_2_v2_1_i1_q_cohort import _consume_object, _history_files, _identity_accumulator

PROTOCOL_PATH = Path("terminal_marl/configs/low_v4_2/i0_r4_repair_protocol.yaml")
# Filled and committed before any formal simulator reset.
EXPECTED_PROTOCOL_SHA = "0a2a561b18cdd54a6ac934bd7fd002675b9bd00b36192ca59b76115d1a6eb41d"


def collection_strata() -> tuple[R3Stratum, ...]:
    # Names describe actual initialization. SourceSpec.phases is a legacy
    # target-selection hint, not a reset constraint; collect every active UAV.
    specs = [item.spec for item in R3_STRATA] + [
        SourceSpec("extra_n5_a", "warm_start", 5, max_steps=12),
        SourceSpec("extra_n5_b", "warm_start", 5, max_steps=12),
        SourceSpec("extra_n12_transition", "warm_start", 12, dynamic=True, max_steps=42),
        SourceSpec("extra_n12_resource", "warm_start", 12, final_capacity=0, max_steps=12),
        SourceSpec("extra_n8_nominal", "warm_start", 8, max_steps=12),
        SourceSpec("extra_n6_nominal", "warm_start", 6, max_steps=12),
    ]
    result = []
    for index, spec in enumerate(specs):
        name = f"slot{index:02d}_N{spec.warm_start_uavs}_{'transition' if spec.dynamic else 'static'}_final{spec.final_capacity}_T{spec.max_steps}"
        result.append(R3Stratum(name, replace(spec, source="i0_r4_" + name, scenario="safe_warm_start", phases=("ring_outer", "ring_middle", "ring_inner"))))
    return tuple(result)


def historical_registry(history_root: Path, output: Path) -> dict:
    """Check JSONL as well as metadata before using fresh seeds."""
    output.mkdir(parents=True, exist_ok=True)
    identities = _identity_accumulator()
    files = _history_files(history_root, output)
    for index, path in enumerate(files):
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            _consume_object(json.loads(line), identities)
            else:
                content = path.read_text(encoding="utf-8")
                try:
                    _consume_object(json.loads(content), identities)
                except json.JSONDecodeError:
                    # Historical logs with .json extension may contain JSONL.
                    for line in content.splitlines():
                        try:
                            _consume_object(json.loads(line), identities)
                        except json.JSONDecodeError:
                            continue
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot inspect historical evidence: {path}") from exc
        if index % 100 == 0:
            _write(output / "registry_progress.json", {"files_read": index + 1, "files_total": len(files)})
    return identities


def run(output: Path, history_root: Path, *, smoke: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    if _sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA:
        raise ValueError("R4 protocol changed after freeze")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    support_protocol, checked = _verify_protocol()
    if not checked["PASS"]:
        raise ValueError("inherited support minima changed")
    strata = collection_strata()
    if smoke:
        strata = tuple(replace(item, spec=replace(item.spec, max_steps=1)) for item in (strata[0], strata[2]))
    seeds = {}
    for index, part in enumerate(("training", "internal_audit")):
        cycles = 1 if smoke else protocol[part]["cycles"]
        start = protocol["smoke_seed_start"] + index * 100 if smoke else protocol[part]["seed_start"]
        seeds[part] = list(range(start, start + cycles * len(strata)))
    historical = historical_registry(history_root, output) if not smoke else _identity_accumulator()
    if any(set(values) & historical["seeds"] for values in seeds.values()):
        raise ValueError("fresh seed range already consumed; preregister another protocol")
    _write(output / "pre_reset_registry.json", {"protocol_sha256": EXPECTED_PROTOCOL_SHA,
           "seeds": seeds, "historical_seed_count": len(historical["seeds"]),
           "historical_identity_check": not smoke, "built_before_first_reset": True,
           "schedule": [{"name": s.name, "spec": asdict(s.spec)} for s in strata]})
    partitions, infos, tapes = {}, {}, []
    for part in seeds:
        schedule = FixedCollectionSchedule(strata, 1 if smoke else protocol[part]["cycles"],
                                           protocol[part]["maximum_rows"], "v2-1-i0-r4")
        rows, info, recorded = _collect_partition(output, part, seeds[part], support_protocol, schedule=schedule)
        path = output / f"{part}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        partitions[part], infos[part] = rows, {**info, "path": str(path), "sha256": _sha(path)}
        tapes.extend(recorded)
        if info["fatal"] or any(info["safety"].values()):
            _write(output / "collection_failure.json", infos)
            raise RuntimeError("collection safety/budget failure; remaining partitions not started")
    rows = partitions["training"] + partitions["internal_audit"]
    replay = []
    for path, spec, name in tapes:
        result = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(_config(spec)), path)
        replay.append({"path": str(path), "stratum": name, "exact": bool(result["exact"])})
        _write(output / "replay_progress.json", {"completed": len(replay), "total": len(tapes), "exact": all(r["exact"] for r in replay)})
    coverage = _coverage(partitions["training"], support_protocol)
    provenance, aliasing = _provenance(rows), _aliasing(rows)
    keys = ("seed", "episode_id", "lifecycle_event_id", "decision_event_id", "full_state_hash")
    independent = all(not ({r[k] for r in partitions["training"]} & {r[k] for r in partitions["internal_audit"]}) for k in keys)
    for key, historic_key in (("seed", "seeds"), ("episode_id", "episode_ids"), ("lifecycle_event_id", "lifecycle_ids"), ("full_state_hash", "full_state_hashes")):
        independent &= not bool({r[key] for r in rows} & historical[historic_key])
    independent &= len({r["decision_event_id"] for r in rows}) == len(rows)
    gates = {
        "fixed_schedule_complete": all(infos[p]["episodes"] == len(seeds[p]) for p in seeds),
        "cardinality": all(protocol[p]["minimum_rows"] <= len(partitions[p]) <= protocol[p]["maximum_rows"] for p in seeds),
        "nominal_anchors": coverage["nominal_anchor_PASS"],
        "all_residual_context_cells": coverage["all_residual_context_cells_PASS"],
        "proposal14_N5_N6_N12": coverage["proposal14_traffic_PASS"],
        "stable_support": coverage["stable_support_PASS"],
        "schema_retry_provenance": provenance["schema_provenance"] == "PASS" and provenance["retry_feature_provenance"] == "PASS",
        "teacher_determinism": all(r["teacher_joint_index"] == r["teacher_repeat_joint_index"] for r in rows),
        "replay": len(replay) == 2 * len(strata) and all(r["exact"] for r in replay),
        "aliasing": aliasing["observation_aliasing"] == "PASS",
        "safety": not any(v for info in infos.values() for v in info["safety"].values()),
        "independence": bool(independent and not smoke),
    }
    if set(gates) != set(protocol["formal_gates"]):
        raise ValueError("qualification gate implementation does not match protocol")
    passed = not smoke and all(gates.values())
    manifest = {"stage": "V4_2_V2_1_I0_R4", "smoke_only": smoke, "protocol_sha256": EXPECTED_PROTOCOL_SHA,
                "observation_schema": protocol["observation_schema"], "observation_dim": 88,
                "action_schema": protocol["action_schema"], "partitions": infos,
                "coverage": coverage, "provenance": provenance, "replay": replay, "aliasing": aliasing,
                "gates": gates, "I0_R4": "PASS" if passed else "FAIL", "I1_PROTOCOL_ALLOWED": passed,
                "PPO_ALLOWED": False, "Q_rows_used": 0, "historical_rows_used": 0}
    _write(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, args.history_root, smoke=args.smoke)
    print(json.dumps({k: result[k] for k in ("I0_R4", "gates", "PPO_ALLOWED")}, indent=2))
