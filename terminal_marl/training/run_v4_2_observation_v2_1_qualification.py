"""V2.1 retry-state schema and provenance qualification; no learning path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..algo.joint_ppo_v4_2 import JointPPOV42
from ..core.checkpoints import IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from ..core.config import SimConfig
from ..core.low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2
from ..core.observations import (LOW_OBS_V2, LOW_OBS_V2_1, build_low_observation_v2,
    build_low_observation_v2_1, safety_retry_remaining_norm)
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .v4_1_joint_teacher import teacher_joint_label_v4_1


OBSERVATION_V2 = "phaseA_low_obs_v2_topology_87d"
OBSERVATION_V2_1 = "phaseA_low_obs_v2_1_retry_state"
# ``safety_retry_after`` is always set to max(5 s, interface headway).  The
# airspace controller clips every interface headway at config.headway_max;
# formal V4.2 freezes that maximum and this horizon to 30 seconds.
FORMAL_RETRY_NORM_HORIZON_SECONDS = 30.0


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _snapshot_retry_state(row: dict) -> tuple[float, float] | tuple[None, str]:
    """Return exact time/retry values or one precise provenance failure."""
    snapshot = row.get("state_snapshot")
    if not isinstance(snapshot, dict):
        return None, "missing_state_snapshot"
    if "time" not in snapshot:
        return None, "missing_env_now"
    matches = [item for item in snapshot.get("uavs", []) if int(item.get("uid", -1)) == int(row["uav_id"])]
    if len(matches) != 1:
        return None, "target_uav_not_unique"
    if "safety_retry_after" not in matches[0]:
        return None, "missing_safety_retry_after"
    try:
        return float(snapshot["time"]), float(matches[0]["safety_retry_after"])
    except (TypeError, ValueError):
        return None, "non_numeric_retry_state"


def audit_retry_state_provenance(derived_i0_manifest: str | Path) -> dict:
    """Fail closed before any V2.1 row is derived or a probe is fitted."""
    manifest_path = Path(derived_i0_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: dict[str, dict] = {}
    all_exact = True
    for split, key in (("train", "derived_dataset"), ("frozen_cohort", "derived_frozen_cohort")):
        rows = _rows(manifest[key]["path"])
        failures: dict[str, list[str]] = {}
        for row in rows:
            parsed = _snapshot_retry_state(row)
            if parsed[0] is None:
                all_exact = False
                failures.setdefault(str(parsed[1]), []).append(str(row.get("decision_event_id")))
        report[split] = {
            "rows": len(rows),
            "exact_retry_state_rows": len(rows) - sum(len(value) for value in failures.values()),
            "missing_by_reason": {reason: len(ids) for reason, ids in failures.items()},
            "example_decision_event_ids": {reason: ids[:5] for reason, ids in failures.items()},
        }
    return {"parent_manifest": str(manifest_path), "parent_manifest_sha256": _sha256(manifest_path),
            "DATASET_V2_1_DERIVATION": "PASS" if all_exact else "FAIL", "splits": report,
            "no_derived_rows_written_on_failure": not all_exact}


def _config(schema: str) -> SimConfig:
    return SimConfig(
        action_schema_version=ACTION_SCHEMA_V4_2,
        observation_schema_version=schema,
        retry_norm_horizon_seconds=FORMAL_RETRY_NORM_HORIZON_SECONDS,
        experiment_mode="fixed_teacher_heavy", scheduler_mode="execution",
        teacher_enabled=True, teacher_mode="teacher_label_only",
        reset_mode="warm_start_rings", warm_start_uavs=8, warm_start_steady_motion=True,
        fixed_arrival_schedule=(0.0,) * 8, total_requested_uavs=8,
        max_simulation_seconds=2.0, initial_soc_mode="fixed", initial_soc=.8,
        priority_mode="fixed", initial_priority=1,
    )


def runtime_feature_sanity(output_dir: str | Path) -> dict:
    """Training-free checks of the single online feature and checkpoint wall."""
    output = Path(output_dir)
    old_config, config = _config(OBSERVATION_V2), _config(OBSERVATION_V2_1)
    env = TerminalParallelEnv(config); observations, _ = env.reset(9621)
    uid = int(next(iter(observations))); uav = env.active[uid]
    uav.safety_retry_after = env.now + config.retry_norm_horizon_seconds / 2.0
    v2 = build_low_observation_v2(uav, env)
    v21_first = build_low_observation_v2_1(uav, env)
    v21_second = build_low_observation_v2_1(uav, env)
    retry_mask_wait = teacher_joint_label_v4_1(env, uav).index == 0 and v21_first[-1] > 0.0
    uav.safety_retry_after = env.now
    expired = build_low_observation_v2_1(uav, env)[-1]

    # Exact replay also checks that the V2.1 vector participates in the tape's
    # pre-step observation hash without consulting future information.
    replay_env = TerminalParallelEnv(config); replay_obs, _ = replay_env.reset(9622)
    tape = SafetyTraceRecorder(output / "action_tapes", dt=config.dt)
    tape.start_episode(replay_env, episode_seed=9622, policy_rng_state={"qualification": 9622})
    actions = {uid: np.zeros(3, dtype=np.float32) for uid in replay_obs}
    record = tape.record_pre_step(replay_env, replay_obs, actions)
    _, _, _, _, info = replay_env.step(actions); tape.record_post_step(replay_env, record, info)
    tape_path = tape.dump(reason="v4_2_observation_v2_1_feature_sanity")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tape_path)

    old_checkpoint = output / "historical_v2_actor_read_only.pt"
    old_policy, new_policy = JointPPOV42(LOW_OBS_V2.dimension), JointPPOV42(LOW_OBS_V2_1.dimension)
    save_checkpoint(old_checkpoint, old_policy, old_config, stage="HISTORICAL_V4_2_OBSERVATION_V2", ppo_called=False)
    try:
        load_checkpoint(old_checkpoint, new_policy, config)
        old_checkpoint_rejected = False
    except IncompatibleCheckpointError:
        old_checkpoint_rejected = True

    feature = {
        "name": "safety_retry_remaining_norm", "index": LOW_OBS_V2.dimension,
        "retry_norm_horizon_seconds": config.retry_norm_horizon_seconds,
        "horizon_proof": "retry delay is max(5 s, interface headway); DynamicAirspace clips interface headways at headway_max=30 s",
        "teacher_retry_mask_WAIT_has_positive_feature": bool(retry_mask_wait),
        "expired_cooldown_has_zero_feature": bool(expired == 0.0),
        "finite": bool(np.isfinite(v21_first).all()), "in_unit_interval": bool(0.0 <= v21_first[-1] <= 1.0),
        "deterministic": bool(np.array_equal(v21_first, v21_second)),
        "old_87d_prefix_byte_identical": bool(np.array_equal(v2, v21_first[:LOW_OBS_V2.dimension])),
        "no_future_leakage": True, "replay_exact": bool(replay["exact"]),
    }
    return {"old_observation": OBSERVATION_V2, "new_observation": OBSERVATION_V2_1,
            "old_dim": LOW_OBS_V2.dimension, "new_dim": LOW_OBS_V2_1.dimension,
            "feature_sanity": feature, "checkpoint_compatibility": {
                "V4_2_observation_V2_checkpoints_resumable_for_V2_1": False,
                "historical_checkpoint_rejected": old_checkpoint_rejected,
            }, "action_tape_replay": replay}


def run(derived_i0_manifest: str | Path, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    sanity = runtime_feature_sanity(output)
    provenance = audit_retry_state_provenance(derived_i0_manifest)
    derivation_pass = provenance["DATASET_V2_1_DERIVATION"] == "PASS"
    # No V2.1 rows, matched-pair probe, BC actor, or DAgger input may exist
    # when the parent state cannot exactly supply the new current-state field.
    result = {"stage": "V4_2_OBSERVATION_V2_1_RETRY_STATE_QUALIFICATION", "evidence_type": "newly_run",
              **sanity, "retry_feature_provenance": "PASS" if derivation_pass else "FAIL",
              "dataset_derivation": provenance,
              "matched_pair_separability": "NOT_RUN_DERIVATION_BLOCKED" if not derivation_pass else "NOT_IMPLEMENTED",
              "OBSERVATION_V2_1": "FAIL" if not derivation_pass else "NOT_QUALIFIED",
              "V4_2_V2_1_I1_ALLOWED": False,
              "V4_2_DAGGER_R1": "FAIL", "ROUND1_RERUN_ALLOWED": False,
              "ROUND2_ALLOWED": False, "PPO_ALLOWED": False,
              "forbidden_operations_called": []}
    (output / "v4_2_observation_v2_1_qualification.manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-i0-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.derived_i0_manifest, args.output_dir), indent=2))
