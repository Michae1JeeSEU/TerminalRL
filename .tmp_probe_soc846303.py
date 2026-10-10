"""Bounded independent A/B witnesses during one exact action-tape replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2
from terminal_marl.core.runtime_traffic_safety import SafetyInvariantError
from terminal_marl.core.safety_trace import SafetyTraceRecorder
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import diagnose_trained_dense30_v2 as diagnostic
from terminal_marl.training import train_random_soc_gate_repair_v1 as repair


TAPE = Path(
    "/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/"
    "pair_dense24_846303/candidate/action_tapes/"
    "safety_trace_seed_846303_step_794.json"
)
CHECKPOINT = Path(
    "/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/"
    "fit846401/random_soc_gate_repair_v1.pt"
)
OUTPUT = Path(
    "/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/"
    "probe_candidate_emergency846303.json"
)
EXPECTED_SHA256 = "e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353"
TARGET_ARRIVAL_ID = 20
TARGET_UID = 148
PROBE_TIMES = {
    202.0, 212.0, 222.0, 232.0, 242.0,
    310.0, 320.0, 330.0, 340.0, 350.0, 360.0,
    437.0, 438.0, 439.0, 440.0, 441.0,
}


def _jsonable(value: Any) -> Any:
    return diagnostic._jsonable(value)


def _lookup(mapping: Mapping[Any, Any], uid: int) -> Any:
    if str(uid) in mapping:
        return mapping[str(uid)]
    return mapping.get(uid)


def _target_key(mapping: Mapping[Any, Any]) -> str | int:
    if str(TARGET_UID) in mapping:
        return str(TARGET_UID)
    if TARGET_UID in mapping:
        return TARGET_UID
    return str(TARGET_UID)


def _eligible(env: TerminalParallelEnv, uav: Any) -> bool:
    return bool(
        uav.committed_plan is None
        and uav.phase.value in {"ring_outer", "ring_middle", "ring_inner"}
        and TARGET_UID not in env.future_release_reservations
    )


def _probe_arm(
    env: TerminalParallelEnv,
    actions: Mapping[Any, Any],
    proposal_shortlists: Mapping[Any, Any],
    *,
    arm: str,
    action: np.ndarray,
    shortlist: tuple[np.ndarray, ...],
    base: dict[str, Any],
) -> dict[str, Any]:
    shadow = copy.deepcopy(env)
    shadow_actions = copy.deepcopy(dict(actions))
    shadow_shortlists = copy.deepcopy(dict(proposal_shortlists))
    key = _target_key(shadow_actions)
    shadow_actions[key] = np.asarray(action, dtype=np.float32).copy()
    shadow_shortlists[key] = tuple(
        np.asarray(item, dtype=np.float32).copy() for item in shortlist
    )
    row = {
        **base,
        "arm": arm,
        "requested_action": _jsonable(shadow_actions[key]),
        "proposal_shortlist": _jsonable(shadow_shortlists[key]),
        "decision": None,
        "action_trace": None,
        "release_intent_telemetry": None,
        "shadow_safety_error": None,
    }
    try:
        _, _, _, _, info = TerminalParallelEnv.step(
            shadow, shadow_actions, shadow_shortlists,
        )
        row["decision"] = _lookup(info.get("release_decisions", {}), TARGET_UID)
        row["action_trace"] = _jsonable(
            _lookup(info.get("action_traces", {}), TARGET_UID)
        )
        row["release_intent_telemetry"] = _jsonable(
            _lookup(info.get("release_intent_telemetry", {}), TARGET_UID)
        )
    except SafetyInvariantError as exc:
        row["shadow_safety_error"] = str(exc)
    return row


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    payload = json.loads(TAPE.read_text(encoding="utf-8"))
    rows = payload.get("action_tape", payload.get("records", []))
    if len(rows) != 795:
        raise ValueError(f"expected 795 action rows, got {len(rows)}")

    config = diagnostic.build_scenario_config(
        "dense24", reconcile_late_merge_slots=False,
    )
    policy, _ = repair.load_repaired_checkpoint(
        CHECKPOINT, config, expected_sha256=EXPECTED_SHA256,
    )
    env = TerminalParallelEnv(config)
    original_step = env.step
    probes: list[dict[str, Any]] = []
    probed_times: set[float] = set()
    calls = 0

    def wrapped_step(
        actions: dict[str, np.ndarray] | None = None,
        proposal_shortlists: dict[str, Any] | None = None,
    ):
        nonlocal calls
        actions = {} if actions is None else actions
        proposal_shortlists = {} if proposal_shortlists is None else proposal_shortlists
        now = float(env.now)
        uav = env.active.get(TARGET_UID)
        if (
            now in PROBE_TIMES
            and now not in probed_times
            and uav is not None
            and int(uav.arrival_id) == TARGET_ARRIVAL_ID
            and _eligible(env, uav)
        ):
            probed_times.add(now)
            obs = env._observations()
            details = diagnostic._trained_release_only_request_details(
                policy, env, obs, shortlist_k=3,
            )
            neural_actions = details[3]
            neural_shortlists = details[9]
            target_action_key = _target_key(actions)
            target_action = np.asarray(actions[target_action_key], dtype=np.float32)
            original_gate = rows[calls].get("requested_gate", {}).get(str(TARGET_UID))
            if original_gate is None:
                original_gate = int(
                    hierarchical_from_normalized_v4_2(target_action).gate
                )
            base = {
                "time": now,
                "soc": float(uav.soc),
                "phase": uav.phase.value,
                "original_gate": int(original_gate),
                "original_state_digest_before": diagnostic._env_state_digest(env),
            }
            neural_action_key = _target_key(neural_actions)
            neural_shortlist_key = _target_key(neural_shortlists)
            probes.append(
                _probe_arm(
                    env, actions, proposal_shortlists,
                    arm="A",
                    action=np.asarray(neural_actions[neural_action_key]),
                    shortlist=tuple(neural_shortlists[neural_shortlist_key]),
                    base=base,
                )
            )
            probes[-1]["original_state_digest_after_arm"] = diagnostic._env_state_digest(env)
            probes.append(
                _probe_arm(
                    env, actions, proposal_shortlists,
                    arm="B",
                    action=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    shortlist=(np.asarray([1.0, 0.0, 0.0], dtype=np.float32),),
                    base=base,
                )
            )
            probes[-1]["original_state_digest_after_arm"] = diagnostic._env_state_digest(env)
            digest_after = diagnostic._env_state_digest(env)
            probes[-2]["original_state_digest_after_all_shadows"] = digest_after
            probes[-1]["original_state_digest_after_all_shadows"] = digest_after
            unchanged = all(
                row["original_state_digest_before"] == row["original_state_digest_after_arm"]
                == row["original_state_digest_after_all_shadows"]
                for row in probes[-2:]
            )
            probes[-2]["original_state_unchanged"] = unchanged
            probes[-1]["original_state_unchanged"] = unchanged
            if not unchanged:
                raise RuntimeError(
                    f"original environment changed during shadow probes at time {now}"
                )
        result = original_step(actions, proposal_shortlists)
        calls += 1
        return result

    env.step = wrapped_step
    replay_result: Any = None
    replay_error: Any = None
    try:
        replay_result = SafetyTraceRecorder.replay_action_tape(env, TAPE)
    except Exception as exc:
        replay_error = {"type": type(exc).__name__, "message": str(exc)}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "tape": str(TAPE),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256_expected": EXPECTED_SHA256,
        "scenario": "dense24",
        "reconcile_late_merge_slots": False,
        "target": {"arrival_id": TARGET_ARRIVAL_ID, "uid": TARGET_UID},
        "probe_times": sorted(PROBE_TIMES),
        "probe_count": len(probes),
        "raw_probes": probes,
        "replay": replay_result,
        "replay_error": replay_error,
        "steps_seen_by_wrapper": calls,
    }
    OUTPUT.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    replay_summary = {
        key: None if replay_result is None else replay_result.get(key)
        for key in ("exact", "steps_replayed", "steps_expected", "reset_state_match", "config_match")
    }
    print(json.dumps({
        "output": str(OUTPUT),
        "replay": replay_summary,
        "probe_count": len(probes),
        "probed_times": sorted(probed_times),
        "original_state_unchanged": all(
            row.get("original_state_unchanged") is True for row in probes
        ),
        "replay_error": replay_error,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
