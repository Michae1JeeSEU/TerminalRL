from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from terminal_marl.core.safety_trace import SafetyTraceRecorder, _digest, _json
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.training import evaluate_p3_adaptive_kl_v7 as evaluator
from terminal_marl.training import terminal_lifecycle_metrics as lifecycle


TAPE = Path(
    "/home/user/training_outputs/terminal/"
    "p3_adaptive_kl_formal_v7_20260926/run1/dev_before_training/"
    "flow6_873101/action_tapes/safety_trace_seed_873101_step_624.json"
)
OUT = Path(
    "/home/user/training_outputs/terminal/"
    "p3_terminal_credit_20260928/reward_hook_debug"
)
LIMIT = 100


def _value(value: Any) -> Any:
    return _json(value)


def _first_diff(left: Any, right: Any, path: str = "") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {
            "path": path or "$",
            "bare": _value(left),
            "lifecycle": _value(right),
            "kind": "type",
        }
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right), key=str):
            key_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return {
                    "path": key_path,
                    "bare": _value(left.get(key)),
                    "lifecycle": _value(right.get(key)),
                    "kind": "missing_key",
                }
            result = _first_diff(left[key], right[key], key_path)
            if result is not None:
                return result
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {
                "path": path or "$",
                "bare": _value(left),
                "lifecycle": _value(right),
                "kind": "length",
            }
        for index, (bare_item, lifecycle_item) in enumerate(zip(left, right)):
            result = _first_diff(bare_item, lifecycle_item, f"{path}[{index}]")
            if result is not None:
                return result
        return None
    if left != right:
        return {
            "path": path or "$",
            "bare": _value(left),
            "lifecycle": _value(right),
            "kind": "value",
        }
    return None


def _reset(env: TerminalParallelEnv, header: dict[str, Any]) -> dict[str, Any]:
    env.reset(int(header["episode_seed"]))
    return SafetyTraceRecorder._apply_reset_recipe(
        env, header.get("reset_recipe", {"kind": "generic_reset", "steps": []})
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(TAPE.read_text(encoding="utf-8"))
    header = {key: value for key, value in payload.items() if key not in {"records", "action_tape"}}
    rows = payload.get("action_tape", payload.get("records", []))[:LIMIT]
    config = evaluator._build_configs(evaluator._protocol())["flow6"]

    bare = TerminalParallelEnv(config)
    hooked = TerminalParallelEnv(config)
    bare_obs = _reset(bare, header)
    lifecycle._capture_terminal_rows(hooked)
    hooked_obs = _reset(hooked, header)
    result: dict[str, Any] = {
        "stage": "P3_TERMINAL_CREDIT_REWARD_HOOK_DEBUG_V1",
        "status": "RUNNING",
        "tape": str(TAPE),
        "limit_steps": LIMIT,
        "reset": {
            "observation_digest_equal": _digest(
                {key: value for key, value in sorted(bare_obs.items())}
            )
            == _digest(
                {key: value for key, value in sorted(hooked_obs.items())}
            ),
            "state_digest_equal": _digest(SafetyTraceRecorder._state(bare))
            == _digest(SafetyTraceRecorder._state(hooked)),
            "state_diff": _first_diff(
                SafetyTraceRecorder._state(bare), SafetyTraceRecorder._state(hooked)
            ),
        },
        "steps_compared": 0,
        "state_mismatch_steps": [],
        "reward_mismatch_steps": [],
        "component_mismatch_steps": [],
        "first_state_difference": None,
        "first_reward_difference": None,
        "first_component_difference": None,
        "errors": [],
        "first_blocked_or_error": None,
    }
    for index, row in enumerate(rows):
        actions = {
            str(uid): np.asarray(action, dtype=np.float32)
            for uid, action in row["a_requested"].items()
        }
        shortlists = {
            str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
            for uid, values in row.get("runtime_proposal_shortlists", {}).items()
        }
        bare_before = SafetyTraceRecorder._state(bare)
        hooked_before = SafetyTraceRecorder._state(hooked)
        before_diff = _first_diff(bare_before, hooked_before)
        try:
            bare_result = bare.step(actions, shortlists)
        except Exception as exc:
            record = {
                "mode": "bare",
                "step": index,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            result["errors"].append(record)
            result["first_blocked_or_error"] = record
            break
        try:
            hooked_result = hooked.step(actions, shortlists)
        except Exception as exc:
            record = {
                "mode": "lifecycle",
                "step": index,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            result["errors"].append(record)
            result["first_blocked_or_error"] = record
            break
        result["steps_compared"] += 1
        if before_diff is not None:
            result["state_mismatch_steps"].append(index)
            if result["first_state_difference"] is None:
                result["first_state_difference"] = {
                    "position": "before_step",
                    "step": index,
                    "difference": before_diff,
                }
        bare_after = SafetyTraceRecorder._state(bare)
        hooked_after = SafetyTraceRecorder._state(hooked)
        after_diff = _first_diff(bare_after, hooked_after)
        if after_diff is not None:
            result["state_mismatch_steps"].append(index)
            if result["first_state_difference"] is None:
                result["first_state_difference"] = {
                    "position": "after_step",
                    "step": index,
                    "difference": after_diff,
                    "tape_post_state_hash": row.get("post_state_hash"),
                    "bare_post_state_hash": _digest(bare_after),
                    "lifecycle_post_state_hash": _digest(hooked_after),
                }
        bare_rewards = _value(bare_result[1])
        hooked_rewards = _value(hooked_result[1])
        if bare_rewards != hooked_rewards:
            result["reward_mismatch_steps"].append(index)
            if result["first_reward_difference"] is None:
                result["first_reward_difference"] = {
                    "step": index,
                    "bare": bare_rewards,
                    "lifecycle": hooked_rewards,
                }
        bare_components = _value(bare_result[4].get("reward_components"))
        hooked_components = _value(hooked_result[4].get("reward_components"))
        if bare_components != hooked_components:
            result["component_mismatch_steps"].append(index)
            if result["first_component_difference"] is None:
                result["first_component_difference"] = {
                    "step": index,
                    "bare": bare_components,
                    "lifecycle": hooked_components,
                }
    result["status"] = "COMPLETED" if not result["errors"] else "FAILED"
    result["bare_final_state_digest"] = _digest(SafetyTraceRecorder._state(bare))
    result["lifecycle_final_state_digest"] = _digest(SafetyTraceRecorder._state(hooked))
    (OUT / "debug.json").write_text(
        json.dumps(_value(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "progress.json").write_text(
        json.dumps(
            {
                "stage": result["stage"],
                "status": result["status"],
                "steps_compared": result["steps_compared"],
                "expected_steps": LIMIT,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "steps_compared": result["steps_compared"],
                "state_mismatch_steps": result["state_mismatch_steps"][:5],
                "reward_mismatch_steps": result["reward_mismatch_steps"][:5],
                "component_mismatch_steps": result["component_mismatch_steps"][:5],
                "first_blocked_or_error": result["first_blocked_or_error"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
