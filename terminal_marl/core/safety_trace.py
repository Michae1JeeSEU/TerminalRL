"""Bounded, serializable action provenance for deterministic safety replay."""
from __future__ import annotations

from base64 import b64decode, b64encode
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from .low_level_action import ACTION_SCHEMA_V3, LowLevelActionV3
from .low_level_joint_action import ACTION_SCHEMA_V4, joint_index_from_normalized


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SafetyTraceRecorder:
    """Capture an exact requested-action tape plus a bounded diagnostic window.

    The recorder is passive: it observes the requested policy action before
    ``env.step`` and the fully executed action trace after it.  This preserves
    PPO requested-action provenance while making future learned-policy safety
    witnesses independently replayable.
    """

    schema_version = "A5V2-L1-S4-SafetyTrace-v4"

    def __init__(self, output_dir: str | Path, *, max_seconds: float = 120.0, dt: float = 1.0) -> None:
        self.output_dir = Path(output_dir)
        self.max_records = max(1, int(round(max_seconds / dt)))
        self._records: deque[dict[str, Any]] = deque(maxlen=self.max_records)
        # Unlike ``_records``, this intentionally covers the whole episode.
        # It is compact (actions, hashes and optional event metadata only), so
        # an event after the 120 s diagnostic window remains replayable from
        # the reset state rather than merely inspectable near the event.
        self._action_tape: list[dict[str, Any]] = []
        self.header: dict[str, Any] = {}
        self.dumped_path: Path | None = None

    @staticmethod
    def _replay_state_value(value: Any) -> Any:
        """Remove opaque runtime IDs from an otherwise exact replay state.

        Scheduler and trajectory identifiers are UUIDs allocated independently
        during replay.  They are not policy/safety semantics, but including
        them in a state digest made a genuinely identical replay fail as soon
        as a release created a reservation.  Preserve the existence and shape
        of each reference while excluding its allocator-specific spelling.
        """
        opaque = {"reservation_id", "trajectory_plan_id", "merge_reservation_id", "plan_id", "forecast_plan_id"}
        if isinstance(value, dict):
            return {str(key): ("<opaque-runtime-id>" if str(key) in opaque else
                               ["<opaque-runtime-id>"] * len(item) if str(key) == "reservation_ids" and isinstance(item, (tuple, list)) else
                               SafetyTraceRecorder._replay_state_value(item))
                    for key, item in value.items()}
        if isinstance(value, (tuple, list, deque)):
            return ["<opaque-runtime-id>" if isinstance(item, str) and item.startswith(("plan-", "reservation-"))
                    else SafetyTraceRecorder._replay_state_value(item) for item in value]
        return value

    @staticmethod
    def _state(env) -> dict[str, Any]:
        reservations = [vars(value).copy() for value in sorted(
            env.scheduler.reservations.values(),
            key=lambda value: (value.uav_id, value.source_stage, value.target_stage, value.merge_time),
        )]
        return {
            "time": float(env.now),
            "step_count": int(env.step_count),
            "uavs": _json(SafetyTraceRecorder._replay_state_value(
                [env._uav_trace(uav) for _, uav in sorted(env.active.items())]
            )),
            "reservations": _json(SafetyTraceRecorder._replay_state_value(reservations)),
            "release_state": _json(SafetyTraceRecorder._replay_state_value(env.release_intents)),
            "future_release_reservations": _json(SafetyTraceRecorder._replay_state_value(env.future_release_reservations)),
        }

    def start_episode(self, env, *, episode_seed: int, policy_rng_state: Any) -> None:
        self._records.clear()
        self._action_tape.clear()
        self.dumped_path = None
        reset_state = self._state(env)
        self.header = {
            "schema_version": self.schema_version,
            "episode_seed": int(episode_seed),
            "reset_state_hash": _digest(reset_state),
            "reset_state": reset_state,
            "environment_config_hash": _digest(vars(env.config)),
            # Pickle is only a lossless local trace encoding, never an input
            # protocol.  It permits exact restoration of NumPy/Torch RNGs.
            "policy_rng_state_b64": b64encode(pickle.dumps(policy_rng_state, protocol=4)).decode("ascii"),
            "max_pre_event_seconds": self.max_records * float(env.config.dt),
        }

    def record_pre_step(self, env, observations: dict[str, np.ndarray], actions: dict[str, np.ndarray],
                        log_probs: dict[str, Any] | None = None) -> dict[str, Any]:
        action_v3: dict[str, Any] = {}; joint_actions: dict[str, int] = {}
        joint_log_prob: dict[str, float] = {}
        head_log_prob: dict[str, list[float]] = {}
        for uid, action in sorted(actions.items()):
            try:
                action_v3[str(uid)] = LowLevelActionV3.from_normalized(action).as_dict()
            except ValueError:
                action_v3[str(uid)] = {"legacy_continuous_action": _json(action)}
            try:
                joint_actions[str(uid)] = int(joint_index_from_normalized(action))
            except ValueError:
                pass
        for uid, value in sorted((log_probs or {}).items()):
            if isinstance(value, dict):
                joint_log_prob[str(uid)] = float(value["joint"])
                head_log_prob[str(uid)] = [float(item) for item in value.get("per_head", ())]
            else:
                joint_log_prob[str(uid)] = float(value)
        action_row = {
            "step": int(env.step_count), "time_before": float(env.now),
            "observation_hash": _digest({key: value for key, value in sorted(observations.items())}),
            "a_requested": _json({key: value for key, value in sorted(actions.items())}),
            "action_schema_version": (ACTION_SCHEMA_V4 if env.config.action_schema_version == ACTION_SCHEMA_V4 else ACTION_SCHEMA_V3),
            "a_requested_v3": action_v3,
            "joint_action_index": joint_actions,
            "joint_log_prob": joint_log_prob,
            "per_head_log_prob": head_log_prob,
            # Read-only compatibility name for archived tooling. New code
            # consumes ``joint_log_prob`` and ``per_head_log_prob``.
            "log_prob": joint_log_prob,
        }
        self._action_tape.append(action_row)
        record = {
            **action_row,
            "action_tape_index": len(self._action_tape) - 1,
            "pre_state": self._state(env),
        }
        self._records.append(record)
        return record

    def record_post_step(self, env, record: dict[str, Any], info: dict[str, Any]) -> None:
        post_state = self._state(env)
        record.update({
            "time_after": float(env.now),
            "a_teacher_shield_applied": _json(info.get("action_traces", {})),
            "shield_telemetry": _json(getattr(env.runtime_traffic, "telemetry", {})),
            "post_state": post_state,
            "conflict_events": _json(info.get("conflict_events", [])),
        })
        tape_index = int(record["action_tape_index"])
        self._action_tape[tape_index].update({
            "time_after": float(env.now),
            "post_state_hash": _digest(post_state),
            "executed_mapped_action": _json({uid: trace.get("execution_applied_action")
                                                for uid, trace in info.get("action_traces", {}).items()}),
            "conflict_events": _json(info.get("conflict_events", [])),
        })
        events = info.get("conflict_events", [])
        if self.dumped_path is None and any(row.get("kind") in {
            "first_hard_separation_trace", "nmac_entry", "collision"
        } for row in events):
            self.dump(reason="hard_safety_event")

    def record_safety_invariant(self, record: dict[str, Any] | None, error: Exception) -> None:
        """Attach a blocked fail-closed decision to the exact action tape."""
        event = {
            "kind": "safety_invariant_error",
            "message": str(error),
            "diagnostic": _json(getattr(error, "diagnostic", {})),
        }
        if record is not None:
            record.setdefault("conflict_events", []).append(event)
            tape_index = int(record["action_tape_index"])
            self._action_tape[tape_index]["safety_event"] = event
        elif self._action_tape:
            self._action_tape[-1]["safety_event"] = event
        self.dump(reason="uncertified_joint_safety_action")

    def dump(self, *, reason: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"safety_trace_seed_{self.header.get('episode_seed', 'unknown')}_step_{self._records[-1]['step'] if self._records else 0}"
        path = self.output_dir / f"{stem}.json"
        payload = {
            **self.header,
            "dump_reason": reason,
            "records": list(self._records),
            "action_tape": self._action_tape,
        }
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        self.dumped_path = path
        return path

    @staticmethod
    def load_action_tape(path: str | Path) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
        """Load an exact requested-action sequence for deterministic replay.

        The caller resets with ``header['episode_seed']`` and must verify the
        returned reset-state hash before accepting a replay as exact.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        header = {key: value for key, value in payload.items() if key not in {"records", "action_tape"}}
        rows = payload.get("action_tape", payload.get("records", []))
        actions = [
            {str(uid): np.asarray(action, dtype=np.float32) for uid, action in row["a_requested"].items()}
            for row in rows
        ]
        return header, actions

    @staticmethod
    def replay_action_tape(env, path: str | Path) -> dict[str, Any]:
        """Reset ``env`` and verify an action tape transition-by-transition.

        This is deliberately action-tape replay, rather than re-sampling the
        policy: it preserves PPO ``a_requested`` provenance even when policy
        implementation details change.  A successful result proves the reset
        state, each pre-step observation, and every recorded post-step state
        agree exactly with the captured episode.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        header = {key: value for key, value in payload.items() if key not in {"records", "action_tape"}}
        rows = payload.get("action_tape", payload.get("records", []))
        observations, _ = env.reset(int(header["episode_seed"]))
        reset_match = _digest(SafetyTraceRecorder._state(env)) == header["reset_state_hash"]
        config_match = _digest(vars(env.config)) == header.get("environment_config_hash", _digest(vars(env.config)))
        steps: list[dict[str, Any]] = []
        exact = bool(reset_match and config_match)
        for row in rows:
            observation_match = _digest({key: value for key, value in sorted(observations.items())}) == row["observation_hash"]
            actions = {str(uid): np.asarray(action, dtype=np.float32) for uid, action in row["a_requested"].items()}
            category_match = True
            if row.get("action_schema_version") == ACTION_SCHEMA_V4:
                for uid, action in actions.items():
                    try:
                        category_match = category_match and int(joint_index_from_normalized(action)) == row.get("joint_action_index", {}).get(uid)
                    except ValueError:
                        category_match = False
            elif row.get("action_schema_version") == ACTION_SCHEMA_V3:
                for uid, action in actions.items():
                    try:
                        expected = LowLevelActionV3.from_normalized(action).as_dict()
                        category_match = category_match and expected == row.get("a_requested_v3", {}).get(uid)
                    except ValueError:
                        category_match = False
            executed_match = True
            try:
                observations, _, _, _, info = env.step(actions)
                post_match = ("post_state_hash" not in row or
                              _digest(SafetyTraceRecorder._state(env)) == row["post_state_hash"])
                if "executed_mapped_action" in row:
                    executed_match = _json({uid: trace.get("execution_applied_action")
                                             for uid, trace in info.get("action_traces", {}).items()}) == row["executed_mapped_action"]
                else:
                    executed_match = True
                blocked = False
            except Exception as error:  # SafetyInvariantError is expected in a blocked event trace.
                expected_event = row.get("safety_event", {})
                post_match = (expected_event.get("kind") == "safety_invariant_error" and
                              expected_event.get("message") == str(error))
                blocked = True
                error_text = str(error)
            steps.append({"step": int(row["step"]), "observation_match": observation_match,
                          "requested_category_match": category_match, "post_state_match": post_match,
                          "executed_mapped_action_match": executed_match, "blocked": blocked,
                          **({"error": error_text} if blocked else {})})
            exact = exact and observation_match and category_match and post_match and executed_match
            if blocked:
                break
        return {"exact": exact, "reset_state_match": reset_match, "config_match": config_match,
                "steps_replayed": len(steps), "steps_expected": len(rows), "steps": steps}

    @staticmethod
    def decode_policy_rng_state(header: dict[str, Any]) -> Any:
        """Return the lossless captured RNG state for a trusted local trace."""
        return pickle.loads(b64decode(header["policy_rng_state_b64"]))
