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
from .low_level_joint_action_v4_1 import ACTION_SCHEMA_V4_1, joint_index_from_normalized_v4_1
from .low_level_joint_action_v4_2 import ACTION_SCHEMA_V4_2, hierarchical_from_normalized_v4_2


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "-Infinity" if value < 0 else "Infinity" if value > 0 else "NaN"
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


_V4_2_REJECTED_RELEASE_HOLD_FIELD = "v4_2_rejected_release_hold_mode"
_LEGACY_CONFIG_COMPATIBILITY = "legacy_requested_default_field"
_WARM_START_LAYOUT_RANDOMIZED_FIELD = "warm_start_layout_randomized"
_WARM_START_LAYOUT_RANDOMIZED_COMPATIBILITY = "warm_start_layout_randomized_default_field"
_COMBINED_LEGACY_CONFIG_COMPATIBILITY = (
    f"{_LEGACY_CONFIG_COMPATIBILITY}+{_WARM_START_LAYOUT_RANDOMIZED_COMPATIBILITY}"
)


def _environment_config_match(config: Any, expected_hash: str) -> tuple[bool, str | None]:
    """Match a tape config hash, allowing only verified default fields.

    Historical tapes were written before one or both of the compatibility
    fields existed.  Each fallback compares the complete config after removing
    exactly the absent default field(s).  Randomized layouts never accept an
    old hash because their layout semantics are seed-dependent.
    """

    values = vars(config)
    if _digest(values) == expected_hash:
        return True, None
    if getattr(config, _WARM_START_LAYOUT_RANDOMIZED_FIELD, False) is not False:
        return False, None
    without_randomized_field = dict(values)
    without_randomized_field.pop(_WARM_START_LAYOUT_RANDOMIZED_FIELD, None)
    if _digest(without_randomized_field) == expected_hash:
        return True, _WARM_START_LAYOUT_RANDOMIZED_COMPATIBILITY
    if getattr(config, _V4_2_REJECTED_RELEASE_HOLD_FIELD, None) == "legacy_requested":
        without_hold_field = dict(values)
        without_hold_field.pop(_V4_2_REJECTED_RELEASE_HOLD_FIELD, None)
        if _digest(without_hold_field) == expected_hash:
            return True, _LEGACY_CONFIG_COMPATIBILITY

        without_legacy_fields = dict(values)
        without_legacy_fields.pop(_V4_2_REJECTED_RELEASE_HOLD_FIELD, None)
        without_legacy_fields.pop(_WARM_START_LAYOUT_RANDOMIZED_FIELD, None)
        if _digest(without_legacy_fields) == expected_hash:
            return True, _COMBINED_LEGACY_CONFIG_COMPATIBILITY
    return False, None


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
        def ring(state):
            return {"role": state.role.value, "radius": float(state.radius), "height": float(state.height),
                    "nominal_speed": float(state.nominal_speed), "derived_capacity": int(state.derived_capacity)}
        transition = env.airspace.transition
        geometry_transition = None if transition is None else {
            "start_time": float(transition.start_time), "end_time": float(transition.end_time),
            "start": {role.value: ring(value) for role, value in transition.start.items()},
            "target": {role.value: ring(value) for role, value in transition.target.items()},
            "affected_interfaces": sorted(transition.affected_interfaces),
        }
        return {
            "time": float(env.now),
            "step_count": int(env.step_count),
            "uavs": _json(SafetyTraceRecorder._replay_state_value(
                [env._uav_trace(uav) for _, uav in sorted(env.active.items())]
            )),
            "reservations": _json(SafetyTraceRecorder._replay_state_value(reservations)),
            "release_state": _json(SafetyTraceRecorder._replay_state_value(env.release_intents)),
            "future_release_reservations": _json(SafetyTraceRecorder._replay_state_value(env.future_release_reservations)),
            "dynamic_airspace": {
                "actual": {role.value: ring(env.airspace.actual_ring(role)) for role in env.airspace._actual},
                "target": {role.value: ring(env.airspace.target_ring(role)) for role in env.airspace._target},
                "headways": _json(env.airspace.headways), "speed_profile_bias": float(env.airspace.speed_profile_bias),
                "last_geometry_update": float(env.airspace.last_geometry_update),
                "occupancy": {role.value: int(value) for role, value in env.airspace._occupancy.items()},
                "committed_incoming": {role.value: int(value) for role, value in env.airspace._committed.items()},
                "transition": geometry_transition,
            },
            "scheduler_last_executed_release": _json(env.scheduler.last_executed_release),
            "landing_zones": [{"lz_id": int(zone.lz_id), "status": zone.status.value,
                               "reserved_by": zone.reserved_by, "reservation_start": zone.reservation_start,
                               "occupied_by": zone.occupied_by, "service_remaining": float(zone.service_remaining)}
                              for zone in env.lz_manager.zones],
            "event_scheduler": _json(env.events),
            "arrival_process": {"generated": int(env.arrivals.generated), "index": int(env.arrivals.index),
                                "wave": int(env.arrivals.wave), "next_time": float(env.arrivals.next_time),
                                "rng_state": _json(env.arrivals.rng.bit_generator.state)},
        }

    def start_episode(self, env, *, episode_seed: int, policy_rng_state: Any,
                      reset_recipe: dict[str, Any] | None = None) -> None:
        self._records.clear()
        self._action_tape.clear()
        self.dumped_path = None
        pre_decision_state = self._state(env)
        recipe = _json(reset_recipe) if reset_recipe is not None else {"kind": "generic_reset", "steps": []}
        self.header = {
            "schema_version": self.schema_version,
            "episode_seed": int(episode_seed),
            # This state is after any collector-specific deterministic setup,
            # i.e. exactly the first low-level decision environment.
            "reset_state_hash": _digest(pre_decision_state),
            "reset_state": pre_decision_state,
            "reset_recipe": recipe,
            "reset_recipe_hash": _digest(recipe),
            "dynamic_airspace_state_hash": _digest(pre_decision_state["dynamic_airspace"]),
            "arrival_rng_state_hash": _digest(pre_decision_state["arrival_process"]["rng_state"]),
            "environment_config_hash": _digest(vars(env.config)),
            # Pickle is only a lossless local trace encoding, never an input
            # protocol.  It permits exact restoration of NumPy/Torch RNGs.
            "policy_rng_state_b64": b64encode(pickle.dumps(policy_rng_state, protocol=4)).decode("ascii"),
            "max_pre_event_seconds": self.max_records * float(env.config.dt),
        }

    @staticmethod
    def _apply_reset_recipe(env, recipe: dict[str, Any]) -> dict[str, np.ndarray]:
        """Restore declared collector setup between ``reset`` and low-level step 0.

        Recipes intentionally contain only deterministic environment operations;
        they are never policy actions and cannot alter requested-action
        provenance. Unknown recipe kinds fail closed rather than silently
        reducing exact replay to a seed-only reset.
        """
        kind = recipe.get("kind", "generic_reset")
        if kind == "generic_reset":
            return env._observations()
        if kind != "high_level_action_after_reset":
            raise ValueError(f"unsupported ActionTape reset recipe: {kind}")
        action = np.asarray(recipe["high_level_action"], dtype=np.float32)
        env.apply_high_level_action(action, geometry_gate=bool(recipe["geometry_gate"]))
        return env._observations()

    def record_pre_step(self, env, observations: dict[str, np.ndarray], actions: dict[str, np.ndarray],
                        log_probs: dict[str, Any] | None = None, *,
                        proposal_shortlists: dict[str, tuple[np.ndarray, ...] | list[np.ndarray]] | None = None) -> dict[str, Any]:
        action_v3: dict[str, Any] = {}; joint_actions: dict[str, int] = {}; hierarchical: dict[str, dict[str, Any]] = {}
        joint_log_prob: dict[str, float] = {}
        head_log_prob: dict[str, list[float]] = {}
        gate_log_prob: dict[str, float] = {}; proposal_log_prob: dict[str, float | None] = {}
        for uid, action in sorted(actions.items()):
            try:
                action_v3[str(uid)] = LowLevelActionV3.from_normalized(action).as_dict()
            except ValueError:
                action_v3[str(uid)] = {"legacy_continuous_action": _json(action)}
            try:
                decoder = joint_index_from_normalized_v4_1 if env.config.action_schema_version in {ACTION_SCHEMA_V4_1, ACTION_SCHEMA_V4_2} else joint_index_from_normalized
                joint_actions[str(uid)] = int(decoder(action))
                if env.config.action_schema_version == ACTION_SCHEMA_V4_2:
                    requested = hierarchical_from_normalized_v4_2(action)
                    hierarchical[str(uid)] = {"gate": requested.gate, "proposal": requested.proposal}
            except ValueError:
                pass
        for uid, value in sorted((log_probs or {}).items()):
            if isinstance(value, dict):
                joint_log_prob[str(uid)] = float(value["joint"])
                head_log_prob[str(uid)] = [float(item) for item in value.get("per_head", ())]
                if "gate" in value:
                    gate_log_prob[str(uid)] = float(value["gate"])
                    proposal_log_prob[str(uid)] = None if value.get("proposal") is None else float(value["proposal"])
            else:
                joint_log_prob[str(uid)] = float(value)
        action_row = {
            "step": int(env.step_count), "time_before": float(env.now),
            "observation_hash": _digest({key: value for key, value in sorted(observations.items())}),
            "a_requested": _json({key: value for key, value in sorted(actions.items())}),
            "action_schema_version": (ACTION_SCHEMA_V4_2 if env.config.action_schema_version == ACTION_SCHEMA_V4_2 else ACTION_SCHEMA_V4_1 if env.config.action_schema_version == ACTION_SCHEMA_V4_1 else ACTION_SCHEMA_V4 if env.config.action_schema_version == ACTION_SCHEMA_V4 else ACTION_SCHEMA_V3),
            "a_requested_v3": action_v3,
            "joint_action_index": joint_actions,
            "joint_log_prob": joint_log_prob,
            "per_head_log_prob": head_log_prob,
            "requested_gate": {uid: value["gate"] for uid, value in hierarchical.items()},
            "requested_proposal": {uid: value["proposal"] for uid, value in hierarchical.items()},
            "gate_logprob": gate_log_prob,
            "proposal_logprob": proposal_log_prob,
            "runtime_proposal_shortlists": _json({
                str(uid): list(values) for uid, values in sorted((proposal_shortlists or {}).items())
            }),
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
        def hierarchy_field(trace: dict[str, Any], field: str) -> dict[str, Any] | None:
            value = trace.get(field)
            return value if isinstance(value, dict) and "gate_index" in value else None
        teacher_hierarchy = {uid: hierarchy_field(trace, "teacher_hierarchical_action")
                             for uid, trace in info.get("action_traces", {}).items()}
        self._action_tape[tape_index].update({
            "time_after": float(env.now),
            "post_state_hash": _digest(post_state),
            "decoded_action": _json({uid: trace.get("decoded_joint_action", trace.get("requested_low_action"))
                                       for uid, trace in info.get("action_traces", {}).items()}),
            "decoded_requested_action": _json({uid: trace.get("decoded_joint_action", trace.get("requested_low_action"))
                                                 for uid, trace in info.get("action_traces", {}).items()}),
            "teacher_joint": _json({uid: trace.get("teacher_joint_action")
                                     for uid, trace in info.get("action_traces", {}).items()}),
            "teacher_gate": _json({uid: None if value is None else value["gate_index"] for uid, value in teacher_hierarchy.items()}),
            "teacher_proposal": _json({uid: None if value is None else value["proposal"] for uid, value in teacher_hierarchy.items()}),
            "applied_action": _json({uid: trace.get("execution_applied_action")
                                      for uid, trace in info.get("action_traces", {}).items()}),
            "shield_action": _json({uid: trace.get("shield_applied_action")
                                     for uid, trace in info.get("action_traces", {}).items()}),
            "executed_mapped_action": _json({uid: trace.get("execution_applied_action")
                                                for uid, trace in info.get("action_traces", {}).items()}),
            "runtime_shortlist_result": _json({
                uid: {
                    "used": bool(trace.get("runtime_shortlist_used")),
                    "rank": trace.get("runtime_shortlist_rank"),
                    "candidates_checked": int(trace.get("runtime_shortlist_candidates_checked", 0)),
                }
                for uid, trace in info.get("action_traces", {}).items()
            }),
            "conflict_events": _json(info.get("conflict_events", [])),
        })
        events = info.get("conflict_events", [])
        if self.dumped_path is None and any(row.get("kind") in {
            "first_hard_separation_trace", "nmac_entry", "collision"
        } for row in events):
            self.dump(reason="hard_safety_event")

    def record_teacher_labels(self, record: dict[str, Any], labels: dict[str, dict[str, Any]]) -> None:
        """Attach external label-only Teacher targets without changing execution.

        The caller computes these labels from the state already captured by
        ``record_pre_step``.  This method is deliberately trace-only: it does
        not pass labels to the environment, scheduler, shield, or policy.
        """
        tape_index = int(record["action_tape_index"])
        gate = {str(uid): int(value["gate"]) for uid, value in labels.items()}
        proposal = {str(uid): value.get("proposal") for uid, value in labels.items()}
        joint = {str(uid): int(value["joint_index_v4_1"]) for uid, value in labels.items()}
        self._action_tape[tape_index].update({"teacher_gate": gate, "teacher_proposal": proposal,
                                               "teacher_joint": joint, "teacher_label_only": True})
        record.update({"teacher_gate": gate, "teacher_proposal": proposal, "teacher_joint": joint,
                       "teacher_label_only": True})

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
        path.write_text(json.dumps(_json(payload), indent=2, allow_nan=False), encoding="utf-8")
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
        tape_schema = rows[0].get("action_schema_version") if rows else None
        if env.config.action_schema_version == ACTION_SCHEMA_V4_1 and tape_schema == ACTION_SCHEMA_V4:
            raise ValueError("historical V4 30-class ActionTape is read-only and cannot replay under V4.1")
        if env.config.action_schema_version == ACTION_SCHEMA_V4_2 and tape_schema in {ACTION_SCHEMA_V4, ACTION_SCHEMA_V4_1}:
            raise ValueError("historical V4/V4.1 ActionTape is read-only and cannot replay under V4.2")
        if tape_schema == ACTION_SCHEMA_V4_1 and env.config.action_schema_version != ACTION_SCHEMA_V4_1:
            raise ValueError("V4.1 ActionTape requires a V4.1 environment schema")
        if tape_schema == ACTION_SCHEMA_V4_2 and env.config.action_schema_version != ACTION_SCHEMA_V4_2:
            raise ValueError("V4.2 ActionTape requires a V4.2 environment schema")
        observations, _ = env.reset(int(header["episode_seed"]))
        recipe = header.get("reset_recipe", {"kind": "generic_reset", "steps": []})
        recipe_hash_match = _digest(recipe) == header.get("reset_recipe_hash", _digest(recipe))
        base_reset_match = (_digest(SafetyTraceRecorder._state(env)) == recipe.get("base_reset_state_hash")
                            if recipe.get("base_reset_state_hash") is not None else True)
        observations = SafetyTraceRecorder._apply_reset_recipe(env, recipe)
        reset_match = _digest(SafetyTraceRecorder._state(env)) == header["reset_state_hash"]
        config_match, config_compatibility = _environment_config_match(
            env.config,
            header.get("environment_config_hash", _digest(vars(env.config))),
        )
        steps: list[dict[str, Any]] = []
        exact = bool(reset_match and config_match and recipe_hash_match and base_reset_match)
        for row in rows:
            observation_match = _digest({key: value for key, value in sorted(observations.items())}) == row["observation_hash"]
            actions = {str(uid): np.asarray(action, dtype=np.float32) for uid, action in row["a_requested"].items()}
            proposal_shortlists = {
                str(uid): tuple(np.asarray(action, dtype=np.float32) for action in values)
                for uid, values in row.get("runtime_proposal_shortlists", {}).items()
            }
            category_match = True
            teacher_label_match = True
            if row.get("action_schema_version") == ACTION_SCHEMA_V4_2:
                for uid, action in actions.items():
                    try:
                        recovered = hierarchical_from_normalized_v4_2(action)
                        category_match = category_match and recovered.gate == row.get("requested_gate", {}).get(uid) and recovered.proposal == row.get("requested_proposal", {}).get(uid)
                    except ValueError:
                        category_match = False
            elif row.get("action_schema_version") == ACTION_SCHEMA_V4_1:
                for uid, action in actions.items():
                    try:
                        category_match = category_match and int(joint_index_from_normalized_v4_1(action)) == row.get("joint_action_index", {}).get(uid)
                    except ValueError:
                        category_match = False
            elif row.get("action_schema_version") == ACTION_SCHEMA_V4:
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
            if row.get("action_schema_version") == ACTION_SCHEMA_V4_2 and row.get("teacher_label_only"):
                # Local import prevents core runtime ownership from depending
                # on the training labeler unless an explicit replay asks for it.
                from ..training.v4_1_joint_teacher import teacher_joint_label_v4_1
                from .low_level_joint_action_v4_2 import hierarchical_from_joint_index_v4_1
                expected_gate, expected_proposal, expected_joint = row.get("teacher_gate", {}), row.get("teacher_proposal", {}), row.get("teacher_joint", {})
                for uid in actions:
                    if int(uid) not in env.active:
                        teacher_label_match = False; continue
                    target = hierarchical_from_joint_index_v4_1(teacher_joint_label_v4_1(env, env.active[int(uid)]).index)
                    teacher_label_match = teacher_label_match and target.gate == expected_gate.get(uid) and target.proposal == expected_proposal.get(uid) and target.joint_index_v4_1 == expected_joint.get(uid)
            executed_match = True
            runtime_shortlist_match = True
            try:
                observations, _, _, _, info = env.step(actions, proposal_shortlists)
                post_match = ("post_state_hash" not in row or
                              _digest(SafetyTraceRecorder._state(env)) == row["post_state_hash"])
                if "executed_mapped_action" in row:
                    executed_match = _json({uid: trace.get("execution_applied_action")
                                             for uid, trace in info.get("action_traces", {}).items()}) == row["executed_mapped_action"]
                else:
                    executed_match = True
                if "runtime_shortlist_result" in row:
                    runtime_shortlist_match = _json({
                        uid: {
                            "used": bool(trace.get("runtime_shortlist_used")),
                            "rank": trace.get("runtime_shortlist_rank"),
                            "candidates_checked": int(trace.get("runtime_shortlist_candidates_checked", 0)),
                        }
                        for uid, trace in info.get("action_traces", {}).items()
                    }) == row["runtime_shortlist_result"]
                blocked = False
            except Exception as error:  # SafetyInvariantError is expected in a blocked event trace.
                expected_event = row.get("safety_event", {})
                post_match = (expected_event.get("kind") == "safety_invariant_error" and
                              expected_event.get("message") == str(error))
                blocked = True
                error_text = str(error)
            steps.append({"step": int(row["step"]), "observation_match": observation_match,
                          "requested_category_match": category_match, "teacher_label_match": teacher_label_match, "post_state_match": post_match,
                          "executed_mapped_action_match": executed_match,
                          "runtime_shortlist_match": runtime_shortlist_match, "blocked": blocked,
                          **({"error": error_text} if blocked else {})})
            exact = exact and observation_match and category_match and teacher_label_match and post_match and executed_match and runtime_shortlist_match
            if blocked:
                break
        return {"exact": exact, "reset_state_match": reset_match, "config_match": config_match,
                "config_compatibility": config_compatibility,
                "reset_recipe_match": recipe_hash_match,
                "base_reset_state_match": base_reset_match,
                "steps_replayed": len(steps), "steps_expected": len(rows), "steps": steps}

    @staticmethod
    def decode_policy_rng_state(header: dict[str, Any]) -> Any:
        """Return the lossless captured RNG state for a trusted local trace."""
        return pickle.loads(b64decode(header["policy_rng_state_b64"]))
