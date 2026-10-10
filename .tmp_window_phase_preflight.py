"""Read-only exact replay probe for UID 42 at steps 99..105."""
from __future__ import annotations

import json

from terminal_marl.training import collect_p3_proposal_head_probe_v1 as c
from terminal_marl.envs.parallel_env import TerminalParallelEnv
from terminal_marl.core.safety_trace import SafetyTraceRecorder


protocol, configs, records, manifest = c._load_source_inputs(c.FORMAL_ROOT)
record = next(row for row in records if int(row["seed"]) == 873103)
env = TerminalParallelEnv(configs["dense24"])
original = env.step
observations: list[dict] = []


def _enum_value(value):
    return getattr(value, "value", value)


def observed_step(*args, **kwargs):
    step = int(env.step_count)
    if step in range(99, 106):
        actor = env.active.get(42)
        observation = {
            "step": step,
            "uid": 42,
            "present": actor is not None,
            "phase": _enum_value(getattr(actor, "phase", None)),
            "arrival_id": getattr(actor, "arrival_id", None),
            "ring_role": _enum_value(getattr(actor, "ring_role", None)),
            "committed_plan": repr(getattr(actor, "committed_plan", None)),
        }
        observations.append(observation)
        print("PRE_STEP " + json.dumps(observation, sort_keys=True, default=str), flush=True)
    return original(*args, **kwargs)


env.step = observed_step  # type: ignore[method-assign]
replay = SafetyTraceRecorder.replay_action_tape(env, record["action_tape"])
print("OBSERVATIONS " + json.dumps(observations, sort_keys=True, default=str), flush=True)
print(
    "EXACT_RESULT "
    + json.dumps(
        {
            "exact": replay.get("exact"),
            "reset_state_match": replay.get("reset_state_match"),
            "config_match": replay.get("config_match"),
            "reset_recipe_match": replay.get("reset_recipe_match"),
            "base_reset_state_match": replay.get("base_reset_state_match"),
            "steps_replayed": replay.get("steps_replayed"),
            "steps_expected": replay.get("steps_expected"),
        },
        sort_keys=True,
        default=str,
    ),
    flush=True,
)
