from pathlib import Path
from types import SimpleNamespace
import inspect
import json

from terminal_marl.training import diagnose_p3_ppo_credit_signal as credit
from terminal_marl.training import run_p3_followup_diagnostics as runner


archive = Path.cwd()
runner_path = Path(runner.__file__).resolve()
credit_path = Path(credit.__file__).resolve()
assert str(runner_path).startswith(str(archive))
assert str(credit_path).startswith(str(archive))
assert not (archive / "terminal_marl/training/diagnose_p3_task_neutral_episode_gradients.py").exists()

capture_signature = inspect.signature(credit._capture_replay)
assert capture_signature.parameters["state_observer"].default is None
assert credit.MAX_REPLAY_STEPS == 1000
assert runner.OBSERVATION_DIM == 88
assert runner.SELECTION_SIZE == 24
assert len(runner.CONTINUATION_SEEDS) == 4
metadata = runner._protocol_metadata({
    "observation_schema_version": "phaseA_low_obs_v2_1_retry_state",
    "action_schema_version": "low_action_schema_v4_2_hierarchical",
})
assert metadata["observation_dimension"] == 88
assert metadata["selection_size"] == 24
assert metadata["continuation_seeds"] == [9303101, 9303102, 9303103, 9303104]

observer = runner._light_state_observer(SimpleNamespace(
    active={},
    lz_manager=SimpleNamespace(zones=[]),
    arrivals=None,
    admission=None,
    service_uavs={},
    telemetry={},
    now=0.0,
    step_count=0,
))
json.dumps(observer, allow_nan=False)

print("ARCHIVE_SOURCE_PREFLIGHT_SCHEMA_CAPTURE_OBSERVER_OK")
print(f"runner_file={runner_path}")
print(f"credit_file={credit_path}")
print(f"state_observer_default={capture_signature.parameters['state_observer'].default!r}")
print(f"MAX_REPLAY_STEPS={credit.MAX_REPLAY_STEPS}")
print(f"observation_dimension={runner.OBSERVATION_DIM}")
print(f"selection_size={runner.SELECTION_SIZE}")
print(f"continuation_seed_count={len(runner.CONTINUATION_SEEDS)}")
print("episodegrad_present=False")
print("observer_json_serializable=True")
