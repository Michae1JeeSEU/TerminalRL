"""Reserved data interface. No learned tail is trained or queried in v1.0."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TailObservation:
    context_id: str
    objective_id: str
    tail_estimator_id: str
    predicted_state: dict
    observed_objective: tuple | None = None


def learned_tail_value(*args, **kwargs):
    raise RuntimeError("TAIL_INFERENCE_DISABLED_BY_FT_RS_S3_INITIAL_1_0")
