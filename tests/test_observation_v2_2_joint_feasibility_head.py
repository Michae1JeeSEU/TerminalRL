import numpy as np

from terminal_marl.training import audit_v4_2_observation_v2_2_joint_feasibility_head_v1 as audit


def _row(value: float, feasible: tuple[int, ...], interface: str):
    return {
        "observation_v2_2": [value] * 94,
        "proposal_set_target": list(feasible),
        "interface": interface,
    }


def test_joint_feasibility_protocol_and_head_shape_are_frozen():
    value = audit.protocol()
    assert value["architecture"]["output"] == "25_independent_proposal_feasibility_logits"
    assert value["architecture"]["runtime_catalogue_search"] is False
    rows = [_row(0.0, (0, 1), "o2m"), _row(0.1, (2,), "m2i")]
    observations, targets = audit.tensors(rows)
    assert observations.shape == (2, 94)
    assert targets.shape == (2, 25)
    assert np.array_equal(targets.sum(dim=1).numpy(), np.asarray([2.0, 1.0]))
    assert audit.JointFeasibilityHead()(observations).shape == (2, 25)


def test_joint_feasibility_probe_uses_fixed_budget(monkeypatch):
    train = [
        _row(0.0, (0, 1), "o2m"), _row(0.1, (2,), "m2i"),
        _row(0.2, (3, 4), "i2f"), _row(0.3, (5,), "o2m"),
    ]
    dev = [
        _row(0.05, (0,), "o2m"), _row(0.15, (2, 3), "m2i"),
        _row(0.25, (4,), "i2f"),
    ]
    monkeypatch.setattr(audit, "EPOCHS", 1)
    monkeypatch.setattr(audit, "BATCH_SIZE", 2)
    result = audit.run_probe(train, dev, seed=801101)
    assert result["optimizer_steps"] == 2
    assert result["state_dev"]["states"] == 3
    assert result["binary_dev"]["rows"] == 75
    assert result["checkpoint_saved"] is False
