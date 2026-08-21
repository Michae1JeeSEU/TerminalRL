from terminal_marl.training.a618_canonical_replay_audit import (
    HISTORICAL_WAITING,
    PPO_AUTHORIZATION,
    _classify,
)


def _row(entry, value):
    return {"entry": entry, "waiting_area": value}


def test_a618_classifies_reproducible_serial_only_difference_as_event_order_drift():
    rows = [_row(entry, HISTORICAL_WAITING) for entry in ("a614", "a615_instrumented", "a617_direct") for _ in range(3)]
    rows += [_row("a617_serial", HISTORICAL_WAITING - 10.0) for _ in range(3)]
    assert _classify(rows, 0.0) == ["EVENT_ORDER_DRIFT"]


def test_a618_classifies_within_entry_nondeterminism_as_rng_stream_drift():
    rows = [_row("a614", HISTORICAL_WAITING), _row("a614", HISTORICAL_WAITING + 1.0)]
    assert _classify(rows, 0.0) == ["RNG_STREAM_DRIFT"]


def test_a618_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
