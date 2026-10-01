from __future__ import annotations

import random

import numpy as np
import pytest

from terminal_marl.core.low_level_joint_action_v4_2 import HierarchicalActionV42, RELEASE_GATE
from terminal_marl.core.merge_scheduler import LowLevelIntent
from terminal_marl.training.p3_prefix_fallback_control import (
    PrefixFallbackContractError,
    prefix_fallback_controller,
)


UID = 7


def _intent(proposal: int) -> LowLevelIntent:
    return LowLevelIntent.from_array(HierarchicalActionV42(RELEASE_GATE, proposal).normalized)


def _source(sampled: int = 7) -> tuple[tuple[int, ...], dict[str, object]]:
    logits = tuple(float(100 - index) for index in range(25))
    order = (sampled,) + tuple(index for index in range(25) if index != sampled)
    return order, {"sampled_proposal": sampled, "source_logits": logits}


class _Uav:
    def __init__(self, uid: int = UID) -> None:
        self.uid = uid


class _Config:
    release_threshold = 0.0


class _FakeEnv:
    config = _Config()

    def __init__(self, accepted: set[int] | None = None, *, raises: bool = False) -> None:
        self.accepted = set() if accepted is None else set(accepted)
        self.raises = raises
        self.calls: list[tuple[int, ...]] = []
        self.last_release_telemetry: dict[int, dict[str, int]] = {}

    def _runtime_shortlist_release(
        self,
        uav: _Uav,
        intent: LowLevelIntent,
        shortlist: tuple[LowLevelIntent, ...],
    ) -> str:
        if self.raises:
            raise RuntimeError("original runtime failure")
        proposal_ids = tuple(_decode(item) for item in shortlist)
        self.calls.append(proposal_ids)
        checked = 0
        selected = None
        for checked, proposal in enumerate(proposal_ids, start=1):
            if proposal in self.accepted:
                selected = proposal
                break
        self.last_release_telemetry[uav.uid] = {
            "runtime_shortlist_candidates_checked": checked,
            "runtime_shortlist_rank": None if selected is None else checked,
        }
        return "approved" if selected is not None else "safety_reject"


def _decode(intent: LowLevelIntent) -> int:
    values = np.asarray(
        [intent.release_request, intent.speed_residual, intent.merge_time_residual],
        dtype=np.float32,
    )
    from terminal_marl.core.low_level_joint_action_v4_2 import hierarchical_from_normalized_v4_2

    action = hierarchical_from_normalized_v4_2(values)
    assert action.proposal is not None
    return int(action.proposal)


def _registration() -> tuple[tuple[int, ...], dict[int, dict[str, object]]]:
    order, metadata = _source()
    return order, {UID: metadata}


def test_prefix_is_preserved_and_original_method_stops_without_rerun() -> None:
    order, metadata = _registration()
    uav = _Uav()

    baseline = _FakeEnv(accepted={order[1]})
    baseline_decision = baseline._runtime_shortlist_release(uav, _intent(order[0]), tuple(_intent(item) for item in order[:3]))

    env = _FakeEnv(accepted={order[1]})
    with prefix_fallback_controller(
        env,
        {UID: order},
        source_metadata=metadata,
    ) as controller:
        decision = env._runtime_shortlist_release(
            uav,
            _intent(order[0]),
            tuple(_intent(item) for item in order[:3]),
        )

    assert decision == baseline_decision == "approved"
    assert env.calls == [order]
    assert baseline.calls == [order[:3]]
    assert env.calls[0][:3] == baseline.calls[0]
    assert controller.records[UID].calls == 1
    assert controller.records[UID].extended is True
    assert controller.records[UID].prefix_proposals == order[:3]
    assert controller.records[UID].full_proposals == order
    assert controller.records[UID].runtime_shortlist_candidates_checked == 2
    assert controller.records[UID].runtime_shortlist_rank == 2


def test_fallback_passes_one_expanded_call_and_reaches_fourth_candidate() -> None:
    order, metadata = _registration()
    env = _FakeEnv(accepted={order[3]})
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata) as controller:
        result = env._runtime_shortlist_release(
            _Uav(),
            _intent(order[0]),
            tuple(_intent(item) for item in order[:3]),
        )

    assert result == "approved"
    assert env.calls == [order]
    assert controller.records[UID].calls == 1
    assert controller.records[UID].runtime_shortlist_candidates_checked == 4
    assert controller.records[UID].runtime_shortlist_rank == 4


def test_original_capacity_stop_is_preserved_without_retrying_remaining_candidates() -> None:
    order, metadata = _registration()
    # The original runtime method stops at the first capacity/reservation
    # decision.  A later candidate would also be acceptable in isolation, but
    # the wrapper must not invoke the original method again after that stop.
    env = _FakeEnv(accepted={order[0], order[3]})
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata) as controller:
        result = env._runtime_shortlist_release(
            _Uav(),
            _intent(order[0]),
            tuple(_intent(item) for item in order[:3]),
        )

    assert result == "approved"
    assert len(env.calls) == 1
    assert env.calls[0] == order
    assert controller.records[UID].runtime_shortlist_candidates_checked == 1
    assert controller.records[UID].runtime_shortlist_rank == 1


def test_dynamic_provider_is_read_at_release_time_and_wait_is_untouched() -> None:
    order, metadata = _registration()
    calls = {"order": 0, "metadata": 0}

    def order_provider(env: _FakeEnv, uav: _Uav, uid: int) -> tuple[int, ...]:
        calls["order"] += 1
        assert env is current_env and uav.uid == uid == UID
        return order

    def metadata_provider(env: _FakeEnv, uav: _Uav, uid: int) -> dict[str, object]:
        calls["metadata"] += 1
        assert env is current_env and uav.uid == uid == UID
        return metadata[UID]

    current_env = _FakeEnv(accepted={order[3]})
    with prefix_fallback_controller(
        current_env,
        order_provider,
        source_metadata=metadata_provider,
    ):
        assert current_env._runtime_shortlist_release(
            _Uav(),
            LowLevelIntent(0.0, 0.0, 0.0),
            (),
        ) == "safety_reject"
        assert calls == {"order": 0, "metadata": 0}
        assert current_env._runtime_shortlist_release(
            _Uav(),
            _intent(order[0]),
            tuple(_intent(item) for item in order[:3]),
        ) == "approved"

    assert calls == {"order": 1, "metadata": 1}
    assert current_env.calls == [(), order]


def test_unranked_singleton_release_passthrough_skips_order_provider() -> None:
    order, metadata = _registration()
    env = _FakeEnv(accepted={5})
    provider_calls = {"order": 0, "metadata": 0}

    def forbidden_order(*_args: object) -> tuple[int, ...]:
        provider_calls["order"] += 1
        raise AssertionError("singleton passthrough must not request an order")

    def forbidden_metadata(*_args: object) -> dict[str, object]:
        provider_calls["metadata"] += 1
        raise AssertionError("singleton passthrough must not request source metadata")

    with prefix_fallback_controller(
        env,
        forbidden_order,
        source_metadata=forbidden_metadata,
    ) as controller:
        singleton = _intent(5)
        result = env._runtime_shortlist_release(_Uav(28), singleton, (singleton,))

    assert result == "approved"
    assert env.calls == [(5,)]
    assert provider_calls == {"order": 0, "metadata": 0}
    record = controller.records[28]
    assert record.calls == 1
    assert record.extended is False
    assert record.native_singleton_passthrough is True
    assert record.request_source == "native_unranked_singleton_release"
    assert record.off_policy_request is False
    assert record.prefix_proposals == (5,)
    assert record.full_proposals == ()

    strict_env = _FakeEnv(accepted={5})
    with prefix_fallback_controller(strict_env, {UID: order}, source_metadata=metadata):
        with pytest.raises(PrefixFallbackContractError, match="either native singleton"):
            strict_env._runtime_shortlist_release(
                _Uav(28),
                _intent(5),
                (_intent(5), _intent(6)),
            )
    assert strict_env.calls == []


def test_mutable_mapping_is_validated_again_at_each_release() -> None:
    order, metadata = _registration()
    live_order = {UID: order}
    live_metadata = {UID: metadata[UID]}
    env = _FakeEnv(accepted={order[3]})
    with prefix_fallback_controller(env, live_order, source_metadata=live_metadata):
        live_order[UID] = tuple(order)
        env._runtime_shortlist_release(_Uav(), _intent(order[0]), tuple(_intent(item) for item in order[:3]))
        live_order[UID] = tuple(list(order[:24]) + [order[0]])
        with pytest.raises(PrefixFallbackContractError, match="unique complete"):
            env._runtime_shortlist_release(_Uav(), _intent(order[0]), tuple(_intent(item) for item in order[:3]))


def test_invalid_registration_and_prefix_fail_closed_before_original_call() -> None:
    order, metadata = _registration()
    duplicate = tuple(list(order[:24]) + [order[0]])
    with pytest.raises(PrefixFallbackContractError, match="unique complete"):
        with prefix_fallback_controller(_FakeEnv(), {UID: duplicate}, source_metadata=metadata):
            pass

    env = _FakeEnv(accepted={order[3]})
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata):
        bad_prefix = (order[0], order[2], order[1])
        with pytest.raises(PrefixFallbackContractError, match="prefix differs"):
            env._runtime_shortlist_release(
                _Uav(),
                _intent(order[0]),
                tuple(_intent(item) for item in bad_prefix),
            )
    assert env.calls == []


def test_wait_does_not_call_provider_and_original_exception_restores_alias() -> None:
    order, metadata = _registration()
    provider_calls = 0

    def provider(_uav: _Uav) -> tuple[int, ...]:
        nonlocal provider_calls
        provider_calls += 1
        return order

    env = _FakeEnv(raises=True)
    original = env._runtime_shortlist_release
    with prefix_fallback_controller(env, provider, source_metadata=lambda _uav: metadata[UID]):
        with pytest.raises(RuntimeError, match="original runtime failure"):
            env._runtime_shortlist_release(_Uav(), LowLevelIntent(0.0, 0.0, 0.0), ())
        with pytest.raises(RuntimeError, match="original runtime failure"):
            env._runtime_shortlist_release(_Uav(), _intent(order[0]), tuple(_intent(item) for item in order[:3]))
    assert provider_calls == 1
    restored = env._runtime_shortlist_release
    assert restored.__self__ is env
    assert restored.__func__ is original.__func__


def test_existing_instance_alias_is_restored_after_success_and_exception() -> None:
    order, metadata = _registration()
    env = _FakeEnv()

    def instance_alias(uav: _Uav, intent: LowLevelIntent, shortlist: tuple[LowLevelIntent, ...]) -> str:
        return "instance-original"

    env._runtime_shortlist_release = instance_alias
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata):
        assert env._runtime_shortlist_release(
            _Uav(),
            _intent(order[0]),
            tuple(_intent(item) for item in order[:3]),
        ) == "instance-original"
    assert env.__dict__["_runtime_shortlist_release"] is instance_alias

    def raising_alias(uav: _Uav, intent: LowLevelIntent, shortlist: tuple[LowLevelIntent, ...]) -> str:
        raise RuntimeError("instance alias failure")

    env._runtime_shortlist_release = raising_alias
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata):
        with pytest.raises(RuntimeError, match="instance alias failure"):
            env._runtime_shortlist_release(
                _Uav(),
                _intent(order[0]),
                tuple(_intent(item) for item in order[:3]),
            )
    assert env.__dict__["_runtime_shortlist_release"] is raising_alias


def test_controller_does_not_change_python_or_numpy_rng() -> None:
    order, metadata = _registration()
    env = _FakeEnv(accepted={order[3]})
    py_state = random.getstate()
    np_state = np.random.get_state()
    with prefix_fallback_controller(env, {UID: order}, source_metadata=metadata):
        env._runtime_shortlist_release(_Uav(), _intent(order[0]), tuple(_intent(item) for item in order[:3]))
    assert random.getstate() == py_state
    after_np = np.random.get_state()
    assert after_np[0] == np_state[0]
    assert np.array_equal(after_np[1], np_state[1])
    assert after_np[2:] == np_state[2:]
