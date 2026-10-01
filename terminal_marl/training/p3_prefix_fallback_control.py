"""Opt-in, instance-scoped prefix-preserving RELEASE fallback control.

This module only adapts the private shortlist seam on one existing
``TerminalParallelEnv`` instance.  The public environment API still receives
the native K=3 shortlist.  When the native K=3 candidates are all rejected,
the wrapped private method sees the caller's pre-registered 25-proposal order
  in one call.  The first three entries are required to carry the same
  catalogue IDs and canonical values as the public request.

The controller does not sample, inspect or change a policy, model, RNG,
reservation, retry rule, or safety decision.  It restores the instance
method even when validation or the wrapped method raises.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import inspect
import types
from typing import Any, Iterator

import numpy as np

from terminal_marl.core.low_level_joint_action_v4_2 import (
    HierarchicalActionV42,
    RELEASE_GATE,
    hierarchical_from_normalized_v4_2,
)
from terminal_marl.core.merge_scheduler import LowLevelIntent


POLICY_REQUESTED_K = 3
CONTROLLER_MAX_PROPOSALS = 25


class PrefixFallbackContractError(ValueError):
    """Raised when the caller's K=3/full-order contract is not satisfied."""


@dataclass(frozen=True)
class ProposalOrderMetadata:
    """Source-forward metadata needed to validate one UID's order.

    ``source_logits`` is the raw 25-way proposal output from the same native
    forward pass that produced ``sampled_proposal``.  Stable descending order
    is reconstructed by proposal index, so equal logits retain lower-index
    order without introducing a second ranking rule.
    """

    sampled_proposal: int
    source_logits: tuple[float, ...]
    requested_k: int = POLICY_REQUESTED_K
    source_forward_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalOrderMetadata":
        if not isinstance(value, Mapping):
            raise PrefixFallbackContractError("source metadata must be a mapping")
        if "sampled_proposal" not in value:
            raise PrefixFallbackContractError("source metadata lacks sampled_proposal")
        sampled = _proposal_id(value["sampled_proposal"], "sampled_proposal")
        requested_k = value.get("requested_k", POLICY_REQUESTED_K)
        if isinstance(requested_k, bool) or int(requested_k) != POLICY_REQUESTED_K:
            raise PrefixFallbackContractError("prefix fallback requires requested_k=3")
        raw_logits = value.get("source_logits")
        if raw_logits is None:
            raise PrefixFallbackContractError("source metadata lacks source_logits")
        try:
            logits = tuple(float(item) for item in raw_logits)
        except (TypeError, ValueError) as exc:
            raise PrefixFallbackContractError("source_logits must be numeric") from exc
        if len(logits) != CONTROLLER_MAX_PROPOSALS or not all(math.isfinite(item) for item in logits):
            raise PrefixFallbackContractError("source_logits must contain 25 finite values")
        return cls(
            sampled_proposal=sampled,
            source_logits=logits,
            requested_k=POLICY_REQUESTED_K,
            source_forward_id=(None if value.get("source_forward_id") is None else str(value["source_forward_id"])),
        )

    def source_order(self) -> tuple[int, ...]:
        ranked = sorted(
            range(CONTROLLER_MAX_PROPOSALS),
            key=lambda proposal: (-self.source_logits[proposal], proposal),
        )
        return (self.sampled_proposal,) + tuple(
            proposal for proposal in ranked if proposal != self.sampled_proposal
        )


@dataclass
class PrefixFallbackRecord:
    """Mechanical per-UID bookkeeping; no execution result is synthesized."""

    uid: int
    policy_requested_k: int = POLICY_REQUESTED_K
    controller_max: int = CONTROLLER_MAX_PROPOSALS
    calls: int = 0
    extended: bool = False
    prefix_proposals: tuple[int, ...] = ()
    full_proposals: tuple[int, ...] = ()
    runtime_shortlist_candidates_checked: int | None = None
    runtime_shortlist_rank: int | None = None
    decision: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "policy_requested_k": self.policy_requested_k,
            "controller_max": self.controller_max,
            "calls": self.calls,
            "extended": self.extended,
            "prefix_proposals": list(self.prefix_proposals),
            "full_proposals": list(self.full_proposals),
            "runtime_shortlist_candidates_checked": self.runtime_shortlist_candidates_checked,
            "runtime_shortlist_rank": self.runtime_shortlist_rank,
            "decision": self.decision,
            "error": self.error,
        }


def _proposal_id(value: Any, name: str = "proposal") -> int:
    if isinstance(value, bool):
        raise PrefixFallbackContractError(f"{name} must be an integer proposal id")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise PrefixFallbackContractError(f"{name} must be an integer proposal id") from exc
    if numeric != value or not 0 <= numeric < CONTROLLER_MAX_PROPOSALS:
        raise PrefixFallbackContractError(f"{name} must be in [0, 25)")
    return numeric


def _decode_intent_proposal(intent: LowLevelIntent) -> int:
    if not isinstance(intent, LowLevelIntent):
        raise PrefixFallbackContractError("shortlist entries must be LowLevelIntent values")
    try:
        action = hierarchical_from_normalized_v4_2(
            np.asarray(
                [intent.release_request, intent.speed_residual, intent.merge_time_residual],
                dtype=np.float32,
            )
        )
    except (TypeError, ValueError) as exc:
        raise PrefixFallbackContractError("shortlist entry is not a V4.2 catalogue action") from exc
    if action.gate != RELEASE_GATE or action.proposal is None:
        raise PrefixFallbackContractError("prefix fallback only accepts RELEASE shortlist entries")
    return int(action.proposal)


def _intent_for_proposal(proposal: int) -> LowLevelIntent:
    action = HierarchicalActionV42(RELEASE_GATE, proposal)
    return LowLevelIntent.from_array(action.normalized)


def _normalize_registration(
    uid: int,
    raw_order: Sequence[Any],
    raw_metadata: Mapping[str, Any],
) -> tuple[tuple[int, ...], ProposalOrderMetadata]:
    """Validate one current UID registration without changing its source data."""

    try:
        values = tuple(_proposal_id(item, "proposal order entry") for item in raw_order)
    except TypeError as exc:
        raise PrefixFallbackContractError("proposal order must be a sequence") from exc
    if len(values) != CONTROLLER_MAX_PROPOSALS or set(values) != set(range(CONTROLLER_MAX_PROPOSALS)):
        raise PrefixFallbackContractError("each proposal order must be a unique complete 25-way catalogue")
    metadata = ProposalOrderMetadata.from_mapping(raw_metadata)
    expected = metadata.source_order()
    if values != expected:
        raise PrefixFallbackContractError(
            f"uid={uid} proposal order does not match sampled-first stable source-logit order"
        )
    return values, metadata


def _normalize_static_registrations(
    orders: Mapping[int, Sequence[Any]],
    source_metadata: Mapping[int, Mapping[str, Any]],
) -> None:
    """Validate mapping inputs once while retaining the mappings for live updates."""

    if not isinstance(orders, Mapping) or not orders:
        raise PrefixFallbackContractError("proposal orders must be a non-empty UID mapping or provider")
    if not isinstance(source_metadata, Mapping):
        raise PrefixFallbackContractError("source_metadata must be a UID mapping or provider")
    for raw_uid, raw_order in orders.items():
        if isinstance(raw_uid, bool):
            raise PrefixFallbackContractError("UID keys must be integers")
        uid = int(raw_uid)
        if uid not in source_metadata:
            raise PrefixFallbackContractError(f"source metadata missing for uid={uid}")
        _normalize_registration(uid, raw_order, source_metadata[uid])


def _provider_value(provider: Callable[..., Any], env: Any, uav: Any, uid: int) -> Any:
    """Call a live provider once using its declared positional arity.

    A provider may be ``(uav)``, ``(env, uav)``, or ``(env, uav, uid)``.  The
    signature is inspected before calling so a callback that mutates state is
    never retried after a ``TypeError``.
    """

    try:
        signature = inspect.signature(provider)
        parameters = tuple(signature.parameters.values())
    except (TypeError, ValueError) as exc:
        raise PrefixFallbackContractError("proposal provider must have an inspectable signature") from exc
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    if has_varargs or len(positional) >= 3:
        return provider(env, uav, uid)
    if len(positional) == 2:
        return provider(env, uav)
    if len(positional) == 1:
        return provider(uav)
    raise PrefixFallbackContractError("proposal provider must accept uav, env/uav, or env/uav/uid")


class PrefixFallbackController:
    """Wrap one env instance's private K3 shortlist method."""

    policy_requested_k = POLICY_REQUESTED_K
    controller_max = CONTROLLER_MAX_PROPOSALS

    def __init__(
        self,
        env: Any,
        proposal_orders: Mapping[int, Sequence[Any]] | Callable[..., Sequence[Any]],
        *,
        source_metadata: Mapping[int, Mapping[str, Any]] | Callable[..., Mapping[str, Any]],
    ) -> None:
        original = getattr(env, "_runtime_shortlist_release", None)
        if not callable(original):
            raise PrefixFallbackContractError("environment lacks _runtime_shortlist_release")
        if not isinstance(proposal_orders, Mapping) and not callable(proposal_orders):
            raise PrefixFallbackContractError("proposal_orders must be a UID mapping or provider")
        if not isinstance(source_metadata, Mapping) and not callable(source_metadata):
            raise PrefixFallbackContractError("source_metadata must be a UID mapping or provider")
        if isinstance(proposal_orders, Mapping) and isinstance(source_metadata, Mapping):
            _normalize_static_registrations(proposal_orders, source_metadata)
        self.env = env
        self._original = original
        self._orders_source = proposal_orders
        self._metadata_source = source_metadata
        self.records: dict[int, PrefixFallbackRecord] = {}

    def _record(self, uid: int) -> PrefixFallbackRecord:
        record = self.records.get(uid)
        if record is None:
            record = PrefixFallbackRecord(uid=uid)
            self.records[uid] = record
        return record

    def _expanded_shortlist(
        self,
        uid: int,
        uav: Any,
        intent: LowLevelIntent,
        shortlist: tuple[LowLevelIntent, ...],
    ) -> tuple[tuple[LowLevelIntent, ...], tuple[int, ...]]:
        if len(shortlist) != POLICY_REQUESTED_K:
            raise PrefixFallbackContractError("public fallback entry must contain exactly the native K=3 prefix")
        prefix = tuple(_decode_intent_proposal(item) for item in shortlist)
        if len(set(prefix)) != POLICY_REQUESTED_K:
            raise PrefixFallbackContractError("native K=3 prefix contains duplicate proposals")
        if isinstance(self._orders_source, Mapping):
            raw_order = self._orders_source.get(uid)
        else:
            raw_order = _provider_value(self._orders_source, self.env, uav, uid)
        if isinstance(self._metadata_source, Mapping):
            raw_metadata = self._metadata_source.get(uid)
        else:
            raw_metadata = _provider_value(self._metadata_source, self.env, uav, uid)
        if raw_order is None or raw_metadata is None:
            raise PrefixFallbackContractError(f"no registered 25-way order for uid={uid}")
        full_order, metadata = _normalize_registration(uid, raw_order, raw_metadata)
        if prefix[0] != metadata.sampled_proposal or prefix[0] != full_order[0]:
            raise PrefixFallbackContractError("native requested proposal does not match sampled-first metadata")
        if prefix != full_order[:POLICY_REQUESTED_K]:
            raise PrefixFallbackContractError("native K=3 prefix differs from registered full order")
        requested_proposal = _decode_intent_proposal(intent)
        if requested_proposal != prefix[0]:
            raise PrefixFallbackContractError("requested intent does not match native first proposal")
        return tuple(_intent_for_proposal(proposal) for proposal in full_order), full_order

    def _invoke(
        self,
        uav: Any,
        intent: LowLevelIntent,
        shortlist: tuple[LowLevelIntent, ...],
    ) -> str:
        uid = int(uav.uid)
        record = self._record(uid)
        if float(intent.release_request) <= float(getattr(self.env.config, "release_threshold", 0.0)):
            # WAIT is outside the RELEASE prefix scope and is passed through
            # with the exact original tuple.
            return self._call_original(record, uav, intent, shortlist, extended=False)
        expanded, full_order = self._expanded_shortlist(uid, uav, intent, shortlist)
        record.prefix_proposals = tuple(_decode_intent_proposal(item) for item in shortlist)
        record.full_proposals = full_order
        return self._call_original(record, uav, intent, expanded, extended=True)

    def _call_original(
        self,
        record: PrefixFallbackRecord,
        uav: Any,
        intent: LowLevelIntent,
        shortlist: tuple[LowLevelIntent, ...],
        *,
        extended: bool,
    ) -> str:
        record.calls += 1
        record.extended = bool(extended)
        try:
            decision = self._original(uav, intent, shortlist)
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        record.decision = str(decision)
        row = getattr(self.env, "last_release_telemetry", {}).get(int(uav.uid))
        if isinstance(row, Mapping):
            checked = row.get("runtime_shortlist_candidates_checked")
            rank = row.get("runtime_shortlist_rank")
            record.runtime_shortlist_candidates_checked = None if checked is None else int(checked)
            record.runtime_shortlist_rank = None if rank is None else int(rank)
        return str(decision)

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_requested_k": self.policy_requested_k,
            "controller_max": self.controller_max,
            "records": {str(uid): record.as_dict() for uid, record in sorted(self.records.items())},
        }

    def _install(self) -> None:
        controller = self

        def wrapped(_env: Any, uav: Any, intent: LowLevelIntent, shortlist: tuple[LowLevelIntent, ...]) -> str:
            return controller._invoke(uav, intent, shortlist)

        self._had_instance_alias = "_runtime_shortlist_release" in getattr(self.env, "__dict__", {})
        self._previous_instance_alias = getattr(self.env, "__dict__", {}).get("_runtime_shortlist_release")
        self.env._runtime_shortlist_release = types.MethodType(wrapped, self.env)

    def _restore(self) -> None:
        if self._had_instance_alias:
            self.env._runtime_shortlist_release = self._previous_instance_alias
        else:
            self.env.__dict__.pop("_runtime_shortlist_release", None)


@contextmanager
def prefix_fallback_controller(
    env: Any,
    proposal_orders: Mapping[int, Sequence[Any]] | Callable[..., Sequence[Any]],
    *,
    source_metadata: Mapping[int, Mapping[str, Any]] | Callable[..., Mapping[str, Any]],
) -> Iterator[PrefixFallbackController]:
    """Install the controller on ``env`` and always restore its private method."""

    controller = PrefixFallbackController(
        env,
        proposal_orders,
        source_metadata=source_metadata,
    )
    installed = False
    try:
        controller._install()
        installed = True
        yield controller
    finally:
        # ``_install`` records the previous alias before assignment.  If the
        # assignment itself raises, restoring is still safe and avoids leaving
        # a partially-installed instance method behind.
        if installed or hasattr(controller, "_had_instance_alias"):
            controller._restore()


__all__ = [
    "CONTROLLER_MAX_PROPOSALS",
    "POLICY_REQUESTED_K",
    "PrefixFallbackContractError",
    "PrefixFallbackController",
    "prefix_fallback_controller",
]
