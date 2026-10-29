"""Frozen search switches; only the reference variant permits overrides."""
from dataclasses import asdict, dataclass, fields
from collections.abc import Mapping


REFERENCE_VARIANT = "opportunity_reference_v4"
PROGRESSIVE_VARIANTS = ("opportunity_cooperative_v2", "opportunity_executable_v3", REFERENCE_VARIANT)
RULE_VARIANTS = ("baseline", "outer_priority_v1", *PROGRESSIVE_VARIANTS)


@dataclass(frozen=True)
class SearchPolicy:
    pad_anchor_priority: bool = False
    target_coverage_order: bool = False
    ordinary_immediate_filter: bool = False

    def __post_init__(self):
        if any(type(getattr(self, f.name)) is not bool for f in fields(self)):
            raise ValueError("search policy switches must be bool")

    def to_dict(self):
        return asdict(self)


def resolve_search_policy(variant, policy=None):
    if variant not in RULE_VARIANTS:
        raise ValueError("unknown selective rule variant")
    default = SearchPolicy(True, True, True) if variant == "opportunity_executable_v3" else SearchPolicy()
    if policy is None:
        return default
    if isinstance(policy, SearchPolicy):
        resolved = policy
    elif isinstance(policy, Mapping):
        names = {f.name for f in fields(SearchPolicy)}
        if set(policy) - names:
            raise ValueError("unknown search policy switch")
        resolved = SearchPolicy(**dict(default.to_dict(), **policy))
    else:
        raise ValueError("search policy must be SearchPolicy or mapping")
    if variant != REFERENCE_VARIANT and resolved != default:
        raise ValueError("search policy overrides require opportunity_reference_v4")
    return resolved
