"""Observable partial-plan features, separated from trace labels and outcomes."""
from math import isfinite
from .learning_features import FEATURE_NAMES as BASE_FEATURE_NAMES, extract_features
from .baselines import objective

SCHEMA = "fixed-terminal-stage7-partial-features-v1"
SEARCH_FEATURE_NAMES = ("search_depth", "search_depth_fraction", "search_catalogue_holder_count",
    "search_remaining_catalogue_holders", "search_current_candidate_count", "search_remaining_candidate_count",
    "search_partial_completed", "search_partial_negative_delay", "search_partial_negative_energy",
    "search_configured_width", "search_remaining_depth_budget", "search_frontier_count",
    "search_selected_fraction", "search_current_holder_soc", "search_remaining_holder_soc_min",
    "search_remaining_holder_soc_mean")
FEATURE_NAMES = BASE_FEATURE_NAMES + SEARCH_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 109


def extract_search_features(env, plans, catalogue, depth, width, frontier_count):
    """Inputs available before expanding a parent; no identities become features.

    Remaining budget denotes known depth*width slots, not an ex-post operation
    count. The frozen catalogue contains only currently observed holders.
    """
    if type(depth) is not int or not 0 <= depth <= len(catalogue):
        raise ValueError("invalid search depth")
    if type(width) is not int or width < 1 or type(frontier_count) is not int or frontier_count < 1:
        raise ValueError("invalid known search budget")
    remaining = list(catalogue)[depth:]
    socs = [env.aircraft[key].soc for key in remaining]
    obj = objective(env, plans)
    extra = [depth, depth/len(catalogue) if catalogue else 0., len(catalogue), len(remaining),
             len(catalogue[remaining[0]]) if remaining else 0, sum(len(catalogue[k]) for k in remaining),
             *obj, width, len(remaining)*width, frontier_count,
             len(plans)/len(catalogue) if catalogue else 0., socs[0] if socs else 0.,
             min(socs, default=0.), sum(socs)/len(socs) if socs else 0.]
    values = extract_features(env, plans) + [float(v) for v in extra]
    if len(values) != FEATURE_DIM or not all(isfinite(v) for v in values):
        raise ValueError("invalid partial-plan feature row")
    return values
