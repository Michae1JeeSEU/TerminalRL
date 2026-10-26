"""Observed catalog diversity heuristics; these are not dominance certificates."""
import math


def _bucket(value):
    return ("finite",math.floor(value / 5.)) if math.isfinite(value) else ("infinite",str(value))


def immediate_start(start, now, effective_time=None):
    return (abs(start-effective_time)<=1.e-8 if effective_time is not None
            else start<=now+5.+1.e-8)


def delayed_start(start, now, effective_time=None):
    return start>(effective_time if effective_time is not None else now+5.)+1.e-8


def coverage_key(column, now, *, effective_time=None):
    plans=column.plans
    return (bool(plans and plans[-1].pad_id is not None),
        bool(plans and immediate_start(plans[0].start_time,now,effective_time)), column.first_target,
        tuple(sorted({p.pad_id for p in plans if p.pad_id is not None})),
        tuple(sorted((repr(r.resource),_bucket(r.start),_bucket(r.end)) for r in column.resources)))


def retain_validated(columns, column, now, target, screening, *, effective_time=None):
    """Only discard a new validated alternative with already observed coverage.

    The target is soft: a new direction, pad or resource window survives it.
    Existing representatives remain present for earlier certified incumbents.
    """
    if any(c.signature==column.signature for c in columns): return False
    moves=[c for c in columns if c.kind=="MOVE"]
    same=[c for c in moves if coverage_key(c,now,effective_time=effective_time)==coverage_key(column,now,effective_time=effective_time)]
    known=lambda c: bool(c.plans and c.plans[-1].pad_id is not None) or getattr(c.reference,"status",None)=="KNOWN"
    if any(known(c) and known(column) and c.objective<=column.objective for c in same):
        screening["catalog_similar_pruned"]+=1
        return False
    if len(moves)>=target and same:
        # Unknown objective estimates cannot establish even this local heuristic.
        screening["catalog_soft_limit_preserved_unknown"]+=1
    columns.append(column)
    return True


def coverage_report(snapshot, columns, *, effective_time=None):
    report={}
    for aid,values in columns.items():
        moves=[c for c in values if c.kind=="MOVE" and c.plans]
        eligible=aid not in snapshot.soft and any(c.kind=="KEEP" and c.node is not None and immediate_start(c.node.time,snapshot.time,effective_time) for c in values)
        missing=[]
        if not any(c.plans[-1].pad_id is not None for c in moves): missing.append("NO_COMPLETE_CHAIN_OBSERVED")
        if eligible and not any(immediate_start(c.plans[0].start_time,snapshot.time,effective_time) for c in moves): missing.append("NO_ELIGIBLE_IMMEDIATE_MOVE_OBSERVED")
        report[aid]=dict(full_chain=any(c.plans[-1].pad_id is not None for c in moves),
            immediate=any(immediate_start(c.plans[0].start_time,snapshot.time,effective_time) for c in moves),
            delayed=any(delayed_start(c.plans[0].start_time,snapshot.time,effective_time) for c in moves),
            first_directions=sorted({repr(c.first_target) for c in moves}),
            pads=sorted({p.pad_id for c in moves for p in c.plans if p.pad_id is not None}),
            resource_windows=sorted({repr(coverage_key(c,snapshot.time,effective_time=effective_time)[-1]) for c in moves}),
            missing_reasons=missing,coverage_claim="observed_bounded_catalog_only")
    return report


def expansion_reasons(snapshot, catalog, result, coverage, context):
    reasons=sorted({reason for row in coverage.values() for reason in row["missing_reasons"]})
    if not result.certified: reasons.append("NO_CERTIFIED_INCUMBENT")
    elif not result.prefix: reasons.append("CERTIFIED_WITHOUT_EXECUTABLE_PREFIX")
    if context.observed_definite_conflicts:
        reasons.append("OBSERVED_DEFINITE_CONFLICT")
    return sorted(set(reasons))
