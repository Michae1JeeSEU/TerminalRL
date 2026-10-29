"""Opt-in, bounded observations and fixed-catalog counterfactual queries.

These helpers never select or publish a policy action. In particular probes do
not invoke CP, generate columns, sample distances, or share validation caches.
"""
from dataclasses import dataclass, asdict, is_dataclass
import hashlib
import math
import time


def plain(value):
    if isinstance(value,float):
        if math.isnan(value): raise ValueError("NaN is not diagnostic evidence")
        if math.isinf(value): return {"$nonfinite_float":"+inf" if value>0 else "-inf"}
    if is_dataclass(value): return plain(asdict(value))
    if hasattr(value, "items"): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)): return sorted(plain(v) for v in value)
    return value


def candidate_hash(column):
    return hashlib.sha256(repr(column.signature).encode()).hexdigest()


def identity(column):
    return dict(aircraft_id=column.aircraft_id, kind=column.kind,
                signature=plain(column.signature), candidate_hash=candidate_hash(column),
                trajectory_hashes=[p.trajectory_hash for p in column.plans])


def result_row(result):
    return dict(accepted=result.accepted, status=result.status, reason=result.reason,
                details=plain(getattr(result, "details", {})),
                refinement_required=bool(getattr(result, "refinement_required", False)))


@dataclass(frozen=True)
class DiagnosticConfig:
    max_candidates: int = 1024
    max_queries: int = 256
    max_trajectory_candidates: int = 64
    max_models: int = 8
    max_constraints: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("diagnostic limits must be nonnegative integers")


class CandidateDiagnostics:
    def __init__(self, config, snapshot=None):
        self.config = config
        self.snapshot = snapshot
        self.candidates = {}; self.trajectories = {}; self.queries = []; self.models = []
        self.dropped = {}; self.wall_s = 0.

    def _drop(self, key, count=1): self.dropped[key] = self.dropped.get(key, 0) + count

    def candidate(self, column, *, trajectory=False):
        key = candidate_hash(column)
        fixed=tuple(p for p in (self.snapshot.active.get(column.aircraft_id),self.snapshot.soft.get(column.aircraft_id))
                    if p is not None) if self.snapshot is not None else ()
        aircraft=self.snapshot.aircraft[column.aircraft_id] if self.snapshot is not None else None
        if key not in self.candidates:
            if len(self.candidates) >= self.config.max_candidates: self._drop("candidates")
            else:
                self.candidates[key] = dict(identity(column), objective=plain(column.objective),
                    node=plain(column.node), first_start=column.plans[0].start_time if column.plans else None,
                    adjustable_node_time=(fixed[-1].end_time if fixed else self.snapshot.time) if self.snapshot is not None else None,
                    snapshot_ring=getattr(aircraft,"ring",None),snapshot_slot=getattr(aircraft,"slot",None),
                    committed_trajectory_hashes=[p.trajectory_hash for p in fixed],
                    first_target=plain(column.first_target), resources=plain(column.resources),
                    downstream=plain(column.downstream),
                    plans=[dict(trajectory_hash=p.trajectory_hash,kind=p.kind,start_time=p.start_time,
                        end_time=p.end_time,source_ring=p.source_ring,source_slot=p.source_slot,
                        target_ring=p.target_ring,target_slot=p.target_slot,pad_id=p.pad_id) for p in column.plans])
        if trajectory and key not in self.trajectories and (column.plans or fixed):
            if len(self.trajectories) >= self.config.max_trajectory_candidates: self._drop("trajectory_candidates")
            else: self.trajectories[key] = dict(candidate=[plain(p.to_dict()) for p in column.plans],
                                              committed=[plain(p.to_dict()) for p in fixed])
        return key

    def query(self, stage, columns, result):
        started = time.perf_counter()
        keys = [self.candidate(c, trajectory=True) for c in columns]
        if len(self.queries) >= self.config.max_queries: self._drop("queries")
        else: self.queries.append(dict(stage=stage,candidates=keys,**result_row(result)))
        self.wall_s += time.perf_counter()-started

    def model(self, columns, exclusions, singles, nogoods, refinement):
        started = time.perf_counter()
        if len(self.models) >= self.config.max_models:
            self._drop("models"); self.wall_s += time.perf_counter()-started; return
        variables = {}; lookup={}; exactly_one = {}
        for aid, rows in columns.items():
            exactly_one[aid] = []
            for n, c in enumerate(rows):
                key = self.candidate(c); variables[c.signature] = key
                lookup[c.signature]=c
                exactly_one[aid].append(dict(variable=f"x_{aid}_{n}",candidate_hash=key))
        constraints = []
        for kind, rows in (("PAIR_AT_MOST_ONE",exclusions),("SINGLE_ZERO",((k,) for k in singles)),
                           ("PREFIX_NOGOOD",nogoods)):
            for keys in rows:
                if len(constraints) >= self.config.max_constraints: self._drop("constraints"); continue
                row=dict(kind=kind,candidates=[variables[k] for k in keys])
                if kind=="PAIR_AT_MOST_ONE":
                    a,b=(lookup[k] for k in keys)
                    row["resource_overlaps"]=[dict(resource=plain(x.resource),interval=[max(x.start,y.start),min(x.end,y.end)])
                        for x in a.resources for y in b.resources
                        if x.resource==y.resource and x.start<y.end and y.start<x.end]
                constraints.append(plain(row))
        self.models.append(dict(refinement_round=refinement,exactly_one=exactly_one,constraints=constraints))
        self.wall_s += time.perf_counter()-started

    def selection(self,catalog,selected):
        from ..stage2_planning import _selection_key
        started=time.perf_counter()
        row=dict(selected_candidate_hashes=[candidate_hash(c) for c in selected],
                 selected_objective=list(_selection_key(catalog,selected)))
        if catalog.downstream is not None:
            row["selected_downstream_evaluation"]=plain(catalog.downstream.evaluate(selected)[1])
        self.wall_s+=time.perf_counter()-started
        return row

    def export(self, catalog, selected):
        from ..stage2_planning import _selection_key, _selection_solver_key
        started = time.perf_counter(); chosen={candidate_hash(c) for c in selected}
        catalog_keys={self.candidate(c) for rows in catalog.columns.values() for c in rows}
        candidates=[dict(row,in_catalog=key in catalog_keys,selected=key in chosen)
                    for key,row in self.candidates.items()]
        score=dict(objective=list(_selection_key(catalog,selected)),
                   integer_objective=list(_selection_solver_key(catalog,selected)),
                   downstream=plain(catalog.downstream.evaluate(selected)[1]) if catalog.downstream is not None else {})
        self.wall_s += time.perf_counter()-started
        return dict(config=asdict(self.config),candidates=candidates,trajectory_payloads=dict(self.trajectories),
                    snapshot_id=getattr(self.snapshot,"version",None),snapshot_time=getattr(self.snapshot,"time",None),
                    validation_queries=list(self.queries),cp_models=list(self.models),dropped=dict(self.dropped),
                    selected_evaluation=score,observation_wall_s=self.wall_s,
                    historical_catalog_reconstruction=False,additional_validation_calls=0,additional_cp_calls=0)


def probe_catalog_choices(snapshot, catalog, forced_choices, *, baseline=None, global_frame=None,
                          baseline_mode="selected",seconds=10., config=None, prefix_evaluator=None):
    """Evaluate UID->candidate_hash/signature/column overrides in one catalog.

    Keys are canonical aircraft IDs, not presentation-layer UID numbers.
    Unspecified IDs retain the actual selected baseline; an all-KEEP baseline
    requires explicit baseline_mode="keep". There is
    no search: callers enumerate the exact MOVE/KEEP combinations they need.
    A separate wall cap bounds validation; timeout remains UNKNOWN. The
    returned objective is descriptive even when the combination is infeasible.
    """
    from ..stage2_planning import EvaluationContext, PlanningBudget, _selection_key, _selection_solver_key
    if not math.isfinite(seconds) or seconds <= 0: raise ValueError("finite positive diagnostic wall allowance required")
    if baseline_mode not in ("selected","keep"): raise ValueError("unknown diagnostic baseline")
    if baseline_mode=="selected" and baseline is None: raise ValueError("actual selected baseline required")
    if set(forced_choices)-set(catalog.columns): raise ValueError("probe UID outside catalog")
    base={c.aircraft_id:c.signature for c in baseline or ()} if baseline_mode=="selected" else {}
    if baseline_mode=="selected" and set(base)!=set(catalog.columns): raise ValueError("complete selected baseline required")
    if set(base)-set(catalog.columns): raise ValueError("baseline UID outside catalog")
    selected=[]; baseline_columns=[]
    for aid, rows in catalog.columns.items():
        base_column=(next((c for c in rows if c.signature==base.get(aid)),None) if baseline_mode=="selected" else
                     next((c for c in rows if c.kind=="KEEP"),None))
        if base_column is None: raise ValueError("baseline candidate missing from catalog for "+aid)
        baseline_columns.append(base_column)
        token=forced_choices.get(aid,base.get(aid))
        if token is None: matches=[c for c in rows if c.kind=="KEEP"][:1]
        else:
            signature=getattr(token,"signature",token)
            matches=[c for c in rows if c.signature==signature or candidate_hash(c)==signature]
        if len(matches)!=1: raise ValueError("probe requires one existing catalog candidate for "+aid)
        selected.append(matches[0])
    selected=tuple(selected); started=time.perf_counter()
    budget=PlanningBudget(soft_seconds=seconds,watchdog_seconds=seconds,diagnostics=config or DiagnosticConfig())
    context=EvaluationContext(snapshot,budget=budget,scope=catalog.scope,global_frame=global_frame,
                              prefix_evaluator=prefix_evaluator)
    batch,conflicts=context.check_batch_conflicts(selected)
    prefix=context.check_commit_prefix(selected) if batch.accepted else None
    integer,downstream=(catalog.downstream.evaluate(selected) if catalog.downstream is not None else
                       (_selection_solver_key(catalog,selected),{}))
    before=_selection_key(catalog,tuple(baseline_columns)); after=_selection_key(catalog,selected)
    before_integer,before_downstream=(catalog.downstream.evaluate(tuple(baseline_columns)) if catalog.downstream is not None else
                                     (_selection_solver_key(catalog,tuple(baseline_columns)),{}))
    return dict(diagnostic_only=True,publication_permitted=False,baseline_mode=baseline_mode,
                snapshot_id=snapshot.version,snapshot_time=snapshot.time,
                baseline_objective=list(before),objective_delta=[b-a for a,b in zip(before,after)],
                baseline_integer_objective=list(before_integer),baseline_downstream=plain(before_downstream),
                integer_objective_delta=[b-a for a,b in zip(before_integer,integer)],
                forced_candidate_hashes={aid:candidate_hash(next(c for c in selected if c.aircraft_id==aid)) for aid in forced_choices},
                selected=[identity(c) for c in selected],objective=list(_selection_key(catalog,selected)),
                integer_objective=list(integer),downstream=plain(downstream),batch=result_row(batch),
                conflicts=[[identity(c) for c in pair] for pair in conflicts],
                prefix=None if prefix is None else result_row(prefix),
                diagnostics=context.diagnostics.export(catalog,selected),wall_s=time.perf_counter()-started,
                wall_allowance_s=seconds,cp_calls=0,formal_cp_budget_consumed_s=0.)
