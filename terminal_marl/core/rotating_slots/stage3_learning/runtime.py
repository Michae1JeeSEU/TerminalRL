"""Bounded advisor hooks over Stage2's unmaterialized ready work.

The caller retains its generator, physics, candidate limits and transaction path.
Fallback uses the same budget object, including its original clock.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import math
import time

from .adapters import node_identity, prefix_identity, snapshot_identity, stable_id
from .schemas import WorkDescriptor

MODES = ("OFF", "RECORD", "SHADOW_OFFLINE", "GUIDED_ORDER", "GUIDED_LIMITED")


def scale_candidate_budget(budget, scale=1.0):
    """Mutate only the three approved limits; never replace/reset the clock."""
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("invalid candidate budget scale")
    if getattr(budget, "_stage3_scale", None) is not None:
        if budget._stage3_scale != scale:
            raise ValueError("candidate budget already scaled differently")
        return budget
    budget.candidate_seconds *= scale
    budget.max_per_aircraft = max(1, math.floor(budget.max_per_aircraft * scale))
    budget.max_attempts = max(1, math.floor(budget.max_attempts * scale))
    budget._stage3_scale = scale
    return budget


class AdvisorRuntime:
    def __init__(self, mode="OFF", advisor=None, feature_encoder=None, fingerprint=None,
                 observer=None, ancestry_root_id=None, evaluation=False, trace_limit=8192,
                 offline_probe=None, deterministic_only=False, trace_mode='collection', feature_schema=None, rank_reuse_limit=None):
        if mode not in MODES:
            raise ValueError("unknown advisor mode")
        self.mode, self.advisor, self.feature_encoder = mode, advisor, feature_encoder
        self.fingerprint = dict(fingerprint or {})
        self.observer, self.ancestry_root_id = observer, ancestry_root_id
        self.evaluation, self.trace_limit = evaluation, trace_limit
        if offline_probe is not None and mode != "SHADOW_OFFLINE":
            raise ValueError("offline probe requires SHADOW_OFFLINE")
        self.offline_probe=offline_probe
        self.deterministic_only=bool(deterministic_only)
        if trace_mode not in ('collection','online'):
            raise ValueError('unknown trace mode')
        self.trace_mode = trace_mode
        self.feature_schema = feature_schema
        from .features import SCHEMA_V2
        self.rank_reuse_limit = (3 if feature_schema == SCHEMA_V2 else 1) if rank_reuse_limit is None else int(rank_reuse_limit)
        if self.rank_reuse_limit not in (1,2,3):
            raise ValueError('rank reuse limit must be 1..3')
        self.retained_context, self.downstream, self.known_conflicts = {}, {}, {}
        self._coverage_epoch = 0
        self._retained_signature = None
        self._query_refs = {}
        self.traces = []
        self.parents, self.nodes = {}, {}
        self._descriptors, self._plans, self._features = {}, {}, {}
        self._serial = 0
        self._query_serial = 0
        self._post_initial = 0
        self.coverage = Counter()
        self._initial = set()
        self._initial_finished=False
        self._rank_cache = None
        self._descriptor_keys={}
        self._successor_states={}

    @property
    def enabled(self):
        return self.mode != "OFF"

    @property
    def guided(self):
        return self.mode in ("GUIDED_ORDER", "GUIDED_LIMITED")

    def emit(self, row):
        if not self.enabled:
            return
        if self.trace_mode == 'online':
            from .online_observer import lightweight_event
            row = lightweight_event(row)
            if row is None:
                return
        row = deepcopy(dict(row))
        if hasattr(self, "snapshot_time"):
            row["snapshot_id"] = self.snapshot_id
            row["as_of"] = self.snapshot_time
        self.traces.append(row)
        if len(self.traces) > self.trace_limit:
            del self.traces[:len(self.traces)-self.trace_limit]
        if self.observer is not None:
            self.observer(row)

    def begin_snapshot(self, snapshot, budget, aircraft_order):
        self.snapshot_id = snapshot_identity(snapshot)
        self.snapshot_time = float(snapshot.time)
        self.parents, self.nodes = {}, {}
        self._descriptors, self._plans, self._features = {}, {}, {}
        self._initial = set(aircraft_order)
        self._initial_finished=False
        self._post_initial = 0
        self._rank_cache = None
        self._descriptor_keys={}
        self._successor_states={}
        self.coverage = Counter()
        self.retained_context, self.downstream, self.known_conflicts = {}, {}, {}
        self._coverage_epoch = 0
        self._retained_signature = None
        self._query_refs = {}
        self.emit(dict(record_type="SnapshotStart", snapshot_id=self.snapshot_id,
                       time=self.snapshot_time,
                       ancestry_root_id=self.ancestry_root_id or self.snapshot_id,
                       mode=self.mode, deterministic_only=self.deterministic_only,
                       eligible_aircraft=list(aircraft_order), budget=budget.settings()))

    def complete_initial_coverage(self):
        self._initial_finished=True

    def register_prefix(self, snapshot, a, node, plans):
        nid = node_identity(snapshot, a.id, node)
        data = dict(materialized=True, soc=node.soc, ring=node.ring, slot=node.slot,
                    end_time=node.time, depth=len(plans), source_node_id=nid, aircraft_id=a.id,
                    version=node.version,
                    peak=node.peak,peak_error_s=node.peak_error_s,error_j=node.error_j,
                    trajectory_hashes=[p.trajectory_hash for p in plans])
        if not plans:
            self.nodes[nid] = data
            return None
        # Require every actual predecessor to have been generated (or explicitly
        # registered as an existing reused chain) before the child becomes ready.
        last = plans[-1]
        if (last.target_ring, last.target_slot, last.end_time) != (node.ring, node.slot, node.time):
            raise ValueError("parent chain does not terminate at source node")
        if any(p.trajectory_hash not in self._plans for p in plans):
            raise ValueError("parent chain has not been materialized")
        expected=self._successor_states.get(last.trajectory_hash)
        if expected is None or expected != (node.ring,node.slot,node.time,node.soc,node.peak,node.version,node.error_j,node.peak_error_s):
            raise ValueError("parent source state has not been computed")
        pid = prefix_identity(snapshot, a.id, plans)
        self.nodes[nid] = data
        self.parents[pid] = data
        return pid

    def descriptor(self, snapshot, a, node, target, start, duration, stage, plans=(), base=None):
        parent = self.register_prefix(snapshot, a, node, plans)
        key=(a.id,node_identity(snapshot,a.id,node),target,start,duration,stage,parent,base)
        if key in self._descriptor_keys:
            return self._descriptors[self._descriptor_keys[key].descriptor_id]
        self._serial += 1
        d = WorkDescriptor(stable_id("work", key), self.snapshot_id,
                           node_identity(snapshot, a.id, node), a.id, node.ring, node.slot,
                           "normal_pad" if node.ring == "inner" else "next_ring_slot", int(target),
                           max(0., start-snapshot.time), float(duration), parent, stage,
                           "resource_aligned" if base is not None and start != base else
                           "delayed" if start > node.time+1.e-8 else "current", self._serial)
        self._descriptors[d.descriptor_id] = d
        self._descriptor_keys[key]=d
        return d

    def search_state(self, budget):
        return dict(parents=self.parents, nodes=self.nodes, attempts=sum(budget.attempts.values()),
                    materialized=self.coverage["attempted"],
                    remaining_budget_fraction=max(0., min(1., 1.-getattr(budget,'candidate_elapsed',budget.elapsed)/max(1.e-12, budget.candidate_seconds))),
                    coverage=dict(self.coverage), retained_context=self.retained_context,
                    downstream=self.downstream, known_conflicts=self.known_conflicts)

    def update_retained_context(self, snapshot, columns_mapping, references_mapping, context, downstream=None):
        """Read retained columns and already computed facts only; never run a validator."""
        retained, estimates = {}, {}
        for aid, columns in columns_mapping.items():
            moves = [c for c in columns if c.kind == 'MOVE' and c.plans]
            full = [c for c in moves if c.plans[-1].pad_id is not None]
            immediate = [c for c in moves if c.plans[0].start_time <= snapshot.time+5. and aid not in snapshot.soft]
            delayed = [c for c in moves if c.plans[0].start_time > snapshot.time+5.]
            pads = {c.plans[-1].pad_id for c in full}
            windows = {int(c.plans[0].start_time//5) for c in moves}
            directions = Counter(repr((p.target_ring,p.target_slot) if p.pad_id is None else ('pad',p.pad_id))
                                 for c in moves for p in c.plans[:1])
            retained[aid] = dict(full_count=len(full),immediate_count=len(immediate),delayed_count=len(delayed),
                pad_count=len(pads),window_count=len(windows),directions=dict(directions),
                missing_coverage_count=sum((not full,not immediate,not delayed)),_pads=tuple(sorted(pads)),_windows=tuple(sorted(windows)))
        # Epoch changes only for coverage novelty/loss, not duplicate multiplicity.
        signature = tuple(sorted((aid,tuple(bool(v[k]) for k in ('full_count','immediate_count','delayed_count')),
            tuple(sorted(v['directions'],key=repr)),v['_pads'],v['_windows']) for aid,v in retained.items()))
        if signature != getattr(self,'_retained_signature',None):
            self._coverage_epoch += 1
            self._retained_signature = signature
        self.retained_context = retained
        for nid, node in self.nodes.items():
            aid = node.get('aircraft_id')
            reference = references_mapping.get(aid)
            if reference is not None and reference.status == 'KNOWN':
                # Cached references must describe this exact node epoch.
                plans = reference.plans
                if plans and (plans[0].source_ring,plans[0].source_slot,plans[0].source_version) == (node['ring'],node['slot'],node['version']) and plans[0].start_time >= node['end_time']-1.e-8:
                    estimates[nid] = dict(status='KNOWN',energy_j=reference.energy_j,duration_s=reference.duration)
            waits = []
            for c in columns_mapping.get(aid,()):
                diag = getattr(c,'downstream',{})
                values = [diag.get(k) for k in ('window_wait_seconds_by_ring','tail_known_wait_seconds_by_ring')]
                if any(isinstance(v,dict) for v in values):
                    parts = [x for v in values if isinstance(v,dict) for x in v.values()]
                    if all(isinstance(x,(int,float)) and math.isfinite(x) for x in parts):
                        waits.append(sum(parts))
            if waits:
                estimates.setdefault(nid,dict(status='UNKNOWN'))['wait_s'] = min(waits)
        self.downstream = estimates
        observed = getattr(context,'observed_definite_conflicts',None)
        if observed is not None:
            counts = Counter()
            seen = set()
            for fact in observed:
                ids = fact.get('aircraft_ids',()) if isinstance(fact,dict) else ()
                key = repr((sorted(ids),fact.get('resources'),fact.get('reason'))) if isinstance(fact,dict) else repr(fact)
                if key in seen:
                    continue
                seen.add(key)
                counts.update(set(ids))
            conflicts = {aid:counts[aid] for aid in columns_mapping}
            if conflicts != self.known_conflicts:
                self._coverage_epoch += 1
            self.known_conflicts = conflicts

    def record_column_validation(self,column,result,wall_s,cpu_s,cache_hit):
        """Attribute an actual single-column check only to its terminal query."""
        if not column.plans:
            return
        dids = self._plans.get(column.plans[-1].trajectory_hash,())
        if not dids:
            return
        self.emit(dict(record_type='ColumnValidationCost',descriptor_id=dids[-1],
            features_before_query_ref=getattr(self,'_query_refs',{}).get(dids[-1]),
            wall_s=wall_s,cpu_s=cpu_s,cache_hit=bool(cache_hit),
            validation_status=getattr(result,'status',None),accepted=bool(getattr(result,'accepted',False)),
            cost_scope='ACTUAL_TERMINAL_SINGLE_COLUMN_VALIDATION'))

    def _ready_state(self,descriptors):
        def frozen(value):
            if isinstance(value,dict):
                return tuple(sorted((k,frozen(v)) for k,v in value.items()))
            if isinstance(value,(tuple,list)):
                return tuple(frozen(v) for v in value)
            return value
        return tuple((d.descriptor_id,(frozen(self.nodes[d.source_node_id]),frozen(self.downstream.get(d.source_node_id))),
            frozen(self.parents.get(d.parent_prefix_id))) for d in descriptors)

    def register_reuse(self,snapshot,a,node,plan,prefix,q):
        """Record a real exact-source cached trajectory, never a new attempt."""
        d=self.descriptor(snapshot,a,node,plan.pad_id if plan.pad_id is not None else plan.target_slot,
                          plan.start_time,plan.end_time-plan.start_time-(13.5 if plan.pad_id is not None else 0.),
                          "alternative",prefix)
        d=replace(d,proposal_origin="reuse")
        self._descriptors[d.descriptor_id]=d
        self._plans.setdefault(plan.trajectory_hash,[])
        if d.descriptor_id not in self._plans[plan.trajectory_hash]:
            self._plans[plan.trajectory_hash].append(d.descriptor_id)
        self.emit(d.as_dict())
        self.emit(dict(record_type="ReuseEvidence",descriptor_id=d.descriptor_id,
                       trajectory_hash=plan.trajectory_hash,source_node_id=d.source_node_id,
                       parent_prefix_id=d.parent_prefix_id,physical_attempts=0,source="exact_stage2_cache"))
        from .features import SCHEMA_V2
        if self.feature_schema==SCHEMA_V2:
            self.record_energy(node,plan,q,True,attribute_query=False)
        else:
            self.record_energy(node,plan,q,True)

    def _encode(self, snapshot, descriptors, budget):
        if self.feature_encoder is None:
            from .features import FeatureEncoder
            self.feature_encoder = FeatureEncoder(**({'schema_id':self.feature_schema} if self.feature_schema else {}))
        with budget.measure("stage3_features"):
            rows = self.feature_encoder.encode(snapshot, descriptors, self.search_state(budget))
        if len(rows) != len(descriptors):
            raise ValueError("feature count differs from descriptor count")
        for d, row in zip(descriptors, rows):
            self._features[d.descriptor_id] = row
        return rows

    def order(self, snapshot, descriptors, budget, observed_search_state=None):
        self._learned_selected=None
        self._heuristic_control_selected=None
        descriptors = list(descriptors)
        if len(descriptors) > 64:
            raise ValueError("ready batch exceeds 64")
        for descriptor in descriptors:
            if descriptor.source_node_id not in self.nodes or (descriptor.parent_prefix_id is not None and descriptor.parent_prefix_id not in self.parents):
                raise ValueError("unmaterialized source/parent cannot enter ready pool")
            if not self.nodes[descriptor.source_node_id].get('materialized') or (descriptor.parent_prefix_id is not None and not self.parents[descriptor.parent_prefix_id].get('materialized')):
                raise ValueError('unmaterialized source/parent cannot enter ready pool')
            if descriptor.snapshot_id != self.snapshot_id or self._descriptors.get(descriptor.descriptor_id) != descriptor:
                raise ValueError('descriptor registry/snapshot mismatch')
        for descriptor in descriptors:
            self.emit(descriptor.as_dict())
        if self.guided and self.deterministic_only:
            ids=tuple(d.descriptor_id for d in descriptors)
            self.emit(dict(record_type="Rank",ready_ids=ids,ordered_ids=ids,policy="heuristic_control"))
            if descriptors: self._heuristic_control_selected=descriptors[0].descriptor_id
            return descriptors
        if self.mode=="SHADOW_OFFLINE" and self.offline_probe is not None and descriptors:
            selected,metadata=self.offline_probe(descriptors,descriptors[0],self.coverage["opportunities"])
            if selected not in descriptors:
                raise ValueError("offline probe returned nonready descriptor")
            self.emit(dict(record_type="OfflineProbe",descriptor_id=selected.descriptor_id,
                           ready_ids=[d.descriptor_id for d in descriptors],**dict(metadata)))
            return [selected]+[d for d in descriptors if d!=selected]
        if not self.guided or len(descriptors) < 2:
            return descriptors
        ids = tuple(d.descriptor_id for d in descriptors)
        ids_set = set(ids)
        reason = None
        # Initial coverage and the last fifth of the original shared budget use
        # inherited order. The same rule applies to the reduced heuristic arm.
        finish = (getattr(budget,'candidate_elapsed',budget.elapsed) >= .8*budget.candidate_seconds or
                  budget.elapsed >= .8*budget.soft_seconds or
                  sum(budget.attempts.values()) >= .8*budget.max_attempts)
        initial=bool(self._initial) and not self._initial_finished
        if initial or finish or self._post_initial % 4 == 3:
            self._rank_cache = None
            self.emit(dict(record_type="Rank", ready_ids=ids, ordered_ids=ids,
                           policy="initial_coverage" if initial else "deterministic_finish" if finish else "heuristic_interleave"))
            return descriptors
        try:
            cache = self._rank_cache
            states = self._ready_state(descriptors)
            if (cache is not None and cache['epoch'] == self._coverage_epoch and cache['uses'] < self.rank_reuse_limit
                    and ids_set == cache['remaining']
                    and all(cache['states'].get(did) == (node,parent) for did,node,parent in states)):
                order = tuple(did for did in cache['order'] if did in ids_set)
                cache['uses'] += 1
                cache['remaining'] = set(order[1:])
                self.coverage['rank_cache_hits'] += 1
                self.coverage['rank_reorders'] += int(order != ids)
                self._learned_selected = order[0]
                self.emit(dict(record_type='Rank',ready_ids=ids,ordered_ids=order,policy='learned_cache'))
                mapping = {d.descriptor_id:d for d in descriptors}
                return [mapping[did] for did in order]
            features = self._encode(snapshot, descriptors, budget)
            if self.advisor is None:
                raise RuntimeError("MODEL_MISSING")
            with budget.measure("stage3_inference"):
                ranked = self.advisor.rank(features, budget)
            order = tuple(ranked.ordered_ids)
            if len(order) != len(ids) or set(order) != set(ids):
                raise RuntimeError("ILLEGAL_DESCRIPTOR_ID")
            if any(not math.isfinite(float(v)) for v in ranked.scores.values()):
                raise RuntimeError("NONFINITE_SCORE")
            mapping = {d.descriptor_id:d for d in descriptors}
            self.emit(dict(record_type="Rank", ready_ids=ids, ordered_ids=order,
                           policy="learned", model_metadata=dict(ranked.model_metadata), scores=dict(ranked.scores)))
            self.coverage["learned_rank_calls"]+=1
            self.coverage['rank_reorders'] += int(order != ids)
            self._rank_cache = dict(epoch=self._coverage_epoch,uses=1,order=order,remaining=set(order[1:]),
                states={did:(node,parent) for did,node,parent in states})
            self._learned_selected=order[0]
            return [mapping[i] for i in order]
        except Exception as exc:
            self._rank_cache = None
            reason = getattr(exc, "reason", None) or (str(exc) if str(exc) in
                ("MODEL_MISSING", "FINGERPRINT_MISMATCH", "INPUT_OOD", "ENTITY_LIMIT", "NONFINITE_SCORE", "ILLEGAL_DESCRIPTOR_ID") else "INFERENCE_ERROR")
            if self.evaluation and reason == "FINGERPRINT_MISMATCH":
                raise
            with budget.measure("stage3_fallback"):
                self.coverage["fallback:"+reason]+=1
                self.emit(dict(record_type="Fallback", reason=reason, ready_ids=ids,
                               remaining_budget=self.search_state(budget)["remaining_budget_fraction"],
                               error_type=type(exc).__name__))
            return descriptors

    def record_before(self, snapshot, descriptor, budget, cache_state):
        if self.trace_mode == 'collection':
            try:
                self._encode(snapshot, [descriptor], budget)
            except Exception as exc:
                self._features[descriptor.descriptor_id] = {"descriptor_id": descriptor.descriptor_id,
                    "missing": True, "reason": getattr(exc, "reason", type(exc).__name__)}
        self._query_serial+=1
        ref = "features:" + descriptor.descriptor_id + ":" + str(self._query_serial)
        if not hasattr(self,'_query_refs'):
            self._query_refs = {}
        self._query_refs[descriptor.descriptor_id] = ref
        self.emit(descriptor.as_dict())
        if self.trace_mode == 'collection':
            self.emit(dict(record_type="FeaturesBeforeQuery", features_before_query_ref=ref,
                       descriptor_id=descriptor.descriptor_id, features=self._features[descriptor.descriptor_id],
                       search_state=self.search_state(budget), cache_state=dict(cache_state)))
        return (time.perf_counter(), time.process_time(), sum(budget.attempts.values()), ref)

    def record_after(self, descriptor, token, ok, plan, budget, cache_state):
        w, c, before, ref = token
        was_initial=bool(self._initial) and not self._initial_finished
        if ok:
            self._initial.discard(descriptor.aircraft_id)
            self._post_initial+=int(not was_initial)
            self.coverage["opportunities"]+=1
            if getattr(self,"_learned_selected",None)==descriptor.descriptor_id:
                self.coverage["learned_selected"]+=1
                self.coverage["learned_stage:"+descriptor.construction_stage]+=1
            if getattr(self,"_heuristic_control_selected",None)==descriptor.descriptor_id:
                self.coverage["heuristic_control_selected"]+=1
        self._learned_selected=None
        self._heuristic_control_selected=None
        self.coverage["attempted"] += int(ok)
        self.coverage["stage:"+descriptor.construction_stage] += int(ok)
        self.emit(dict(record_type="FactualResult", descriptor_id=descriptor.descriptor_id,
                       context_id=descriptor.source_node_id, ancestry_root_id=self.ancestry_root_id or self.snapshot_id,
                       fingerprint=self.fingerprint, outcome="UNKNOWN" if ok else "NOT_ATTEMPTED",
                       reason="GENERATED_PENDING_ENERGY_CHECK" if plan is not None else "BOUNDED_GENERATOR_NO_CERTIFICATE" if ok else "BUDGET_LIMIT",
                       wall_s=time.perf_counter()-w, cpu_s=time.process_time()-c, censored=not ok,
                       cache_state=dict(cache_state), features_before_query_ref=ref,
                       physical_attempts=sum(budget.attempts.values())-before,
                       trajectory_hash=None if plan is None else plan.trajectory_hash,
                       parent_prefix_id=descriptor.parent_prefix_id))
        if plan is not None:
            self._plans.setdefault(plan.trajectory_hash, []).append(descriptor.descriptor_id)

    def record_energy(self, node, plan, q, cache_hit, wall_s=None, cpu_s=None,attribute_query=True):
        finite=all(math.isfinite(v) for v in q)
        if finite and plan.target_ring is not None:
            self._successor_states[plan.trajectory_hash]=(plan.target_ring,plan.target_slot,plan.end_time,
                node.soc-q[1],node.peak+q[2],node.version+1,node.error_j+q[3],node.peak_error_s+q[4])
        for did in self._plans.get(plan.trajectory_hash, ())[-1:]:
            valid = finite and node.soc-q[1] >= (.25 if plan.pad_id is not None else 0.) and node.peak+q[2] <= 120.
            self.emit(dict(record_type="SingleVehicleEvidence", descriptor_id=did,
                           features_before_query_ref=self._query_refs.get(did) if attribute_query else None,
                           outcome="UNKNOWN" if not finite else "SINGLE_VALID" if valid else "SINGLE_INVALID_CERTIFIED",
                           reason="STAGE2_MOVE_ENERGY", energy_j=q[0] if math.isfinite(q[0]) else None,
                           soc_used=q[1] if math.isfinite(q[1]) else None, energy_cache_hit=cache_hit,
                           wall_s=wall_s,cpu_s=cpu_s,
                           validity_label_mask=finite))

    def record_column(self, column, result=None, retained=None):
        ids = []
        for p in column.plans:
            ids.extend(self._plans.get(p.trajectory_hash, ())[-1:])
        self.emit(dict(record_type="ColumnAncestry", column_id=stable_id("column", column.signature),
                       aircraft_id=column.aircraft_id, descriptor_ids=ids,
                       trajectory_hashes=[p.trajectory_hash for p in column.plans],
                       retained=retained, validation_status=getattr(result, "status", None),
                       validation_reason=getattr(result, "reason", None),
                       objective=list(column.objective)))

    def finish_snapshot(self, budget):
        self.emit(dict(record_type="Coverage", snapshot_id=self.snapshot_id, counts=dict(self.coverage),
                       unserved_aircraft=sorted(self._initial), unserved_aircraft_count=len(self._initial),
                       rankable_stages=["progress", "reference", "alternative"],
                       guided_stages=sorted(key.split(":",1)[1] for key,value in self.coverage.items() if key.startswith("learned_stage:") and value),
                       unguided_paths=["previous_chain_reuse", "delayed_reference_regeneration", "compatibility_successors"],
                       geometry_screening=dict(budget.screening), geometry_timing=budget.timings.get("geometry_template_screen")))
