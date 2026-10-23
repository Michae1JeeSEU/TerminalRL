"""Continuous interval certificates. Unresolved boxes always reject as UNKNOWN."""
import math
import heapq
import sys
import numpy as np
from collections import OrderedDict
from fractions import Fraction
from .config import load_config
from .geometry import ring_angular_speed, slot_state, slot_phase, ring_config, pad_position
from .models import ValidationResult, KinematicState
from .trajectory import Interval, digest, poly, _intrinsic_bounds_from_polar, TrajectoryPlan
from .energy import evaluate_power, power_bounds, _numeric_count

_NUMERIC_CERTIFICATES={}
_PLAN_CERTIFICATES={}
_SEGMENT_CERTIFICATES={}

def _norm_upper(box): return sum((Interval(a,b).square() for a,b in box),Interval(0)).sqrt().hi
def _norm_lower(box):
    value=sum((Interval(a,b).square() for a,b in box),Interval(0)).sqrt().lo
    return max(0.,value)
def _close(a,b,tol): return math.dist(a,b)<=tol

def _kinematic_witness(plan,segment,times):
    """Point values witness rejection only; absence of a violation proves none."""
    _numeric_count('early_check_calls')
    if segment.kind=='GROUND': return None
    for t in times:
        state=segment.state(t); v=state.v; a=state.a; j=state.jerk
        violations=[(math.hypot(*v[:2])>25.+1e-10,'HORIZONTAL_SPEED'),
            (math.hypot(*a[:2])>1.5+1e-10,'HORIZONTAL_ACCELERATION'),
            (abs(a[2])>1.+1e-10,'VERTICAL_ACCELERATION'),
            (math.hypot(*j[:2])>1.+1e-10,'HORIZONTAL_JERK'),
            (abs(j[2])>.5+1e-10,'VERTICAL_JERK'),(abs(v[2])>2.+1e-10,'VERTICAL_SPEED')]
        u=(t-segment.start_time)/segment.duration
        if plan.kind in ('INBOUND','TRANSFER','FINAL_NORMAL','FINAL_EMERGENCY'):
            violations += [(poly(segment.r,u,1)/segment.duration>1e-9,'RADIUS_INCREASE'),
                (poly(segment.theta,u,1)/segment.duration>1e-9,'REVERSE_ROTATION'),(v[2]>1e-9,'HEIGHT_INCREASE')]
        if plan.kind in ('INBOUND','TRANSFER'):
            violations.append((math.hypot(*v[:2])<2.-1e-10,'MIN_HORIZONTAL_SPEED'))
        for violated,reason in violations:
            if violated:
                _numeric_count('kinematic_witness_rejects')
                return ValidationResult(False,'UNSAFE',reason,{'witness_time':t})
    return None

def validate_plan(plan,config=None,use_cache=True):
    _numeric_count('validate_plan_calls')
    _numeric_count('plan_interval_computes',0)
    c=config or load_config()
    if plan.profile_hash!=c.profile_hash or plan.config_hash!=c.config_hash: return ValidationResult(False,'INVALID','PROFILE_HASH_MISMATCH')
    if plan.trajectory_hash!=digest(plan._payload()): return ValidationResult(False,'INVALID','TRAJECTORY_HASH_MISMATCH')
    exact_key=(plan.trajectory_hash,c.config_hash)
    if use_cache and exact_key in _PLAN_CERTIFICATES:
        _numeric_count('plan_certificate_hits')
        return ValidationResult(True,'SAFE','',dict(_PLAN_CERTIFICATES[exact_key]))
    allowed=('INBOUND','TRANSFER','FINAL_NORMAL','FINAL_EMERGENCY','STABLE','GROUND')
    if plan.kind not in allowed: return ValidationResult(False,'INVALID','UNKNOWN_MANEUVER')
    if plan.end_time<=plan.start_time: return ValidationResult(False,'INVALID','INVALID_TIME')
    if plan.kind=='TRANSFER' and (plan.source_ring,plan.target_ring) not in (('outer','middle'),('middle','inner')): return ValidationResult(False,'INVALID','NORMAL_TOPOLOGY')
    if plan.kind=='FINAL_NORMAL' and plan.source_ring!='inner': return ValidationResult(False,'INVALID','NORMAL_TOPOLOGY')
    if plan.kind=='FINAL_EMERGENCY' and plan.source_ring not in ('outer','middle','inner'): return ValidationResult(False,'INVALID','EMERGENCY_SOURCE_RING')
    if plan.kind=='INBOUND' and (plan.target_ring!='outer' or plan.source_ring is not None): return ValidationResult(False,'INVALID','INBOUND_TOPOLOGY')
    try:
        if plan.kind=='STABLE' and not _canonical_stable(plan,c): return ValidationResult(False,'INVALID','NONCANONICAL_STABLE')
        if plan.kind=='GROUND' and not _canonical_ground(plan,c): return ValidationResult(False,'INVALID','NONCANONICAL_GROUND')
        if plan.kind!='GROUND' and any(s.kind=='GROUND' for s in plan.segments): return ValidationResult(False,'INVALID','AIRBORNE_GROUND_LABEL')
        if plan.kind!='STABLE' and any(s.kind=='STABLE' for s in plan.segments): return ValidationResult(False,'INVALID','MANEUVER_STABLE_LABEL')
        if plan.kind not in ('GROUND',) and any(all(all(x==0 for x in coeff[1:]) for coeff in (s.r,s.theta,s.z)) for s in plan.segments): return ValidationResult(False,'INVALID','HOVER_WAIT')
        # C2 is an endpoint constraint independently checked against metadata.
        for a,b in zip(plan.segments,plan.segments[1:]):
            if abs(a.end_time-b.start_time)>1e-9: return ValidationResult(False,'INVALID','SEGMENT_TIME_GAP')
            sa=a.state(a.end_time); sb=b.state(b.start_time)
            if any(not _close(getattr(sa,k),getattr(sb,k),tol) for k,tol in (('p',1e-7),('v',1e-7),('a',1e-7))): return ValidationResult(False,'INVALID','C2_DISCONTINUITY')
        first=plan.state(plan.start_time); last=plan.state(plan.end_time)
        if plan.source_ring is not None:
            expected=slot_state(plan.source_ring,plan.source_slot,plan.start_time,c)
            if any(not _close(getattr(first,k),getattr(expected,k),tol) for k,tol in (('p',1e-6),('v',1e-7),('a',1e-7))): return ValidationResult(False,'INVALID','SOURCE_STATE_MISMATCH')
        if plan.target_ring is not None:
            expected=slot_state(plan.target_ring,plan.target_slot,plan.end_time,c)
            if any(not _close(getattr(last,k),getattr(expected,k),tol) for k,tol in (('p',1e-6),('v',1e-7),('a',1e-7))): return ValidationResult(False,'INVALID','TARGET_STATE_MISMATCH')
        if plan.kind.startswith('FINAL'):
            if len(plan.segments)!=4: return ValidationResult(False,'INVALID','ATOMIC_DESCENT_REQUIRED')
            capture=plan.segments[0].state(plan.segments[0].end_time); p=pad_position(plan.pad_id,c)
            if not _close(capture.p,(p[0],p[1],15.),1e-6) or not _close(capture.v,(0,0,-.5),1e-7) or not _close(capture.a,(0,0,0),1e-7): return ValidationResult(False,'INVALID','CAPTURE_STATE_MISMATCH')
            from .trajectory import descent_segments
            expected=descent_segments(plan.pad_id,plan.segments[0].end_time,c)
            if tuple(s.to_dict() for s in plan.segments[1:])!=tuple(s.to_dict() for s in expected): return ValidationResult(False,'INVALID','NONCANONICAL_DESCENT')
            if not _close(last.p,p,1e-6) or not _close(last.v,(0,0,0),1e-7): return ValidationResult(False,'INVALID','TOUCHDOWN_MISMATCH')
        if plan.kind=='INBOUND':
            r=math.hypot(*first.p[:2]); gate_angle=math.atan2(first.p[1],first.p[0]); matches=[]
            for angle in c.entry.gate_angles_rad:
                p=(900*math.cos(angle),900*math.sin(angle),110.)
                vr=-21/math.sqrt(2); vt=-21/math.sqrt(2)
                v=(vr*math.cos(angle)-vt*math.sin(angle),vr*math.sin(angle)+vt*math.cos(angle),0)
                matches.append(_close(first.p,p,1e-6) and _close(first.v,v,1e-7) and _close(first.a,(0,0,0),1e-7))
            if not any(matches): return ValidationResult(False,'INVALID','ENTRY_STATE_MISMATCH')
    except (ValueError,TypeError,AttributeError) as exc: return ValidationResult(False,'INVALID','INVALID_METADATA',{'error':str(exc)})
    numeric_key=(c.config_hash,plan.kind,plan.source_ring,tuple((s.duration,s.r,s.theta[1:],s.z,s.kind) for s in plan.segments))
    if use_cache and numeric_key in _NUMERIC_CERTIFICATES:
        _numeric_count('numeric_certificate_hits')
        details=dict(_NUMERIC_CERTIFICATES[numeric_key],trajectory_hash=plan.trajectory_hash,profile_hash=plan.profile_hash)
        if len(_PLAN_CERTIFICATES)>=512: _PLAN_CERTIFICATES.pop(next(iter(_PLAN_CERTIFICATES)))
        _PLAN_CERTIFICATES[exact_key]=dict(details)
        return ValidationResult(True,'SAFE','',details)
    counts={'intervals':0,'max_depth':0,'power_intervals':0}
    def check(s,a,b,depth):
        _numeric_count('plan_interval_computes')
        counts['intervals']+=1; counts['max_depth']=max(depth,counts['max_depth'])
        if s.kind=='GROUND': return None
        if depth:
            result=_kinematic_witness(plan,s,(a,(a+b)/2,b))
            if result is not None: return result
        polar=s.polar_bounds(a,b); r,theta,z=polar
        vb=_intrinsic_bounds_from_polar(polar,1); ab=_intrinsic_bounds_from_polar(polar,2); jb=_intrinsic_bounds_from_polar(polar,3)
        # Polynomial coordinate enclosures include floating evaluation padding;
        # 1e-9 is solely algebraic roundoff on exact endpoint equalities, never
        # an allowance on the 80m separation boundary.
        eps=1e-9
        checks=[(r[0].lo>=-eps and r[0].hi<=900+eps,'AIRSPACE_RADIUS'),(z[0].lo>=-eps and z[0].hi<=120+eps,'AIRSPACE_HEIGHT'),
            (_norm_upper(vb[:2])<=25.,'HORIZONTAL_SPEED'),(_norm_upper(ab[:2])<=1.5,'HORIZONTAL_ACCELERATION'),
            (max(abs(ab[2][0]),abs(ab[2][1]))<=1.,'VERTICAL_ACCELERATION'),
            (_norm_upper(jb[:2])<=1.,'HORIZONTAL_JERK'),(max(abs(jb[2][0]),abs(jb[2][1]))<=.5,'VERTICAL_JERK'),
            (vb[2][0]>=-2. and vb[2][1]<=2.,'VERTICAL_SPEED')]
        if plan.kind in ('INBOUND','TRANSFER','FINAL_NORMAL','FINAL_EMERGENCY'):
            checks += [(r[1].hi<=eps,'RADIUS_INCREASE'),(theta[1].hi<=eps,'REVERSE_ROTATION'),(z[1].hi<=eps,'HEIGHT_INCREASE')]
        if plan.kind in ('INBOUND','TRANSFER'):
            # Orthogonal polar components give a sharper speed lower bound.
            speed2=r[1].square()+(r[0]*theta[1]).square()
            checks.append((speed2.lo>=4.,'MIN_HORIZONTAL_SPEED'))
        if plan.kind=='INBOUND': checks += [(r[0].lo>=600-eps,'INBOUND_INNER_BOUNDARY'),(z[0].lo>=110-eps and z[0].hi<=110+eps,'INBOUND_HEIGHT')]
        if plan.kind.startswith('FINAL') or plan.kind=='TRANSFER':
            source=ring_config(plan.source_ring,c)
            checks.append((r[0].hi<=source.radius_m+eps,'OUTWARD_PATH'))
        pb=power_bounds(vb,ab,c); counts['power_intervals']+=1
        if pb is not None:
            checks += [(pb['axial_descent_ratio'][0]>=c.energy.minimum_axial_descent_ratio,'ENERGY_MODEL_DOMAIN'),(pb['battery_w'][0]>0,'NONPOSITIVE_POWER'),
                (pb['current_a'][1]<=c.energy.reference_max_battery_current_A,'BATTERY_CURRENT'),
                (pb['shaft_w'][1]<=c.vehicle.rotor_count*c.energy.reference_motor_max_power_kW_each*1000,'SHAFT_POWER')]
        else: checks.append((False,'ENERGY_MODEL_DOMAIN'))
        if all(ok for ok,_ in checks): return None
        # Finite point checks can witness rejection, but never prove acceptance.
        for t in (a,(a+b)/2,b):
            st=s.state(t); power=evaluate_power(st.v,st.a,c)
            violations=[(not power.valid,power.status),(power.valid and power.current_a>2016,'BATTERY_CURRENT'),(power.valid and power.shaft_w>288000,'SHAFT_POWER')]
            for violation,reason in violations:
                if violation: return ValidationResult(False,'UNSAFE',reason,{'witness_time':t})
        if depth>=24 or b-a<=1e-6:
            return ValidationResult(False,'UNKNOWN',next(reason for ok,reason in checks if not ok),{'interval':[a,b]})
        mid=(a+b)/2
        return check(s,a,mid,depth+1) or check(s,mid,b,depth+1)
    pending=[]
    for s in plan.segments:
        # Canonical descents were checked exactly against the frozen builder
        # above. Their constraints and power do not depend on pad angle/time.
        segment_key=(c.config_hash,plan.kind,plan.source_ring,s.duration,s.r,s.theta[1:],s.z,s.kind)
        if s.kind=='DESCENT' and plan.kind.startswith('FINAL'):
            segment_key=(c.config_hash,'canonical_descent',s.duration,s.r,s.theta[1:],s.z)
        if use_cache and segment_key in _SEGMENT_CERTIFICATES:
            _numeric_count('segment_certificate_hits')
            continue
        pending.append((s,segment_key))
    # Check every uncached segment before evaluating any segment's power.
    # Structure and endpoint metadata above retain INVALID precedence.
    for s,_ in pending:
        result=_kinematic_witness(plan,s,tuple(s.start_time+s.duration*f for f in (0.,.25,.5,.75,1.)))
        if result is not None: return result
    for s,segment_key in pending:
        result=check(s,s.start_time,s.end_time,0)
        if result is not None: return result
        if use_cache:
            if len(_SEGMENT_CERTIFICATES)>=1024: _SEGMENT_CERTIFICATES.pop(next(iter(_SEGMENT_CERTIFICATES)))
            _SEGMENT_CERTIFICATES[segment_key]=True
    certificate=dict(counts,method='outward_polynomial_Bernstein_intrinsic_frame_and_induced_root_intervals')
    details=dict(certificate,trajectory_hash=plan.trajectory_hash,profile_hash=plan.profile_hash)
    if use_cache:
        if len(_NUMERIC_CERTIFICATES)>=256: _NUMERIC_CERTIFICATES.pop(next(iter(_NUMERIC_CERTIFICATES)))
        if len(_PLAN_CERTIFICATES)>=512: _PLAN_CERTIFICATES.pop(next(iter(_PLAN_CERTIFICATES)))
        _NUMERIC_CERTIFICATES[numeric_key]=dict(certificate)
        _PLAN_CERTIFICATES[exact_key]=dict(details)
    return ValidationResult(True,'SAFE','',details)

class _ExecutionQueryCache:
    """Bounded pure-query values for one physical execution interval only.

    Keep actual base-plan objects alive; identity is never substituted by a
    payload hash. Estimate cache-owned containers, bindings, keys and values;
    authoritative plan payloads remain owned by the caller and are not copied.
    FIFO eviction changes only recomputation, never a certificate or pair order.
    """
    def __init__(self, plans, *, max_entries=4096, max_bytes=8*1024*1024):
        if not 1<=max_entries<=4096 or not 8192<=max_bytes<=8*1024*1024:
            raise ValueError('execution query cache limits exceed the private budget')
        self.max_entries=max_entries; self.max_bytes=max_bytes
        self._plans={id(p):p for p in plans if type(p) is TrajectoryPlan}
        self._values=OrderedDict()
        self._binding_bytes=(4096+sys.getsizeof(self._plans)+sys.getsizeof(self._values)+
            sum(sys.getsizeof(key)+sys.getsizeof(plan) for key,plan in self._plans.items()))
        # If even bindings exceed the budget, retain no bindings or query data.
        if self._binding_bytes>max_bytes:
            self._plans.clear()
            self._binding_bytes=4096+sys.getsizeof(self._plans)+sys.getsizeof(self._values)
        self.estimated_bytes=self._binding_bytes
        self.counts={'state_hits':0,'state_misses':0,'bounds_hits':0,'bounds_misses':0,
                     'bypasses':0,'oversized_values':0,'evictions':0}

    @property
    def entries(self): return len(self._values)

    @staticmethod
    def _value_bytes(value):
        if type(value) is KinematicState:
            return sys.getsizeof(value)+sum(_ExecutionQueryCache._value_bytes(getattr(value,k))
                                            for k in ('p','v','a','jerk'))
        if type(value) is tuple:
            return sys.getsizeof(value)+sum(_ExecutionQueryCache._value_bytes(x) for x in value)
        return sys.getsizeof(value)

    def _query(self, plan, kind, times, derivative=0):
        eligible=(self._plans.get(id(plan)) is plan and
                  all(type(t) is float and math.isfinite(t) for t in times) and
                  (kind=='state' or type(derivative) is int))
        if not eligible:
            self.counts['bypasses']+=1
            return plan.state(times[0]) if kind=='state' else plan.bounds(*times,derivative)
        key=(id(plan),kind,*(t.hex() for t in times),derivative)
        found=self._values.get(key)
        if found is not None:
            self.counts[kind+'_hits']+=1
            return found[0]
        self.counts[kind+'_misses']+=1
        value=plan.state(times[0]) if kind=='state' else plan.bounds(*times,derivative)
        # These are the immutable original return forms, never mutable boxes.
        immutable=(type(value) is KinematicState if kind=='state' else
                   type(value) is tuple and all(type(pair) is tuple and
                       all(type(x) is float for x in pair) for pair in value))
        if not immutable:
            self.counts['bypasses']+=1
            return value
        # 512 covers the entry's mapping/record bookkeeping conservatively;
        # keys/values are charged recursively, including repeated references.
        cost=512+self._value_bytes(key)+self._value_bytes(value)
        if self._binding_bytes+cost>self.max_bytes:
            self.counts['oversized_values']+=1
            return value
        while self._values and (len(self._values)>=self.max_entries or
                                 self.estimated_bytes+cost>self.max_bytes):
            _,(_,previous_cost)=self._values.popitem(last=False)
            self.estimated_bytes-=previous_cost
            self.counts['evictions']+=1
        self._values[key]=(value,cost)
        self.estimated_bytes+=cost
        return value

    def state(self, plan, time): return self._query(plan,'state',(time,))

    def bounds(self, plan, start, end, derivative=0):
        return self._query(plan,'bounds',(start,end),derivative)


def validate_separation_original(plan_a,plan_b,start,end,config=None,max_depth=32,min_interval=1e-7,*,_query_cache=None):
    """Unmodified decision path for offline comparisons, without broad phases."""
    _numeric_count('separation_original_calls')
    c=config or load_config(); hard=c.safety.hard_separation_m
    if end<start or not all(math.isfinite(t) for t in (start,end)): return ValidationResult(False,'INVALID','INVALID_TIME')
    exact=_constant_vertical_offset(plan_a,plan_b,start,end)
    if exact is not None:
        distance=abs(exact); safe=distance>=Fraction(hard)
        return ValidationResult(safe,'SAFE' if safe else 'UNSAFE','' if safe else 'HARD_SEPARATION',{'lower_bound_m':float(distance),'upper_bound_m':float(distance),'method':'identical_horizontal_polynomials_exact_vertical_offset'})
    # Equal clocks retain constant distance. Different clocks use the
    # concentric radial/vertical lower bound, or the continuous checker.
    if _canonical_stable(plan_a,c) and _canonical_stable(plan_b,c):
        ra=ring_config(plan_a.target_ring,c); rb=ring_config(plan_b.target_ring,c)
        if ring_angular_speed(ra.id,c)!=ring_angular_speed(rb.id,c):
            # Concentric circles retain this bound for every relative phase.
            lower=math.nextafter(math.hypot(ra.radius_m-rb.radius_m,ra.height_agl_m-rb.height_agl_m)-1e-10,-math.inf)
            if lower>=hard:
                witness=math.dist(plan_a.state(start).p,plan_b.state(start).p)
                return ValidationResult(True,'SAFE','',{'lower_bound_m':lower,'upper_bound_m':math.nextafter(witness+1e-10,math.inf),'method':'concentric_ring_radial_vertical_bound','checked_until':end})
        else:
            delta=slot_phase(ra.id,plan_a.target_slot,0,c)-slot_phase(rb.id,plan_b.target_slot,0,c)
            dsq=(ra.radius_m-rb.radius_m)**2+4*ra.radius_m*rb.radius_m*math.sin(delta/2)**2+(ra.height_agl_m-rb.height_agl_m)**2
            d=math.sqrt(dsq); low=math.nextafter(d-1e-10,-math.inf); high=math.nextafter(d+1e-10,math.inf)
            if low>=hard or high<hard:
                safe=low>=hard
                return ValidationResult(safe,'SAFE' if safe else 'UNSAFE','' if safe else 'HARD_SEPARATION',{'minimum_distance_m':d,'lower_bound_m':low,'upper_bound_m':high,'method':'common_rotation_structure_with_roundoff_enclosure'})
    best_upper=math.inf; min_lower=math.inf; count=0
    def check(a,b,depth):
        _numeric_count('separation_fine_intervals')
        nonlocal best_upper,min_lower,count
        count+=1; mid=(a+b)/2
        if _query_cache is None:
            sa=plan_a.state(mid); sb=plan_b.state(mid)
        else:
            sa=_query_cache.state(plan_a,mid); sb=_query_cache.state(plan_b,mid)
        d=math.dist(sa.p,sb.p); best_upper=min(best_upper,d)
        dlow=math.nextafter(d-4*math.ulp(d),-math.inf); dhigh=math.nextafter(d+4*math.ulp(d),math.inf)
        if dhigh<hard: return ValidationResult(False,'UNSAFE','HARD_SEPARATION',{'witness_time':mid,'distance_m':d,'upper_bound_m':dhigh})
        if _query_cache is None:
            ba=plan_a.bounds(a,b,0); bb=plan_b.bounds(a,b,0)
        else:
            ba=_query_cache.bounds(plan_a,a,b,0); bb=_query_cache.bounds(plan_b,a,b,0)
        rel=tuple((Interval(*x)-Interval(*y)).pair() for x,y in zip(ba,bb)); lower=_norm_lower(rel)
        # Taylor enclosure at midpoint reduces dependency on common motion.
        if _query_cache is None:
            va=plan_a.bounds(a,b,1); vb=plan_b.bounds(a,b,1)
        else:
            va=_query_cache.bounds(plan_a,a,b,1); vb=_query_cache.bounds(plan_b,a,b,1)
        vr=tuple((Interval(*x)-Interval(*y)).pair() for x,y in zip(va,vb)); lower=max(lower,math.nextafter(dlow-math.nextafter(_norm_upper(vr),math.inf)*(b-a)/2,-math.inf))
        if lower>=hard:
            min_lower=min(min_lower,lower); return None
        if depth>=max_depth or b-a<=min_interval:
            return ValidationResult(False,'UNKNOWN','UNRESOLVED_SEPARATION',{'interval':[a,b],'lower_bound_m':lower,'upper_bound_m':d})
        return check(a,mid,depth+1) or check(mid,b,depth+1)
    result=check(start,end,0)
    if result is not None: return result
    return ValidationResult(True,'SAFE','',{'lower_bound_m':min_lower,'upper_bound_m':best_upper,'intervals':count,'method':'continuous_interval_relative_velocity','checked_until':end})


def validate_separation(plan_a,plan_b,start,end,config=None,max_depth=32,min_interval=1e-7,*,_query_cache=None):
    """Conservative pair broad phases, then the original interval checker.

    Displacement bounds are derived from actual trajectory position enclosures,
    including future segments, rather than assuming a validated speed limit or
    continuity for arbitrary, not-yet-validated input plans.
    Both position boxes are evaluated over the *same* time slab.
    """
    _numeric_count('separation_calls')
    c=config or load_config(); hard=c.safety.hard_separation_m
    fallback=lambda a,b: validate_separation_original(plan_a,plan_b,a,b,c,max_depth,min_interval,_query_cache=_query_cache)
    if end<=start or not all(math.isfinite(t) for t in (start,end)):
        return fallback(start,end)
    # Submission wraps holds in _Traffic. Only an exact wrapper with no future
    # plans represents its canonical stable base for the entire query window.
    from .world import _Traffic
    def stable_base(plan):
        if _canonical_stable(plan,c): return plan
        if type(plan) is _Traffic and not plan.plans and _canonical_stable(plan.base,c): return plan.base
        return None
    stable_a=stable_base(plan_a); stable_b=stable_base(plan_b)
    if stable_a is not None and stable_b is not None:
        _numeric_count('separation_analytic_calls')
        return validate_separation_original(stable_a,stable_b,start,end,c,max_depth,min_interval,_query_cache=_query_cache)
    # Preserve exact boundary and analytic common-rotation proofs.
    if _constant_vertical_offset(plan_a,plan_b,start,end) is not None:
        _numeric_count('separation_analytic_calls')
        return fallback(start,end)
    def bounds(p,a,b,k):
        return p.bounds(a,b,k) if _query_cache is None else _query_cache.bounds(p,a,b,k)
    mid=(start+end)/2
    sa=plan_a.state(mid) if _query_cache is None else _query_cache.state(plan_a,mid)
    sb=plan_b.state(mid) if _query_cache is None else _query_cache.state(plan_b,mid)
    distance=math.dist(sa.p,sb.p)
    ba=bounds(plan_a,start,end,0); bb=bounds(plan_b,start,end,0)
    witness_upper=math.nextafter(distance+8*math.ulp(distance),math.inf)
    def displacement(box,state):
        delta=tuple((Interval(*x)-value).pair() for x,value in zip(box,state.p))
        return _norm_upper(delta)
    reach=displacement(ba,sa)+displacement(bb,sb)
    box_lower=_norm_lower(tuple((Interval(*x)-Interval(*y)).pair() for x,y in zip(ba,bb)))
    lower=max(box_lower,math.nextafter(distance-8*math.ulp(distance)-math.nextafter(reach,math.inf),-math.inf))
    if lower>=hard:
        _numeric_count('separation_displacement_accepts')
        return ValidationResult(True,'SAFE','',{'lower_bound_m':lower,'upper_bound_m':witness_upper,'witness_time':mid,'method':'certified_maximum_displacement','checked_until':end})
    cuts={start,end}
    # _Traffic exposes base/plans and conservative bounds across junctions.
    # No world import or assumption about the current active plan is needed.
    for p in (plan_a,plan_b):
        contained=(p,) if isinstance(p,TrajectoryPlan) else (getattr(p,'base',None),*getattr(p,'plans',()))
        for plan in contained:
            if plan is None: continue
            for s in getattr(plan,'segments',()):
                cuts.update(t for t in (s.start_time,s.end_time) if start<t<end)
    cuts.update(start+(end-start)*i/8 for i in range(1,8))
    cuts=sorted(cuts); minimum=math.inf; fine=0
    for a,b in zip(cuts,cuts[1:]):
        _numeric_count('separation_box_intervals')
        ba=bounds(plan_a,a,b,0); bb=bounds(plan_b,a,b,0)
        low=_norm_lower(tuple((Interval(*x)-Interval(*y)).pair() for x,y in zip(ba,bb)))
        if low>=hard:
            _numeric_count('separation_box_accepts'); minimum=min(minimum,low)
            continue
        fine+=1; result=fallback(a,b)
        if not result.accepted: return result
        minimum=min(minimum,result.details.get('lower_bound_m',hard))
        witness_upper=min(witness_upper,result.details.get('upper_bound_m',math.inf))
    return ValidationResult(True,'SAFE','',{'lower_bound_m':minimum,'upper_bound_m':witness_upper,'method':'segmented_spacetime_boxes_with_original_fallback','fine_slabs':fine,'checked_until':end})

def _canonical_stable(plan,c):
    try:
        if plan.kind!='STABLE' or len(plan.segments)!=1 or plan.source_ring!=plan.target_ring or plan.source_slot!=plan.target_slot: return False
        from .trajectory import _stable_segment
        expected=_stable_segment(plan.target_ring,plan.target_slot,plan.start_time,plan.end_time,c)
        return plan.segments[0].to_dict()==expected.to_dict() and plan.profile_hash==c.profile_hash and plan.config_hash==c.config_hash
    except (AttributeError,ValueError): return False

def _canonical_ground(plan,c):
    try:
        if plan.kind!='GROUND' or len(plan.segments)!=1: return False
        from .trajectory import make_ground_plan
        expected=make_ground_plan(plan.aircraft_id,plan.pad_id,plan.start_time,plan.end_time,c)
        return plan.segments[0].to_dict()==expected.segments[0].to_dict()
    except (AttributeError,ValueError): return False


_EXECUTION_SPEED_CERTIFICATES=OrderedDict()

def _execution_distance_screen(plans,start,end,config=None):
    """Safe pair exclusions for one <=.2s execution step, without position boxes.

    A successful complete plan validation justifies the frozen 3D speed bound.
    This helper never substitutes a current-distance test for future scheduling.
    """
    _numeric_count('execution_screen_calls')
    c=config or load_config()
    if not all(math.isfinite(t) for t in (start,end)): return {}
    dt=end-start; rounding=min(1e-10,8*max(math.ulp(start),math.ulp(end),math.ulp(.2)))
    if not 0<dt<=.2+rounding: return {}
    records=[]
    for aircraft_id,plan in plans.items():
        if type(plan) is not TrajectoryPlan: continue
        stable=_canonical_stable(plan,c); ground=_canonical_ground(plan,c)
        if not stable and not ground and not (plan.start_time<=start and end<=plan.end_time): continue
        try: position=plan.state(start).p
        except (ValueError,TypeError,AttributeError,OverflowError): continue
        if not all(math.isfinite(x) for x in position): continue
        records.append((aircraft_id,plan,stable,ground,position))
    if len(records)<2 or all(record[2] for record in records): return {}
    positions=np.asarray([record[4] for record in records],dtype=float)
    _numeric_count('execution_screen_state_calls',len(records))
    delta=positions[:,None,:]-positions[None,:,:]
    distances=np.sqrt(np.sum(delta*delta,axis=2))
    speeds={}
    for _,plan,_,ground,_ in records:
        key=(id(plan),id(c)); cached=_EXECUTION_SPEED_CERTIFICATES.get(key)
        if cached is not None and cached[0] is plan and cached[1] is c:
            _numeric_count('execution_screen_certificate_hits')
        else:
            _numeric_count('execution_screen_validation_computes')
            if not validate_plan(plan,c).accepted: continue
            if len(_EXECUTION_SPEED_CERTIFICATES)>=512: _EXECUTION_SPEED_CERTIFICATES.popitem(last=False)
            _EXECUTION_SPEED_CERTIFICATES[key]=(plan,c)
        speeds[id(plan)]=0. if ground else math.nextafter(math.hypot(25.,2.),math.inf)
    result={}; eps=np.finfo(float).eps
    for i,(id_a,plan_a,stable_a,_,_) in enumerate(records):
        for j in range(i+1,len(records)):
            id_b,plan_b,stable_b,_,_=records[j]
            _numeric_count('execution_screen_pairs')
            if stable_a and stable_b: continue
            if id(plan_a) not in speeds or id(plan_b) not in speeds: continue
            distance=float(distances[i,j])
            if not math.isfinite(distance): continue
            # Pad subtraction/norm and endpoint position roundoff outward.
            scale=max(1.,distance,float(np.max(np.abs(positions[[i,j]]))))
            pad=64*eps*scale+8*math.ulp(distance)
            travel=math.nextafter((speeds[id(plan_a)]+speeds[id(plan_b)])*dt,math.inf)
            lower=math.nextafter(distance-pad-travel,-math.inf)
            if lower<=c.safety.hard_separation_m: continue
            _numeric_count('execution_screen_accepts')
            result[(id_a,id_b)]=ValidationResult(True,'SAFE','',{
                'lower_bound_m':lower,'upper_bound_m':math.nextafter(distance+pad,math.inf),
                'witness_time':start,'checked_until':end,'method':'validated_execution_speed_distance_screen'})
    return result

def _constant_vertical_offset(a,b,start,end):
    try:
        aa=list(a.intervals(start,end)); bb=list(b.intervals(start,end))
        if len(aa)!=len(bb) or not aa: return None
        offset=None
        for x,y in zip(aa,bb):
            if x.start_time!=y.start_time or x.duration!=y.duration or x.r!=y.r or x.theta!=y.theta or x.z[1:]!=y.z[1:]: return None
            value=Fraction(x.z[0])-Fraction(y.z[0])
            if offset is not None and value!=offset: return None
            offset=value
        return offset
    except (AttributeError,ValueError): return None

def certified_minimum_distance(plan_a,plan_b,start,end,config=None,tolerance_m=.001,max_intervals=100000):
    """Continuous minimum enclosure, separate from the fast safety decision.

    Witness points supply upper bounds only. Exhausting the refinement budget
    returns UNKNOWN even when the sampled apparent minimum looks safe.
    """
    c=config or load_config()
    if end<start or tolerance_m<=0 or not all(math.isfinite(x) for x in (start,end,tolerance_m)):
        raise ValueError('invalid minimum-distance query')
    exact=_constant_vertical_offset(plan_a,plan_b,start,end)
    if exact is not None:
        d=float(abs(exact))
        return {'status':'CERTIFIED','lower_bound_m':d,'upper_bound_m':d,'witness_time':start,'witness_distance_m':d,'intervals':0,'method':'exact_constant_vertical_offset'}
    structural=validate_separation(plan_a,plan_b,start,end,c,max_depth=0)
    if structural.details.get('method')=='common_rotation_structure_with_roundoff_enclosure':
        return dict(structural.details,status='CERTIFIED',witness_time=start,witness_distance_m=structural.details['minimum_distance_m'],intervals=0)
    upper=math.inf; witness_time=start; witness_distance=math.inf; serial=0; evaluated=0
    def box(a,b):
        nonlocal upper,witness_time,witness_distance,evaluated
        evaluated+=1; mid=(a+b)/2
        for t in (a,mid,b):
            d=math.dist(plan_a.state(t).p,plan_b.state(t).p)
            hi=math.nextafter(d+4*math.ulp(d),math.inf)
            if hi<upper: upper=hi; witness_time=t; witness_distance=d
        pa=plan_a.bounds(a,b,0); pb=plan_b.bounds(a,b,0)
        rel=[Interval(*x)-Interval(*y) for x,y in zip(pa,pb)]
        low=_norm_lower([x.pair() for x in rel])
        va=plan_a.bounds(a,b,1); vb=plan_b.bounds(a,b,1)
        vel=[Interval(*x)-Interval(*y) for x,y in zip(va,vb)]
        # |d/dt ||r||²| = |2 r·v|, sharper than speed near a smooth minimum.
        slope=2*sum((x*y for x,y in zip(rel,vel)),Interval(0))
        d=math.dist(plan_a.state(mid).p,plan_b.state(mid).p)
        dl=max(0.,math.nextafter(d-4*math.ulp(d),-math.inf))
        sq=Interval(dl).square()-Interval(max(abs(slope.lo),abs(slope.hi)))*(b-a)/2
        low=max(low,max(0.,sq.sqrt().lo))
        return low
    heap=[(box(start,end),serial,start,end,0)]
    while heap and upper-heap[0][0]>tolerance_m and evaluated<max_intervals:
        low,_,a,b,depth=heapq.heappop(heap)
        if low>=upper: continue
        if depth>=48 or b-a<=1e-10:
            heapq.heappush(heap,(low,serial,a,b,depth)); break
        mid=(a+b)/2
        for x,y in ((a,mid),(mid,b)):
            lower=box(x,y); serial+=1
            if lower<=upper: heapq.heappush(heap,(lower,serial,x,y,depth+1))
    lower=min(upper,heap[0][0] if heap else upper)
    return {'status':'CERTIFIED' if upper-lower<=tolerance_m else 'UNKNOWN','lower_bound_m':lower,'upper_bound_m':upper,'witness_time':witness_time,'witness_distance_m':witness_distance,'intervals':evaluated,'method':'continuous_squared_distance_interval_branch_and_bound'}
