"""Immutable analytic polar trajectories and deterministic candidate construction.

Coefficients are in normalized local time. Certificates operate on polynomial
enclosures and trigonometric intervals, never on the execution sample grid.
"""
from dataclasses import dataclass, replace
from collections import OrderedDict
from functools import lru_cache
import math
import sys
import numpy as np
from .config import load_config, digest
from .geometry import slot_phase, slot_state, ring_config, pad_position
from .models import KinematicState, CandidateResult

def poly(c,u,k=0):
    c=tuple(c)
    for _ in range(k): c=tuple(i*c[i] for i in range(1,len(c)))
    return sum(x*u**i for i,x in enumerate(c))

def _poly_bounds_uncached(c,lo,hi,k=0,scale=1.):
    """Power-to-Bernstein hull on the requested subinterval (outward padded)."""
    c=tuple(c)
    for _ in range(k): c=tuple(i*c[i] for i in range(1,len(c)))
    if not c: return (0.,0.)
    n=len(c)-1; width=hi-lo
    power=[sum(c[j]*math.comb(j,i)*lo**(j-i)*width**i for j in range(i,n+1)) for i in range(n+1)]
    bern=[sum(power[j]*math.comb(i,j)/math.comb(n,j) for j in range(i+1)) for i in range(n+1)]
    error=32*np.finfo(float).eps*sum(abs(x) for x in c)*max(1.,abs(lo),abs(hi))**n
    return (math.nextafter((min(bern)-error)*scale,-math.inf), math.nextafter((max(bern)+error)*scale,math.inf))


class _BernsteinMetadataCache:
    """FIFO coefficient-only metadata; no interval certificates or world state."""
    __slots__=('entries','entry_bytes','hits','misses','evictions','bypasses',
               'enabled','max_entries','max_bytes')
    POLICY='original_bernstein_product_order_v1'
    FIXED_BYTES=4096
    def __init__(self,max_entries=4096,max_bytes=16*1024*1024):
        self.entries=OrderedDict(); self.entry_bytes=0
        self.hits=self.misses=self.evictions=self.bypasses=0
        self.enabled=True; self.max_entries=max_entries; self.max_bytes=max_bytes
        if type(max_entries) is not int or not 1<=max_entries<=4096: raise ValueError('invalid metadata entry limit')
        if type(max_bytes) is not int or self.owned_bytes()>max_bytes or max_bytes>16*1024*1024: raise ValueError('invalid metadata byte limit')
    @staticmethod
    def _size(value):
        # Deliberately count repeated immutable references repeatedly: conservative.
        return sys.getsizeof(value)+(sum(_BernsteinMetadataCache._size(x) for x in value) if isinstance(value,tuple) else 0)
    def owned_bytes(self):
        # Includes the cache object, current dict allocation, all retained records,
        # and conservative fixed allowance for scalar fields/counters/containers.
        return self.FIXED_BYTES+sys.getsizeof(self)+sys.getsizeof(self.entries)+self.entry_bytes
    def clear(self):
        self.entries.clear(); self.entry_bytes=0
        self.hits=self.misses=self.evictions=self.bypasses=0
    def stats(self):
        return dict(entries=len(self.entries),bytes=self.owned_bytes(),max_entries=self.max_entries,
                    max_bytes=self.max_bytes,hits=self.hits,misses=self.misses,
                    evictions=self.evictions,bypasses=self.bypasses,enabled=self.enabled,policy=self.POLICY)
    def get(self,c,k):
        if not self.enabled:
            self.bypasses+=1; return None
        # The integer itself disambiguates distinct builtin ints sharing float.hex.
        key=(self.POLICY,k,tuple((type(x).__name__,float(x).hex(),x if type(x) is int else None) for x in c))
        if key in self.entries:
            self.hits+=1; return self.entries[key][0]
        self.misses+=1
        for _ in range(k): c=tuple(i*c[i] for i in range(1,len(c)))
        if not c: record=(c,(),(),(),0)
        else:
            n=len(c)-1
            products=tuple(tuple(c[j]*math.comb(j,i) for j in range(i,n+1)) for i in range(n+1))
            numerator=tuple(tuple(math.comb(i,j) for j in range(i+1)) for i in range(n+1))
            denominator=tuple(math.comb(n,j) for j in range(n+1))
            record=(c,products,numerator,denominator,sum(abs(x) for x in c))
        # Pair tuple and recorded size integer are owned too; 128 overcharges
        # their scalar size bookkeeping, including the temporary zero placeholder.
        size=self._size((key,(record,0)))+128
        empty_bytes=self.FIXED_BYTES+sys.getsizeof(self)+sys.getsizeof(OrderedDict())
        if empty_bytes+size>self.max_bytes:
            self.bypasses+=1; return None
        while self.entries and (len(self.entries)>=self.max_entries or self.owned_bytes()+size>self.max_bytes):
            _,(_,old_size)=self.entries.popitem(last=False); self.entry_bytes-=old_size; self.evictions+=1
        self.entries[key]=(record,size); self.entry_bytes+=size
        # Account for dict growth after insertion, deterministically evict FIFO.
        while self.entries and self.owned_bytes()>self.max_bytes:
            _,(_,old_size)=self.entries.popitem(last=False); self.entry_bytes-=old_size; self.evictions+=1
        if key not in self.entries:
            self.bypasses+=1; return None
        return record

_BERNSTEIN_METADATA=_BernsteinMetadataCache()

def bernstein_cache_clear():
    """Clear only coefficient metadata and its counters; leave poly_bounds LRU."""
    _BERNSTEIN_METADATA.clear()

def bernstein_cache_stats(): return _BERNSTEIN_METADATA.stats()

def bernstein_cache_enable(enabled=True):
    """Set only metadata policy; an already cached interval remains unchanged."""
    if type(enabled) is not bool: raise TypeError('enabled must be builtin bool')
    _BERNSTEIN_METADATA.enabled=enabled

def _bernstein_eligible(c,lo,hi,k,scale):
    if type(c) is not tuple or len(c)>128 or type(k) is not int or not 0<=k<=3: return False
    if any(type(x) is not float or not math.isfinite(x) for x in (lo,hi,scale)): return False
    try: return all(type(x) in (int,float) and math.isfinite(x) for x in c)
    except OverflowError: return False

@lru_cache(maxsize=16384)
def poly_bounds(c,lo,hi,k=0,scale=1.):
    """Original arithmetic with bounded reusable coefficient-only metadata."""
    if not _bernstein_eligible(c,lo,hi,k,scale): return _poly_bounds_uncached(c,lo,hi,k,scale)
    try: metadata=_BERNSTEIN_METADATA.get(c,k)
    except (OverflowError,ValueError): return _poly_bounds_uncached(c,lo,hi,k,scale)
    if metadata is None: return _poly_bounds_uncached(c,lo,hi,k,scale)
    c,products,numerator,denominator,absolute_sum=metadata
    if not c: return (0.,0.)
    n=len(c)-1; width=hi-lo
    lo_powers=tuple(lo**p for p in range(n+1))
    width_powers=tuple(width**p for p in range(n+1))
    power=[sum(products[i][j-i]*lo_powers[j-i]*width_powers[i] for j in range(i,n+1)) for i in range(n+1)]
    bern=[sum(power[j]*numerator[i][j]/denominator[j] for j in range(i+1)) for i in range(n+1)]
    error=32*np.finfo(float).eps*absolute_sum*max(1.,abs(lo),abs(hi))**n
    return (math.nextafter((min(bern)-error)*scale,-math.inf), math.nextafter((max(bern)+error)*scale,math.inf))


def hermite(p0,v0,a0,p1,v1,a1,T):
    c0=p0; c1=v0*T; c2=a0*T*T/2
    d=p1-c0-c1-c2; e=v1*T-c1-2*c2; f=a1*T*T-2*c2
    return (c0,c1,c2,10*d-4*e+f/2,-15*d+7*e-f,6*d-3*e+f/2)

def final_height(height,T):
    """Nonincreasing altitude with the prescribed -.5m/s capture velocity.

    A nonnegative endpoint velocity basis plus a nonnegative beta bump avoids
    the altitude overshoot of a quintic when a long horizontal final is needed.
    """
    loss=height-15.; k=max(2,math.ceil(T/loss-2))
    remainder=max(0.,loss-T/(k+2)); coeff=[0.]*(max(5,k+2)+1); coeff[0]=height
    for i,value in ((3,-10*remainder),(4,15*remainder),(5,-6*remainder),(k+1,-.5*T),(k+2,.5*T*k/(k+2))): coeff[i]+=value
    return tuple(coeff)

class Interval:
    __slots__=('lo','hi')
    def __init__(self,lo,hi=None): self.lo=float(lo); self.hi=float(lo if hi is None else hi)
    @classmethod
    def of(cls,x): return x if isinstance(x,cls) else cls(x)
    def __add__(self,x):
        x=self.of(x); return Interval(math.nextafter(self.lo+x.lo,-math.inf),math.nextafter(self.hi+x.hi,math.inf))
    __radd__=__add__
    def __neg__(self): return Interval(-self.hi,-self.lo)
    def __sub__(self,x): return self+-self.of(x)
    def __rsub__(self,x): return self.of(x)+-self
    def __mul__(self,x):
        x=self.of(x); values=[self.lo*x.lo,self.lo*x.hi,self.hi*x.lo,self.hi*x.hi]
        return Interval(math.nextafter(min(values),-math.inf),math.nextafter(max(values),math.inf))
    __rmul__=__mul__
    def __truediv__(self,x):
        x=self.of(x)
        if x.lo<=0<=x.hi: return Interval(-math.inf,math.inf)
        return self*Interval(1/x.hi,1/x.lo)
    def square(self): return Interval(0. if self.lo<=0<=self.hi else math.nextafter(min(self.lo**2,self.hi**2),-math.inf),math.nextafter(max(self.lo**2,self.hi**2),math.inf))
    def sqrt(self): return Interval(math.nextafter(math.sqrt(max(0.,self.lo)),-math.inf),math.nextafter(math.sqrt(max(0.,self.hi)),math.inf))
    def sin(self):
        if self.hi-self.lo>=2*math.pi: return Interval(-1.,1.)
        values=[math.sin(self.lo),math.sin(self.hi)]
        for k in range(math.ceil((self.lo-math.pi/2)/math.pi),math.floor((self.hi-math.pi/2)/math.pi)+1): values.append((-1.)**k)
        return Interval(math.nextafter(min(values),-math.inf),math.nextafter(max(values),math.inf))
    def cos(self): return (self+math.pi/2).sin()
    def pair(self): return (self.lo,self.hi)

def _cartesian(r,th,z):
    co=th[0].cos() if isinstance(th[0],Interval) else math.cos(th[0])
    si=th[0].sin() if isinstance(th[0],Interval) else math.sin(th[0])
    rr=[r[0],r[1],r[2]-r[0]*th[1]*th[1],r[3]-3*r[1]*th[1]*th[1]-3*r[0]*th[1]*th[2]]
    tt=[0,r[0]*th[1],2*r[1]*th[1]+r[0]*th[2],3*r[2]*th[1]+3*r[1]*th[2]+r[0]*th[3]-r[0]*th[1]*th[1]*th[1]]
    return tuple((rr[k]*co-tt[k]*si,rr[k]*si+tt[k]*co,z[k]) for k in range(4))

def _cartesian_component(r,th,z,derivative=0):
    """Compute one Cartesian derivative from the unchanged read-only polar jet."""
    k=(0,1,2,3)[derivative]
    co=th[0].cos() if isinstance(th[0],Interval) else math.cos(th[0])
    si=th[0].sin() if isinstance(th[0],Interval) else math.sin(th[0])
    if k==0: rr=r[0]; tt=0
    elif k==1: rr=r[1]; tt=r[0]*th[1]
    elif k==2: rr=r[2]-r[0]*th[1]*th[1]; tt=2*r[1]*th[1]+r[0]*th[2]
    elif k==3: rr=r[3]-3*r[1]*th[1]*th[1]-3*r[0]*th[1]*th[2]; tt=3*r[2]*th[1]+3*r[1]*th[2]+r[0]*th[3]-r[0]*th[1]*th[1]*th[1]
    else: raise TypeError('Cartesian derivative must be an integer index')
    return (rr*co-tt*si,rr*si+tt*co,z[k])

def _intrinsic_bounds_from_polar(polar,derivative=1):
    """Derive rotation-invariant motion bounds from a read-only polar jet."""
    r,t,z=polar
    if derivative==1: values=(r[1],r[0]*t[1],z[1])
    elif derivative==2: values=(r[2]-r[0]*t[1]*t[1],2*r[1]*t[1]+r[0]*t[2],z[2])
    elif derivative==3: values=(r[3]-3*r[1]*t[1]*t[1]-3*r[0]*t[1]*t[2],3*r[2]*t[1]+3*r[1]*t[2]+r[0]*t[3]-r[0]*t[1]*t[1]*t[1],z[3])
    else: raise ValueError('intrinsic bounds support derivatives 1,2,3')
    return tuple(x.pair() for x in values)

@dataclass(frozen=True)
class Segment:
    start_time: float
    duration: float
    r: tuple
    theta: tuple
    z: tuple
    kind: str = 'POLAR'
    @property
    def end_time(self): return self.start_time+self.duration
    def __post_init__(self):
        for name in ('r','theta','z'): object.__setattr__(self,name,tuple(float(x) for x in getattr(self,name)))
        if self.duration<=0 or not math.isfinite(self.duration) or not math.isfinite(self.start_time) or not all(math.isfinite(x) for name in ('r','theta','z') for x in getattr(self,name)): raise ValueError('invalid segment')
    def state(self,t):
        u=(min(max(t,self.start_time),self.end_time)-self.start_time)/self.duration
        parts=[[poly(c,u,k)/self.duration**k for k in range(4)] for c in (self.r,self.theta,self.z)]
        return KinematicState(*_cartesian(*parts))
    def polar_bounds(self,start,end):
        lo=max(0.,(start-self.start_time)/self.duration); hi=min(1.,(end-self.start_time)/self.duration)
        return tuple(tuple(Interval(*poly_bounds(c,lo,hi,k,1/self.duration**k)) for k in range(4)) for c in (self.r,self.theta,self.z))
    def bounds(self,start,end,derivative=0): return tuple(x.pair() for x in _cartesian_component(*self.polar_bounds(start,end),derivative))
    def intrinsic_bounds(self,start,end,derivative=1):
        """Motion in the instantaneous orthonormal radial/tangential frame.

        Horizontal norms and power are rotation invariant in this no-wind,
        isotropic-drag model; separation still uses global Cartesian bounds.
        """
        return _intrinsic_bounds_from_polar(self.polar_bounds(start,end),derivative)
    def to_dict(self): return {'start_time':self.start_time,'duration':self.duration,'r':list(self.r),'theta':list(self.theta),'z':list(self.z),'kind':self.kind}

@dataclass(frozen=True)
class TrajectoryPlan:
    aircraft_id: str
    kind: str
    start_time: float
    end_time: float
    source_ring: object = None
    source_slot: object = None
    target_ring: object = None
    target_slot: object = None
    pad_id: object = None
    source_version: int = 0
    profile_hash: str = ''
    trajectory_hash: str = ''
    segments: tuple = ()
    config_hash: str = ''
    def __post_init__(self):
        object.__setattr__(self,'segments',tuple(self.segments))
        if not self.segments or not math.isfinite(self.start_time) or self.end_time<self.start_time: raise ValueError('invalid plan time/segments')
        if abs(self.segments[0].start_time-self.start_time)>1e-8 or abs(self.segments[-1].end_time-self.end_time)>1e-8: raise ValueError('segment times mismatch')
        expected=digest(self._payload())
        if self.trajectory_hash and self.trajectory_hash!=expected: raise ValueError('trajectory hash mismatch')
        object.__setattr__(self,'trajectory_hash',expected)
    def _payload(self):
        return {name:([s.to_dict() for s in self.segments] if name=='segments' else getattr(self,name)) for name in self.__dataclass_fields__ if name!='trajectory_hash'}
    def to_dict(self): return dict(self._payload(),trajectory_hash=self.trajectory_hash)
    @classmethod
    def from_dict(cls,data,config=None):
        d=dict(data); d['segments']=tuple(Segment(**s) for s in d['segments']); plan=cls(**d)
        if config is not None and plan.profile_hash != config.profile_hash: raise ValueError('profile hash mismatch')
        return plan
    def _segment(self,t):
        for s in self.segments:
            if t<=s.end_time: return s
        return self.segments[-1]
    def state(self,t):
        if t<self.start_time and self.source_ring is not None:
            return slot_state(self.source_ring,self.source_slot,t)
        if t<=self.end_time: return self._segment(t).state(t)
        if self.target_ring is not None: return slot_state(self.target_ring,self.target_slot,t)
        return KinematicState(self.segments[-1].state(self.end_time).p)
    def intervals(self,start,end):
        """Segments including physical continuation beyond the maneuver endpoint."""
        if start<self.start_time:
            if self.source_ring is not None:
                yield _stable_segment(self.source_ring,self.source_slot,start,min(end,self.start_time))
            else:
                raise ValueError('inbound state before admission is undefined')
        for s in self.segments:
            if s.start_time<end and s.end_time>start: yield s
        if end>self.end_time:
            begin=max(start,self.end_time)
            if self.target_ring is not None: yield _stable_segment(self.target_ring,self.target_slot,begin,end)
            else:
                p=self.state(self.end_time).p
                yield Segment(begin,end-begin,(math.hypot(*p[:2]),),(math.atan2(p[1],p[0]),),(p[2],),'GROUND')
    def bounds(self,start,end,derivative=0):
        values=[s.bounds(max(start,s.start_time),min(end,s.end_time),derivative) for s in self.intervals(start,end)]
        if not values: return tuple((x,x) for x in (self.state(start).p,self.state(start).v,self.state(start).a,self.state(start).jerk)[derivative])
        return tuple((min(v[i][0] for v in values),max(v[i][1] for v in values)) for i in range(3))

def _stable_segment(ring,slot,start,end,config=None):
    c=config or load_config(); r=ring_config(ring,c)
    return Segment(start,end-start,(r.radius_m,),(slot_phase(ring,slot,start,c),-c.airspace.common_angular_speed_radps*(end-start)),(r.height_agl_m,),'STABLE')

def make_stable_plan(aircraft_id,ring,slot,start_time=0.,end_time=240.,source_version=0,config=None):
    c=config or load_config(); s=_stable_segment(ring,slot,start_time,end_time,c)
    return TrajectoryPlan(str(aircraft_id),'STABLE',start_time,end_time,ring,slot,ring,slot,None,source_version,c.profile_hash,'',(s,),c.config_hash)

def make_ground_plan(aircraft_id,pad_id,start_time=0.,end_time=240.,config=None):
    c=config or load_config(); p=pad_position(pad_id,c)
    s=Segment(start_time,end_time-start_time,(c.pads.layout_radius_m,),(math.atan2(p[1],p[0]),),(0.,),'GROUND')
    return TrajectoryPlan(str(aircraft_id),'GROUND',start_time,end_time,None,None,None,None,pad_id,0,c.profile_hash,'',(s,),c.config_hash)

def descent_segments(pad_id,start_time,config=None):
    c=config or load_config(); p=pad_position(pad_id,c); theta=(math.atan2(p[1],p[0]),); r=(c.pads.layout_radius_m,)
    # z(u) is the analytic integral of the fixed smoothstep velocity.
    result=[]; t=start_time; z=15.
    for T,v0,dv in ((4.,-.5,-1.),(31/6,-1.5,0.),(13/3,-1.5,1.5)):
        coeff=(z,T*v0,0.,0.,T*dv*2.5,-T*dv*3.,T*dv)
        s=Segment(t,T,r,theta,coeff,'DESCENT'); result.append(s); z=poly(coeff,1); t+=T
    return tuple(result)

def _candidate(aircraft_id,kind,start,source_ring,source_slot,target_ring,target_slot,pad_id,version,segments,c):
    return TrajectoryPlan(str(aircraft_id),kind,start,segments[-1].end_time,source_ring,source_slot,target_ring,target_slot,pad_id,version,c.profile_hash,'',tuple(segments),c.config_hash)

_TRANSFER_FEASIBILITY = OrderedDict()
_TRANSFER_FEASIBILITY_LIMIT = 4096

def _transfer_endpoints_match(plan,c):
    # A cache entry proves only invariant single-aircraft numerics. Check the
    # actual generated epoch's endpoints before using it; large absolute clocks
    # can otherwise introduce position roundoff beyond metadata tolerances.
    for t,ring,slot in ((plan.start_time,plan.source_ring,plan.source_slot),
                        (plan.end_time,plan.target_ring,plan.target_slot)):
        actual=plan.state(t); expected=slot_state(ring,slot,t,c)
        for name,tol in (('p',1e-6),('v',1e-7),('a',1e-7)):
            if math.dist(getattr(actual,name),getattr(expected,name))>tol: return False
    return True

def _search(builder,durations,c,*,_transfer_cache_context=None):
    from .safety import validate_plan
    from .energy import _ConfigIdentity, _numeric_count
    failed=[]
    for T in (tuple(durations) if durations is not None else tuple(range(20,181,5))):
        if not math.isfinite(T) or T<=0: return CandidateResult('INVALID_INPUT',reason='duration must be positive finite')
        for branch,plan in enumerate(builder(float(T))):
            key=None
            if _transfer_cache_context is not None:
                _numeric_count('transfer_feasibility_calls')
                key=(_ConfigIdentity(c),c.config_hash,c.profile_hash,*_transfer_cache_context,float(T),branch,
                     tuple((s.duration,s.r,s.theta[1:],s.z,s.kind) for s in plan.segments))
                cached=_TRANSFER_FEASIBILITY.get(key)
                if cached is not None and _transfer_endpoints_match(plan,c):
                    _numeric_count('transfer_feasibility_hits')
                    accepted,reason,details=cached
                    if accepted:
                        return CandidateResult('FEASIBLE',plan,details=dict(details,trajectory_hash=plan.trajectory_hash,profile_hash=plan.profile_hash))
                    failed.append({'duration':T,'reason':reason})
                    continue
                _numeric_count('transfer_feasibility_computes')
            validation=validate_plan(plan,c)
            # Only this internal builder opts in. INVALID metadata failures can
            # depend on the absolute epoch and must never poison an invariant
            # entry. Rejected numeric details contain absolute witness times;
            # retain only the reason consumed by the duration search.
            if key is not None and validation.status!='INVALID':
                details={k:v for k,v in validation.details.items() if k not in ('trajectory_hash','profile_hash')} if validation.accepted else {}
                if key not in _TRANSFER_FEASIBILITY and len(_TRANSFER_FEASIBILITY)>=_TRANSFER_FEASIBILITY_LIMIT:
                    _TRANSFER_FEASIBILITY.popitem(last=False)
                _TRANSFER_FEASIBILITY[key]=(validation.accepted,validation.reason,details)
            if validation.accepted: return CandidateResult('FEASIBLE',plan,details=validation.details)
            failed.append({'duration':T,'reason':validation.reason})
    return CandidateResult('NOT_FOUND',reason='candidate duration/phase search exhausted; no infeasibility proof',details={'attempts':failed})

def generate_transfer(aircraft_id,source_ring,source_slot,target_slot,start_time,source_version=0,config=None,durations=None):
    c=config or load_config()
    try:
        targets={'outer':'middle','middle':'inner'}
        if source_ring not in targets: return CandidateResult('INVALID_INPUT',reason='normal transfers require adjacent inward rings')
        target=targets[source_ring]; a=ring_config(source_ring,c); b=ring_config(target,c)
        th0=slot_phase(source_ring,source_slot,start_time,c); slot_phase(target,target_slot,start_time,c)
        w=c.airspace.common_angular_speed_radps
        def builder(T):
            # The shared slot clock cancels exactly. Build local angular
            # coefficients independently of absolute epoch/rotation so the
            # single-aircraft certificate and energy tables are reusable.
            raw=slot_phase(target,target_slot,0.,c)-slot_phase(source_ring,source_slot,0.,c)
            # Nearest clockwise phase offset; also a neighboring unwrapped offset.
            delta=raw%(2*math.pi)
            for d in (delta if delta<=math.pi else delta-2*math.pi,delta-2*math.pi):
                s=Segment(start_time,T,hermite(a.radius_m,0,0,b.radius_m,0,0,T),(th0,-w*T,0.,10*d,-15*d,6*d),hermite(a.height_agl_m,0,0,b.height_agl_m,0,0,T))
                yield _candidate(aircraft_id,'TRANSFER',start_time,source_ring,source_slot,target,target_slot,None,source_version,(s,),c)
        return _search(builder,durations,c,_transfer_cache_context=(source_ring,source_slot,target,target_slot))
    except (ValueError,TypeError,IndexError) as exc: return CandidateResult('INVALID_INPUT',reason=str(exc))

def generate_inbound(aircraft_id,gate_id,target_slot,start_time,source_version=0,config=None,durations=None):
    c=config or load_config()
    try:
        if not isinstance(gate_id,int) or not 0<=gate_id<4: raise ValueError('invalid gate')
        th0=c.entry.gate_angles_rad[gate_id]; r0=c.entry.radius_m; speed=c.entry.initial_horizontal_speed_mps
        rd=-speed/math.sqrt(2); td=-speed/math.sqrt(2)/r0; rdd=r0*td*td; tdd=-2*rd*td/r0
        slot_phase('outer',target_slot,start_time,c); w=c.airspace.common_angular_speed_radps
        def builder(T):
            raw=slot_phase('outer',target_slot,start_time+T,c)
            end=th0+(raw-th0)%(2*math.pi)-2*math.pi
            for theta_end in (end,end+2*math.pi):
                s=Segment(start_time,T,hermite(r0,rd,rdd,600.,0,0,T),hermite(th0,td,tdd,theta_end,-w,0,T),(110.,))
                yield _candidate(aircraft_id,'INBOUND',start_time,None,None,'outer',target_slot,None,source_version,(s,),c)
        return _search(builder,durations,c)
    except (ValueError,TypeError,IndexError) as exc: return CandidateResult('INVALID_INPUT',reason=str(exc))

def generate_final(aircraft_id,source_ring,source_slot,pad_id,start_time,emergency=False,source_version=0,config=None,durations=None):
    c=config or load_config()
    try:
        if not emergency and source_ring!='inner': return CandidateResult('INVALID_INPUT',reason='normal final requires inner ring')
        r=ring_config(source_ring,c); th0=slot_phase(source_ring,source_slot,start_time,c); p=pad_position(pad_id,c); w=c.airspace.common_angular_speed_radps
        target=math.atan2(p[1],p[0]); delta=(target-th0)%(2*math.pi)-2*math.pi
        def builder(T):
            # One clockwise passage only; no resource-waiting loops.
            s=Segment(start_time,T,hermite(r.radius_m,0,0,c.pads.layout_radius_m,0,0,T),hermite(th0,-w,0,th0+delta,0,0,T),final_height(r.height_agl_m,T))
            yield _candidate(aircraft_id,'FINAL_EMERGENCY' if emergency else 'FINAL_NORMAL',start_time,source_ring,source_slot,None,None,pad_id,source_version,(s,)+descent_segments(pad_id,start_time+T,c),c)
        # 20..180 is the normative initial fixture grid, not a physical bound.
        search_durations=tuple(range(20,181,5))+tuple(range(185,301,5)) if durations is None else durations
        return _search(builder,search_durations,c)
    except (ValueError,TypeError,IndexError) as exc: return CandidateResult('INVALID_INPUT',reason=str(exc))
