"""Immutable analytic polar trajectories and deterministic candidate construction.

Coefficients are in normalized local time. Certificates operate on polynomial
enclosures and trigonometric intervals, never on the execution sample grid.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import numpy as np
from .config import load_config, digest
from .geometry import slot_phase, slot_state, ring_config, pad_position
from .models import KinematicState, CandidateResult

def poly(c,u,k=0):
    c=tuple(c)
    for _ in range(k): c=tuple(i*c[i] for i in range(1,len(c)))
    return sum(x*u**i for i,x in enumerate(c))

@lru_cache(maxsize=16384)
def poly_bounds(c,lo,hi,k=0,scale=1.):
    """Power-to-Bernstein hull on the requested subinterval (outward padded)."""
    c=tuple(c)
    for _ in range(k): c=tuple(i*c[i] for i in range(1,len(c)))
    if not c: return (0.,0.)
    n=len(c)-1; width=hi-lo
    power=[sum(c[j]*math.comb(j,i)*lo**(j-i)*width**i for j in range(i,n+1)) for i in range(n+1)]
    bern=[sum(power[j]*math.comb(i,j)/math.comb(n,j) for j in range(i+1)) for i in range(n+1)]
    error=32*np.finfo(float).eps*sum(abs(x) for x in c)*max(1.,abs(lo),abs(hi))**n
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
    def bounds(self,start,end,derivative=0): return tuple(x.pair() for x in _cartesian(*self.polar_bounds(start,end))[derivative])
    def intrinsic_bounds(self,start,end,derivative=1):
        """Motion in the instantaneous orthonormal radial/tangential frame.

        Horizontal norms and power are rotation invariant in this no-wind,
        isotropic-drag model; separation still uses global Cartesian bounds.
        """
        r,t,z=self.polar_bounds(start,end)
        if derivative==1: values=(r[1],r[0]*t[1],z[1])
        elif derivative==2: values=(r[2]-r[0]*t[1]*t[1],2*r[1]*t[1]+r[0]*t[2],z[2])
        elif derivative==3: values=(r[3]-3*r[1]*t[1]*t[1]-3*r[0]*t[1]*t[2],3*r[2]*t[1]+3*r[1]*t[2]+r[0]*t[3]-r[0]*t[1]*t[1]*t[1],z[3])
        else: raise ValueError('intrinsic bounds support derivatives 1,2,3')
        return tuple(x.pair() for x in values)
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

def _search(builder,durations,c):
    from .safety import validate_plan
    failed=[]
    for T in (tuple(durations) if durations is not None else tuple(range(20,181,5))):
        if not math.isfinite(T) or T<=0: return CandidateResult('INVALID_INPUT',reason='duration must be positive finite')
        for plan in builder(float(T)):
            validation=validate_plan(plan,c)
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
            raw=slot_phase(target,target_slot,start_time+T,c)-(th0-w*T)
            # Nearest clockwise phase offset; also a neighboring unwrapped offset.
            delta=raw%(2*math.pi)
            for d in (delta if delta<=math.pi else delta-2*math.pi,delta-2*math.pi):
                s=Segment(start_time,T,hermite(a.radius_m,0,0,b.radius_m,0,0,T),hermite(th0,-w,0,th0-w*T+d,-w,0,T),hermite(a.height_agl_m,0,0,b.height_agl_m,0,0,T))
                yield _candidate(aircraft_id,'TRANSFER',start_time,source_ring,source_slot,target,target_slot,None,source_version,(s,),c)
        return _search(builder,durations,c)
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
