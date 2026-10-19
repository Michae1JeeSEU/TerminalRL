"""Appendix A research surrogate; charge-fraction SOC and shared quadrature."""
from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache, wraps
import math
import sys
import bisect
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from .config import load_config, Config
from .trajectory import Interval, Segment, _intrinsic_bounds_from_polar

@dataclass(frozen=True)
class VehicleProfile:
    profile_id: str = 'eh216s_reference_pm16_v1_0'
    mass_kg: float = 620.
    rotor_count: int = 16
    rotor_diameter_m: float = 1.575

@dataclass(frozen=True)
class EnergyProfile:
    profile_id: str = 'eh216s_lit_proxy_v1_0'
    capacity_ah: float = 252.
    voltage_v: float = 85.
    @property
    def capacity_j(self): return self.capacity_ah*self.voltage_v*3600

@dataclass(frozen=True)
class PowerResult:
    battery_w: float
    shaft_w: float
    current_a: float
    status: str = 'VALID'
    reason: str = ''
    induced_velocity_mps: float = 0.
    induced_relative_residual: float = 0.
    axial_descent_ratio: float = 0.
    @property
    def valid(self): return self.status=='VALID'

@dataclass(frozen=True)
class EnergyResult:
    joules: float
    soc_used: float
    peak_seconds: float
    valid: bool
    status: str = 'VALID'
    reason: str = ''
    error_j: float = 0.

# Pure scalar acceleration state: never part of a world or checkpoint.
_ROOT_CACHE_POLICY = 'scalar-induced-positive-branch-tolerance-v2'
_ROOT_CACHE_MAX_ENTRIES = 65536
_ROOT_CACHE_MAX_BYTES = 32 * 1024 * 1024
_ROOT_CACHE = OrderedDict()
_ROOT_CACHE_PAYLOAD_BYTES = 0
_ROOT_CACHE_ENABLED = True
_ROOT_CACHE_REQUESTS = _ROOT_CACHE_COMPUTES = _ROOT_CACHE_HITS = _ROOT_CACHE_EVICTIONS = 0

_PRACTICAL_STATS = {}

class _ConfigIdentity:
    """Retain the immutable config as part of a bounded cache's identity key."""
    __slots__=('config',)
    def __init__(self,config): self.config=config
    def __hash__(self): return id(self.config)
    def __eq__(self,other): return type(other) is _ConfigIdentity and self.config is other.config

def _numeric_count(name, value=1):
    _PRACTICAL_STATS[name] = _PRACTICAL_STATS.get(name, 0) + value

def practical_numeric_stats():
    """Process-local operation counts and actual precomputation wall seconds."""
    return dict(_PRACTICAL_STATS)

def reset_practical_numeric_stats():
    """Reset observations only; reusable derived numerical data remain warm."""
    _PRACTICAL_STATS.clear()


def root_cache_clear():
    """Worker/cold reset of this derived cache and its performance counters."""
    global _ROOT_CACHE_PAYLOAD_BYTES
    global _ROOT_CACHE_REQUESTS, _ROOT_CACHE_COMPUTES, _ROOT_CACHE_HITS, _ROOT_CACHE_EVICTIONS
    _ROOT_CACHE.clear()
    _root_bracket.cache_clear()
    _ROOT_CACHE_PAYLOAD_BYTES = 0
    _ROOT_CACHE_REQUESTS = _ROOT_CACHE_COMPUTES = _ROOT_CACHE_HITS = _ROOT_CACHE_EVICTIONS = 0


def root_cache_set_enabled(enabled):
    """Disable lookup/insertion without changing the scalar evaluator or policy."""
    global _ROOT_CACHE_ENABLED
    if type(enabled) is not bool:
        raise TypeError('root cache enabled flag must be bool')
    _ROOT_CACHE_ENABLED = enabled


def root_cache_stats():
    """Detached performance data; bytes conservatively count shared key objects."""
    return {'enabled': _ROOT_CACHE_ENABLED, 'logical_requests': _ROOT_CACHE_REQUESTS,
            'actual_computes': _ROOT_CACHE_COMPUTES, 'hits': _ROOT_CACHE_HITS,
            'evictions': _ROOT_CACHE_EVICTIONS, 'entries': len(_ROOT_CACHE),
            'estimated_bytes': sys.getsizeof(_ROOT_CACHE) + _ROOT_CACHE_PAYLOAD_BYTES,
            'max_entries': _ROOT_CACHE_MAX_ENTRIES, 'max_bytes': _ROOT_CACHE_MAX_BYTES,
            'policy': _ROOT_CACHE_POLICY, 'eviction_policy': 'FIFO'}


def _root_entry_bytes(key, value):
    # Keys contain only a tuple of immutable str/int fields; values float/None.
    return sys.getsizeof(key) + sum(sys.getsizeof(item) for item in key) + sys.getsizeof(value)


def _memoize_root(function):
    @wraps(function)
    def cached(vp, vn, rhs, bisections=80):
        global _ROOT_CACHE_PAYLOAD_BYTES
        global _ROOT_CACHE_REQUESTS, _ROOT_CACHE_COMPUTES, _ROOT_CACHE_HITS, _ROOT_CACHE_EVICTIONS
        _ROOT_CACHE_REQUESTS += 1
        eligible = (_ROOT_CACHE_ENABLED and type(bisections) is int and bisections > 0
                    and all(type(x) is float and math.isfinite(x) for x in (vp, vn, rhs)))
        key = ((_ROOT_CACHE_POLICY, vp.hex(), vn.hex(), rhs.hex(), bisections)
               if eligible else None)
        if eligible and key in _ROOT_CACHE:
            _ROOT_CACHE_HITS += 1
            return _ROOT_CACHE[key]
        _ROOT_CACHE_COMPUTES += 1
        result = function(vp, vn, rhs, bisections)
        if eligible and (result is None or type(result) is float):
            cost = _root_entry_bytes(key, result)
            # A result that cannot fit even alone is always recomputed.
            if _ROOT_CACHE_MAX_ENTRIES > 0 and cost + sys.getsizeof(OrderedDict()) <= _ROOT_CACHE_MAX_BYTES:
                while len(_ROOT_CACHE) >= _ROOT_CACHE_MAX_ENTRIES:
                    oldest, value = _ROOT_CACHE.popitem(last=False)
                    _ROOT_CACHE_PAYLOAD_BYTES -= _root_entry_bytes(oldest, value)
                    _ROOT_CACHE_EVICTIONS += 1
                _ROOT_CACHE[key] = result
                _ROOT_CACHE_PAYLOAD_BYTES += cost
                # Include dict allocation after growth, not an average per entry.
                while (_ROOT_CACHE and sys.getsizeof(_ROOT_CACHE) + _ROOT_CACHE_PAYLOAD_BYTES
                       > _ROOT_CACHE_MAX_BYTES):
                    oldest, value = _ROOT_CACHE.popitem(last=False)
                    _ROOT_CACHE_PAYLOAD_BYTES -= _root_entry_bytes(oldest, value)
                    _ROOT_CACHE_EVICTIONS += 1
                if not _ROOT_CACHE:
                    _ROOT_CACHE.clear()  # Release the now-empty dict allocation.
        return result
    return cached


@lru_cache(maxsize=65536)
def _root_bracket(vp,vn,rhs,bisections=80):
    """Certified endpoints on the monotone vi >= max(0,-vn) branch.

    Stop well inside the frozen 1e-8 residual requirement. Interval consumers
    use these endpoints, never an arbitrary padding around an approximate root.
    """
    _numeric_count('root_bracket_computes')
    lo=max(0.,-vn); hi=max(1.,lo+1.,math.sqrt(rhs)*2+abs(vn))
    f=lambda vi:vi*math.hypot(vp,vn+vi)-rhs
    if f(lo)>max(1.,rhs)*1e-12: return None
    while f(hi)<0: hi*=2
    for _ in range(bisections):
        _numeric_count('root_iterations')
        mid=(lo+hi)/2
        if f(mid)>0: hi=mid
        else: lo=mid
        if hi-lo<=1e-11*max(1.,mid) and abs(f(mid))<=1e-11*max(1.,rhs): break
    # Certify the final signs using outward interval arithmetic: rounded f
    # evaluations used for iteration alone are insufficient for root bounds.
    def enclosure(vi):
        return Interval(vi)*(Interval(vp).square()+(Interval(vn)+vi).square()).sqrt()-rhs
    branch=max(0.,-vn)
    while lo>branch and enclosure(lo).hi>0:
        lo=max(branch,math.nextafter(lo,-math.inf))
    while enclosure(hi).lo<0:
        hi=math.nextafter(hi,math.inf)
    return math.nextafter(lo,-math.inf),math.nextafter(hi,math.inf)

@_memoize_root
def _root(vp,vn,rhs,bisections=80):
    bracket=_root_bracket(vp,vn,rhs,bisections)
    return None if bracket is None else (bracket[0]+bracket[1])/2

_root_uncached = _root.__wrapped__

def evaluate_power(v,a,config=None):
    _numeric_count('power_calls')
    c=config or load_config(); e=c.energy; m=c.vehicle.fixed_flight_mass_kg; rho=e.air_density_kgpm3
    try:
        vx,vy,vz=map(float,v); ax,ay,az=map(float,a)
        if not all(math.isfinite(x) for x in (vx,vy,vz,ax,ay,az)): raise ValueError('nonfinite motion')
        vh=math.hypot(vx,vy)
        thrust=(m*ax+.5*rho*e.effective_horizontal_drag_area_m2*vh*vx,m*ay+.5*rho*e.effective_horizontal_drag_area_m2*vh*vy,m*(az+c.coordinates.gravity_mps2)+.5*rho*e.effective_vertical_drag_area_m2*abs(vz)*vz)
        T=math.sqrt(sum(x*x for x in thrust))
        if T<=0 or thrust[2]<=0: raise ValueError('nonpositive upward thrust')
        dot=sum(x*y for x,y in zip(thrust,(vx,vy,vz))); vn=dot/T; vp=math.sqrt(max(0.,vx*vx+vy*vy+vz*vz-vn*vn))
        area=math.pi*(c.vehicle.rotor_diameter_m/2)**2; rhs=T/(c.vehicle.rotor_count*2*rho*area); ratio=vn/math.sqrt(2*rhs)
        if ratio<e.minimum_axial_descent_ratio: raise ValueError('axial descent ratio outside shallow-descent domain')
        vi=_root(vp,vn,rhs,e.induced_root_bisections)
        if vi is None: raise ValueError('no root on positive-power induced branch')
        residual=abs(vi*math.hypot(vp,vn+vi)-rhs)/max(1.,rhs)
        if residual>e.induced_root_relative_residual_tolerance: raise ValueError('induced root did not converge')
        pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
        shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
        if shaft<=0 or battery<=0: raise ValueError('nonpositive power; clipping is prohibited')
        if battery/e.equivalent_voltage_V>e.reference_max_battery_current_A or shaft>c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000:
            return PowerResult(battery,shaft,battery/e.equivalent_voltage_V,'POWER_LIMIT','battery current or shaft power exceeds frozen limit',vi,residual,ratio)
        return PowerResult(battery,shaft,battery/e.equivalent_voltage_V,'VALID','',vi,residual,ratio)
    except (ValueError,OverflowError) as exc:
        return PowerResult(math.nan,math.nan,math.nan,'ENERGY_MODEL_DOMAIN',str(exc))

def _norm(values): return sum((v.square() for v in values),Interval(0)).sqrt()

def power_bounds(v,a,config=None):
    """Conservative interval extension of Appendix A, including induced root.

    On vi >= max(0,-Vn), vi*hypot(Vp,Vn+vi) is monotone in vi.
    Its bounding roots therefore enclose all instantaneous physical roots.
    """
    _numeric_count('power_bounds_calls')
    c=config or load_config(); e=c.energy; rho=e.air_density_kgpm3; m=c.vehicle.fixed_flight_mass_kg
    v=tuple(Interval(*x) if not isinstance(x,Interval) else x for x in v); a=tuple(Interval(*x) if not isinstance(x,Interval) else x for x in a)
    vh=_norm(v[:2]); vz_abs=Interval(0. if v[2].lo<=0<=v[2].hi else min(abs(v[2].lo),abs(v[2].hi)),max(abs(v[2].lo),abs(v[2].hi)))
    thrust=(m*a[0]+.5*rho*e.effective_horizontal_drag_area_m2*vh*v[0],m*a[1]+.5*rho*e.effective_horizontal_drag_area_m2*vh*v[1],m*(a[2]+c.coordinates.gravity_mps2)+.5*rho*e.effective_vertical_drag_area_m2*vz_abs*v[2])
    T=_norm(thrust)
    if T.lo<=0 or thrust[2].lo<=0: return None
    dot=sum((x*y for x,y in zip(thrust,v)),Interval(0)); vn=dot/T
    vp=(_norm(v).square()-vn.square()).sqrt()
    area=math.pi*(c.vehicle.rotor_diameter_m/2)**2; rhs=T/(c.vehicle.rotor_count*2*rho*area)
    ratio=vn/(2*rhs).sqrt()
    # The lower bound root may lie off the selected branch for the enclosing
    # box. Zero is then a conservative lower bound, never an accepted root.
    lower=_root_bracket(vp.hi,vn.hi,rhs.lo)
    upper=_root_bracket(vp.lo,vn.lo,rhs.hi)
    vi=Interval(max(0.,lower[0]) if lower is not None else 0.,
                upper[1] if upper is not None else math.nextafter(max(0.,-vn.lo),math.inf))
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
    return {'battery_w':battery.pair(),'shaft_w':shaft.pair(),'current_a':(battery/e.equivalent_voltage_V).pair(),'axial_descent_ratio':ratio.pair(),'positive_upward_thrust':True}

def _power_array(s,t,c):
    """Vector evaluation of exactly the same scalar Appendix A equations."""
    _numeric_count('power_array_calls')
    u=(t-s.start_time)/s.duration
    values=[]
    for coeff in (s.r,s.theta,s.z):
        cc=np.asarray(coeff); derivatives=[]
        for k in range(3):
            derivatives.append((np.polynomial.polynomial.polyval(u,cc) if len(cc) else np.zeros_like(u))/s.duration**k)
            cc=np.arange(1,len(cc))*cc[1:]
        values.append(derivatives)
    r,th,z=values; co=np.cos(th[0]); si=np.sin(th[0])
    vx=r[1]*co-r[0]*th[1]*si; vy=r[1]*si+r[0]*th[1]*co; vz=z[1]
    radial=r[2]-r[0]*th[1]**2; tangent=2*r[1]*th[1]+r[0]*th[2]
    ax=radial*co-tangent*si; ay=radial*si+tangent*co; az=z[2]
    e=c.energy; rho=e.air_density_kgpm3; m=c.vehicle.fixed_flight_mass_kg; vh=np.hypot(vx,vy)
    tx=m*ax+.5*rho*e.effective_horizontal_drag_area_m2*vh*vx; ty=m*ay+.5*rho*e.effective_horizontal_drag_area_m2*vh*vy
    tz=m*(az+c.coordinates.gravity_mps2)+.5*rho*e.effective_vertical_drag_area_m2*np.abs(vz)*vz
    thrust=np.sqrt(tx*tx+ty*ty+tz*tz); work=tx*vx+ty*vy+tz*vz
    if np.any(thrust<=0) or np.any(tz<=0): raise ValueError('nonpositive upward thrust')
    vn=work/thrust; vp=np.sqrt(np.maximum(0.,vx*vx+vy*vy+vz*vz-vn*vn)); area=math.pi*(c.vehicle.rotor_diameter_m/2)**2
    rhs=thrust/(c.vehicle.rotor_count*2*rho*area)
    if np.any(vn/np.sqrt(2*rhs)<e.minimum_axial_descent_ratio): raise ValueError('axial descent ratio outside model domain')
    left=np.maximum(0.,-vn); right=np.maximum.reduce((np.ones_like(left),left+1,2*np.sqrt(rhs)+np.abs(vn)))
    if np.any(left*np.hypot(vp,vn+left)-rhs>np.maximum(1.,rhs)*1e-12): raise ValueError('induced root branch unavailable')
    for _ in range(e.induced_root_bisections):
        _numeric_count('vector_root_iterations')
        mid=(left+right)/2; positive=mid*np.hypot(vp,vn+mid)>rhs
        right=np.where(positive,mid,right); left=np.where(positive,left,mid)
        if np.all((right-left<=1e-11*np.maximum(1.,mid)) &
                  (np.abs(mid*np.hypot(vp,vn+mid)-rhs)<=1e-11*np.maximum(1.,rhs))): break
    vi=(left+right)/2
    if np.any(np.abs(vi*np.hypot(vp,vn+vi)-rhs)>e.induced_root_relative_residual_tolerance*np.maximum(1.,rhs)):
        raise ValueError('induced root did not converge')
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*thrust*vi+work+pp
    if np.any(shaft<=0): raise ValueError('nonpositive power')
    if np.any(shaft>c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000) or np.any(shaft/e.motor_drive_efficiency/e.equivalent_voltage_V>e.reference_max_battery_current_A):
        raise ValueError('power exceeds frozen limit')
    return shaft/e.motor_drive_efficiency

@lru_cache(maxsize=512)
def _integrable_plan(plan):
    # Shared approval cache prevents execution from integrating an unvalidated
    # rehashed trajectory or from bypassing limits with a STABLE/GROUND label.
    from .safety import validate_plan
    return validate_plan(plan,load_config())

_GAUSS={n:np.polynomial.legendre.leggauss(n) for n in (4,8)}

def _quadrature_cells(s,left,right,c,depth=0):
    mid=(left+right)/2; half=(right-left)/2; estimates=[]
    for n in (4,8):
        nodes,weights=_GAUSS[n]
        estimates.append(half*np.sum(_power_array(s,(mid[:,None]+half[:,None]*nodes).ravel(),c).reshape(len(left),n)*weights,axis=1))
    error=np.abs(estimates[1]-estimates[0]); bad=error>np.maximum(1e-5,2e-10*np.abs(estimates[1]))
    if np.any(bad):
        if depth>=12: raise ValueError('energy quadrature failed to converge')
        lval,lerr=_quadrature_cells(s,left[bad],mid[bad],c,depth+1); rval,rerr=_quadrature_cells(s,mid[bad],right[bad],c,depth+1)
        estimates[1][bad]=lval+rval; error[bad]=lerr+rerr
    return estimates[1],error

@lru_cache(maxsize=256)
def _segment_table(duration,r,theta,z,max_dt,profile_hash):
    # Isotropic drag/gravity make power invariant under absolute time and
    # horizontal rotation. The cache key deliberately omits theta's constant.
    began=time.perf_counter(); _numeric_count('energy_table_computes')
    s=Segment(0.,duration,r,(0.,)+theta[1:],z)
    n=max(1,math.ceil(duration/max_dt)); knots=np.linspace(0.,duration,n+1)
    values,errors=_quadrature_cells(s,knots[:-1],knots[1:],load_config())
    _numeric_count('precompute_seconds',time.perf_counter()-began)
    return knots,np.concatenate(([0.],np.cumsum(values))),np.concatenate(([0.],np.cumsum(errors)))

@lru_cache(maxsize=8192)
def _segment_primitive(duration,r,theta,z,max_dt,profile_hash,t):
    _numeric_count('energy_primitive_computes')
    knots,energy,errors=_segment_table(duration,r,theta,z,max_dt,profile_hash)
    i=min(len(knots)-1,bisect.bisect_right(knots,t)-1)
    value=float(energy[i]); error=float(errors[i])
    if t>knots[i]:
        s=Segment(0.,duration,r,theta,z)
        v,e=_quadrature_cells(s,np.asarray([knots[i]]),np.asarray([t]),load_config())
        value+=float(v[0]); error+=float(e[0])
    return value,error

def _segment_joules(s,lo,hi,max_dt,c):
    key=(s.duration,s.r,(0.,)+s.theta[1:],s.z,max_dt,c.profile_hash)
    _numeric_count('energy_primitive_calls',2)
    a,ea=_segment_primitive(*key,max(0.,lo-s.start_time))
    b,eb=_segment_primitive(*key,min(s.duration,hi-s.start_time))
    return b-a,ea+eb

@lru_cache(maxsize=128)
def _stable_power_cached(radius,angular_speed,config_key):
    c=config_key.config
    _numeric_count('stable_power_computes')
    return evaluate_power((0.,radius*angular_speed,0.),(-radius*angular_speed**2,0.,0.),c)

def stable_power(ring,config=None):
    """Power for a canonical slot; invariant under slot phase and absolute time."""
    from .geometry import ring_config
    c=config or load_config(); _numeric_count('stable_power_calls')
    return _stable_power_cached(ring_config(ring,c).radius_m,-c.airspace.common_angular_speed_radps,_ConfigIdentity(c))

@lru_cache(maxsize=256)
def _peak_table(duration,r,theta,z,threshold,config_key):
    """Cumulative upper measure, reusing the same threshold crossing enclosure.

    Unresolved leaves count wholly as peak, as in _peak_measure. Subinterval
    queries intersect this fixed coverage, so overlapping queries are additive.
    """
    c=config_key.config
    began=time.perf_counter(); _numeric_count('peak_table_computes')
    s=Segment(0.,duration,r,theta,z); active=[]
    def visit(a,b,depth):
        _numeric_count('peak_table_nodes')
        polar=s.polar_bounds(a,b)
        bounds=power_bounds(_intrinsic_bounds_from_polar(polar,1),_intrinsic_bounds_from_polar(polar,2),c)
        if bounds is not None:
            low,high=bounds['shaft_w']
            if high<=threshold: return
            above=low>threshold
        else: above=False
        if above or b-a<1e-7 or depth>=26:
            if active and active[-1][1]==a: active[-1]=(active[-1][0],b)
            else: active.append((a,b))
            return
        mid=(a+b)/2; visit(a,mid,depth+1); visit(mid,b,depth+1)
    visit(0.,duration,0)
    starts=tuple(a for a,b in active); ends=tuple(b for a,b in active)
    prefix=(0.,)+tuple(np.cumsum([b-a for a,b in active]))
    _numeric_count('precompute_seconds',time.perf_counter()-began)
    return starts,ends,prefix

def _segment_peak(s,lo,hi,threshold,c):
    _numeric_count('peak_primitive_calls',2)
    starts,ends,prefix=_peak_table(s.duration,s.r,(0.,)+s.theta[1:],s.z,threshold,_ConfigIdentity(c))
    def primitive(t):
        i=bisect.bisect_right(ends,t)
        return prefix[i]+(max(0.,t-starts[i]) if i<len(starts) else 0.)
    return max(0.,primitive(hi-s.start_time)-primitive(lo-s.start_time))

def integrate_energy(plan,start,end,max_dt=.2,config=None):
    _numeric_count('integrate_energy_calls')
    c=config or load_config()
    if not 0<max_dt<=.2: raise ValueError('energy maximum segment must be in (0,0.2]')
    if not math.isfinite(start) or not math.isfinite(end) or end<start: raise ValueError('invalid integration interval')
    from .trajectory import TrajectoryPlan
    if isinstance(plan,TrajectoryPlan):
        validity=_integrable_plan(plan)
        if not validity.accepted: return EnergyResult(0.,0.,0.,False,validity.status,validity.reason)
    total=error=peak=0.
    threshold=c.vehicle.rotor_count*c.energy.reference_motor_continuous_power_kW_each*1000
    try:
        for s in plan.intervals(start,end):
            lo=max(start,s.start_time); hi=min(end,s.end_time)
            if s.kind=='GROUND': continue
            if s.kind=='STABLE':
                if len(s.r)==1 and len(s.z)==1 and len(s.theta)==2:
                    _numeric_count('stable_power_calls')
                    # TrajectoryPlan validation establishes the canonical clock;
                    # avoid duration division reintroducing epoch-dependent ulps.
                    omega=(-c.airspace.common_angular_speed_radps if isinstance(plan,TrajectoryPlan)
                           else s.theta[1]/s.duration)
                    result=_stable_power_cached(s.r[0],omega,_ConfigIdentity(c))
                else:
                    st=s.state((lo+hi)/2); result=evaluate_power(st.v,st.a,c)
                if not result.valid: return EnergyResult(total,total/(85*252*3600),peak,False,result.status,result.reason,error)
                total+=result.battery_w*(hi-lo); peak+=(hi-lo) if result.shaft_w>threshold else 0.
                continue
            value,err=_segment_joules(s,lo,hi,max_dt,c); total+=value; error+=err
            peak+=_segment_peak(s,lo,hi,threshold,c)
        return EnergyResult(total,total/(c.energy.equivalent_voltage_V*c.energy.reference_capacity_Ah*3600),peak,True,error_j=error)
    except ValueError as exc: return EnergyResult(total,total/(85*252*3600),peak,False,'ENERGY_MODEL_DOMAIN',str(exc),error)

# Exact duplicate top-level peak queries only; recursive nodes are not cached.
_PEAK_CACHE_POLICY = 'peak-r0-order-depth26-width1e-7-v1'
_PEAK_CACHE_MAX_ENTRIES = 4096
_PEAK_CACHE_MAX_BYTES = 16 * 1024 * 1024
_PEAK_CACHE = OrderedDict()
_PEAK_CACHE_PAYLOAD_BYTES = 0
_PEAK_CACHE_ENABLED = False
_PEAK_CACHE_REQUESTS = _PEAK_CACHE_COMPUTES = _PEAK_CACHE_HITS = _PEAK_CACHE_EVICTIONS = 0
_PEAK_CACHE_NON_TOP_COMPUTES = 0


def peak_cache_clear():
    """Discard only derived peak answers and their performance counters."""
    global _PEAK_CACHE_PAYLOAD_BYTES, _PEAK_CACHE_NON_TOP_COMPUTES
    global _PEAK_CACHE_REQUESTS, _PEAK_CACHE_COMPUTES, _PEAK_CACHE_HITS, _PEAK_CACHE_EVICTIONS
    _PEAK_CACHE.clear()
    _PEAK_CACHE_PAYLOAD_BYTES = _PEAK_CACHE_NON_TOP_COMPUTES = 0
    _PEAK_CACHE_REQUESTS = _PEAK_CACHE_COMPUTES = _PEAK_CACHE_HITS = _PEAK_CACHE_EVICTIONS = 0


def peak_cache_set_enabled(enabled):
    global _PEAK_CACHE_ENABLED
    if type(enabled) is not bool:
        raise TypeError('peak cache enabled flag must be bool')
    _PEAK_CACHE_ENABLED = enabled


def peak_cache_stats():
    """Detached counters; actual_computes counts top-level original evaluations."""
    return {'enabled': _PEAK_CACHE_ENABLED, 'logical_requests': _PEAK_CACHE_REQUESTS,
            'actual_computes': _PEAK_CACHE_COMPUTES, 'hits': _PEAK_CACHE_HITS,
            'non_top_level_computes': _PEAK_CACHE_NON_TOP_COMPUTES,
            'actual_node_computes': _PEAK_CACHE_COMPUTES + _PEAK_CACHE_NON_TOP_COMPUTES,
            'evictions': _PEAK_CACHE_EVICTIONS, 'entries': len(_PEAK_CACHE),
            'estimated_bytes': sys.getsizeof(_PEAK_CACHE) + _PEAK_CACHE_PAYLOAD_BYTES,
            'max_entries': _PEAK_CACHE_MAX_ENTRIES, 'max_bytes': _PEAK_CACHE_MAX_BYTES,
            'policy': _PEAK_CACHE_POLICY, 'eviction_policy': 'FIFO'}


def _peak_exact_key(value):
    """Immutable, type-tagged values; no rounding, sorting or identity shortcut."""
    if value is None: return ('none',)
    if type(value) is bool: return ('bool', value)
    if type(value) is int: return ('int', value)
    if type(value) is float and math.isfinite(value): return ('float.hex', value.hex())
    if type(value) is str: return ('str', value)
    if type(value) in (tuple, list):
        return ('tuple' if type(value) is tuple else 'list', tuple(_peak_exact_key(x) for x in value))
    if type(value) is dict:
        return ('dict', tuple((_peak_exact_key(k), _peak_exact_key(v)) for k, v in value.items()))
    raise TypeError('unsupported exact peak cache key value')


def _peak_owned_bytes(value):
    # The key contains only immutable tuples/scalars. Counting shared objects
    # repeatedly is conservative relative to recursive deduplicated accounting.
    return sys.getsizeof(value) + (sum(_peak_owned_bytes(x) for x in value) if type(value) is tuple else 0)


def _memoize_peak(function):
    @wraps(function)
    def cached(s, a, b, threshold, c, depth=0):
        global _PEAK_CACHE_PAYLOAD_BYTES, _PEAK_CACHE_NON_TOP_COMPUTES
        global _PEAK_CACHE_REQUESTS, _PEAK_CACHE_COMPUTES, _PEAK_CACHE_HITS, _PEAK_CACHE_EVICTIONS
        if type(depth) is not int or depth != 0:
            _PEAK_CACHE_NON_TOP_COMPUTES += 1
            return function(s, a, b, threshold, c, depth)
        _PEAK_CACHE_REQUESTS += 1
        key = None
        if (_PEAK_CACHE_ENABLED and type(s) is Segment and type(c) is Config
                and all(type(x) is float and math.isfinite(x) for x in (a, b, threshold))):
            try:
                key = (_PEAK_CACHE_POLICY, _peak_exact_key((s.start_time, s.duration, s.r, s.theta, s.z, s.kind)),
                       _peak_exact_key(c.to_dict()), a.hex(), b.hex(), threshold.hex(), depth)
            except (TypeError, ValueError, AttributeError, OverflowError, RecursionError):
                # Unsupported/tampered payloads retain original evaluation.
                key = None
        if key is not None and key in _PEAK_CACHE:
            _PEAK_CACHE_HITS += 1
            return _PEAK_CACHE[key]
        _PEAK_CACHE_COMPUTES += 1
        result = function(s, a, b, threshold, c, depth)
        if key is not None and type(result) is float:
            cost = _peak_owned_bytes(key) + sys.getsizeof(result)
            if _PEAK_CACHE_MAX_ENTRIES > 0 and cost + sys.getsizeof(OrderedDict()) <= _PEAK_CACHE_MAX_BYTES:
                while len(_PEAK_CACHE) >= _PEAK_CACHE_MAX_ENTRIES:
                    oldest, value = _PEAK_CACHE.popitem(last=False)
                    _PEAK_CACHE_PAYLOAD_BYTES -= _peak_owned_bytes(oldest) + sys.getsizeof(value)
                    _PEAK_CACHE_EVICTIONS += 1
                _PEAK_CACHE[key] = result
                _PEAK_CACHE_PAYLOAD_BYTES += cost
                while (_PEAK_CACHE and sys.getsizeof(_PEAK_CACHE) + _PEAK_CACHE_PAYLOAD_BYTES
                       > _PEAK_CACHE_MAX_BYTES):
                    oldest, value = _PEAK_CACHE.popitem(last=False)
                    _PEAK_CACHE_PAYLOAD_BYTES -= _peak_owned_bytes(oldest) + sys.getsizeof(value)
                    _PEAK_CACHE_EVICTIONS += 1
                if not _PEAK_CACHE:
                    _PEAK_CACHE.clear()
        return result
    return cached


@_memoize_peak
def _peak_measure(s,a,b,threshold,c,depth=0):
    polar=s.polar_bounds(a,b)
    bounds=power_bounds(_intrinsic_bounds_from_polar(polar,1),_intrinsic_bounds_from_polar(polar,2),c)
    if bounds is not None:
        low,high=bounds['shaft_w']
        if low>threshold: return b-a
        if high<=threshold: return 0.
    if b-a<1e-7 or depth>=26:
        # Conservative upper measure of unresolved threshold-crossing intervals.
        return b-a
    mid=(a+b)/2
    return _peak_measure(s,a,mid,threshold,c,depth+1)+_peak_measure(s,mid,b,threshold,c,depth+1)

_peak_measure_uncached = _peak_measure.__wrapped__
