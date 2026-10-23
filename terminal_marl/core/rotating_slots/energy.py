"""Appendix A research surrogate; charge-fraction SOC and shared quadrature."""
from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache, wraps
import math
import sys
import bisect
import heapq
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
    peak_error_s: float = 0.

# Pure scalar acceleration state: never part of a world or checkpoint.
_ROOT_CACHE_POLICY = 'scalar-induced-positive-branch-residual-v3'
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
    def cached(vp, vn, rhs, bisections=80, tol=1e-8):
        global _ROOT_CACHE_PAYLOAD_BYTES
        global _ROOT_CACHE_REQUESTS, _ROOT_CACHE_COMPUTES, _ROOT_CACHE_HITS, _ROOT_CACHE_EVICTIONS
        _ROOT_CACHE_REQUESTS += 1
        eligible = (_ROOT_CACHE_ENABLED and type(bisections) is int and bisections > 0
                    and all(type(x) is float and math.isfinite(x) for x in (vp, vn, rhs, tol))
                    and tol > 0)
        key = ((_ROOT_CACHE_POLICY, vp.hex(), vn.hex(), rhs.hex(), bisections, tol.hex())
               if eligible else None)
        if eligible and key in _ROOT_CACHE:
            _ROOT_CACHE_HITS += 1
            return _ROOT_CACHE[key]
        _ROOT_CACHE_COMPUTES += 1
        result = function(vp, vn, rhs, bisections, tol)
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
def _root_bracket(vp,vn,rhs,bisections=80,tol=1e-8):
    """Certified endpoints on the monotone vi >= max(0,-vn) branch.

    Stop on the returned midpoint's configured residual. Interval consumers use
    signed outward endpoints, never padding around an approximate scalar root.
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
        candidate=(lo+hi)/2
        if abs(f(candidate))<=tol*max(1.,rhs): break
    # Certify the final signs using outward interval arithmetic: rounded f
    # evaluations used for iteration alone are insufficient for root bounds.
    def enclosure(vi):
        return Interval(vi)*(Interval(vp).square()+(Interval(vn)+vi).square()).sqrt()-rhs
    branch=max(0.,-vn)
    while lo>branch and enclosure(lo).hi>0:
        lo=max(branch,math.nextafter(lo,-math.inf))
    while enclosure(hi).lo<0:
        hi=math.nextafter(hi,math.inf)
    return max(branch,math.nextafter(lo,-math.inf)),math.nextafter(hi,math.inf)

@_memoize_root
def _root(vp,vn,rhs,bisections=80,tol=1e-8):
    bracket=_root_bracket(vp,vn,rhs,bisections,tol)
    if bracket is None: return None
    value=(bracket[0]+bracket[1])/2
    return value if abs(value*math.hypot(vp,vn+value)-rhs)<=tol*max(1.,rhs) else None

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
        tol=e.induced_root_relative_residual_tolerance
        vi=_root(vp,vn,rhs,e.induced_root_bisections,tol)
        if vi is None: raise ValueError('no root on positive-power induced branch')
        residual=abs(vi*math.hypot(vp,vn+vi)-rhs)/max(1.,rhs)
        if residual>e.induced_root_relative_residual_tolerance: raise ValueError('induced root did not converge')
        pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
        shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
        max_shaft=min(c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000,
                      e.reference_max_battery_current_A*e.equivalent_voltage_V*e.motor_drive_efficiency)
        bracket=_root_bracket(vp,vn,rhs,e.induced_root_bisections,tol)
        shaft_box=(1+e.coaxial_chi)*Interval(T)*Interval(*bracket)+dot+pp
        if shaft_box.lo<=0<=shaft_box.hi or shaft_box.lo<=max_shaft<=shaft_box.hi:
            _numeric_count('power_limit_root_refinements')
            fine_tol=min(tol,1e-14)
            vi=_root(vp,vn,rhs,e.induced_root_bisections,fine_tol)
            if vi is None: raise ValueError('fine induced root did not converge')
            bracket=_root_bracket(vp,vn,rhs,e.induced_root_bisections,fine_tol)
            shaft_box=(1+e.coaxial_chi)*Interval(T)*Interval(*bracket)+dot+pp
            shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
            residual=abs(vi*math.hypot(vp,vn+vi)-rhs)/max(1.,rhs)
            if shaft_box.lo<=0: raise ValueError('nonpositive power cannot be excluded')
            if shaft_box.hi>max_shaft:
                return PowerResult(battery,shaft,battery/e.equivalent_voltage_V,'POWER_LIMIT','power limit cannot be excluded after fine root',vi,residual,ratio)
        if shaft<=0 or battery<=0: raise ValueError('nonpositive power; clipping is prohibited')
        if battery/e.equivalent_voltage_V>e.reference_max_battery_current_A or shaft>c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000:
            return PowerResult(battery,shaft,battery/e.equivalent_voltage_V,'POWER_LIMIT','battery current or shaft power exceeds frozen limit',vi,residual,ratio)
        return PowerResult(battery,shaft,battery/e.equivalent_voltage_V,'VALID','',vi,residual,ratio)
    except (ValueError,OverflowError) as exc:
        return PowerResult(math.nan,math.nan,math.nan,'ENERGY_MODEL_DOMAIN',str(exc))

def _norm(values): return sum((v.square() for v in values),Interval(0)).sqrt()

def power_bounds(v,a,config=None,*,fine=False):
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
    tolerance=min(e.induced_root_relative_residual_tolerance,1e-12) if fine else e.induced_root_relative_residual_tolerance
    lower=_root_bracket(vp.hi,vn.hi,rhs.lo,e.induced_root_bisections,tolerance)
    upper=_root_bracket(vp.lo,vn.lo,rhs.hi,e.induced_root_bisections,tolerance)
    vi=Interval(max(0.,lower[0]) if lower is not None else 0.,
                upper[1] if upper is not None else math.nextafter(max(0.,-vn.lo),math.inf))
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
    return {'battery_w':battery.pair(),'shaft_w':shaft.pair(),'current_a':(battery/e.equivalent_voltage_V).pair(),'axial_descent_ratio':ratio.pair(),'positive_upward_thrust':True}

def _power_array(s,t,c,fine=False):
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
    tolerance=min(e.induced_root_relative_residual_tolerance,1e-11) if fine else e.induced_root_relative_residual_tolerance
    for _ in range(e.induced_root_bisections):
        _numeric_count('vector_root_iterations')
        mid=(left+right)/2; positive=mid*np.hypot(vp,vn+mid)>rhs
        right=np.where(positive,mid,right); left=np.where(positive,left,mid)
        candidate=(left+right)/2
        if np.all(np.abs(candidate*np.hypot(vp,vn+candidate)-rhs)<=tolerance*np.maximum(1.,rhs)): break
    vi=(left+right)/2
    if np.any(np.abs(vi*np.hypot(vp,vn+vi)-rhs)>e.induced_root_relative_residual_tolerance*np.maximum(1.,rhs)):
        raise ValueError('induced root did not converge')
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*thrust*vi+work+pp
    # Only samples whose valid root enclosure straddles a power boundary need
    # scalar fine classification. This changes no power/domain limit.
    low_shaft=(1+e.coaxial_chi)*thrust*left+work+pp
    high_shaft=(1+e.coaxial_chi)*thrust*right+work+pp
    maximum=min(c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000,
                e.reference_max_battery_current_A*e.equivalent_voltage_V*e.motor_drive_efficiency)
    near=((low_shaft<=maximum)&(maximum<=high_shaft))|((low_shaft<=0)&(0<=high_shaft))
    if np.any(near):
        for i in np.flatnonzero(near):
            result=evaluate_power((vx[i],vy[i],vz[i]),(ax[i],ay[i],az[i]),c)
            if not result.valid: raise ValueError(result.reason)
            shaft[i]=result.shaft_w
    if np.any(shaft<=0): raise ValueError('nonpositive power')
    if np.any(shaft>c.vehicle.rotor_count*e.reference_motor_max_power_kW_each*1000) or np.any(shaft/e.motor_drive_efficiency/e.equivalent_voltage_V>e.reference_max_battery_current_A):
        raise ValueError('power exceeds frozen limit')
    return shaft/e.motor_drive_efficiency

@lru_cache(maxsize=512)
def _integrable_plan(plan,config_key=None):
    # Shared approval cache prevents execution from integrating an unvalidated
    # rehashed trajectory or from bypassing limits with a STABLE/GROUND label.
    from .safety import validate_plan
    return validate_plan(plan,load_config() if config_key is None else config_key.config)

_GAUSS={n:np.polynomial.legendre.leggauss(n) for n in (4,8)}

def _gauss_estimates(s,left,right,c,fine=False):
    mid=(left+right)/2; half=(right-left)/2; estimates=[]
    for n in (4,8):
        nodes,weights=_GAUSS[n]
        samples=_power_array(s,(mid[:,None]+half[:,None]*nodes).ravel(),c,fine=fine)
        estimates.append(half*np.sum(samples.reshape(len(left),n)*weights,axis=1))
    return estimates[1],np.abs(estimates[1]-estimates[0])

def _quadrature_cells(s,left,right,c,budgets,depth=0,fine=False):
    """Refine against allocated whole-plan shares, never a fresh per-leaf floor."""
    values,error=_gauss_estimates(s,left,right,c,fine)
    bad=error>budgets
    if np.any(bad):
        if depth>=20: raise ValueError('energy quadrature failed allocated numerical budget')
        mid=(left[bad]+right[bad])/2
        lval,lerr=_quadrature_cells(s,left[bad],mid,c,budgets[bad]/2,depth+1,fine)
        rval,rerr=_quadrature_cells(s,mid,right[bad],c,budgets[bad]/2,depth+1,fine)
        values[bad]=lval+rval; error[bad]=lerr+rerr
    return values,error

@lru_cache(maxsize=512)
def _segment_initial(duration,r,theta,z,max_dt,config_key,fine):
    s=Segment(0.,duration,r,theta,z)
    n=max(1,math.ceil(duration/max_dt)); knots=np.linspace(0.,duration,n+1)
    values,errors=_gauss_estimates(s,knots[:-1],knots[1:],config_key.config,fine)
    return knots,values,errors

@lru_cache(maxsize=256)
def _segment_table(duration,r,theta,z,max_dt,config_key,budget,fine=False):
    # Isotropic power omits absolute time/angle; the complete immutable config,
    # initial grid, allocated plan share, and precision remain in the cache key.
    began=time.perf_counter(); _numeric_count('energy_table_computes')
    s=Segment(0.,duration,r,theta,z)
    knots,initial,initial_errors=_segment_initial(duration,r,theta,z,max_dt,config_key,fine)
    values=initial.copy(); errors=initial_errors.copy()
    shares=budget*np.diff(knots)/duration
    bad=errors>shares
    if np.any(bad):
        values[bad],errors[bad]=_quadrature_cells(s,knots[:-1][bad],knots[1:][bad],
                                                config_key.config,shares[bad],fine=fine)
    _numeric_count('precompute_seconds',time.perf_counter()-began)
    return knots,np.concatenate(([0.],np.cumsum(values))),np.concatenate(([0.],np.cumsum(errors)))

@lru_cache(maxsize=8192)
def _segment_primitive(duration,r,theta,z,max_dt,config_key,budget,fine,t):
    _numeric_count('energy_primitive_computes')
    knots,energy,errors=_segment_table(duration,r,theta,z,max_dt,config_key,budget,fine)
    t=min(duration,max(0.,t))
    i=min(len(knots)-1,bisect.bisect_right(knots,t)-1)
    value=float(energy[i])
    if t>knots[i]:
        s=Segment(0.,duration,r,theta,z)
        share=budget*(t-knots[i])/duration
        v,_=_quadrature_cells(s,np.asarray([knots[i]]),np.asarray([t]),config_key.config,
                               np.asarray([share]),fine=fine)
        value+=float(v[0])
    # This is the allocated cumulative numerical-error allowance, not a strict
    # interval certificate. Actual Gauss4/8 estimates qualify the complete table.
    return value,budget*t/duration

def _shape(s):
    return (s.duration,s.r,(0.,)+s.theta[1:],s.z,s.kind)

def _constant_segment_power(shape,c,fine=False):
    duration,r,theta,z,kind=shape
    if kind=='GROUND': return 0.
    if kind=='STABLE' and len(r)==1 and len(z)==1 and len(theta)==2:
        result=_stable_power_cached(r[0],theta[1]/duration,_ConfigIdentity(c))
        if not result.valid: raise ValueError(result.reason)
        return result.battery_w
    return None

@lru_cache(maxsize=256)
def _plan_numerical_policy(shapes,max_dt,config_key,fine,absolute_floor=True):
    """One budget for all finite segments; reused by every query of the plan."""
    c=config_key.config; duration=math.fsum(s[0] for s in shapes)
    energies=[]
    for shape in shapes:
        constant=_constant_segment_power(shape,c,fine)
        if constant is not None:
            energies.append(constant*shape[0])
        else:
            initial=_segment_initial(*shape[:4],max_dt,config_key,fine)[1]
            energies.append(math.fsum(map(float,initial)))
    relative=2e-10 if fine else 1e-6
    floor=(1e-5 if fine else 1.) if absolute_floor else 0.
    budget=max(floor,relative*math.fsum(energies))
    for attempt in range(6):
        shares=tuple(budget*shape[0]/duration for shape in shapes)
        final=[]; actual_error=0.
        for shape,share,initial_energy in zip(shapes,shares,energies):
            if _constant_segment_power(shape,c,fine) is not None:
                final.append(initial_energy)
            else:
                table=_segment_table(*shape[:4],max_dt,config_key,share,fine)
                final.append(float(table[1][-1])); actual_error+=float(table[2][-1])
        final_target=max(floor,relative*math.fsum(final))
        if actual_error<=final_target and budget<=final_target:
            _numeric_count('energy_plan_policies')
            return shares,tuple(final),actual_error,budget
        budget=min(budget*.5 if actual_error>final_target else budget,final_target*(1-1e-12))
    raise ValueError('whole-plan energy numerical budget did not converge')

def _segment_joules(s,lo,hi,max_dt,c,budget=None,fine=False):
    shape=_shape(s); identity=_ConfigIdentity(c)
    if budget is None:
        budget=_plan_numerical_policy((shape,),max_dt,identity,fine,False)[0][0]
    key=(*shape[:4],max_dt,identity,budget,fine)
    _numeric_count('energy_primitive_calls',2)
    a,ea=_segment_primitive(*key,max(0.,lo-s.start_time))
    b,eb=_segment_primitive(*key,min(s.duration,hi-s.start_time))
    return b-a,max(0.,eb-ea)

@lru_cache(maxsize=128)
def _stable_power_cached(radius,angular_speed,config_key):
    c=config_key.config
    _numeric_count('stable_power_computes')
    return evaluate_power((0.,radius*angular_speed,0.),(-radius*angular_speed**2,0.,0.),c)

def stable_power(ring,config=None):
    """Power for a canonical slot; invariant under slot phase and absolute time."""
    from .geometry import ring_angular_speed, ring_config
    c=config or load_config(); _numeric_count('stable_power_calls')
    return _stable_power_cached(ring_config(ring,c).radius_m,-ring_angular_speed(ring,c),_ConfigIdentity(c))

def _coverage(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and merged[-1][1]==a: merged[-1]=(merged[-1][0],b)
        else: merged.append((a,b))
    starts=tuple(a for a,b in merged); ends=tuple(b for a,b in merged)
    prefix=(0.,)+tuple(np.cumsum([b-a for a,b in merged]))
    return starts,ends,prefix

def _coverage_primitive(coverage,t):
    starts,ends,prefix=coverage
    i=bisect.bisect_right(ends,t)
    return float(prefix[i])+(max(0.,t-starts[i]) if i<len(starts) else 0.)

@lru_cache(maxsize=256)
def _peak_table(duration,r,theta,z,threshold,config_key,budget=.01,fine=False):
    """Cumulative upper measure and additive unresolved-width measure.

    Continuous power bounds classify intervals. The remaining unresolved total
    width consumes the segment's share of one plan budget, including all leaves.
    """
    c=config_key.config; began=time.perf_counter(); _numeric_count('peak_table_computes')
    s=Segment(0.,duration,r,theta,z); active=[]; pending=[]; unresolved=0.
    def classify(a,b,depth):
        _numeric_count('peak_table_nodes')
        polar=s.polar_bounds(a,b)
        bounds=power_bounds(_intrinsic_bounds_from_polar(polar,1),
                            _intrinsic_bounds_from_polar(polar,2),c,fine=fine)
        if bounds is not None:
            low,high=bounds['shaft_w']
            if high<=threshold: return
            if low>threshold:
                active.append((a,b)); return
        heapq.heappush(pending,(-(b-a),a,b,depth))
    classify(0.,duration,0)
    unresolved=math.fsum(-row[0] for row in pending)
    nodes=1
    while pending and unresolved>budget:
        _,a,b,depth=heapq.heappop(pending)
        if depth>=45 or nodes>=200000:
            raise ValueError('peak unresolved width exceeds allocated numerical budget')
        unresolved-=b-a
        mid=(a+b)/2
        classify(a,mid,depth+1); classify(mid,b,depth+1); nodes+=2
        # The heap can reorder on insertion, so use its complete measure rather
        # than treating newly appended heap positions as a contiguous suffix.
        unresolved=math.fsum(-row[0] for row in pending)
    uncertain=[(a,b) for _,a,b,_ in pending]
    _numeric_count('precompute_seconds',time.perf_counter()-began)
    _numeric_count('peak_unresolved_seconds',math.fsum(b-a for a,b in uncertain))
    return _coverage(active+uncertain),_coverage(uncertain)

def _segment_peak(s,lo,hi,threshold,c,budget=.01,fine=False,return_error=False):
    _numeric_count('peak_primitive_calls',2)
    covered,uncertain=_peak_table(s.duration,s.r,(0.,)+s.theta[1:],s.z,threshold,_ConfigIdentity(c),budget,fine)
    a,b=lo-s.start_time,hi-s.start_time
    upper=max(0.,_coverage_primitive(covered,b)-_coverage_primitive(covered,a))
    error=max(0.,_coverage_primitive(uncertain,b)-_coverage_primitive(uncertain,a))
    return (upper,error) if return_error else upper

def integrate_energy(plan,start,end,max_dt=1.0,config=None,*,fine=False):
    """Shared cumulative energy/peak estimates with a single finite-plan budget.

    error_j is the allocated additive numerical-error allowance, not a rigorous
    quadrature certificate. peak_seconds includes all unresolved threshold width;
    peak_error_s reports that conservative measure's additive uncertainty width.
    fine uses independent tighter tables, without changing the physical clock.
    """
    _numeric_count('integrate_energy_calls')
    from .trajectory import TrajectoryPlan
    from .config import config_for_hash
    c=config or (config_for_hash(plan.config_hash) if isinstance(plan,TrajectoryPlan) else load_config())
    if not 0<max_dt<=1.: raise ValueError('energy initial segment must be in (0,1]')
    if type(fine) is not bool: raise TypeError('fine must be bool')
    if not math.isfinite(start) or not math.isfinite(end) or end<start: raise ValueError('invalid integration interval')
    if isinstance(plan,TrajectoryPlan):
        validity=_integrable_plan(plan,_ConfigIdentity(c))
        if not validity.accepted: return EnergyResult(0.,0.,0.,False,validity.status,validity.reason)
    if start==end: return EnergyResult(0.,0.,0.,True)
    total=error=peak=peak_error=0.
    threshold=c.vehicle.rotor_count*c.energy.reference_motor_continuous_power_kW_each*1000
    capacity=c.energy.equivalent_voltage_V*c.energy.reference_capacity_Ah*3600
    try:
        intervals=tuple(plan.intervals(start,end))
        complete=getattr(plan,'segments',None)
        known=bool(complete) and all(isinstance(s,Segment) for s in complete)
        finite=tuple(complete) if known else intervals
        if finite:
            shapes=tuple(_shape(s) for s in finite)
            shares,_,_,_=_plan_numerical_policy(shapes,max_dt,_ConfigIdentity(c),fine,known)
            energy_shares=dict(zip(shapes,shares))
            finite_duration=math.fsum(s.duration for s in finite)
        else:
            energy_shares={}; finite_duration=0.
        for s in intervals:
            lo=max(start,s.start_time); hi=min(end,s.end_time)
            if s.kind=='GROUND': continue
            if s.kind=='STABLE':
                if len(s.r)==1 and len(s.z)==1 and len(s.theta)==2:
                    _numeric_count('stable_power_calls')
                    omega=(-c.airspace.common_angular_speed_radps if isinstance(plan,TrajectoryPlan) and c.rotation_profile=='equal_angular' else s.theta[1]/s.duration)
                    result=_stable_power_cached(s.r[0],omega,_ConfigIdentity(c))
                else:
                    st=s.state((lo+hi)/2); result=evaluate_power(st.v,st.a,c)
                if not result.valid:
                    return EnergyResult(total,total/capacity,peak,False,result.status,result.reason,error,peak_error)
                total+=result.battery_w*(hi-lo); peak+=(hi-lo) if result.shaft_w>threshold else 0.
                continue
            shape=_shape(s)
            budget=energy_shares.get(shape)
            if budget is None:
                budget=_plan_numerical_policy((shape,),max_dt,_ConfigIdentity(c),fine,False)[0][0]
            value,err=_segment_joules(s,lo,hi,max_dt,c,budget,fine); total+=value; error+=err
            # Unknown wrappers receive per-segment relative shares without an
            # absolute .01s floor repeated independently for each query/segment.
            peak_budget=((1e-7 if fine else .01)*s.duration/finite_duration if known else
                         min(1e-7 if fine else .01, (1e-7 if fine else .01)*s.duration/1000.))
            upper,uncertainty=_segment_peak(s,lo,hi,threshold,c,peak_budget,fine,True)
            peak+=upper; peak_error+=uncertainty
        return EnergyResult(total,total/capacity,peak,True,error_j=error,peak_error_s=peak_error)
    except ValueError as exc:
        return EnergyResult(total,total/capacity,peak,False,'ENERGY_MODEL_DOMAIN',str(exc),error,peak_error)

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
