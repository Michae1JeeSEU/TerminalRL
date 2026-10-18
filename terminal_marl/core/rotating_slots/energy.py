"""Appendix A research surrogate; charge-fraction SOC and shared quadrature."""
from dataclasses import dataclass
from functools import lru_cache
import math
import bisect
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from .config import load_config
from .trajectory import Interval, _intrinsic_bounds_from_polar

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

def _root(vp,vn,rhs,bisections=80):
    lo=max(0.,-vn); hi=max(1.,lo+1.,math.sqrt(rhs)*2+abs(vn))
    f=lambda vi:vi*math.hypot(vp,vn+vi)-rhs
    if f(lo)>max(1.,rhs)*1e-12: return None
    while f(hi)<0: hi*=2
    for _ in range(bisections):
        mid=(lo+hi)/2
        if f(mid)>0: hi=mid
        else: lo=mid
    return (lo+hi)/2

def evaluate_power(v,a,config=None):
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
    lower=_root(vp.hi,vn.hi,rhs.lo)
    lower=0. if lower is None else lower
    upper=_root(vp.lo,vn.lo,rhs.hi)
    upper=max(0.,-vn.lo) if upper is None else upper
    vi=Interval(math.nextafter(max(0.,lower-1e-10),-math.inf),math.nextafter(upper+1e-10,math.inf))
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*T*vi+dot+pp; battery=shaft/e.motor_drive_efficiency
    return {'battery_w':battery.pair(),'shaft_w':shaft.pair(),'current_a':(battery/e.equivalent_voltage_V).pair(),'axial_descent_ratio':ratio.pair(),'positive_upward_thrust':True}

def _power_array(s,t,c):
    """Vector evaluation of exactly the same scalar Appendix A equations."""
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
    for _ in range(80):
        mid=(left+right)/2; positive=mid*np.hypot(vp,vn+mid)>rhs
        right=np.where(positive,mid,right); left=np.where(positive,left,mid)
    vi=(left+right)/2
    pp=c.vehicle.rotor_count*rho/8*e.profile_solidity*e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp
    shaft=(1+e.coaxial_chi)*thrust*vi+work+pp
    if np.any(shaft<=0): raise ValueError('nonpositive power')
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
    from .trajectory import Segment
    s=Segment(0.,duration,r,(0.,)+theta[1:],z)
    n=max(1,math.ceil(duration/max_dt)); knots=np.linspace(0.,duration,n+1)
    values,errors=_quadrature_cells(s,knots[:-1],knots[1:],load_config())
    return knots,np.concatenate(([0.],np.cumsum(values))),np.concatenate(([0.],np.cumsum(errors)))

def _segment_joules(s,lo,hi,max_dt,c):
    theta=(0.,)+s.theta[1:]
    knots,energy,errors=_segment_table(s.duration,s.r,theta,s.z,max_dt,c.profile_hash)
    a=lo-s.start_time; b=hi-s.start_time
    ia=bisect.bisect_left(knots,a); ib=bisect.bisect_right(knots,b)-1
    if ia>ib:
        value,error=_quadrature_cells(s,np.asarray([lo]),np.asarray([hi]),c)
        return float(value[0]),float(error[0])
    value=float(energy[ib]-energy[ia]); error=float(errors[ib]-errors[ia])
    for x,y in ((a,float(knots[ia])),(float(knots[ib]),b)):
        if y>x:
            v,e=_quadrature_cells(s,np.asarray([s.start_time+x]),np.asarray([s.start_time+y]),c); value+=float(v[0]); error+=float(e[0])
    return value,error

def integrate_energy(plan,start,end,max_dt=.2,config=None):
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
                st=s.state((lo+hi)/2); result=evaluate_power(st.v,st.a,c)
                if not result.valid: return EnergyResult(total,total/(85*252*3600),peak,False,result.status,result.reason,error)
                total+=result.battery_w*(hi-lo); peak+=(hi-lo) if result.shaft_w>threshold else 0.
                continue
            value,err=_segment_joules(s,lo,hi,max_dt,c); total+=value; error+=err
            peak+=_peak_measure(s,lo,hi,threshold,c)
        return EnergyResult(total,total/(c.energy.equivalent_voltage_V*c.energy.reference_capacity_Ah*3600),peak,True,error_j=error)
    except ValueError as exc: return EnergyResult(total,total/(85*252*3600),peak,False,'ENERGY_MODEL_DOMAIN',str(exc),error)

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
