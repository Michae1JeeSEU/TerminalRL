"""Inactive exact-output candidate: sufficient convex-hull check, exact fallback."""
import math
import numpy as np
from terminal_marl.core.polynomial_trajectory import QuinticSegment, prime_quintic_bounds
from terminal_marl.core.trajectory_generator import TrajectoryInfeasibleError
from terminal_marl.core.trajectory_time_parameterization import TrajectoryTimeParameterizer


def _matrix(degree):
    return np.asarray([[math.comb(k,i)/math.comb(degree,i) if i<=k else 0.
                        for i in range(degree+1)] for k in range(degree+1)])


_VELOCITY_MATRIX=_matrix(4)
_ACCELERATION_MATRIX=_matrix(3)
_POSITION_MATRIX=_matrix(5)
_ORIGINAL_SEGMENTS_FEASIBLE=TrajectoryTimeParameterizer._v3_segments_feasible


def hull_bounds(segments):
    """Conservative continuous bounds, never installed as exact cache values."""
    coefficients=np.stack([s.coefficients for s in segments])
    duration=np.asarray([s.duration for s in segments])[:,None,None]
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        velocity=coefficients[:,1:]*np.arange(1,6)[None,:,None]/duration
        acceleration=coefficients[:,2:]*np.asarray([2.,6.,12.,20.])[None,:,None]/duration**2
        values=[]
        for power,matrix in ((coefficients,_POSITION_MATRIX),(velocity,_VELOCITY_MATRIX),(acceleration,_ACCELERATION_MATRIX)):
            control=np.einsum('ki,nid->nkd',matrix,power)
            margin=1e-8*(1.+np.linalg.norm(power,axis=-1).sum(axis=-1))
            values.append((control,margin))
        (p,pm),(v,vm),(a,am)=values
        columns=dict(max_horizontal_speed=np.linalg.norm(v[:,:,:2],axis=-1).max(axis=-1)+vm,
            max_horizontal_accel=np.linalg.norm(a[:,:,:2],axis=-1).max(axis=-1)+am,
            max_climb_rate=np.maximum(0.,v[:,:,2].max(axis=-1))+vm,
            max_descent_rate=np.maximum(0.,-v[:,:,2].min(axis=-1))+vm,
            max_vertical_accel_up=np.maximum(0.,a[:,:,2].max(axis=-1))+am,
            max_vertical_accel_down=np.maximum(0.,-a[:,:,2].min(axis=-1))+am,
            min_altitude=p[:,:,2].min(axis=-1)-pm,max_altitude=p[:,:,2].max(axis=-1)+pm)
    return [{key:float(value[i]) for key,value in columns.items()} for i in range(len(segments))]


def segments_feasible(self,segments):
    segments=tuple(segments)
    if not segments:return _ORIGINAL_SEGMENTS_FEASIBLE(self,segments)
    try:
        bounds=hull_bounds(segments)
    except (TypeError,ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
        return _ORIGINAL_SEGMENTS_FEASIBLE(self,segments)
    profile=self.dynamics.profile;tolerance=1e-3
    limits=dict(max_horizontal_speed=profile.max_horizontal_speed,
        max_horizontal_accel=min(profile.max_horizontal_accel,profile.max_horizontal_decel,profile.max_lateral_accel),
        max_climb_rate=profile.max_climb_rate,max_descent_rate=profile.max_descent_rate,
        max_vertical_accel_up=profile.max_vertical_accel_up,max_vertical_accel_down=profile.max_vertical_accel_down,
        max_altitude=profile.max_operating_altitude_agl_m)
    pending=[]
    for segment,bound in zip(segments,bounds):
        certified=(all(np.isfinite(x) for x in bound.values()) and bound['min_altitude']>=-tolerance
            and all(bound[key]<=limit+tolerance for key,limit in limits.items()))
        if not certified:
            pending.append(segment);continue
        # Keep the original finite-state sanity check even for certified hulls.
        for moment in (segment.start_time,(segment.start_time+segment.end_time)/2.,segment.end_time):
            position,velocity,acceleration=segment.sample(moment)
            if not all(np.isfinite(x).all() for x in (position,velocity,acceleration)):
                return False
            if (np.linalg.norm(velocity[:2])>limits['max_horizontal_speed']+tolerance
                or velocity[2]>limits['max_climb_rate']+tolerance or -velocity[2]>limits['max_descent_rate']+tolerance
                or np.linalg.norm(acceleration[:2])>limits['max_horizontal_accel']+tolerance
                or acceleration[2]>limits['max_vertical_accel_up']+tolerance
                or -acceleration[2]>limits['max_vertical_accel_down']+tolerance
                or position[2]<-tolerance or position[2]>limits['max_altitude']+tolerance):return False
    return not pending or _ORIGINAL_SEGMENTS_FEASIBLE(self,pending)


def horizontal_hull_certifies(segments, speed_limit, acceleration_limit):
    """Only sufficient acceptance; ambiguous bounds use the old exact solver.

    A polynomial lies in its Bernstein control-point convex hull. The norm is
    convex, so the largest control-vector norm bounds the entire horizontal
    derivative. An outward arithmetic margin deliberately favors fallback.
    This does not populate or replace the segment's exact bounds cache.
    """
    if not segments:
        return False
    coefficients=np.stack([s.coefficients for s in segments])
    durations=np.asarray([s.duration for s in segments])[:,None,None]
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        velocity=coefficients[:,1:,:2]*np.arange(1,6)[None,:,None]/durations
        acceleration=coefficients[:,2:,:2]*np.asarray([2.,6.,12.,20.])[None,:,None]/durations**2
        for values,matrix,limit in ((velocity,_VELOCITY_MATRIX,speed_limit),
                                    (acceleration,_ACCELERATION_MATRIX,acceleration_limit)):
            controls=np.einsum('ki,nid->nkd',matrix,values)
            bound=np.linalg.norm(controls,axis=-1).max(axis=-1)
            margin=1e-8*(1.+np.linalg.norm(values,axis=-1).sum(axis=-1))
            if not np.isfinite(bound+margin).all() or not np.all(bound+margin<=limit):
                return False
    return True


def ring_hold_points(self,ring,theta,current_speed,target_speed,start,duration,*,initial_acceleration=None):
    duration=float(duration)
    if not math.isfinite(duration) or duration<=0.:
        raise RuntimeError('ring_hold_duration_invalid')
    ease,minimum_ramp,*_=self._v3_ring_transition_parameters(ring,theta,current_speed,target_speed,
        initial_acceleration=initial_acceleration)
    if duration+1e-9<ease+minimum_ramp:
        raise RuntimeError('ring_hold_duration_shorter_than_required')
    candidates=[minimum_ramp];room=max(0.,duration-ease)
    for fraction in (.20,.40,.60,.80,1.):
        candidates.append(min(room,minimum_ramp*(1.+fraction)))
    last_error=None;profile=self.dynamics.profile
    for ramp in candidates:
        if ease+ramp>duration+1e-9:continue
        points=self._v3_ring_hold_points_for_ramp(ring,theta,current_speed,target_speed,start,duration,
            initial_acceleration=initial_acceleration,ramp_duration=ramp)
        try:
            segments=tuple(QuinticSegment.from_boundary(left.t,right.t,left.position,left.velocity,left.acceleration,
                right.position,right.velocity,right.acceleration) for left,right in zip(points,points[1:]))
            if horizontal_hull_certifies(segments,profile.max_horizontal_speed+1e-3,profile.max_horizontal_accel+1e-3):
                return points
            prime_quintic_bounds(segments)
            if all(segment.bounds()['max_horizontal_speed']<=profile.max_horizontal_speed+1e-3
                and segment.bounds()['max_horizontal_accel']<=profile.max_horizontal_accel+1e-3 for segment in segments):
                return points
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            last_error=exc
    if last_error is not None:
        raise TrajectoryInfeasibleError('ring_hold_interpolation_infeasible') from last_error
    raise TrajectoryInfeasibleError('ring_hold_interpolation_infeasible')
