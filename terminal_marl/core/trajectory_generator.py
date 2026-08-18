"""The single authoritative plan generator for checking and execution."""
from __future__ import annotations
from dataclasses import dataclass,field
import math,uuid,numpy as np
from .config import RingRole

@dataclass
class TrajectoryPoint: t:float; position:np.ndarray; velocity:np.ndarray|None=None; yaw:float|None=None
@dataclass
class TrajectoryPlan:
    plan_id:str;uav_id:int;mode:str;points:list[TrajectoryPoint];start_time:float;end_time:float;reservation_ids:list[str]=field(default_factory=list);metadata:dict=field(default_factory=dict)
    def sample(self,now):
        if now<=self.start_time:return self.points[0]
        if now>=self.end_time:return self.points[-1]
        for a,b in zip(self.points,self.points[1:]):
            if a.t<=now<=b.t:
                p=(now-a.t)/(b.t-a.t);pos=a.position+(b.position-a.position)*p;vel=(b.position-a.position)/(b.t-a.t);return TrajectoryPoint(now,pos,vel)
        return self.points[-1]

class TrajectoryGenerator:
    def __init__(self,config):self.config=config
    def _plan(self,uav,mode,points,start,end,reservation=None,metadata=None):
        ts=np.linspace(start,end,len(points)); out=[TrajectoryPoint(float(t),np.asarray(p,dtype=float)) for t,p in zip(ts,points)]; return TrajectoryPlan(str(uuid.uuid4()),uav.uid,mode,out,start,end,[] if reservation is None else [reservation.reservation_id],metadata or {})
    def inbound(self,uav,gate,airspace,now):
        r=airspace.actual_ring(RingRole.OUTER); a=gate.angle;target=np.array((r.radius*math.cos(a),r.radius*math.sin(a),r.height));dist=np.linalg.norm(target-uav.position)
        duration=max(1.,dist/28,abs(target[2]-uav.position[2])/self.config.max_descent_rate)
        return self._plan(uav,"inbound",[uav.position,target],now,now+duration,metadata={"target_phase":"ring_outer"})
    def ring_transfer(self,uav,source,target,reservation,airspace,now, speed_residual_mps=0.0):
        current=np.asarray(uav.position,dtype=float); a=reservation.merge_angle or 0.; ring=airspace.actual_ring(target); p3=np.array((ring.radius*math.cos(a),ring.radius*math.sin(a),ring.height))
        source_ring=airspace.actual_ring(source); source_theta=uav.ring_progress if uav.ring_progress is not None else math.atan2(current[1],current[0])
        source_speed=np.clip(source_ring.nominal_speed+airspace.speed_profile_bias+speed_residual_mps,self.config.speed_min,self.config.max_speed)
        commanded_speed=np.clip(ring.nominal_speed+airspace.speed_profile_bias+speed_residual_mps,self.config.speed_min,self.config.max_speed)

        def build_curve(departure: float):
            theta=(source_theta+source_speed*(departure-now)/source_ring.radius)%(2*math.pi)
            p0=np.array((source_ring.radius*math.cos(theta),source_ring.radius*math.sin(theta),source_ring.height))
            t0=np.array((-math.sin(theta),math.cos(theta),0.)); t1=np.array((-math.sin(a),math.cos(a),0.))
            length=np.linalg.norm(p3-p0); tangent=min(.45*length,180.); p1=p0+tangent*t0; p2=p3-tangent*t1
            ss=np.linspace(0.,1.,41); curve=[]
            for s in ss:
                point=(1-s)**3*p0+3*(1-s)**2*s*p1+3*(1-s)*s*s*p2+s**3*p3
                smooth=10*s**3-15*s**4+6*s**5; point[2]=p0[2]+(p3[2]-p0[2])*smooth; curve.append(point)
            curve_length=sum(float(np.linalg.norm(right-left)) for left,right in zip(curve,curve[1:]))
            duration=max(curve_length/max(1.,commanded_speed),abs(p3[2]-p0[2])/max(1e-6,self.config.max_climb_rate)*1.90,1.)
            return theta,p0,t0,t1,curve,duration

        # Solve the arrival-slot/departure coupling twice.  The reservation is
        # an arrival-at-capture slot, so any waiting is a real source-ring arc
        # in this very plan, never an instantaneous chord or future-position
        # jump after SafetyFilter approval.
        _,_,_,_,_,duration=build_curve(now)
        start=max(now,reservation.merge_time-duration)
        theta,p0,t0,t1,curve,duration=build_curve(start)
        start=max(now,reservation.merge_time-duration)
        theta,p0,t0,t1,curve,duration=build_curve(start)
        end=start+duration
        points=[TrajectoryPoint(float(now),current)]
        if start>now+1e-9:
            hold_times=np.arange(now+1.,start,1.)
            for moment in hold_times:
                hold_theta=(source_theta+source_speed*(moment-now)/source_ring.radius)%(2*math.pi)
                hold=np.array((source_ring.radius*math.cos(hold_theta),source_ring.radius*math.sin(hold_theta),source_ring.height))
                points.append(TrajectoryPoint(float(moment),hold))
            points.append(TrajectoryPoint(float(start),p0))
        transfer_times=np.linspace(start,end,len(curve)); points.extend(TrajectoryPoint(float(t),np.asarray(point,dtype=float)) for t,point in zip(transfer_times[1:],curve[1:]))
        return TrajectoryPlan(str(uuid.uuid4()),uav.uid,"ring_transfer",points,now,end,[reservation.reservation_id],{"target_phase": "ring_middle" if target is RingRole.MIDDLE else "ring_inner","source_tangent":t0,"target_tangent":t1,"speed_residual_mps":float(speed_residual_mps),"scheduled_departure":start,"merge_time":reservation.merge_time})
    def final_approach(self,uav,lz,reservation,now, *, scheduled_delay: float = 0.0, speed_residual_mps: float = 0.0, airspace=None):
        """Generate the checked final trajectory, including any future ring hold.

        A delayed I2F release is represented by explicit INNER-ring arc points
        in the same plan that is passed to SafetyFilter and later executed.
        This rules out a hidden time jump between a checked future reservation
        and the actual final approach.
        """
        start = now + max(0.0, scheduled_delay)
        p0=np.asarray(uav.position,dtype=float)
        points=[TrajectoryPoint(float(now), p0)]
        if start > now + 1e-9 and uav.ring_role is RingRole.INNER:
            # ``airspace`` is supplied by the environment for a delayed
            # release.  Legacy callers never request a delay and remain
            # unchanged.
            ring=airspace.actual_ring(RingRole.INNER) if airspace is not None else None
            if ring is not None:
                theta=uav.ring_progress if uav.ring_progress is not None else math.atan2(p0[1],p0[0])
                speed=np.clip(ring.nominal_speed+speed_residual_mps,self.config.speed_min,self.config.max_speed)
                for moment in np.arange(now+1.,start,1.):
                    angle=(theta+speed*(moment-now)/ring.radius)%(2*math.pi)
                    points.append(TrajectoryPoint(float(moment),np.array((ring.radius*math.cos(angle),ring.radius*math.sin(angle),ring.height))))
                angle=(theta+speed*(start-now)/ring.radius)%(2*math.pi)
                p0=np.array((ring.radius*math.cos(angle),ring.radius*math.sin(angle),ring.height))
                points.append(TrajectoryPoint(float(start),p0))
        entry=np.array((0.,-150.,120.));stabilized=np.array((lz.position[0],lz.position[1]-80.,35.));p3=np.asarray(lz.position);length=np.linalg.norm(entry-p0)+np.linalg.norm(stabilized-entry)+np.linalg.norm(p3-stabilized)
        end=max(start+length/10.,reservation.merge_time)
        final_points=[p0,entry,stabilized,p3]
        times=np.linspace(start,end,len(final_points))
        if not points or abs(points[-1].t-start)>1e-9:
            points.append(TrajectoryPoint(float(start),p0))
        points.extend(TrajectoryPoint(float(t),np.asarray(point,dtype=float)) for t,point in zip(times[1:],final_points[1:]))
        return TrajectoryPlan(str(uuid.uuid4()),uav.uid,"final_approach",points,now,end,[reservation.reservation_id],{"target_phase":"landing","scheduled_departure":start,"scheduled_delay":max(0.0,scheduled_delay),"speed_residual_mps":float(speed_residual_mps)})
    def validate_kinematics(self,plan):
        errors=[]
        for a,b in zip(plan.points,plan.points[1:]):
            dt=b.t-a.t; speed=np.linalg.norm(b.position-a.position)/dt
            if speed>self.config.max_speed+1e-6:errors.append("speed")
            vr=abs(b.position[2]-a.position[2])/dt
            if vr>max(self.config.max_climb_rate,self.config.max_descent_rate)+1e-6:errors.append("vertical_rate")
        return errors
