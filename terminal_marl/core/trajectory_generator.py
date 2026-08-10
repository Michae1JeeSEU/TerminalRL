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
        p0=np.asarray(uav.position,dtype=float); a=reservation.merge_angle or 0.; ring=airspace.actual_ring(target);p3=np.array((ring.radius*math.cos(a),ring.radius*math.sin(a),ring.height));
        src_angle=math.atan2(p0[1],p0[0]);t0=np.array((-math.sin(src_angle),math.cos(src_angle),0.));t1=np.array((-math.sin(a),math.cos(a),0.));d=np.linalg.norm(p3-p0);L=min(0.45*d,180.);p1=p0+L*t0;p2=p3-L*t1
        ss=np.linspace(0,1,41);curve=[]
        for s in ss:
            xy=(1-s)**3*p0+3*(1-s)**2*s*p1+3*(1-s)*s*s*p2+s**3*p3; q=10*s**3-15*s**4+6*s**5; xy[2]=p0[2]+(p3[2]-p0[2])*q;curve.append(xy)
        length=sum(np.linalg.norm(b-a) for a,b in zip(curve,curve[1:]))
        # quintic smoothstep peaks at 1.875 times its average vertical rate.
        vertical_duration=abs(p3[2]-p0[2]) / max(1e-6, self.config.max_climb_rate) * 1.90
        commanded_speed=np.clip(ring.nominal_speed + speed_residual_mps, self.config.speed_min, self.config.max_speed)
        end=max(now+length/max(1.,commanded_speed), now+vertical_duration, reservation.merge_time)
        return self._plan(uav,"ring_transfer",curve,now,end,reservation,{"target_phase": "ring_middle" if target is RingRole.MIDDLE else "ring_inner","source_tangent":t0,"target_tangent":t1,"speed_residual_mps":float(speed_residual_mps)})
    def final_approach(self,uav,lz,reservation,now):
        p0=np.asarray(uav.position); entry=np.array((0.,-150.,120.));stabilized=np.array((lz.position[0],lz.position[1]-80.,35.));p3=np.asarray(lz.position);length=np.linalg.norm(entry-p0)+np.linalg.norm(stabilized-entry)+np.linalg.norm(p3-stabilized);return self._plan(uav,"final_approach",[p0,entry,stabilized,p3],now,max(now+length/10.,reservation.merge_time),reservation,{"target_phase":"landing"})
    def validate_kinematics(self,plan):
        errors=[]
        for a,b in zip(plan.points,plan.points[1:]):
            dt=b.t-a.t; speed=np.linalg.norm(b.position-a.position)/dt
            if speed>self.config.max_speed+1e-6:errors.append("speed")
            vr=abs(b.position[2]-a.position[2])/dt
            if vr>max(self.config.max_climb_rate,self.config.max_descent_rate)+1e-6:errors.append("vertical_rate")
        return errors
