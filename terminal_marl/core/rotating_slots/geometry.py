import math
from .config import load_config
from .models import KinematicState

def ring_config(ring,config=None):
    c=config or load_config()
    for r in c.airspace.rings:
        if r.id == ring: return r
    raise ValueError('unknown ring: '+str(ring))

def ring_angular_speed(ring,config=None):
    """Positive clockwise speed for a ring in the selected frozen profile."""
    c=config or load_config(); r=ring_config(ring,c)
    return (c.airspace.common_tangential_speed_mps/r.radius_m
            if c.rotation_profile=='equal_linear_21_v1' else c.airspace.common_angular_speed_radps)

def maximum_ring_period(config=None):
    """Longest retained ring period; cross-ring safety uses radial bounds."""
    c=config or load_config()
    return 2*math.pi/min(ring_angular_speed(r.id,c) for r in c.airspace.rings)

def slot_phase(ring,slot,t,config=None):
    c=config or load_config(); r=ring_config(ring,c)
    if not isinstance(slot,int) or not 0<=slot<r.slot_count: raise ValueError('invalid slot')
    if not math.isfinite(t): raise ValueError('nonfinite time')
    return r.initial_phase_rad+2*math.pi*slot/r.slot_count-ring_angular_speed(ring,c)*t

def slot_state(ring,slot,t,config=None):
    c=config or load_config(); r=ring_config(ring,c); a=slot_phase(ring,slot,t,c)
    w=ring_angular_speed(ring,c); x=r.radius_m*math.cos(a); y=r.radius_m*math.sin(a)
    return KinematicState((x,y,r.height_agl_m),(w*y,-w*x,0),(-w*w*x,-w*w*y,0),(-w**3*y,w**3*x,0))

def pad_position(pad,config=None):
    c=config or load_config()
    if not isinstance(pad,int) or not 0<=pad<c.pads.count: raise ValueError('invalid pad')
    angle=c.pads.initial_phase_rad+2*math.pi*pad/c.pads.count
    return (c.pads.layout_radius_m*math.cos(angle),c.pads.layout_radius_m*math.sin(angle),c.pads.ground_height_agl_m)
