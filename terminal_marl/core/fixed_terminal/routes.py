"""Fixed routes, independently selectable ring windows, and SOC costs.

Normal paths leave each ring at angle 0; emergency leaves the current ring at
pi and uses a dedicated outward corridor and outside descent. Window choices
are whole extra laps on every visited ring, indexed inner-to-current. No episode
cutoff is used as a route feasibility deadline.
"""
from math import pi, cos, sin
from .domain import Segment, Plan, Reservation, distance


def hold_segment(aircraft, now, end, config):
    point = aircraft.position(now, config)
    return Segment(now, end, "hold", point, point, aircraft.ring,
                   aircraft.theta, config.angular_speed)


def build_plan(aircraft, now, config, wait_cycles=(), emergency=False):
    if aircraft.state != "holding" or aircraft.ring is None:
        raise ValueError("only holding aircraft can propose terminal routes")
    if isinstance(aircraft.ring, bool) or not isinstance(aircraft.ring, int) or not 0 <= aircraft.ring < len(config.rings):
        raise ValueError("ring index must be an integer inside the configured rings")
    n = 1 if emergency else aircraft.ring+1
    cycles = tuple(wait_cycles) if wait_cycles else (0,)*n
    if len(cycles) != n or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in cycles):
        raise ValueError("one nonnegative integer wait cycle per visited ring required")
    target = pi if emergency else 0.
    wait = ((aircraft.theta-target) % (2*pi))/config.angular_speed
    if wait > config.lap_time-1e-8:
        wait = 0.
    wait += cycles[-1]*config.lap_time
    segments, reservations = [], []
    time, point = now, aircraft.position(now, config)

    def hold(duration, ring_index, theta):
        nonlocal point, time
        if duration > 1e-10:
            segment = Segment(time, time+duration, "hold", point, point,
                              ring_index, theta, config.angular_speed)
            segments.append(segment)
            time += duration
            point = segment.position(time)

    hold(wait, aircraft.ring, aircraft.theta)
    reservations.append(Reservation(f"ring:{aircraft.ring}", now, time+config.merge_window))
    # Roundoff from full laps is removed at the exact fixed node.
    ring = config.rings[aircraft.ring]
    point = (ring.radius*cos(target), ring.radius*sin(target), ring.height)

    def line(destination, mode, resource, ring_index=None):
        nonlocal point, time
        duration = max(distance(point, destination)/config.transfer_speed,
                       abs(point[2]-destination[2])/config.max_descent, 1e-8)
        if mode == "landing":
            duration = abs(point[2]-destination[2])/config.landing_speed
        segments.append(Segment(time, time+duration, mode, point, destination, ring_index))
        reservations.append(Reservation(resource, time-config.temporal_spacing/2,
                                        time+duration+config.temporal_spacing/2))
        point, time = destination, time+duration

    if emergency:
        pad = config.emergency_pad
        line((pad[0], pad[1], point[2]), "transfer", "emergency_corridor")
        line((pad[0], pad[1], config.landing_height), "descent", "emergency_approach")
    else:
        for target_ring in range(aircraft.ring-1, -1, -1):
            ring = config.rings[target_ring]
            line((ring.radius, 0., ring.height), "transfer", f"transfer:{target_ring+1}:{target_ring}", target_ring)
            merge_time = time
            reservations.append(Reservation(f"merge:{target_ring}", time-config.merge_window, time+config.merge_window))
            hold(cycles[target_ring]*config.lap_time, target_ring, 0.)
            reservations.append(Reservation(f"ring:{target_ring}", merge_time-config.merge_window, time+config.merge_window))
        pad = config.normal_pad
        line((pad[0], pad[1], config.landing_height), "descent", "normal_approach")
    pad_name = "emergency_pad" if emergency else "normal_pad"
    landing_start = time
    line(pad, "landing", pad_name)
    reservations.append(Reservation(pad_name, landing_start-config.temporal_spacing/2,
                                    time+config.pad_service+config.temporal_spacing/2))
    return Plan(aircraft.aircraft_id, emergency, cycles, tuple(segments), tuple(reservations))


def segment_energy(segment, config, start=None, end=None):
    start = segment.start if start is None else max(start, segment.start)
    end = segment.end if end is None else min(end, segment.end)
    rate = getattr(config.energy, segment.mode)
    if segment.mode == "transfer" and segment.p1[2] < segment.p0[2]:
        rate += config.energy.descent
    return max(0., end-start)*rate


def plan_energy(plan, config, start=None, end=None):
    return sum(segment_energy(s, config, start, end) for s in plan.segments)


def required_energy(aircraft, now, config, wait_cycles=(), emergency=False):
    return plan_energy(build_plan(aircraft, now, config, wait_cycles, emergency), config)+config.energy.reserve
