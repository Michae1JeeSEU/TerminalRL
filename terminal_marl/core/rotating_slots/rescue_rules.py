"""Pure, versioned rescue classification shared by execution and planning.

A finite trajectory search is never a proof of normal-route infeasibility.
Only a certified completion or a valid lower energy bound resolves that state.
"""
from dataclasses import dataclass
import math

LEGACY_PROFILE = "legacy_v5"
RESCUE_PROFILE_V6 = "rescue_opportunity_v6"
RESCUE_PROFILE_V7 = "rescue_need_v7"
MONITOR_SECONDS_V6 = 10.
DISPATCH_SECONDS_V6 = 5.
RESCUE_BUFFER_SECONDS_V6 = MONITOR_SECONDS_V6 + DISPATCH_SECONDS_V6
RESCUE_LOOKAHEAD_SECONDS_V6 = 120.
RESCUE_DURATION_ALTERNATIVES_V6 = 3


def normal_energy_lower_bound(config, ring):
    """Necessary SOC cost for any normal completion from a stable ring.

    On the selected induced-root branch vi >= max(0,-vn), chi >= 0,
    so shaft=T*((1+chi)*vi+vn)+pp >= pp. Altitude divided by the
    maximum descent speed is a necessary remaining flight time. This bound
    deliberately omits transfers, acceleration and holding, and cannot prove
    feasibility. It changes no power or trajectory calculation.
    """
    e, v = config.energy, config.vehicle
    if e.coaxial_chi < 0:
        return 0.
    height = next(r.height_agl_m for r in config.airspace.rings if r.id == ring)
    area = math.pi*(v.rotor_diameter_m/2.)**2
    pp = (v.rotor_count*e.air_density_kgpm3/8.*e.profile_solidity
          *e.profile_mean_drag_coefficient*area*e.profile_tip_speed_mps**3*e.profile_Fp)
    soc_rate = pp/(e.motor_drive_efficiency*e.equivalent_voltage_V*e.reference_capacity_Ah*3600.)
    return math.nextafter(soc_rate*height/v.max_descent_speed_mps, 0.)


@dataclass(frozen=True)
class RescueAssessment:
    warning: bool
    normal_status: str
    normal_margin: float | None
    rescue_required: bool
    rescue_reason: str | None
    rescue_slack_seconds: float | None


def assess_rescue(soc, holding_soc_per_second, *, normal_energy_lower_bound=0.,
                  normal_completion_soc=None, normal_path_soc=None, normal_path_wait_seconds=0.,
                  rescue_soc=None, rescue_wait_seconds=0.,
                  rescue_feasible=False, declared=False, pending=False,
                  profile=RESCUE_PROFILE_V6):
    """Assess using only supplied, certified evidence (v6 by default).

    ``rescue_soc`` is the validated flight cost, excluding the future holding
    interval. ``rescue_feasible`` certifies that exact opportunity including its
    pad window and hold cost. The zero lower bound is universally conservative;
    callers must justify any stronger normal completion energy lower bound.
    """
    if not math.isfinite(soc) or not math.isfinite(holding_soc_per_second) or holding_soc_per_second <= 0:
        raise ValueError("finite SOC and positive holding rate required")
    if normal_energy_lower_bound is not None and (not math.isfinite(normal_energy_lower_bound) or normal_energy_lower_bound < 0):
        raise ValueError("normal energy lower bound must be nonnegative")
    impossible = normal_energy_lower_bound is not None and soc < .25 + normal_energy_lower_bound
    if impossible:
        status, margin = "INFEASIBLE", soc - .25 - normal_energy_lower_bound
    elif normal_completion_soc is not None and math.isfinite(normal_completion_soc) and normal_completion_soc >= .25:
        status, margin = "FEASIBLE", normal_completion_soc - .25
    else:
        status, margin = "UNKNOWN", None
        if normal_path_soc is not None and math.isfinite(normal_path_soc):
            margin = soc-normal_path_soc-normal_path_wait_seconds*holding_soc_per_second-.25
    known = rescue_soc is not None and math.isfinite(rescue_soc) and rescue_soc >= 0
    slack = ((soc-rescue_soc) / holding_soc_per_second - rescue_wait_seconds
             if known and rescue_feasible else None)
    reason = ("DECLARED" if declared else "PENDING" if pending else
              "NORMAL_RESERVE_IMPOSSIBLE" if impossible and (rescue_feasible or profile == RESCUE_PROFILE_V7) else
              "LAST_RESCUE_START" if slack is not None and slack <= RESCUE_BUFFER_SECONDS_V6 else None)
    warning = (soc <= .35 or status == "INFEASIBLE"
               or (margin is not None and margin <= RESCUE_BUFFER_SECONDS_V6*holding_soc_per_second))
    return RescueAssessment(warning, status, margin,
                            reason is not None, reason, slack)


def pad_opportunities_overlap(plan, other, aircraft, config):
    """Exact half-open pad windows used by the authoritative world calendars."""
    from .config import service_duration
    return (plan.pad_id is not None and plan.pad_id == other.pad_id
            and plan.end_time-13.5 < other.end_time + service_duration(config, aircraft[other.aircraft_id].service_class)
            and other.end_time-13.5 < plan.end_time + service_duration(config, aircraft[plan.aircraft_id].service_class))
