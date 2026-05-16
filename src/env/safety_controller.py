"""Reflexive collision avoidance layer between RL actions and UAV kinematics.

SafetyController sits between the policy's actions and UAV.step(), applying
speed/heading adjustments to prevent collisions. It does NOT do path planning
— only reflexive slowdown and minor course divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.config import SafetyConfig
from src.env.uav import UAV, UAVStatus, LowLevelAction


@dataclass
class SafetyAdjustment:
    """Per-UAV safety modification applied on top of the original action."""
    speed_mult: float = 1.0
    psi_offset: float = 0.0       # radians
    gamma_offset: float = 0.0     # radians
    blocked: bool = False         # force CIRCLE
    active_rules: list[str] = field(default_factory=list)


class SafetyController:
    """Reflexive collision avoidance with 5 priority-aware safety rules."""

    def __init__(self, config: SafetyConfig, airspace, conflict_detector):
        self.config = config
        self.airspace = airspace
        self.conflict_detector = conflict_detector

    # ---- Public API ----

    def evaluate(self, uavs: list[UAV],
                 actions: dict[int, LowLevelAction]) -> dict[int, SafetyAdjustment]:
        """Run all safety rules and return per-UAV adjustments."""
        per_rule: list[dict[int, SafetyAdjustment]] = [
            self._rule1_angular_separation(uavs),
            self._rule2_circling_yields_descending(uavs, actions),
            self._rule3_dual_descent_divergence(uavs, actions),
            self._rule4_universal_brake(uavs),
            self._rule5_emergency_override(uavs),
        ]
        return self._compose_adjustments(per_rule)

    def apply(self, actions: dict[int, LowLevelAction],
              adjustments: dict[int, SafetyAdjustment]) -> dict[int, LowLevelAction]:
        """Return new actions with safety adjustments applied."""
        result: dict[int, LowLevelAction] = {}
        for uav_id, action in actions.items():
            adj = adjustments.get(uav_id, SafetyAdjustment())
            new_action = LowLevelAction(
                control=0 if adj.blocked else action.control,
                psi=action.psi + adj.psi_offset,
                gamma=action.gamma + adj.gamma_offset,
                speed=action.speed * adj.speed_mult,
                landing_zone=action.landing_zone,
            )
            result[uav_id] = new_action
        return result

    # ---- Priority ----

    @staticmethod
    def _is_emergency(uav: UAV) -> bool:
        return (uav.base_priority >= 5 or
                uav.battery <= uav.config.battery_critical * uav.config.max_battery)

    @staticmethod
    def _classify_priority(uav1: UAV, uav2: UAV) -> tuple[UAV, UAV] | None:
        """Return (higher_priority, lower_priority). None if equal.

        Emergency > high base_priority > low base_priority.
        """
        e1 = SafetyController._is_emergency(uav1)
        e2 = SafetyController._is_emergency(uav2)
        if e1 and not e2:
            return (uav1, uav2)
        if e2 and not e1:
            return (uav2, uav1)
        if e1 and e2:
            return None  # both emergency: equal
        if uav1.base_priority > uav2.base_priority:
            return (uav1, uav2)
        if uav2.base_priority > uav1.base_priority:
            return (uav2, uav1)
        return None  # equal priority

    # ---- Rule 1: Same-ring angular separation ----

    def _rule1_angular_separation(self, uavs: list[UAV]) -> dict[int, SafetyAdjustment]:
        """Existing logic: slow circling UAVs that are too close on the same ring."""
        result: dict[int, SafetyAdjustment] = {}
        safe_dist = self.config.angular_separation_threshold

        ring_circlers: dict[int, list[UAV]] = {1: [], 2: [], 3: []}
        for uav in uavs:
            if uav.status != UAVStatus.CRUISING or uav.ring_id == 0:
                continue
            ring_circlers[uav.ring_id].append(uav)

        for ring_id, circlers in ring_circlers.items():
            if len(circlers) < 2:
                continue
            circlers.sort(key=lambda u: u.angle)
            n = len(circlers)
            r_i = self.airspace.radii[ring_id - 1]
            safe_angle = safe_dist / max(r_i, 1e-6)
            for i, uav in enumerate(circlers):
                prev_uav = circlers[(i - 1) % n]
                next_uav = circlers[(i + 1) % n]
                gap_prev = (uav.angle - prev_uav.angle) % (2 * np.pi)
                gap_next = (next_uav.angle - uav.angle) % (2 * np.pi)
                min_gap = min(gap_prev, gap_next)
                if min_gap < safe_angle:
                    result[uav.id] = SafetyAdjustment(
                        speed_mult=0.95, active_rules=["R1_angular"]
                    )
                elif min_gap < 2 * safe_angle:
                    result[uav.id] = SafetyAdjustment(
                        speed_mult=0.98, active_rules=["R1_angular"]
                    )
        return result

    # ---- Rule 2: Circling yields to descending ----

    def _rule2_circling_yields_descending(
        self, uavs: list[UAV], actions: dict[int, LowLevelAction]
    ) -> dict[int, SafetyAdjustment]:
        """When a descending UAV will cross a ring, slow circling UAVs nearby."""
        result: dict[int, SafetyAdjustment] = {}
        lookahead = self.config.descend_crossing_lookahead
        slowdown = self.config.descend_crossing_slowdown

        descenders = [u for u in uavs if u.status == UAVStatus.DESCENDING]
        circlers = [u for u in uavs if u.status == UAVStatus.CRUISING]

        for duav in descenders:
            d_action = actions.get(duav.id)
            if d_action is None or d_action.control != 1:
                continue
            # Predict descending UAV position in lookahead seconds
            vel = duav.velocity
            if np.linalg.norm(vel) < 0.1:
                continue
            future_pos = duav.position + vel * lookahead

            # Which ring(s) will the descending UAV cross through?
            for ring_id in [1, 2, 3]:
                r_i, h_i = self.airspace.get_ring_orbit(ring_id)
                # Check if descent path crosses this ring's altitude band
                z_now = duav.position[2]
                z_future = future_pos[2]
                if not (min(z_now, z_future) <= h_i + 20.0 and
                        max(z_now, z_future) >= h_i - 20.0):
                    continue

                # Find the angular position where the descending UAV crosses this ring
                t_cross = None
                if abs(vel[2]) > 0.01:
                    t_cross = (h_i - duav.position[2]) / vel[2]
                if t_cross is None or t_cross < 0 or t_cross > lookahead:
                    continue
                cross_pos = duav.position + vel * t_cross
                cross_angle = np.arctan2(cross_pos[1], cross_pos[0])
                cross_angle = cross_angle % (2 * np.pi)

                # Slow circling UAVs whose angular position is near the crossing point
                angular_window = 0.15  # ~8.6 degrees
                for cuav in circlers:
                    if cuav.ring_id != ring_id:
                        continue
                    # How far is the circling UAV from the crossing angle?
                    angle_diff = abs((cuav.angle - cross_angle + np.pi) % (2 * np.pi) - np.pi)
                    # Estimate time until the circling UAV reaches the crossing
                    if cuav.ring_id > 0:
                        omega = self.airspace.speeds[ring_id - 1] / max(r_i, 1e-6)
                        time_to_cross = angle_diff / max(omega, 1e-6)
                        if time_to_cross < t_cross + 5.0 and angle_diff < angular_window * 3:
                            existing = result.get(cuav.id)
                            new_mult = min(existing.speed_mult, slowdown) if existing else slowdown
                            rules = (existing.active_rules if existing else []) + ["R2_circle_yield"]
                            result[cuav.id] = SafetyAdjustment(
                                speed_mult=new_mult, active_rules=rules
                            )
        return result

    # ---- Rule 3: Dual descent divergence ----

    def _rule3_dual_descent_divergence(
        self, uavs: list[UAV], actions: dict[int, LowLevelAction]
    ) -> dict[int, SafetyAdjustment]:
        """When two descending UAVs converge, apply psi divergence."""
        result: dict[int, SafetyAdjustment] = {}
        threshold = self.config.dual_descend_threshold
        psi_offset = np.deg2rad(self.config.dual_descend_psi_offset)

        descenders = [(u, actions.get(u.id)) for u in uavs
                      if u.status == UAVStatus.DESCENDING]
        n = len(descenders)
        for i in range(n):
            for j in range(i + 1, n):
                ua, act_a = descenders[i]
                ub, act_b = descenders[j]
                if act_a is None or act_b is None:
                    continue
                dist = float(np.linalg.norm(ua.position - ub.position))
                if dist >= threshold:
                    continue
                # Check closing speed: relative velocity projected onto separation vector
                sep = ub.position - ua.position
                rel_vel = ua.velocity - ub.velocity
                closing = float(np.dot(rel_vel, sep)) / max(dist, 1e-6)
                if closing <= 0:
                    continue  # already separating

                priority = self._classify_priority(ua, ub)
                if priority is None:
                    # Equal: both diverge — turn away from each other
                    a_offset = psi_offset if sep[1] >= 0 else -psi_offset
                    b_offset = -a_offset
                else:
                    # Lower priority yields more
                    _higher, lower = priority
                    if lower.id == ua.id:
                        a_offset = psi_offset * 1.5
                        b_offset = 0.0
                    else:
                        a_offset = 0.0
                        b_offset = -psi_offset * 1.5

                for uid, offset in [(ua.id, a_offset), (ub.id, b_offset)]:
                    if abs(offset) < 1e-6:
                        continue
                    existing = result.get(uid)
                    if existing and abs(existing.psi_offset) >= abs(offset):
                        continue
                    rules = ["R3_dual_descent"]
                    result[uid] = SafetyAdjustment(
                        psi_offset=float(offset), active_rules=rules
                    )
        return result

    # ---- Rule 4: Universal proximity brake ----

    def _rule4_universal_brake(self, uavs: list[UAV]) -> dict[int, SafetyAdjustment]:
        """When any two UAVs are within 30m, brake the closing one(s)."""
        result: dict[int, SafetyAdjustment] = {}
        threshold = self.config.universal_brake_threshold
        active = [u for u in uavs if u.status != UAVStatus.LANDED]

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ua, ub = active[i], active[j]
                dist = float(np.linalg.norm(ua.position - ub.position))
                if dist >= threshold or dist < 1e-6:
                    continue

                sep = ub.position - ua.position
                rel_vel = ua.velocity - ub.velocity
                closing = float(np.dot(rel_vel, sep)) / dist

                priority = self._classify_priority(ua, ub)
                if priority is None:
                    mult_low = self.config.universal_brake_low
                    mult_high = self.config.universal_brake_low  # same
                else:
                    _higher, _lower = priority
                    mult_high = self.config.universal_brake_high
                    mult_low = self.config.universal_brake_low

                if closing > 0:
                    # Both approaching each other: brake both
                    # Lower priority UAV brakes harder
                    if priority is None:
                        ma, mb = mult_low, mult_low
                    else:
                        _h, _l = priority
                        ma = mult_low if _l.id == ua.id else mult_high
                        mb = mult_low if _l.id == ub.id else mult_high

                    for u, mult in [(ua, ma), (ub, mb)]:
                        existing = result.get(u.id)
                        new_mult = min(existing.speed_mult, mult) if existing else mult
                        rules = (existing.active_rules if existing else []) + ["R4_brake"]
                        result[u.id] = SafetyAdjustment(
                            speed_mult=new_mult, active_rules=rules
                        )
        return result

    # ---- Rule 5: Emergency override ----

    def _rule5_emergency_override(self, uavs: list[UAV]) -> dict[int, SafetyAdjustment]:
        """Emergency UAVs are never slowed. Non-emergency yield aggressively."""
        result: dict[int, SafetyAdjustment] = {}
        emergencies = [u for u in uavs if self._is_emergency(u) and u.status != UAVStatus.LANDED]
        non_emergencies = [u for u in uavs if not self._is_emergency(u) and u.status != UAVStatus.LANDED]

        for euav in emergencies:
            # Cancel any existing slowdown for emergency UAVs
            result[euav.id] = SafetyAdjustment(
                speed_mult=1.0, psi_offset=0.0, blocked=False, active_rules=["R5_emergency"]
            )
            # Aggressive yield from nearby non-emergency UAVs
            for nuav in non_emergencies:
                dist = float(np.linalg.norm(euav.position - nuav.position))
                if dist < self.config.universal_brake_threshold * 2:
                    # Point away from emergency UAV
                    dx = nuav.position[0] - euav.position[0]
                    dy = nuav.position[1] - euav.position[1]
                    evade_angle = np.arctan2(dy, dx)
                    current_psi = nuav.angle  # approximate heading
                    # Offset: turn away from emergency UAV
                    psi_diff = (evade_angle - current_psi + np.pi) % (2 * np.pi) - np.pi
                    offset = (np.deg2rad(self.config.emergency_yield_psi)
                              if psi_diff > 0
                              else -np.deg2rad(self.config.emergency_yield_psi))
                    existing = result.get(nuav.id, SafetyAdjustment())
                    new_mult = min(existing.speed_mult, self.config.emergency_yield_slowdown)
                    rules = existing.active_rules + ["R5_yield"]
                    result[nuav.id] = SafetyAdjustment(
                        speed_mult=new_mult,
                        psi_offset=offset,
                        active_rules=rules,
                    )
        return result

    # ---- Composition ----

    @staticmethod
    def compute_adjustment(adjustments: list[SafetyAdjustment]) -> SafetyAdjustment:
        """Combine multiple adjustments: most conservative speed, largest |psi|."""
        result = SafetyAdjustment()
        for a in adjustments:
            result.speed_mult = min(result.speed_mult, a.speed_mult)
            if abs(a.psi_offset) > abs(result.psi_offset):
                result.psi_offset = a.psi_offset
            if abs(a.gamma_offset) > abs(result.gamma_offset):
                result.gamma_offset = a.gamma_offset
            result.blocked = result.blocked or a.blocked
            result.active_rules.extend(a.active_rules)
        return result

    def _compose_adjustments(
        self, per_rule: list[dict[int, SafetyAdjustment]]
    ) -> dict[int, SafetyAdjustment]:
        """For each UAV, compose adjustments from all rules."""
        all_ids: set[int] = set()
        for rule_dict in per_rule:
            all_ids.update(rule_dict.keys())

        result: dict[int, SafetyAdjustment] = {}
        for uid in all_ids:
            applicable = [r[uid] for r in per_rule if uid in r]
            result[uid] = self.compute_adjustment(applicable)
        return result
