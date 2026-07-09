
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .config import SimConfig
from .geometry import distance_3d
from .uav import UAVState, FlightPhase
from .observations import progress_error


@dataclass
class RewardTracker:
    config: SimConfig
    prev_error: Dict[int, float] = field(default_factory=dict)
    prev_phase: Dict[int, int] = field(default_factory=dict)
    prev_ring_slot: Dict[int, bool] = field(default_factory=dict)
    prev_finished: Dict[int, bool] = field(default_factory=dict)
    prev_commands: Dict[int, np.ndarray] = field(default_factory=dict)
    prev_ring_level: Dict[int, int] = field(default_factory=dict)
    ring_dwell_steps: Dict[int, int] = field(default_factory=dict)
    prev_alt_error: Dict[int, float] = field(default_factory=dict)
    final_dwell_steps: Dict[int, int] = field(default_factory=dict)
    prev_xy_error: Dict[int, float] = field(default_factory=dict)
    prev_path_length: Dict[int, float] = field(default_factory=dict)
    prev_waiting_ring_distance: Dict[int, float] = field(default_factory=dict)
    prev_turn_angle_rad: Dict[int, float] = field(default_factory=dict)

    def _ring_level(self, ac: UAVState) -> int:
        """
        0 -> FINAL
        1 -> inner ring
        2 -> outer ring
        99 -> other phases
        """
        if ac.phase == FlightPhase.FINAL:
            return 0

        if ac.phase != FlightPhase.RING:
            return 99

        cx, cy, _ = self.config.center
        x, y, z = ac.position
        r = np.hypot(x - cx, y - cy)

        inner = self.config.rings[0]
        outer = self.config.rings[1]

    # 综合半径误差 + 高度误差，判断更接近哪一层环
        inner_score = abs(r - inner.radius) + 0.5 * abs(z - inner.height)
        outer_score = abs(r - outer.radius) + 0.5 * abs(z - outer.height)

        return 1 if inner_score <= outer_score else 2


    def _target_altitude(self, ac: UAVState) -> float:
        level = self._ring_level(ac)
        if level == 2:   # outer ring
            return self.config.rings[1].height
        if level == 1:   # inner ring
            return self.config.rings[0].height
        if level == 0:   # final
            return self.config.final_entry_hold_altitude
        return float(ac.position[2])

    def reset(self, aircraft: List[UAVState]) -> None:
        self.prev_error = {a.uid: progress_error(a, self.config) for a in aircraft}
        self.prev_phase = {a.uid: int(a.phase) for a in aircraft}
        self.prev_ring_slot = {a.uid: bool(a.ring_slot_active) for a in aircraft}
        self.prev_finished = {a.uid: bool(a.finished) for a in aircraft}
        self.prev_commands = {a.uid: np.zeros(3, dtype=np.float32) for a in aircraft}
        self.prev_path_length = {a.uid: float(a.path_length_total) for a in aircraft}
        self.prev_waiting_ring_distance = {a.uid: float(a.waiting_ring_distance) for a in aircraft}
        self.prev_turn_angle_rad = {a.uid: float(a.cumulative_turn_angle_rad) for a in aircraft}

    # 新增：记录上一时刻所在环层
        self.prev_ring_level = {a.uid: self._ring_level(a) for a in aircraft}

    # 新增：记录每架机当前已在该环层停留了多少步
        self.ring_dwell_steps = {a.uid: 0 for a in aircraft}

    # 新增：记录上一时刻的高度误差
        self.prev_alt_error = {
            a.uid: abs(float(a.position[2]) - self._target_altitude(a))
            for a in aircraft
        }
        self.final_dwell_steps = {a.uid: 0 for a in aircraft}
        self.prev_xy_error = {
            a.uid: np.hypot(a.position[0] - self.config.center[0],
                            a.position[1] - self.config.center[1])
            for a in aircraft
        }

    def compute(
        self,
        aircraft: List[UAVState],
        applied_actions: Dict[int, np.ndarray],
        nmac_pairs: Dict[int, int],
        collision_pairs: Dict[int, int],
    ) -> Dict[int, float]:
        rewards: Dict[int, float] = {}

    # ===== 可调系数：先用这组 =====
        OUTER_TO_INNER_BONUS = 4.0
        INNER_TO_FINAL_BONUS = 5.0
        ALT_PROGRESS_WEIGHT = 0.03
        FINAL_XY_PROGRESS_WEIGHT = 0.05
        ACTION_MAG_PENALTY_WEIGHT = self.config.residual_action_penalty_weight

        OUTER_RING_OVERSTAY_PENALTY = 0.05
        INNER_RING_OVERSTAY_PENALTY = 0.06

        FINAL_DWELL_PENALTY_1 = 0.05
        FINAL_DWELL_PENALTY_2 = 0.10

        for ac in aircraft:
            r = self.config.step_penalty

        # 1) 进度奖励
            current_error = progress_error(ac, self.config)
            prev_error = self.prev_error.get(ac.uid, current_error)
            r += self.config.progress_weight * (prev_error - current_error)

        # 路径长度惩罚
            prev_path = self.prev_path_length.get(ac.uid, float(ac.path_length_total))
            delta_path = max(0.0, float(ac.path_length_total) - prev_path)

            r -= self.config.path_length_penalty_weight * delta_path

            prev_waiting = self.prev_waiting_ring_distance.get(ac.uid, float(ac.waiting_ring_distance))
            delta_waiting = max(0.0, float(ac.waiting_ring_distance) - prev_waiting)
            r -= self.config.waiting_ring_penalty_weight * delta_waiting

            prev_turn = self.prev_turn_angle_rad.get(ac.uid, float(ac.cumulative_turn_angle_rad))
            delta_turn_rad = max(0.0, float(ac.cumulative_turn_angle_rad) - prev_turn)
            r -= self.config.turn_angle_penalty_weight * float(np.degrees(delta_turn_rad))

        # 2) 安全间隔惩罚
            prox_pen = 0.0
            for other in aircraft:
                if other.uid == ac.uid or other.finished:
                    continue
                d = distance_3d(ac.position, other.position)
                if d < self.config.safe_distance:
                    prox_pen += (self.config.safe_distance - d) / max(1.0, self.config.safe_distance)

            r += self.config.separation_penalty_weight * prox_pen
            r += self.config.nmac_penalty * nmac_pairs.get(ac.uid, 0)
            r += self.config.collision_penalty * collision_pairs.get(ac.uid, 0)

        # 3) 阶段信息
            prev_phase = self.prev_phase.get(ac.uid, int(ac.phase))
            ring_level = self._ring_level(ac)
            prev_ring_level = self.prev_ring_level.get(ac.uid, ring_level)

        # 4) 进入 FINAL 的阶段奖励
            if prev_phase != int(ac.phase):
                if ac.phase == FlightPhase.FINAL:
                    r += self.config.final_entry_bonus

        # 5) 第一次入环奖励
            if ac.ring_slot_active and not self.prev_ring_slot.get(ac.uid, False):
                r += self.config.ring_capture_bonus

        # 6) 显式奖励“往里走”
        # 外环(2) -> 内环(1)
            if prev_ring_level == 2 and ring_level == 1:
                r += OUTER_TO_INNER_BONUS

        # 内环(1) -> FINAL(0)
            if prev_ring_level == 1 and ring_level == 0:
                r += INNER_TO_FINAL_BONUS

        # 7) 完成奖励
            if ac.finished and not self.prev_finished.get(ac.uid, False):
                r += self.config.finish_bonus

        # 8) 高度进展 shaping：鼓励向目标层高度靠近
            alt_err = abs(float(ac.position[2]) - self._target_altitude(ac))
            prev_alt_err = self.prev_alt_error.get(ac.uid, alt_err)
            r += ALT_PROGRESS_WEIGHT * (prev_alt_err - alt_err)

        # 9) FINAL 水平收敛 shaping：鼓励向 vertiport 中心靠近
            cx, cy, _ = self.config.center
            xy_err = np.hypot(ac.position[0] - cx, ac.position[1] - cy)
            prev_xy_err = self.prev_xy_error.get(ac.uid, xy_err)

            if ac.phase == FlightPhase.FINAL and not ac.finished:
                r += FINAL_XY_PROGRESS_WEIGHT * (prev_xy_err - xy_err)

        # 10) 环上停留计数
            if ac.phase == FlightPhase.RING and ring_level == prev_ring_level:
                self.ring_dwell_steps[ac.uid] = self.ring_dwell_steps.get(ac.uid, 0) + 1
            else:
                self.ring_dwell_steps[ac.uid] = 0

            dwell = self.ring_dwell_steps[ac.uid]

        # 11) FINAL 停留计数
            if ac.phase == FlightPhase.FINAL and not ac.finished:
                self.final_dwell_steps[ac.uid] = self.final_dwell_steps.get(ac.uid, 0) + 1
            else:
                self.final_dwell_steps[ac.uid] = 0

            final_dwell = self.final_dwell_steps[ac.uid]

        # 12) 环上超时盘旋惩罚
            if ac.phase == FlightPhase.RING:
            # 外环待太久
                if ring_level == 2 and dwell > self.config.outer_to_inner_min_dwell_steps + 12:
                    r -= OUTER_RING_OVERSTAY_PENALTY

            # 内环待太久
                if ring_level == 1 and dwell > self.config.inner_to_final_min_dwell_steps + 10:
                    r -= INNER_RING_OVERSTAY_PENALTY

        # 13) FINAL 超时未落地惩罚
            if ac.phase == FlightPhase.FINAL and not ac.finished:
                if final_dwell > 12:
                    r -= FINAL_DWELL_PENALTY_1
                if final_dwell > 20:
                    r -= FINAL_DWELL_PENALTY_2

        # 14) 平滑惩罚 + 动作幅值惩罚
            this_cmd = applied_actions.get(ac.uid, np.zeros(3, dtype=np.float32))
            prev_cmd = self.prev_commands.get(ac.uid)
            if prev_cmd is None or prev_cmd.shape != this_cmd.shape:
                prev_cmd = np.zeros_like(this_cmd, dtype=np.float32)

            r += self.config.smoothness_penalty_weight * float(np.linalg.norm(this_cmd - prev_cmd))
            r -= ACTION_MAG_PENALTY_WEIGHT * float(np.linalg.norm(this_cmd))

            self.prev_commands[ac.uid] = this_cmd.astype(np.float32)

            rewards[ac.uid] = float(r)

        # 15) 更新缓存
            self.prev_error[ac.uid] = current_error
            self.prev_phase[ac.uid] = int(ac.phase)
            self.prev_ring_slot[ac.uid] = bool(ac.ring_slot_active)
            self.prev_finished[ac.uid] = bool(ac.finished)
            self.prev_ring_level[ac.uid] = ring_level
            self.prev_alt_error[ac.uid] = alt_err
            self.prev_xy_error[ac.uid] = xy_err
            self.prev_path_length[ac.uid] = float(ac.path_length_total)
            self.prev_waiting_ring_distance[ac.uid] = float(ac.waiting_ring_distance)
            self.prev_turn_angle_rad[ac.uid] = float(ac.cumulative_turn_angle_rad)

        return rewards
