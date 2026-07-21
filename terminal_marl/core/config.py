
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


#同心环规格
@dataclass
class RingSpec:
    index: int
    radius: float
    capacity: int
    height: float
    speed: float | None = None


#全局仿真参数
@dataclass
class SimConfig:
    # Experiment routing.  The legacy default preserves existing checkpoints.
    config_name: str = "python_default"
    experiment_mode: str = "legacy_residual"
    action_mode: str | None = None
    simulation_only: bool = True
    safety_enabled: bool = True
    high_level_interval_steps: int = 10
    metrics_sampling_interval_seconds: float = 20.0
    decision_interval_seconds: float = 10.0
    reconfiguration_dwell_seconds: float = 10.0
    geometry_transition_seconds: float = 0.0
    world_boundary: Optional[Tuple[float, float, float, float, float, float]] = None
    arrival_rate: float = 0.0
    traffic_demand_level: str = "medium"
    arrival_model: str = "deterministic_interval"
    max_active_uavs: int = 40

    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)#终端区中心坐标
    dt: float = 1#仿真每一步时间间隔
    max_steps: int = 1100#一回合最大仿真数

    # Compact 3D terminal geometry
    approach_threshold_radius: float = 1000.0#进近阈值半径
    gate_count: int = 4#进近门数量
    gate_height: float = 500.0#进近门高度
    ring_direction: int = 1#环飞方向（逆时针）
    vertiport_radius: float = 50.0#垂直起降场半径
    final_queue_capacity: int = 1#最终进近段容纳飞机数

    vertiport_physical_radius: float = 25.0#起降场真实半径
    touchdown_accept_radius: float = 40.0
    touchdown_max_height = 15.0
    touchdown_hold_steps_required = 2
    touchdown_max_speed = 8.0
    final_speed_min: float = 6.0

    # Ring capture and transfer timing
    ring_capture_radius_tolerance: float = 30.0#进入环的半径允许误差
    ring_capture_height_tolerance: float = 20.0#进入环的高度允许误差
    outer_to_inner_min_dwell_steps: int = 18#外环飞满多少步才能切进内环
    inner_to_final_min_dwell_steps: int = 12#内环飞满多少步才能最终进近
    outer_to_inner_release_offset_deg: float = 90.0#外环到内环放行角度偏移
    inner_to_final_release_offset_deg: float = 180.0#内环到降落放行角度偏移
    ring_merge_spacing_deg: float = 32.0#环间合并最小角度间隔

    # Time-based release logic
    outer_to_inner_release_horizon_s: float = 14.0#外环到内环放行时间窗口
    inner_to_final_release_horizon_s: float = 9.0#内环到降落放行时间窗口
    ring_to_ring_min_time_gap_s: float = 7.0#环间切换最小时间间隔
    final_min_time_gap_s: float = 12.0#进入Final最小间隔
    release_time_alignment_tolerance_s: float = 0.4#放行时间对齐允许误差

    # Merge-point protection
    merge_point_protection_deg: float = 30.0#合并点保护角度范围
    merge_reservation_steps: int = 7#合并点占用保护步数

    # Ring-following guidance
    ring_tangent_gain: float = 1.0#切向跟踪增益
    ring_radial_gain: float = 1.8#径向修正增益
    ring_radial_error_scale: float = 70.0#径向误差缩放系数
    ring_heading_smoothing: float = 0.92#航向平滑系数
    ring_speed_reduction_max: float = 3.0#环内最大减速值

    # Same-ring spacing control
    same_ring_lookahead_deg: float = 75.0#同环前向探测角度
    same_ring_desired_spacing_deg: float = 60.0#同环期望角度间隔
    same_ring_min_spacing_deg: float = 30.0#同环最小安全间隔
    same_ring_speed_reduction_max: float = 6.0#同环跟驰最大减速

    # Final entry staging
    final_entry_hold_radius: float = 100.0#Final入口等待点半径
    final_entry_hold_altitude: float = 140.0#Final入口等待高度
    final_entry_capture_radius: float = 80.0#Final入口捕获半径
    final_entry_speed: float = 10.0#进入Final速度
    final_approach_speed: float = 7.0#最终进近速度

    # Time-target speed guidance
    time_guidance_blend: float = 0.60#时间引导混合系数
    time_guidance_min_remaining_s: float = 1.5#最小剩余时间触发
    time_guidance_max_speedup: float = 3.0#时间引导最大加速

    # Smooth merge arc / transition curve
    merge_curve_tangent_fraction: float = 0.32#过渡弧切向比例
    merge_curve_tangent_min: float = 70.0#过渡弧切向角度范围
    merge_curve_tangent_max: float = 190.0#过渡弧切向角度范围
    merge_curve_lookahead: float = 0.08#过渡弧前瞻系数
    merge_curve_capture_radius: float = 35.0# 过渡弧捕获半径
    transition_heading_smoothing: float = 0.86#过渡航向平滑
    merge_curve_speed_reduction: float = 2.5#过渡段减速

    # Transition-curve occupancy control
    transition_same_target_radius: float = 80.0#过渡目标半径
    transition_front_desired_spacing: float = 80.0#过渡段期望间距
    transition_front_min_spacing: float = 50.0#过渡段最小间距
    transition_speed_reduction_max: float = 8.0#过渡段最大减速
    transition_endpoint_hold_radius: float = 100.0#过渡终点等待半径
    transition_endpoint_progress_guard: float = 0.72#过渡终点进度保护系数
    transition_target_ring_lookahead_deg: float = 34.0#过渡目标环前瞻角度

    # Aircraft limits
    speed_min: float = 8.0#最小速度
    speed_max: float = 36.0#最大速度
    nominal_speed: float = 28.0#标称速度
    max_acceleration: float = 4.0
    max_turn_rate_deg: float = 12.0#最大转弯率
    max_climb_rate: float = 5.0#最大爬升率
    max_descent_rate: float = 7.0#最大下降率

    # Safety thresholds
    safe_distance: float = 80.0#安全距离
    nmac_distance: float = 40.0#NMAC 距离
    collision_distance: float = 20.0#碰撞距离

    # RL-facing action scaling (residual control around baseline)
    max_action_heading_delta_deg: float = 6.0#航向最大微调角度
    max_action_speed_delta: float = 2.0#速度最大微调量
    max_action_altitude_delta: float = 10.0#高度最大微调量
    max_slot_speed_delta: float = 0.75
    intention_release_threshold: float = -0.5
    timing_release_bias: float = 0.25

    # Observation settings
    neighbor_k: int = 3#每个无人机观测几架邻机
    world_xy_limit: float = 1500.0#仿真世界边界
    world_z_limit: float = 700.0#仿真世界边界

    # Reward shaping
    step_penalty: float = -0.035#每步时间惩罚
    progress_weight: float = 0.05#进近进度奖励
    ring_capture_bonus: float = 0.8#成功入环奖励
    transfer_bonus: float = 0#成功切环奖励
    final_entry_bonus: float = 5.0#进入Final奖励
    finish_bonus: float = 60.0#成功着陆大额奖励
    nmac_penalty: float = -2.0#NMAC惩罚
    collision_penalty: float = -10.0#碰撞惩罚
    separation_penalty_weight: float = -0.02#间隔不足惩罚
    smoothness_penalty_weight: float = -0.02#动作不平滑惩罚
    path_length_penalty_weight: float = 0.012#路径增大惩罚
    waiting_ring_penalty_weight: float = 0.010#等待环内增量距离惩罚
    turn_angle_penalty_weight: float = 0.003#累计转弯角度惩罚
    residual_action_penalty_weight: float = 0.004#residual动作幅值惩罚

    #同心环参数
    rings: List[RingSpec] = field(default_factory=lambda: [
        RingSpec(index=1, radius=200.0, capacity=2, height=200.0, speed=20.0),
        RingSpec(index=2, radius=400.0, capacity=6, height=400.0, speed=28.0),
    ])

    # Initial 2x4 queue at airway exit
    initial_aircraft_count: int = 6#初始飞机总数8架
    formation_rows: int = 3#队形4行2列
    formation_cols: int = 2#队形4行2列
    row_spacing: float = 70.0#行列间距
    col_spacing: float = 60.0#行列间距
    spawn_margin: float = 250.0#初始生成位置偏移
    inbound_bearing_deg: float = 180.0#进入方向
    spawn_altitude: float = 600.0#生成高度

    def gate_angles_rad(self) -> List[float]:
        return [2.0 * math.pi * i / self.gate_count for i in range(self.gate_count)]
