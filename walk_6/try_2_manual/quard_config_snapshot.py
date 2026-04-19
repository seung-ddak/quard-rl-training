# -*- coding: utf-8 -*-
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# -----------------------------------------------------------------------------
# Common robot setup shared by walk_0 and all walk curriculum levels
# -----------------------------------------------------------------------------

_STANCE_SHOULDER = 0.0
_STANCE_HIP = 0.70
_STANCE_KNEE = -0.44

_BASE_HEIGHT = 0.235
_SPAWN_Z = 0.25

_DEFAULT_ANGLES = {
    "j_L_shoulder": _STANCE_SHOULDER,
    "j_LF_hip": _STANCE_HIP,
    "j_LF_knee": _STANCE_KNEE,
    "j_RF_shoulder": _STANCE_SHOULDER,
    "j_RF_hip": _STANCE_HIP,
    "j_RF_knee": _STANCE_KNEE,
    "j_LR_shoulder": _STANCE_SHOULDER,
    "j_LR_hip": -_STANCE_HIP,
    "j_LR_knee": -_STANCE_KNEE,
    "j_RR_shoulder": _STANCE_SHOULDER,
    "j_RR_hip": -_STANCE_HIP,
    "j_RR_knee": -_STANCE_KNEE,
}


class _QuardCommonCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        episode_length_s = 20.0
        env_spacing = 2.5

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, _SPAWN_Z]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = dict(_DEFAULT_ANGLES)

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"shoulder": 20.0, "hip": 28.0, "knee": 32.0}
        damping = {"shoulder": 0.70, "hip": 1.00, "knee": 1.20}
        decimation = 4
        action_scale = 0.25

    class asset(LeggedRobotCfg.asset):
        file = "/mnt/gstore/home/xiangyue/RL/sim/quard_rl.urdf"
        name = "quard"
        foot_name = "calf"
        penalize_contacts_on = ["shoulder", "thigh"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        push_robots = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 4.0
        startup_hold_time_s = 0.0
        min_command_norm = 0.10
        class ranges:
            lin_vel_x = [0.20, 0.45]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = _BASE_HEIGHT
        base_height_sigma = 0.025
        target_air_time = 0.10
        swing_height_target = 0.028
        swing_height_sigma = 0.012
        tripod_airtime_threshold = 0.18
        tripod_height_threshold = 0.035
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.30
            tracking_ang_vel = 0.00
            feet_air_time = 0.60
            swing_height = 0.80
            diagonal_gait = 0.50
            all_feet_stepping = 0.40

            base_height = 0.0
            base_height_gaussian = 1.40
            dof_pos_limits = -6.0
            support_balance = 0.30
            left_right_stance_symmetry = 0.0
            front_rear_stance_symmetry = 0.15
            orientation = -0.60
            lin_vel_z = -2.0
            ang_vel_xy = -0.12

            collision = -10.0
            feet_slip = -0.60
            rear_feet_slip = -0.30
            feet_stumble = -1.50
            stand_when_should_walk = -0.80
            tripod_penalty = 0.0

            torques = -0.00015
            dof_vel = -0.0012
            dof_acc = -3.0e-7
            action_rate = -0.012

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class _QuardCommonCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.006

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 1800
        save_interval = 50
        resume = False


# =============================================================================
# ---- walk_0 : 원래 기본 학습 / legacy direct walk ----
# =============================================================================
class QuardWalk0Cfg(_QuardCommonCfg):
    pass


class QuardWalk0CfgPPO(_QuardCommonCfgPPO):
    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_0"
        experiment_name = "walk_0"
        max_iterations = 1000
        resume = False


# =============================================================================
# ---- walk_1 : 전진 유지 + 끌기 보행 감소 ----
# =============================================================================
class QuardWalk1Cfg(_QuardCommonCfg):
    pass


class QuardWalk1CfgPPO(_QuardCommonCfgPPO):
    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_1"
        experiment_name = "walk_1"
        max_iterations = 800
        resume = True
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_0/Apr13_16-51-25_walk_0"
        checkpoint = 950


# =============================================================================
# ---- walk_2 : 대각 보행 강화 + tripod 꼼수 억제 ----
# =============================================================================
class QuardWalk2Cfg(_QuardCommonCfg):
    class commands(_QuardCommonCfg.commands):
        class ranges:
            lin_vel_x = [0.25, 0.60]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.15, 0.15]
            heading = [0.0, 0.0]

    class rewards(_QuardCommonCfg.rewards):
        target_air_time = 0.11
        swing_height_target = 0.030
        tripod_airtime_threshold = 0.16
        tripod_height_threshold = 0.032
        only_positive_rewards = True

        class scales(_QuardCommonCfg.rewards.scales):
            tracking_lin_vel = 1.35
            tracking_ang_vel = 0.20
            feet_air_time = 0.90
            swing_height = 1.00
            diagonal_gait = 0.90
            all_feet_stepping = 0.80

            base_height = 0.0
            base_height_gaussian = 1.35
            dof_pos_limits = -6.5
            support_balance = 0.35
            left_right_stance_symmetry = 0.0
            front_rear_stance_symmetry = 0.18
            orientation = -0.65
            lin_vel_z = -2.2
            ang_vel_xy = -0.14

            collision = -12.0
            feet_slip = -1.10
            rear_feet_slip = -0.70
            feet_stumble = -1.80
            stand_when_should_walk = -1.10
            tripod_penalty = -0.80

            torques = -0.00018
            dof_vel = -0.0013
            dof_acc = -3.5e-7
            action_rate = -0.013

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class QuardWalk2CfgPPO(_QuardCommonCfgPPO):
    class algorithm(_QuardCommonCfgPPO.algorithm):
        entropy_coef = 0.0055

    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_2"
        experiment_name = "walk_2"
        max_iterations = 1200
        resume = True
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_1/Apr13_17-43-12_walk_1"
        checkpoint = 1750


# =============================================================================
# ---- walk_3 : gait 품질 정제 + 명령 범위 확대 ----
# =============================================================================
class QuardWalk3Cfg(_QuardCommonCfg):
    class commands(_QuardCommonCfg.commands):
        class ranges:
            lin_vel_x = [0.30, 0.70]
            lin_vel_y = [-0.05, 0.05]
            ang_vel_yaw = [-0.25, 0.25]
            heading = [0.0, 0.0]

    class rewards(_QuardCommonCfg.rewards):
        base_height_sigma = 0.022
        target_air_time = 0.12
        swing_height_target = 0.032
        swing_height_sigma = 0.011
        tripod_airtime_threshold = 0.15
        tripod_height_threshold = 0.030
        only_positive_rewards = False

        class scales(_QuardCommonCfg.rewards.scales):
            tracking_lin_vel = 1.45
            tracking_ang_vel = 0.35
            feet_air_time = 1.10
            swing_height = 1.20
            diagonal_gait = 1.30
            all_feet_stepping = 1.00

            base_height = 0.0
            base_height_gaussian = 1.20
            dof_pos_limits = -7.0
            support_balance = 0.40
            left_right_stance_symmetry = 0.0
            front_rear_stance_symmetry = 0.20
            orientation = -0.75
            lin_vel_z = -2.4
            ang_vel_xy = -0.16

            collision = -15.0
            feet_slip = -1.70
            rear_feet_slip = -1.10
            feet_stumble = -2.20
            stand_when_should_walk = -1.40
            tripod_penalty = -1.60

            torques = -0.00020
            dof_vel = -0.0015
            dof_acc = -4.0e-7
            action_rate = -0.015

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class QuardWalk3CfgPPO(_QuardCommonCfgPPO):
    class algorithm(_QuardCommonCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_3"
        experiment_name = "walk_3"
        max_iterations = 1500
        resume = True
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_2/Apr14_05-42-35_walk_2"
        checkpoint = 2950


# =============================================================================
# ---- walk_4 : 직진 정제 단계 / gait 품질 활성화 ----
#   - 측면(lin_vel_y) 및 회전(ang_vel_yaw) 명령 완전 차단 → 순수 직진 refinement
#   - left_right_stance_symmetry 재활성화 (좌우 비대칭 직접 페널티)
#   - tracking_lin_vel 약화, diagonal_gait / swing_height / tripod_penalty 강화
# =============================================================================
class QuardWalk4Cfg(_QuardCommonCfg):
    class commands(_QuardCommonCfg.commands):
        class ranges:
            lin_vel_x = [0.30, 0.55]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(_QuardCommonCfg.rewards):
        base_height_sigma = 0.022
        target_air_time = 0.13
        # swing_height now uses the MEAN over swinging feet (two-sided Gaussian),
        # so target/sigma are interpreted as the expected trot step height.
        swing_height_target = 0.028
        swing_height_sigma = 0.020
        left_right_pose_sigma = 0.15
        # foot_ground_usage: any foot airborne longer than this (seconds) starts
        # accumulating penalty. Normal trot swing is ~0.2 s, so 0.45 s gives
        # ample slack for a single missed step but catches the "permanent-air"
        # failure mode seen in try_2.
        foot_max_air_time = 0.45
        tripod_airtime_threshold = 0.14
        tripod_height_threshold = 0.030
        only_positive_rewards = False

        class scales(_QuardCommonCfg.rewards.scales):
            tracking_lin_vel = 1.20
            tracking_ang_vel = 1.00
            feet_air_time = 1.30
            swing_height = 1.40
            diagonal_gait = 1.80
            all_feet_stepping = 1.30

            base_height = 0.0
            base_height_gaussian = 1.20
            dof_pos_limits = -7.0
            support_balance = 0.45
            left_right_stance_symmetry = 0.60
            front_rear_stance_symmetry = 0.25
            orientation = -2.50
            lin_vel_z = -2.4
            ang_vel_xy = -0.18

            collision = -15.0
            feet_slip = -1.80
            rear_feet_slip = -1.20
            feet_stumble = -2.30
            stand_when_should_walk = -1.50
            tripod_penalty = -2.00
            foot_ground_usage = -3.00

            torques = -0.00020
            dof_vel = -0.0015
            dof_acc = -4.0e-7
            action_rate = -0.015

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class QuardWalk4CfgPPO(_QuardCommonCfgPPO):
    class algorithm(_QuardCommonCfgPPO.algorithm):
        entropy_coef = 0.0045

    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_4"
        experiment_name = "walk_4"
        max_iterations = 1500
        resume = True
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_3/Apr14_06-32-46_walk_3"
        checkpoint = 4450


# =============================================================================
# ---- walk_5 : 발 높이 강화 / max-clearance straight walk ----
#   - domain randomization은 아직 적용하지 않음
#   - walk_4의 직진성과 대칭성은 유지
#   - 발을 "조금 드는 정도"가 아니라, 한 스윙 구간에서 실제로 더 높은
#     clearance를 만들도록 max-clearance 기반 swing reward를 사용
# =============================================================================
class QuardWalk5Cfg(_QuardCommonCfg):
    class domain_rand(_QuardCommonCfg.domain_rand):
        randomize_friction = False
        push_robots = False

    class commands(_QuardCommonCfg.commands):
        class ranges:
            lin_vel_x = [0.28, 0.50]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(_QuardCommonCfg.rewards):
        base_height_sigma = 0.022
        # try_13 (2026-04-18): try_12 plateau (swing_height raw 0.279).
        # Diagnosis: control-penalty trade-off saturates lift; one diagonal
        # pair scuffs while the other lifts. Fix = make lifting cheap and
        # force both pairs to actually swing.
        target_air_time = 0.18                # try_12: 0.16 -> longer real trot swing
        swing_height_target = 0.040
        swing_height_sigma = 0.030            # try_12: 0.025 -> gradient at low effective
        swing_height_min_mode = True          # per-foot min + stuck-aware clearance
        swing_peak_stance_reset = 0.30        # try_12: 0.32 -> tighter
        foot_stance_max_time = 0.30           # try_12: 0.35 -> force scuffing pair to lift
        all_feet_stepping_denom = 4.0
        left_right_mode = "ema_all"
        left_right_pose_sigma = 0.30
        pose_ema_alpha = 0.02
        foot_max_air_time = 0.45
        tripod_airtime_threshold = 0.14
        tripod_height_threshold = 0.030
        only_positive_rewards = False

        class scales(_QuardCommonCfg.rewards.scales):
            tracking_lin_vel = 1.30           # try_12: 1.10 -> both pairs must propel
            tracking_ang_vel = 1.00
            feet_air_time = 2.40              # try_12: 1.70 -> stronger swing-time pull
            swing_height = 6.00               # try_12: 4.50 -> overcome control penalty
            diagonal_gait = 2.00              # try_12: 1.60 -> perfect_trot pattern
            all_feet_stepping = 2.60          # try_12: 2.20

            base_height = 0.0
            base_height_gaussian = 1.15
            dof_pos_limits = -7.0
            support_balance = 0.42
            left_right_stance_symmetry = 1.20
            front_rear_stance_symmetry = 0.25
            orientation = -2.40
            lin_vel_z = -2.5
            ang_vel_xy = -0.20

            collision = -15.0
            feet_slip = -2.00
            rear_feet_slip = -1.35
            feet_stumble = -2.40
            stand_when_should_walk = -1.60
            tripod_penalty = -2.30
            stuck_foot_penalty = -5.00        # try_12: -3.50 -> stronger anti-scuff

            # try_13: control penalties cut ~30% so foot lifting is cheap.
            torques = -0.00014                # try_12: -0.00020
            dof_vel = -0.0010                 # try_12: -0.0015
            dof_acc = -2.8e-7                 # try_12: -4.0e-7
            action_rate = -0.011              # try_12: -0.015

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class QuardWalk5CfgPPO(_QuardCommonCfgPPO):
    class algorithm(_QuardCommonCfgPPO.algorithm):
        entropy_coef = 0.0050             # try_12: 0.0040 -> escape plateau

    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_5"
        experiment_name = "walk_5"
        max_iterations = 2000             # try_13: longer budget for clearance breakthrough
        resume = True
        # try_13: per user request, resume from walk_4 final (ckpt 5950) again,
        # not from any prior walk_5 try, to escape try_12 plateau.
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_4/Apr14_16-06-11_walk_4"
        checkpoint = 5950


# =============================================================================
# ---- walk_6 : 품질 재설계 단계 (Quality Redesign, re-defined 2026-04-18) ----
#   walk_5 try_13 결과:
#     - mean_reward 133→171 (+38), 대각 위상/scuff 해소 OK
#     - swing_height raw 0.41 plateau (min over feet 한계: worst foot ≈ 0)
#     - 직진 deviation 페널티 부재 (지형 외란 시 휘어 나갈 위험)
#   walk_6 도입:
#     - swing_height MIX mode (w_min=0.6, mean=0.4) → worst foot cap 완화
#     - _reward_diagonal_propulsion 신규 → 양 쌍 z-force 균형 강제
#     - _reward_straight_line_deviation 신규 → 위치 기반 직진 strict
#     - DR은 walk_7로 미룸 (품질 먼저, 강건성 다음)
# =============================================================================
class QuardWalk6Cfg(_QuardCommonCfg):
    class domain_rand(_QuardCommonCfg.domain_rand):
        randomize_friction = False
        push_robots = False

    class commands(_QuardCommonCfg.commands):
        class ranges:
            lin_vel_x = [0.28, 0.50]          # walk_5와 동일 (직진 정제 유지)
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(_QuardCommonCfg.rewards):
        base_height_sigma = 0.022
        target_air_time = 0.18
        swing_height_target = 0.040
        swing_height_sigma = 0.030
        # walk_5 try_13 flags 유지
        swing_height_min_mode = True
        # try_2: try_1 mix mode w_min=0.6 효과 미미 → 0.4로 더 mean에 비중
        swing_height_min_weight = 0.40
        swing_peak_stance_reset = 0.30
        foot_stance_max_time = 0.30
        all_feet_stepping_denom = 4.0
        left_right_mode = "ema_all"
        left_right_pose_sigma = 0.30
        pose_ema_alpha = 0.02
        foot_max_air_time = 0.45
        tripod_airtime_threshold = 0.14
        tripod_height_threshold = 0.030
        # try_2: straight dev sigma 0.30→0.60 (drift 더 허용, 페널티 완화)
        straight_dev_sigma = 0.60
        pair_force_ema_alpha = 0.02
        # try_2: pair sigma 0.12→0.20 (ratio 0.3~0.7도 부분 보상)
        pair_force_sigma = 0.20
        only_positive_rewards = False

        class scales(_QuardCommonCfg.rewards.scales):
            tracking_lin_vel = 1.20
            tracking_ang_vel = 1.00
            feet_air_time = 2.40
            swing_height = 6.00
            diagonal_gait = 2.00
            all_feet_stepping = 2.60

            base_height = 0.0
            base_height_gaussian = 1.15
            dof_pos_limits = -7.0
            support_balance = 0.42
            left_right_stance_symmetry = 1.20
            front_rear_stance_symmetry = 0.25
            orientation = -2.40
            lin_vel_z = -2.5
            ang_vel_xy = -0.20

            collision = -15.0
            feet_slip = -2.00
            rear_feet_slip = -1.35
            feet_stumble = -2.40
            stand_when_should_walk = -1.60
            tripod_penalty = -2.30
            stuck_foot_penalty = -5.00

            # try_2: straight_line scale -1.5 → -0.50 (정책 혼란 방지)
            straight_line_deviation = -0.50
            # try_2: diagonal_propulsion scale 1.5 → 2.50 (sigma 확대와 함께 강화)
            diagonal_propulsion = 2.50

            torques = -0.00014
            dof_vel = -0.0010
            dof_acc = -2.8e-7
            action_rate = -0.011

            pose_hold = 0.0
            knee_range_violation = 0.0
            stand_still = 0.0
            termination = 0.0


class QuardWalk6CfgPPO(_QuardCommonCfgPPO):
    class algorithm(_QuardCommonCfgPPO.algorithm):
        # try_2: try_1 action_noise_std 0.40 폭증 → entropy 더 낮춰 안정화
        entropy_coef = 0.0035

    class runner(_QuardCommonCfgPPO.runner):
        run_name = "walk_6"
        experiment_name = "walk_6"
        max_iterations = 2000
        resume = True
        # try_2: try_1 정책 망가져 폐기. walk_5 try_13 final ckpt에서 다시 시작.
        load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_5/Apr18_09-39-52_walk_5"
        checkpoint = 7950


# =============================================================================
# ---- optional compatibility aliases ----
# =============================================================================
QuardDirectWalkCfg = QuardWalk0Cfg
QuardDirectWalkCfgPPO = QuardWalk0CfgPPO
