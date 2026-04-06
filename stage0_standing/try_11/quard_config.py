# -*- coding: utf-8 -*-
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


# =============================================================================
# Stage 0: Standing balance
# =============================================================================
class QuardStage0Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.22]

        default_joint_angles = {
            "j_L_shoulder": 0.0,
            "j_LF_hip": 1.00,
            "j_LF_knee": -1.00,
            "j_RF_shoulder": 0.0,
            "j_RF_hip": 1.00,
            "j_RF_knee": -1.00,
            "j_LR_shoulder": 0.0,
            "j_LR_hip": 1.00,
            "j_LR_knee": -1.00,
            "j_RR_shoulder": 0.0,
            "j_RR_hip": 1.00,
            "j_RR_knee": -1.00,
        }

        randomize_dof_reset = False
        dof_pos_reset_scale_low = 0.99
        dof_pos_reset_scale_high = 1.01

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"shoulder": 30.0, "hip": 30.0, "knee": 30.0}
        damping = {"shoulder": 1.0, "hip": 1.0, "knee": 1.0}
        action_scale = 0.15
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "/mnt/gstore/home/xiangyue/RL/sim/quard_rl.urdf"
        name = "quard"
        foot_name = "calf"
        penalize_contacts_on = ["thigh", "shoulder"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.90
        base_height_target = 0.20
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            upright = 4.0
            base_height_gaussian = 4.0
            stand_still = 1.0
            four_feet_contact = 2.0

            orientation = -0.5
            ang_vel_xy = -0.2
            dof_vel = -0.01
            action_rate = -0.01
            torques = -0.0002
            collision = -1.0
            dof_pos_limits = -2.0

            low_velocity = 0.0
            feet_force_symmetry = 0.0
            feet_symmetry = 0.0

            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            base_height = 0.0
            dof_acc = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0
            termination = 0.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False


class QuardStage0CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "quard_stage0"
        max_iterations = 1000
        resume = False
        load_run = ""
        checkpoint = -1


# =============================================================================
# Stage 1: Flat ground walking
# =============================================================================
class QuardStage1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.22]
        default_joint_angles = {
            "j_L_shoulder": 0.0,
            "j_LF_hip": 1.0,
            "j_LF_knee": -1.0,
            "j_RF_shoulder": 0.0,
            "j_RF_hip": 1.0,
            "j_RF_knee": -1.0,
            "j_LR_shoulder": 0.0,
            "j_LR_hip": 1.0,
            "j_LR_knee": -1.0,
            "j_RR_shoulder": 0.0,
            "j_RR_hip": 1.0,
            "j_RR_knee": -1.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"shoulder": 25.0, "hip": 25.0, "knee": 25.0}
        damping = {"shoulder": 0.8, "hip": 0.8, "knee": 0.8}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "/mnt/gstore/home/xiangyue/RL/sim/quard_rl.urdf"
        name = "quard"
        foot_name = "calf"
        penalize_contacts_on = ["thigh", "shoulder"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        collapse_fixed_joints = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [-0.3, 0.5]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.20
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -0.5
            torques = -0.0002
            dof_vel = 0.0
            dof_acc = -2.5e-7
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            dof_pos_limits = -10.0
            stand_still = 0.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False


class QuardStage1CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "quard_stage1"
        max_iterations = 1500
        resume = True
        load_run = "Apr03_15-03-52_"
        checkpoint = 2500


# =============================================================================
# Aliases
# =============================================================================
QuardRoughCfg = QuardStage0Cfg
QuardRoughCfgPPO = QuardStage0CfgPPO
QuardFlatCfg = QuardStage0Cfg
QuardFlatCfgPPO = QuardStage0CfgPPO