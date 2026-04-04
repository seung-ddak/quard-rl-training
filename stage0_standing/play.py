# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ------------------------------------------------------------
    # Test-time overrides
    # ------------------------------------------------------------
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # ------------------------------------------------------------
    # IMPORTANT: set the exact trained run/checkpoint here
    # ------------------------------------------------------------
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = "Apr04_18-10-58_"   # change this
    train_cfg.runner.checkpoint = 1000              # change this
    # ------------------------------------------------------------

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    logger = Logger(env.dt)

    robot_index = 0
    joint_index = 1
    stop_state_log = 200
    stop_rew_log = env.max_episode_length + 1

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # ------------------------------------------------------------
    # Contact debug options
    # ------------------------------------------------------------
    CONTACT_THRESHOLD_Z = 0.5
    PRINT_CONTACT_EVERY = 10
    START_PRINT_AFTER_STEP = 50
    STOP_ON_FIRST_RESET = True

    num_feet = len(env.feet_indices)
    contact_counter = np.zeros(num_feet, dtype=np.int64)
    all_contact_counter = 0
    valid_steps = 0

    print("=" * 100)
    print("PLAY DEBUG STARTED")
    print(f"Task                : {args.task}")
    print(f"Experiment name     : {train_cfg.runner.experiment_name}")
    print(f"Requested load_run  : {train_cfg.runner.load_run}")
    print(f"Requested checkpoint: {train_cfg.runner.checkpoint}")
    print(f"Feet indices        : {env.feet_indices.detach().cpu().numpy()}")
    print(f"Contact threshold z : {CONTACT_THRESHOLD_Z}")
    print(f"Max episode length  : {env.max_episode_length}")
    print("=" * 100)

    # If available, print resolved resume path
    resume_path = getattr(train_cfg.runner, "resume_path", None)
    if resume_path is not None:
        print(f"Resolved resume path: {resume_path}")

    for i in range(10 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # --------------------------------------------------------
        # Correct target joint position
        # actual P target = default_dof_pos + action * action_scale
        # --------------------------------------------------------
        dof_pos_target = (
            env.default_dof_pos[0, joint_index].item()
            + actions[robot_index, joint_index].item() * env.cfg.control.action_scale
        )

        # --------------------------------------------------------
        # Contact debug
        # --------------------------------------------------------
        contact_z = env.contact_forces[robot_index, env.feet_indices, 2].detach().cpu().numpy()
        contact_bool = contact_z > CONTACT_THRESHOLD_Z
        num_contact = int(np.sum(contact_bool))
        all_feet_contact = bool(num_contact == num_feet)

        base_z = env.root_states[robot_index, 2].item()
        base_vel = env.base_lin_vel[robot_index].detach().cpu().numpy()
        base_ang = env.base_ang_vel[robot_index].detach().cpu().numpy()

        # Only count stable, pre-reset region
        if i >= START_PRINT_AFTER_STEP and not dones[robot_index].item():
            contact_counter += contact_bool.astype(np.int64)
            if all_feet_contact:
                all_contact_counter += 1
            valid_steps += 1

        if i >= START_PRINT_AFTER_STEP and (i % PRINT_CONTACT_EVERY == 0):
            print(
                f"[step {i:05d}] "
                f"base_z={base_z:.4f} | "
                f"base_lin={np.round(base_vel, 4)} | "
                f"base_ang={np.round(base_ang, 4)} | "
                f"contact_z={np.round(contact_z, 4)} | "
                f"contact={contact_bool.astype(int)} | "
                f"num_contact={num_contact}/{num_feet} | "
                f"all_feet_contact={all_feet_contact}"
            )

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # --------------------------------------------------------
        # Logger data
        # --------------------------------------------------------
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": dof_pos_target,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "base_z": base_z,
                    "contact_forces_z": contact_z.copy(),
                    "num_contact_feet": num_contact,
                    "all_feet_contact": int(all_feet_contact),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # --------------------------------------------------------
        # Reset handling
        # --------------------------------------------------------
        if dones[robot_index].item():
            print("-" * 100)
            print(
                f"[RESET DETECTED at step {i}] "
                f"base_z={base_z:.4f} | "
                f"contact_z={np.round(contact_z, 4)} | "
                f"contact={contact_bool.astype(int)} | "
                f"num_contact={num_contact}/{num_feet}"
            )
            print("-" * 100)

            if STOP_ON_FIRST_RESET:
                break

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    print("=" * 100)
    print("PLAY DEBUG SUMMARY")
    print(f"Valid pre-reset steps counted: {valid_steps}")
    if valid_steps > 0:
        foot_contact_ratio = contact_counter / valid_steps
        print(f"Per-foot contact ratio : {np.round(foot_contact_ratio, 4)}")
        print(f"All-feet-contact ratio : {all_contact_counter / valid_steps:.4f}")
    else:
        print("No valid pre-reset steps were accumulated.")
    print("=" * 100)


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)