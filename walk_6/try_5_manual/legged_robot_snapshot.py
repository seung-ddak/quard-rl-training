# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()

        self._calibrate_init_height()

        self.init_done = True

    def _calibrate_init_height(self):
        """
        Automatically adjust the initial base height before training.

        Rules:
          - settled_z < 0.05         -> robot is buried in the ground
          - settled_z > target * 1.5 -> robot starts too high
        """
        CALIB_STEPS = 150
        BURIED_THRESH = 0.05
        FLOAT_FACTOR = 1.5
        MAX_RETRY = 3

        target_z = getattr(self.cfg.rewards, "base_height_target", 0.20)

        print("\n[Calibration] Initial height check started...")
        print(f"[Calibration] Config pos z = {self.base_init_state[2].item():.4f} m")
        print(f"[Calibration] base_height_target = {target_z:.4f} m")

        for attempt in range(MAX_RETRY):
            zero_actions = torch.zeros(
                self.num_envs,
                self.num_actions,
                dtype=torch.float,
                device=self.device,
            )

            for _ in range(CALIB_STEPS):
                torques = self._compute_torques(zero_actions)
                self.gym.set_dof_actuation_force_tensor(
                    self.sim, gymtorch.unwrap_tensor(torques)
                )
                self.gym.simulate(self.sim)
                if self.device == "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)

            self.gym.refresh_actor_root_state_tensor(self.sim)
            settled_z = self.root_states[:, 2].mean().item()

            print(f"[Calibration] Attempt {attempt + 1}: settled base z = {settled_z:.4f} m")

            buried = settled_z < BURIED_THRESH
            too_high = settled_z > target_z * FLOAT_FACTOR

            if not buried and not too_high:
                print("[Calibration] Height looks good. No correction needed.")
                break

            if buried:
                correction = target_z - settled_z + 0.02
                reason = f"robot is buried (z={settled_z:.4f})"
            else:
                correction = target_z - settled_z
                reason = f"robot starts too high (z={settled_z:.4f})"

            new_z = self.base_init_state[2].item() + correction
            print(f"[Calibration] Warning: {reason} -> updating init z to {new_z:.4f} m")

            self.base_init_state[2] = new_z

            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self._reset_dofs(all_env_ids)
            self._reset_root_states(all_env_ids)

        else:
            print(f"[Calibration] Failed to converge after {MAX_RETRY} attempts. Using current value.")

        final_z = self.base_init_state[2].item()
        print(f"[Calibration] Final init z = {final_z:.4f} m")
        print(f"[Calibration] Update quard_config.py pos z to {final_z:.4f} if needed.\n")

        try:
            record_root = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "training_record"
            )
            record_root = os.path.abspath(record_root)
            os.makedirs(record_root, exist_ok=True)

            calib_path = os.path.join(record_root, "calibrated_init_height.txt")
            with open(calib_path, "w", encoding="utf-8") as f:
                f.write(f"calibrated_init_z = {final_z:.6f}\n")
                f.write(f"base_height_target = {target_z:.6f}\n")
                f.write(f"timestamp = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"[Calibration] Saved result: {calib_path}")
        except Exception as e:
            print(f"[Calibration] Failed to save calibration file: {e}")

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self._update_gait_tracking()
        self.check_termination()
        self.compute_reward()
        self.last_step_contacts[:] = self.contact_forces[:, self.feet_indices, 2] > 1.0

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _update_gait_tracking(self):
        """Per-step gait-tracking buffers used by walk_6+ rewards."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.last_step_contacts
        airborne = ~contact
        touchdown = contact & (~prev_contact)
        # foot_positions[:,:,2] is in a link-origin frame where stance feet
        # sit at NEGATIVE z (around -0.13 for this asset). So we work with
        # lift-relative-to-ground = z - min(z across feet). Stuck feet → 0
        # lift; airborne feet → positive lift. This was a long-standing bug
        # that froze _max_swing_height at 0 for all walk_5/6 trainings.
        foot_heights_raw = self.foot_positions[:, :, 2]
        ground_ref = torch.min(foot_heights_raw, dim=1, keepdim=True).values
        foot_heights = torch.clamp(foot_heights_raw - ground_ref, min=0.0)

        # running max clearance during current swing
        self._max_swing_height = torch.where(
            airborne,
            torch.maximum(self._max_swing_height, foot_heights),
            self._max_swing_height,
        )
        # at touchdown, freeze the peak and reset running max
        self._last_swing_peak = torch.where(
            touchdown, self._max_swing_height, self._last_swing_peak
        )
        self._max_swing_height = torch.where(
            touchdown, torch.zeros_like(self._max_swing_height), self._max_swing_height
        )

        # per-foot stance time
        self._foot_stance_time = torch.where(
            contact,
            self._foot_stance_time + self.dt,
            torch.zeros_like(self._foot_stance_time),
        )

        # dof position EMA
        alpha = float(getattr(self.cfg.rewards, "pose_ema_alpha", 0.02))
        if not self._dof_pos_ema_initialized:
            self._dof_pos_ema[:] = self.dof_pos
            self._dof_pos_ema_initialized = True
        else:
            self._dof_pos_ema.mul_(1.0 - alpha).add_(self.dof_pos, alpha=alpha)

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.last_step_contacts[env_ids] = False
        self.startup_ready[env_ids] = (self.startup_hold_steps == 0)
        self.startup_stable_steps[env_ids] = 0
        if hasattr(self, "_max_swing_height"):
            self._max_swing_height[env_ids] = 0.0
        if hasattr(self, "_foot_stance_time"):
            self._foot_stance_time[env_ids] = 0.0
        if hasattr(self, "_last_swing_peak"):
            self._last_swing_peak[env_ids] = 0.0
        if hasattr(self, "_dof_pos_ema"):
            self._dof_pos_ema[env_ids] = self.dof_pos[env_ids]
        if hasattr(self, "_spawn_xy"):
            self._spawn_xy[env_ids] = self.root_states[env_ids, :2]
            fwd = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
            fwd_xy = fwd[:, :2]
            fwd_norm = torch.norm(fwd_xy, dim=1, keepdim=True).clamp(min=1e-6)
            self._spawn_forward[env_ids] = fwd_xy / fwd_norm
        if hasattr(self, "_pair_force_ema"):
            self._pair_force_ema[env_ids] = 0.5

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed: [None, plane, heightfield, trimesh]")

        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0],
                    friction_range[1],
                    (num_buckets, 1),
                    device="cpu",
                )
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2.0
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.startup_hold_steps > 0:
            startup_thr = float(getattr(self.cfg.rewards, "startup_contact_threshold", 0.5))
            foot_contact = self.contact_forces[:, self.feet_indices, 2] > startup_thr
            all_contact = torch.all(foot_contact, dim=1)
            upright_ok = (-self.projected_gravity[:, 2]) > 0.80
            low_motion = (torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.20) & (torch.norm(self.base_ang_vel[:, :2], dim=1) < 0.60)
            stable = all_contact & upright_ok & low_motion

            waiting = ~self.startup_ready
            self.startup_stable_steps = torch.where(
                waiting,
                torch.where(stable, self.startup_stable_steps + 1, torch.zeros_like(self.startup_stable_steps)),
                self.startup_stable_steps
            )
            self.startup_ready |= self.startup_stable_steps >= self.startup_hold_steps

            waiting_envs = ~self.startup_ready
            self.commands[waiting_envs, :3] = 0.0
            if self.commands.shape[1] > 3:
                self.commands[waiting_envs, 3] = 0.0

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)

        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > self.min_command_norm
        ).unsqueeze(1)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        randomize_reset = getattr(self.cfg.init_state, "randomize_dof_reset", True)
        scale_low = getattr(self.cfg.init_state, "dof_pos_reset_scale_low", 0.98)
        scale_high = getattr(self.cfg.init_state, "dof_pos_reset_scale_high", 1.02)

        if randomize_reset:
            rand_scales = torch_rand_float(
                scale_low,
                scale_high,
                (len(env_ids), self.num_dof),
                device=self.device,
            )
            reset_pos = self.default_dof_pos.repeat(len(env_ids), 1) * rand_scales
        else:
            reset_pos = self.default_dof_pos.repeat(len(env_ids), 1)

        lower = self.dof_pos_limits[:, 0].unsqueeze(0) + 1e-4
        upper = self.dof_pos_limits[:, 1].unsqueeze(0) - 1e-4
        reset_pos = torch.clamp(reset_pos, min=lower, max=upper)

        self.dof_pos[env_ids] = reset_pos
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0,
                1.0,
                (len(env_ids), 2),
                device=self.device,
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel,
            max_vel,
            (self.num_envs, 2),
            device=self.device,
        )
        self.gym.set_actor_root_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
        )

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2],
            dim=1,
        )

        move_up = distance > self.terrain.env_length / 2
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        ) * ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids],
        ]

    def update_command_curriculum(self, env_ids):
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0

        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )

        return noise_vec

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx),
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )

        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.last_step_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self._max_swing_height = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # walk_6: per-foot stance-time counter (parallels feet_air_time but for ground)
        self._foot_stance_time = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # walk_6: peak clearance of the most recently completed swing, per foot
        self._last_swing_peak = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # walk_6: EMA of dof positions for time-averaged left/right symmetry
        self._dof_pos_ema = torch.zeros_like(self.dof_pos)
        self._dof_pos_ema_initialized = False
        # walk_6 (re-defined): straight-line deviation reward needs spawn pose
        self._spawn_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._spawn_forward = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._spawn_forward[:, 0] = 1.0
        # walk_6: EMA of diagonal-pair vertical force ratio (pair_a / total)
        self._pair_force_ema = torch.full((self.num_envs,), 0.5, dtype=torch.float, device=self.device, requires_grad=False)
        self.startup_ready = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        if self.startup_hold_steps == 0:
            self.startup_ready[:] = True
        self.startup_stable_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.foot_positions = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_states[:, self.feet_indices, 7:10]
        # Separate front/rear foot views (names-based ordering, robust to IsaacGym body order)
        self.front_foot_positions = self.rigid_body_states[:, self.front_feet_indices, 0:3]
        self.front_foot_velocities = self.rigid_body_states[:, self.front_feet_indices, 7:10]
        self.rear_foot_positions = self.rigid_body_states[:, self.rear_feet_indices, 0:3]
        self.rear_foot_velocities = self.rigid_body_states[:, self.rear_feet_indices, 7:10]

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True

            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gains for joint {name} were not defined. Set to zero.")

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        missing_rewards = []

        for name in list(self.reward_scales.keys()):
            if name == "termination":
                continue
            func_name = "_reward_" + name
            if not hasattr(self, func_name):
                missing_rewards.append(name)
                self.reward_scales.pop(name)
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, func_name))

        if missing_rewards:
            print(f"[WARN] Skipping reward scales without matching functions: {missing_rewards}")

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows,
            self.terrain.tot_cols,
        ).to(self.device)

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )

        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows,
            self.terrain.tot_cols,
        ).to(self.device)

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        front_feet_names = [
            s for s in feet_names
            if ("lf" in s.lower()) or ("rf" in s.lower()) or ("front" in s.lower())
        ]
        rear_feet_names = [
            s for s in feet_names
            if ("lr" in s.lower()) or ("rr" in s.lower()) or ("rear" in s.lower()) or ("hind" in s.lower()) or ("back" in s.lower())
        ]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        print(f"[Asset] foot_name match ('{self.cfg.asset.foot_name}') -> {feet_names}")
        print(f"[Asset] front feet -> {front_feet_names}")
        print(f"[Asset] rear feet  -> {rear_feet_names}")
        print(f"[Asset] penalized bodies -> {penalized_contact_names}")
        print(f"[Asset] termination bodies -> {termination_contact_names}")

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list,
            device=self.device,
            requires_grad=False,
        )

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim,
                env_lower,
                env_upper,
                int(np.sqrt(self.num_envs)),
            )

            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle,
                actor_handle,
                body_props,
                recomputeInertia=True,
            )

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.actor_handles[0],
                feet_names[i],
            )

        # Robust front/rear split for quadrupeds. Fallback to first/last two if names do not match.
        if len(front_feet_names) == 2 and len(rear_feet_names) == 2:
            self.front_feet_indices = torch.zeros(2, dtype=torch.long, device=self.device, requires_grad=False)
            self.rear_feet_indices = torch.zeros(2, dtype=torch.long, device=self.device, requires_grad=False)
            for i in range(2):
                self.front_feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], front_feet_names[i]
                )
                self.rear_feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], rear_feet_names[i]
                )
        else:
            self.front_feet_indices = self.feet_indices[:2].clone()
            self.rear_feet_indices = self.feet_indices[2:].clone()

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.actor_handles[0],
                penalized_contact_names[i],
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.actor_handles[0],
                termination_contact_names[i],
            )

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            self.terrain_levels = torch.randint(
                0,
                max_init_level + 1,
                (self.num_envs,),
                device=self.device,
            )
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))

            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.startup_hold_steps = int(np.ceil(getattr(self.cfg.commands, "startup_hold_time_s", 0.0) / self.dt))
        self.min_command_norm = getattr(self.cfg.commands, "min_command_norm", 0.2)
        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def _draw_debug_vis(self):
        if not self.terrain.cfg.measure_heights:
            return

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))

        for i in range(self.num_envs):
            base_pos = self.root_states[i, :3].cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(
                self.base_quat[i].repeat(heights.shape[0]),
                self.height_points[i],
            ).cpu().numpy()

            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Cannot measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + self.root_states[env_ids, :3].unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points),
                self.height_points,
            ) + self.root_states[:, :3].unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]

        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt

        target_air_time = getattr(self.cfg.rewards, "target_air_time", 0.12)
        rew_air_time = torch.sum((self.feet_air_time - target_air_time) * first_contact, dim=1)
        walk_mask = (self.episode_length_buf >= self.startup_hold_steps).float()
        rew_air_time *= (torch.norm(self.commands[:, :2], dim=1) > 0.05).float() * walk_mask
        self.feet_air_time *= ~contact_filt

        return rew_air_time

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_base_height_gaussian(self):
        base_height = self.root_states[:, 2]
        height_error = base_height - self.cfg.rewards.base_height_target
        sigma = getattr(self.cfg.rewards, "base_height_sigma", 0.10)
        return torch.exp(-torch.square(height_error) / (2.0 * sigma * sigma))


    def _reward_upright(self):
        upright = -self.projected_gravity[:, 2]
        return torch.clamp(upright, min=0.0, max=1.0)

    def _reward_forward_vel(self):
        # 아주 미세한 전진은 보상하지 않음
        return torch.clamp(self.base_lin_vel[:, 0] - 0.08, min=0.0)

    def _reward_no_lateral_motion(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_no_yaw_motion(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _startup_mask(self):
        return (~self.startup_ready).float()

    def _reward_startup_stable_contact(self):
        mask = self._startup_mask()
        startup_thr = float(getattr(self.cfg.rewards, "startup_contact_threshold", 0.5))
        contact = self.contact_forces[:, self.feet_indices, 2] > startup_thr
        return mask * torch.all(contact, dim=1).float()

    def _reward_startup_stand_still(self):
        mask = self._startup_mask()
        pose_err = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        vel_err = torch.sum(torch.square(self.dof_vel), dim=1)
        return mask * torch.exp(-5.0 * pose_err - 0.05 * vel_err)

    def _reward_startup_upright(self):
        mask = self._startup_mask()
        upright = -self.projected_gravity[:, 2]
        return mask * torch.clamp(upright, min=0.0, max=1.0)

    def _reward_startup_low_motion(self):
        mask = self._startup_mask()
        lin_vel_sq = torch.sum(torch.square(self.base_lin_vel), dim=1)
        ang_vel_sq = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return mask * torch.exp(-2.0 * lin_vel_sq - 0.5 * ang_vel_sq)

    def _reward_startup_support_balance(self):
        mask = self._startup_mask()
        startup_thr = float(getattr(self.cfg.rewards, "startup_contact_threshold", 0.5))
        contact = self.contact_forces[:, self.feet_indices, 2] > startup_thr
        all_contact = torch.all(contact, dim=1)
        z_forces = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
        total_force = z_forces.sum(dim=1).clamp(min=1e-6)
        force_ratio = z_forces / total_force.unsqueeze(1)
        balance = 1.0 - torch.std(force_ratio, dim=1) * 2.0
        return mask * torch.clamp(balance, min=0.0) * all_contact.float()

    def _reward_pose_hold(self):
        sigma = getattr(self.cfg.rewards, "pose_hold_sigma", 0.10)
        pose_err = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return torch.exp(-pose_err / (2.0 * sigma * sigma))

    def _reward_support_balance(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        z_forces = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
        total_force = z_forces.sum(dim=1).clamp(min=1e-6)
        force_ratio = z_forces / total_force.unsqueeze(1)
        balance = 1.0 - torch.std(force_ratio, dim=1) * 1.5
        balance = torch.clamp(balance, min=0.0)
        num_contact = torch.sum(contact.float(), dim=1)

        move_cmd = (torch.norm(self.commands[:, :2], dim=1) > self.min_command_norm).float()
        walk_contact_score = 0.75 * torch.exp(-1.2 * torch.square(num_contact - 2.0)) + 0.25 * torch.exp(-1.0 * torch.square(num_contact - 3.0))
        stand_contact_score = (num_contact >= 4.0).float()
        return move_cmd * balance * walk_contact_score + (1.0 - move_cmd) * balance * stand_contact_score

    def _reward_backward_motion(self):
        return torch.clamp(-self.base_lin_vel[:, 0], min=0.0)

    def _reward_heading_forward(self):
        world_forward = quat_apply(self.base_quat, self.forward_vec)
        return torch.clamp(world_forward[:, 0], min=0.0, max=1.0)

    def _reward_stand_when_should_walk(self):
        post_startup = 1.0 - self._startup_mask()
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.5
        num_contact = torch.sum(contact.float(), dim=1)
        four_feet = (num_contact >= 4.0).float()
        three_feet = (num_contact == 3.0).float()
        low_forward_speed = (torch.abs(self.base_lin_vel[:, 0]) < 0.08).float()
        return post_startup * move_cmd * (1.5 * four_feet + 0.5 * three_feet + 0.5 * low_forward_speed * (num_contact >= 3.0).float())

    def _reward_swing_height(self):
        # walk_0..walk_5: original touchdown-avg behavior preserved.
        # walk_6+:        stuck-aware per-foot min. Enable via cfg.rewards.swing_height_min_mode = True
        post_startup = 1.0 - self._startup_mask()
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        target = getattr(self.cfg.rewards, "swing_height_target", 0.040)
        sigma = getattr(self.cfg.rewards, "swing_height_sigma", 0.012)

        if getattr(self.cfg.rewards, "swing_height_min_mode", False):
            # Each foot's "effective recent clearance":
            #   - current running max while airborne
            #   - last swing peak if not currently airborne
            #   - forced to 0 if the foot has been in stance longer than stance_reset
            stance_reset = float(getattr(self.cfg.rewards, "swing_peak_stance_reset", 0.30))
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
            airborne = ~contact
            effective = torch.where(airborne, self._max_swing_height, self._last_swing_peak)
            stuck = self._foot_stance_time > stance_reset
            effective = torch.where(stuck, torch.zeros_like(effective), effective)
            under = torch.clamp(target - effective, min=0.0)
            per_foot = torch.exp(-torch.square(under) / (2.0 * sigma * sigma))
            # walk_6 (re-defined): mix mode = w_min·min + (1-w_min)·mean.
            # Pure min caps reward when one foot stays at 0 even if other 3
            # lift well (walk_5 try_13 plateau). Mix gives partial credit for
            # the 3 good feet so policy keeps lifting while still pushing the
            # worst one up. w_min=1 falls back to legacy min mode.
            w_min = float(getattr(self.cfg.rewards, "swing_height_min_weight", 1.0))
            min_reward, _ = torch.min(per_foot, dim=1)
            mean_reward = torch.mean(per_foot, dim=1)
            mixed = w_min * min_reward + (1.0 - w_min) * mean_reward
            return post_startup * move_cmd * mixed

        # Legacy (walk_0..walk_5): touchdown-averaged one-sided Gaussian.
        # `_update_gait_tracking` freezes the swing peak into `_last_swing_peak`
        # at the touchdown step, so use that (not the already-reset running max).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.last_step_contacts
        touchdown = contact & (~prev_contact)
        clearance_err = torch.clamp(target - self._last_swing_peak, min=0.0)
        clearance_reward = torch.exp(-torch.square(clearance_err) / (2.0 * sigma * sigma))
        touchdown_count = torch.sum(touchdown.float(), dim=1)
        touchdown_reward = torch.sum(clearance_reward * touchdown.float(), dim=1) / torch.clamp(touchdown_count, min=1.0)
        touchdown_any = (touchdown_count > 0).float()
        return post_startup * move_cmd * touchdown_any * touchdown_reward

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        near_ground = self.foot_positions[:, :, 2] < 0.045
        slip_mask = (contact | near_ground).float()
        foot_speed_xy = torch.norm(self.foot_velocities[:, :, :2], dim=2)
        return torch.sum(foot_speed_xy * slip_mask, dim=1)

    def _reward_left_right_stance_symmetry(self):
        # Trot has LEFT and RIGHT feet 180 deg out of phase, so instantaneous
        # hip/knee comparison collapses to noise. Two modes:
        #   legacy (walk_4/5): shoulder-only instantaneous comparison.
        #   walk_6 (left_right_mode="ema_all"): compare EMA of L vs R for all
        #   three joints — the EMA averages out the swing alternation, so a
        #   truly symmetric trot has EMA_L ≈ EMA_R everywhere, while a
        #   scuffing pair leaves a persistent bias.
        sigma = getattr(self.cfg.rewards, "left_right_pose_sigma", 0.15)
        mode = getattr(self.cfg.rewards, "left_right_mode", "shoulder_only")
        if mode == "ema_all":
            ema = self._dof_pos_ema
            # Shoulder: same sign convention L vs R → direct compare
            # Hip/Knee: mirrored joint axis → L = -R when symmetric
            #   LF(0,1,2) vs RF(3,4,5): shoulder direct, hip/knee sign-flip
            #   LR(6,7,8) vs RR(9,10,11): shoulder direct, hip/knee sign-flip
            front = (torch.square(ema[:, 0] - ema[:, 3])
                   + torch.square(ema[:, 1] + ema[:, 4])
                   + torch.square(ema[:, 2] + ema[:, 5]))
            rear = (torch.square(ema[:, 6] - ema[:, 9])
                  + torch.square(ema[:, 7] + ema[:, 10])
                  + torch.square(ema[:, 8] + ema[:, 11]))
            return torch.exp(-(front + rear) / (2.0 * sigma * sigma))
        # Legacy path: shoulder-only instantaneous comparison.
        shoulder_err = torch.square(self.dof_pos[:, 0] - self.dof_pos[:, 3]) \
                     + torch.square(self.dof_pos[:, 6] - self.dof_pos[:, 9])
        return torch.exp(-shoulder_err / (2.0 * sigma * sigma))

    def _reward_stuck_foot_penalty(self):
        # walk_6: penalize any foot that stays in stance much longer than the
        # expected cycle time — this is the failure mode of walk_5 try_2 where
        # one diagonal pair never actually lifts.
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        stance_max = float(getattr(self.cfg.rewards, "foot_stance_max_time", 0.35))
        over = torch.clamp(self._foot_stance_time - stance_max, min=0.0)
        return move_cmd * torch.sum(torch.square(over), dim=1)

    def _reward_straight_line_deviation(self):
        # walk_6 (re-defined): penalize cumulative lateral position drift from
        # spawn point, projected onto the spawn-time forward direction. Position-
        # based, not velocity-based, so transient sway is fine but slow drift
        # accumulates penalty. Returns a NEGATIVE quantity meant to be paired
        # with a POSITIVE scale in cfg (so the final contribution is negative).
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        delta_xy = self.root_states[:, :2] - self._spawn_xy
        forward = self._spawn_forward
        # Perpendicular (rotate forward by +90 deg in world XY plane)
        perp_x = -forward[:, 1]
        perp_y = forward[:, 0]
        lateral = delta_xy[:, 0] * perp_x + delta_xy[:, 1] * perp_y
        sigma = float(getattr(self.cfg.rewards, "straight_dev_sigma", 0.30))
        return move_cmd * torch.square(lateral) / (sigma * sigma)

    def _reward_swing_cycle_peak(self):
        # walk_6 try_4 try (DEPRECATED, returned 0 throughout training).
        # Kept for cfg backwards-compat — set scale 0 to disable.
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        prev_contact = self.last_step_contacts
        touchdown = contact & (~prev_contact)
        target = float(getattr(self.cfg.rewards, "swing_cycle_target", 0.030))
        peak_norm = torch.clamp(self._last_swing_peak / target, 0.0, 1.5)
        per_foot_event = peak_norm * touchdown.float()
        return move_cmd * torch.sum(per_foot_event, dim=1)

    def _reward_swing_air_height(self):
        # walk_6 try_5+: continuous airborne foot-height reward.
        # Important: foot_positions[:,:,2] returns values in a frame where
        # stance feet sit around z ≈ -0.13 (link origin offset). So we use
        # the per-env MIN foot Z across the four feet as a dynamic ground
        # reference. Lifted feet rise above this; stuck feet stay at it.
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        airborne = (~contact).float()
        target = float(getattr(self.cfg.rewards, "swing_air_target", 0.030))
        foot_z = self.foot_positions[:, :, 2]
        ground_ref = torch.min(foot_z, dim=1, keepdim=True).values
        lift = torch.clamp(foot_z - ground_ref, min=0.0)
        height_norm = torch.clamp(lift / target, 0.0, 1.5)
        per_foot = height_norm * airborne
        return move_cmd * torch.sum(per_foot, dim=1)

    def _reward_diagonal_propulsion(self):
        # walk_6 (re-defined): both diagonal pairs must contribute z-force.
        # walk_5 try_13 ended with diagonal_gait phase OK but only one pair
        # carrying load — the other pair scuffed without pushing. We track
        # the EMA of pair_a/(pair_a+pair_b) z-force ratio and reward staying
        # near 0.5 (balanced load between the two pairs).
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        f = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
        # Asset feet order: [LR, LF, RF, RR] per quard_config.py
        # Diagonal pair A = LF + RR (idx 1, 3), pair B = LR + RF (idx 0, 2)
        pair_a = f[:, 1] + f[:, 3]
        pair_b = f[:, 0] + f[:, 2]
        total = (pair_a + pair_b).clamp(min=1e-6)
        ratio = pair_a / total
        alpha = float(getattr(self.cfg.rewards, "pair_force_ema_alpha", 0.02))
        self._pair_force_ema.mul_(1.0 - alpha).add_(ratio, alpha=alpha)
        sigma = float(getattr(self.cfg.rewards, "pair_force_sigma", 0.10))
        err = self._pair_force_ema - 0.5
        return move_cmd * torch.exp(-torch.square(err) / (2.0 * sigma * sigma))

    def _reward_front_rear_stance_symmetry(self):
        sigma = getattr(self.cfg.rewards, "front_rear_pose_sigma", 0.15)
        front_hip = (self.dof_pos[:, 1] + self.dof_pos[:, 4]) * 0.5
        rear_hip = -(self.dof_pos[:, 7] + self.dof_pos[:, 10]) * 0.5
        front_knee = (self.dof_pos[:, 2] + self.dof_pos[:, 5]) * 0.5
        rear_knee = -(self.dof_pos[:, 8] + self.dof_pos[:, 11]) * 0.5
        err = torch.square(front_hip - rear_hip) + torch.square(front_knee - rear_knee)
        return torch.exp(-err / (2.0 * sigma * sigma))

    def _reward_rear_swing_height(self):
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.rear_feet_indices, 2] > 1.0
        swing_mask = (~contact).float()
        foot_heights = self.rear_foot_positions[:, :, 2]
        target = getattr(self.cfg.rewards, "rear_swing_height_target", 0.028)
        sigma = getattr(self.cfg.rewards, "rear_swing_height_sigma", 0.010)
        height_reward = torch.exp(-torch.square(foot_heights - target) / (2.0 * sigma * sigma))
        swing_count = torch.sum(swing_mask, dim=1).clamp(min=1.0)
        return move_cmd * torch.sum(height_reward * swing_mask, dim=1) / swing_count

    def _reward_rear_air_presence(self):
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        rear_contact = self.contact_forces[:, self.rear_feet_indices, 2] > 1.0
        return move_cmd * (~rear_contact).any(dim=1).float()

    def _reward_front_only_air(self):
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        front_contact = self.contact_forces[:, self.front_feet_indices, 2] > 1.0
        rear_contact = self.contact_forces[:, self.rear_feet_indices, 2] > 1.0
        front_air = (~front_contact).any(dim=1).float()
        rear_all_ground = rear_contact.all(dim=1).float()
        return move_cmd * front_air * rear_all_ground

    def _reward_rear_feet_slip(self):
        contact = self.contact_forces[:, self.rear_feet_indices, 2] > 1.0
        near_ground = self.rear_foot_positions[:, :, 2] < 0.045
        slip_mask = (contact | near_ground).float()
        foot_speed_xy = torch.norm(self.rear_foot_velocities[:, :, :2], dim=2)
        return torch.sum(foot_speed_xy * slip_mask, dim=1)

    def _reward_knee_range_violation(self):
        knee_min = getattr(self.cfg.rewards, "knee_soft_min", -0.80)
        knee_max = getattr(self.cfg.rewards, "knee_soft_max", -0.10)
        front_knee_pos = self.dof_pos[:, [2, 5]]
        f_below = (knee_min - front_knee_pos).clip(min=0.0)
        f_above = (front_knee_pos - knee_max).clip(min=0.0)
        rear_knee_pos = self.dof_pos[:, [8, 11]]
        r_below = (-knee_max - rear_knee_pos).clip(min=0.0)
        r_above = (rear_knee_pos - (-knee_min)).clip(min=0.0)
        return (torch.sum(torch.square(f_below) + torch.square(f_above), dim=1)
                + torch.sum(torch.square(r_below) + torch.square(r_above), dim=1))

    def _reward_feet_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5.0 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        ).float()

    def _reward_diagonal_gait(self):
        post_startup = 1.0 - self._startup_mask()
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        lf, rf, lr, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
        pair_a = torch.stack([lf.float(), rr.float()], dim=1).mean(dim=1)
        pair_b = torch.stack([rf.float(), lr.float()], dim=1).mean(dim=1)
        pair_match = ((lf == rr).float() + (rf == lr).float()) * 0.5
        anti_phase = (torch.abs(pair_a - pair_b) > 0.5).float()
        num_contact = contact.float().sum(dim=1)
        two_or_three = ((num_contact == 2.0) | (num_contact == 3.0)).float()
        perfect_trot = (pair_match > 0.99).float() * anti_phase * two_or_three
        return post_startup * move_cmd * (0.35 * pair_match + 0.65 * perfect_trot)

    def _reward_all_feet_stepping(self):
        post_startup = 1.0 - self._startup_mask()
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_touch = ((self.feet_air_time > 0.06) & contact).float()
        stepped_feet = torch.sum(first_touch, dim=1)
        num_contact = torch.sum(contact.float(), dim=1)
        # walk_6 tightens this: require all 4 feet to step for full credit (denom 4.0).
        # Earlier walks used 2.0 (saturated with a single diagonal pair stepping).
        denom = float(getattr(self.cfg.rewards, "all_feet_stepping_denom", 2.0))
        step_score = torch.clamp(stepped_feet / denom, 0.0, 1.0)
        support_score = 0.7 * torch.exp(-1.1 * torch.square(num_contact - 2.0)) + 0.3 * torch.exp(-1.0 * torch.square(num_contact - 3.0))
        return post_startup * move_cmd * (0.6 * step_score + 0.4 * support_score)



    def _reward_tripod_penalty(self):
        post_startup = 1.0 - self._startup_mask()
        move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        num_contact = torch.sum(contact.float(), dim=1)
        air_mask = (~contact).float()
        max_air_height = torch.max(self.foot_positions[:, :, 2] * air_mask, dim=1).values
        max_air_time = torch.max(self.feet_air_time, dim=1).values
        air_time_thr = getattr(self.cfg.rewards, "tripod_airtime_threshold", 0.15)
        air_height_thr = getattr(self.cfg.rewards, "tripod_height_threshold", 0.030)
        tripod = (num_contact == 3.0).float()
        long_single_swing = (max_air_time > air_time_thr).float()
        clear_lift = (max_air_height > air_height_thr).float()
        return post_startup * move_cmd * tripod * torch.clamp(0.7 * long_single_swing + 0.3 * clear_lift, 0.0, 1.0)
