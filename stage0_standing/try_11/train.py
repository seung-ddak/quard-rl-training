# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import re
import sys
import shutil

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TRAINING_RECORD_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "training_record"
)
TRAINING_RECORD_ROOT = os.path.abspath(TRAINING_RECORD_ROOT)

TASK_STAGE_MAP = {
    "quard_stage0": ("stage0_standing", "Stage0 Standing"),
    "quard_stage1": ("stage1_basic_walk", "Stage1 Basic Walk"),
    "quard_stage2": ("stage2_robust_walk", "Stage2 Robust Walk"),
    "quard_stage3": ("stage3_rough_terrain", "Stage3 Rough Terrain"),
}


class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        clean = re.sub(r"\033\[[0-9;]*m", "", data)
        self.file.write(clean)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


def get_next_try_number(stage_dir):
    if not os.path.exists(stage_dir):
        return 1

    existing = [d for d in os.listdir(stage_dir) if d.startswith("try_")]
    if not existing:
        return 1

    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass

    return max(nums) + 1 if nums else 1


def get_stage_info(task_name):
    for key, (stage_dir, stage_label) in TASK_STAGE_MAP.items():
        if key in task_name:
            return stage_dir, stage_label
    return task_name, task_name


def get_init_at_random_ep_len(experiment_name, stage_dir_name):
    exp = str(experiment_name).lower()
    stage = str(stage_dir_name).lower()

    if "quard_stage0" in exp or "stage0_standing" in stage:
        return False
    return True


def save_reward_config(env_cfg, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Reward Configuration ===\n\n")

        if hasattr(env_cfg, "rewards"):
            rewards = env_cfg.rewards
            f.write(
                f"only_positive_rewards: "
                f"{getattr(rewards, 'only_positive_rewards', 'N/A')}\n"
            )
            f.write(
                f"base_height_target: "
                f"{getattr(rewards, 'base_height_target', 'N/A')}\n"
            )
            f.write(
                f"soft_dof_pos_limit: "
                f"{getattr(rewards, 'soft_dof_pos_limit', 'N/A')}\n\n"
            )

            if hasattr(rewards, "scales"):
                f.write("--- Reward Scales ---\n")
                scales = rewards.scales
                for attr in sorted(dir(scales)):
                    if attr.startswith("_"):
                        continue
                    val = getattr(scales, attr)
                    if isinstance(val, (int, float)):
                        f.write(f"  {attr}: {val}\n")


def save_init_config(env_cfg, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Init State Configuration ===\n\n")

        if hasattr(env_cfg, "init_state"):
            init = env_cfg.init_state
            f.write(f"pos (config): {getattr(init, 'pos', 'N/A')}\n")
            f.write("default_joint_angles:\n")
            angles = getattr(init, "default_joint_angles", {})
            for k, v in angles.items():
                f.write(f"  {k}: {v}\n")

        calib_path = os.path.join(TRAINING_RECORD_ROOT, "calibrated_init_height.txt")
        if os.path.exists(calib_path):
            f.write("\n--- Calibration Result ---\n")
            with open(calib_path, "r", encoding="utf-8") as cf:
                f.write(cf.read())


def save_run_config(train_cfg, experiment_name, stage_dir_name, save_path, init_at_random_ep_len):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Run Configuration ===\n\n")
        f.write(f"experiment_name: {experiment_name}\n")
        f.write(f"stage_dir_name: {stage_dir_name}\n")
        f.write(f"max_iterations: {getattr(train_cfg.runner, 'max_iterations', 'N/A')}\n")
        f.write(f"resume: {getattr(train_cfg.runner, 'resume', 'N/A')}\n")
        f.write(f"load_run: {getattr(train_cfg.runner, 'load_run', 'N/A')}\n")
        f.write(f"checkpoint: {getattr(train_cfg.runner, 'checkpoint', 'N/A')}\n")
        f.write(f"init_at_random_ep_len: {init_at_random_ep_len}\n")


def generate_plots_and_summary(log_dir, try_dir, stage_label, try_num):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        print(f"[WARNING] Plot generation failed (missing module): {e}")
        return

    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("[WARNING] No TensorBoard scalar data found.")
        return

    def get_scalar(tag):
        if tag in tags:
            events = ea.Scalars(tag)
            return [e.step for e in events], [e.value for e in events]
        return [], []

    steps, values = get_scalar("Train/mean_reward")
    steps_len, values_len = get_scalar("Train/mean_episode_length")
    steps_noise, values_noise = get_scalar("Policy/mean_noise_std")
    episode_tags = [t for t in tags if t.startswith("Episode/")]

    if steps:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.title(f"{stage_label} - Try {try_num}: Mean Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(try_dir, "mean_reward.png"), dpi=150)
        plt.close()

    if steps_len:
        plt.figure(figsize=(10, 6))
        plt.plot(steps_len, values_len, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Episode Length")
        plt.title(f"{stage_label} - Try {try_num}: Episode Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(try_dir, "episode_length.png"), dpi=150)
        plt.close()

    if steps_noise:
        plt.figure(figsize=(10, 6))
        plt.plot(steps_noise, values_noise, linewidth=0.8)
        plt.xlabel("Iteration")
        plt.ylabel("Noise Std")
        plt.title(f"{stage_label} - Try {try_num}: Action Noise Std")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(try_dir, "noise_std.png"), dpi=150)
        plt.close()

    if episode_tags:
        plt.figure(figsize=(12, 8))
        for tag in episode_tags:
            s, v = get_scalar(tag)
            if s:
                plt.plot(s, v, linewidth=0.8, label=tag.replace("Episode/", ""))
        plt.xlabel("Iteration")
        plt.ylabel("Reward Component")
        plt.title(f"{stage_label} - Try {try_num}: Reward Components")
        plt.legend(fontsize=8, loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(try_dir, "reward_components.png"), dpi=150)
        plt.close()

    summary_path = os.path.join(try_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=== {stage_label} - Try {try_num} Training Summary ===\n\n")
        if steps:
            f.write(f"Total iterations: {steps[-1]}\n")
        if values_len:
            f.write(f"Final episode length: {values_len[-1]:.1f}\n")
            f.write(f"Max episode length: {max(values_len):.1f}\n")
        if values:
            f.write(f"Final mean reward: {values[-1]:.4f}\n")
            f.write(f"Max mean reward: {max(values):.4f}\n")
        if values_noise:
            f.write(f"Final noise std: {values_noise[-1]:.4f}\n")

        f.write("\n--- Final Reward Components ---\n")
        for tag in episode_tags:
            s, v = get_scalar(tag)
            if v:
                f.write(f"  {tag.replace('Episode/', '')}: {v[-1]:.4f}\n")

    print("[Record] training_summary.txt saved.")


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    experiment_name = getattr(train_cfg.runner, "experiment_name", args.task)
    stage_dir_name, stage_label = get_stage_info(experiment_name)
    stage_dir = os.path.join(TRAINING_RECORD_ROOT, stage_dir_name)
    os.makedirs(stage_dir, exist_ok=True)

    try_num = get_next_try_number(stage_dir)
    try_dir = os.path.join(stage_dir, f"try_{try_num}")
    os.makedirs(try_dir, exist_ok=True)

    init_at_random_ep_len = get_init_at_random_ep_len(experiment_name, stage_dir_name)

    print("\n" + "=" * 60)
    print(f" Auto record: {stage_label} - Try {try_num}")
    print(f" Save path: {try_dir}")
    print(f" init_at_random_ep_len: {init_at_random_ep_len}")
    print("=" * 60 + "\n")

    train_log_path = os.path.join(try_dir, "train_log.txt")
    tee = TeeOutput(train_log_path)
    sys.stdout = tee

    try:
        max_iter = getattr(train_cfg.runner, "max_iterations", 1000)

        print(f"[Train] experiment_name: {experiment_name}")
        print(f"[Train] stage_dir_name: {stage_dir_name}")
        print(f"[Train] max_iterations: {max_iter}")
        print(f"[Train] init_at_random_ep_len: {init_at_random_ep_len}")

        ppo_runner.learn(
            num_learning_iterations=max_iter,
            init_at_random_ep_len=init_at_random_ep_len,
        )
    finally:
        tee.close()

    print("\n[Record] Training done. Saving records...")

    save_reward_config(env_cfg, os.path.join(try_dir, "reward_config.txt"))
    print("[Record] reward_config.txt saved.")

    save_init_config(env_cfg, os.path.join(try_dir, "init_config.txt"))
    print("[Record] init_config.txt saved.")

    save_run_config(
        train_cfg=train_cfg,
        experiment_name=experiment_name,
        stage_dir_name=stage_dir_name,
        save_path=os.path.join(try_dir, "run_config.txt"),
        init_at_random_ep_len=init_at_random_ep_len,
    )
    print("[Record] run_config.txt saved.")

    log_dir = ppo_runner.log_dir
    if log_dir and os.path.exists(log_dir):
        generate_plots_and_summary(log_dir, try_dir, stage_label, try_num)
        print("[Record] 4 plots saved.")
    else:
        print(f"[WARNING] log_dir not found: {log_dir}")

    if log_dir and os.path.exists(log_dir):
        model_files = sorted(
            f for f in os.listdir(log_dir)
            if f.startswith("model_") and f.endswith(".pt")
        )
        if model_files:
            latest_model = model_files[-1]
            shutil.copy2(
                os.path.join(log_dir, latest_model),
                os.path.join(try_dir, latest_model),
            )
            print(f"[Record] {latest_model} saved.")
        else:
            print(f"[WARNING] No model .pt file found in: {log_dir}")

    print("\n" + "=" * 60)
    print(f" Records saved: {try_dir}")
    print(" Files:")
    for f in sorted(os.listdir(try_dir)):
        print(f"   - {f}")
    print("=" * 60 + "\n")

    calib_path = os.path.join(TRAINING_RECORD_ROOT, "calibrated_init_height.txt")
    if os.path.exists(calib_path):
        print(f"[Calibration] Result: {calib_path}")
        with open(calib_path, "r", encoding="utf-8") as f:
            print(f.read())
        print("[Calibration] Update pos z in quard_config.py to the calibrated_init_z value above.")


if __name__ == "__main__":
    args = get_args()
    train(args)