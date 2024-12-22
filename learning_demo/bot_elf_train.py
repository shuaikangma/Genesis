import argparse
import os
import pickle
import shutil

from bot_elf_env import BotElfEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles" : {
            'l_hip_z_joint': 0.,
            'l_hip_x_joint': 0.,
            'l_hip_y_joint': -0.5,
            'l_knee_y_joint': 0.9,  # 0.6
            'l_ankle_y_joint': -0.45,
            'l_ankle_x_joint': 0.,
            
            'r_hip_z_joint': 0.,
            'r_hip_x_joint': 0.,
            'r_hip_y_joint': -0.5,
            'r_knee_y_joint': 0.9,  # 0.6
            'r_ankle_y_joint': -0.45,
            'r_ankle_x_joint': 0.,
        },
        "dof_names": [
            "l_hip_z_joint",
            "l_hip_x_joint",
            "l_hip_y_joint",
            "l_knee_y_joint",
            "l_ankle_y_joint",
            "l_ankle_x_joint",
            "r_hip_z_joint",
            "r_hip_x_joint",
            "r_hip_y_joint",
            "r_knee_y_joint",
            "r_ankle_y_joint",
            "r_ankle_x_joint",
        ],
        # PD
        "kp": [25,25,30,40,3,3, 25,25,30,40,3,3],
        "kd": [2.5,2.5,3,4,0.3,0.3, 2.5,2.5,3,4,0.3,0.3],
        # termination
        "termination_if_roll_greater_than": 80,  # degree
        "termination_if_pitch_greater_than": 80,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.9],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 20.0,
        "foot_names": ["l_ankle_x_link", "r_ankle_x_link"],
        "knee_names": ["l_knee_y_link", "r_knee_y_link"],
        "penalize_contacts_on": ["l_hip_y_link","l_knee_y_link","r_hip_y_link","r_knee_y_link"]
    }
    obs_cfg = {
        "num_single_obs": 61,
        "num_single_privileged_obs": 64,
        "frame_stack": 5,
        "c_frame_stack": 5,
        "num_obs": 61*5,
        "num_privileged_obs": 64*5,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "cycle_time": 0.56,
        "target_joint_pos_scale": 0.1,
        "max_contact_force": 200.,
        "tracking_sigma": 5.,
        "base_height_target": 0.83,
        "feet_height_target": 0.08,
        "soft_torque_limit": 0.9,
        "min_distance": 0.1,
        "max_distance": 0.5,
        "reward_scales": {
            "joint_pos": 1.5,
            "feet_contact_number": 1.0,
            
            "feet_air_time": 1.0,
            "foot_slip": -0.05,
            "feet_clearance": 1.0,
            "feet_distance": 0.2,
            "knee_distance": 0.2,
                        
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 1.0,
            "vel_mismatch_exp": 0.5,
            "low_speed": 0.2,
            "track_vel_hard": 0.5,
            
            "default_joint_pos": 0.5,
            "orientation": 1.0,
            "base_height": 0.2,
            
            "base_acc": 0.2,
            "feet_contact_forces": 0,#-2e-5,
            "action_smoothness": 0,#-2e-3,
            "torques": 0,#-2e-3,
            "dof_vel": 0,#-5e-4,
            "dof_acc": 0,#-1e-7,
            "collision": -1.,
            "torque_rate": 0,#-2e-5
            "torque_limits": -1., 
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
    # command_cfg = {
    #     "num_commands": 3,
    #     "lin_vel_x_range": [0.5, 0.5],
    #     "lin_vel_y_range": [0., 0.],
    #     "ang_vel_range": [0.0, 0.0],
    # }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="botelf-walking-all")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = BotElfEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=not args.headless,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python learning_demo/bot_elf_train.py
"""
