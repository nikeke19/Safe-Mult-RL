import argparse
import numpy as np
import wandb
from envs.lunar_lander import LunarLander
import os

from util.monitors.lunar_lander_monitor import LunarLanderMonitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack
from util.callbacks.lunar_lander_eval_cb import LunarLanderEvalCallback
from models.common.util import get_linear_fn
from util.callbacks.callback_template import EvalCallbackTemplate


from models import PPO, SAC
from wandb.integration.sb3 import WandbCallback
from typing import Any, Dict, Optional, Type, Union
import warnings


def join_two_dicts(dict1, dict2):
    dict_joined = dict1.copy()
    # Finding same keys in both dicts and concatenating
    for key, value in dict_joined.items():
        if key in dict2:
            # join dicts
            if isinstance(value, dict):
                dict_joined[key] = {**value, **dict2[key]}
            else:
                warnings.warn("Two dicts have the same key: " + key)
    # Adding keys of dict2
    for key, value in dict2.items():
        if key not in dict_joined:
            dict_joined[key] = value
    return dict_joined


class LunarLanderTrainer:
    def __init__(self, opt):

        self.model_type = opt.model
        self.model_architecture = opt.model_architecture
        self.load_model_dir = opt.load_model_dir
        self.save_model = opt.save_model
        self.eval_freq = opt.eval_freq
        self.time_steps = opt.time_steps
        self.n_test_episodes = opt.n_test_episodes
        self.collision_reward = opt.collision_reward
        self.video_freq = opt.video_freq
        self.gamma_col_net = opt.gamma_col_net
        self.safe_lagrange = opt.safe_lagrange
        self.target_kl = opt.kl_target
        self.clip_range = opt.clip_range
        self.lr_schedule_step = opt.lr_schedule_step
        self.wandb_on = opt.wandb_on
        self.use_bayes = opt.use_bayes
        self.use_beta_dist = opt.use_beta_dist
        self.safe_mult = opt.safe_mult
        self.seed = None
        self.env, self.env_video, self.env_eval = None, None, None
        self.log_dir = None
        self.run = None
        self.model = None
        self.callbacks = None
        self.model_kwargs = None
        self.cb_eval_class = None
        self.opt = opt

    def setup_env(self, record_video=False, normalize=False):
        def environment():
            kwargs = dict(continuous=False if self.model_type == "DQN" or self.model_type == "DoubleDQN" else True,
                          hard_mode=True, normalize=normalize, easy_landing=self.opt.easy_landing)
            env = LunarLander(**kwargs)
            env.seed(self.seed)
            path = f"{self.log_dir}/runs/{self.run.id}/" if self.wandb_on else self.log_dir
            os.makedirs(path, exist_ok=True)
            return LunarLanderMonitor(env, path, col_reward=-1 if normalize else -100)

        env = DummyVecEnv([environment])
        if record_video:
            folder = f"{self.log_dir}/videos/{self.run.id}" if self.wandb_on else self.log_dir + "experimental"
            env = VecVideoRecorder(env, folder, record_video_trigger=lambda x: x % 1000 == 0, video_length=1000)
        env = VecFrameStack(env, n_stack=4)

        return env

    def get_sac_kwargs(self):
        return {"learning_rate": get_linear_fn(3e-4, 1e-8),
                "tau": 0.01,
                "learning_starts": 10000,
                "safe_version": self.opt.safe_version,
                "optimize_gamma_col_net": self.opt.optimize_gamma_col_net,
                "gamma_col_net_target": self.opt.gamma_col_net_target,
                "safe_lagrange_baseline": self.safe_lagrange,
                }

    def get_ppo_kwargs(self):
        self.model_kwargs["policy"] = "Custom" + self.model_kwargs["policy"]
        return {"n_steps": 1024,
                "gae_lambda": 0.98,
                # "ent_coef": 0.0,
                "ent_coef": get_linear_fn(0.02, 0, start_fraction=0.5, end_fraction=1),
                "policy_kwargs": dict(log_std_init=0),
                "l_multiplier_init": self.opt.l_multiplier_init,
                "advantage_mode": self.opt.advantage_mode,
                "n_lagrange_samples": self.opt.n_lagrange_samples,
                "n_col_value_samples": self.opt.n_col_value_samples,
                "optimize_gamma_col_net": self.opt.optimize_gamma_col_net,
                "gamma_col_net_target": self.opt.gamma_col_net_target,
                "learning_rate": self.opt.learning_rate,
                }

    def train(self):
        seeds = np.arange(self.opt.start_seed, self.opt.end_seed)
        for seed in seeds:
            self.seed = int(seed)

            config = {"collision_reward": self.opt.collision_reward,
                      "model": self.opt.model,
                      "model_architecture": self.opt.model_architecture,
                      "time_steps": self.opt.time_steps,
                      "safe_rl_on": self.safe_lagrange or self.safe_mult,
                      "lr_schedule_step": self.lr_schedule_step,
                      "ent_schedule": "0.01-0",
                      "normalize_reward": self.opt.normalize_reward,
                      "easy_landing": self.opt.easy_landing,
                      "opt": self.opt}
            if self.wandb_on:
                self.run = wandb.init(project=self.opt.wandb_project, config=config, sync_tensorboard=True,
                                      monitor_gym=True, group=self.opt.wandb_group, save_code=False)
                wandb.run.name = self.opt.wandb_name + "_" + str(seed)
            self.setup_log_dir()
            self.env = self.setup_env(normalize=self.opt.normalize_reward)
            self.env_eval = self.setup_env(normalize=False)
            self.env_video = self.setup_env(record_video=True, normalize=False)

            self.register_eval_callback_class(LunarLanderEvalCallback)
            self.callbacks = self.setup_callback()
            self.model = self.setup_model() if self.opt.load_model_dir is None else self.load_model()

            self.model.learn(total_timesteps=self.opt.time_steps, log_interval=5, callback=self.callbacks)
            self.env.close()
            self.env_video.close()
            if self.wandb_on:
                self.run.finish()

    def setup_log_dir(self):
        if self.safe_mult or self.safe_lagrange:
            self.log_dir = "logs/safe_rl/"
        else:
            self.log_dir = "logs/normal_rl/"
        os.makedirs(self.log_dir, exist_ok=True)

    def register_eval_callback_class(self, eval_callback_class: EvalCallbackTemplate) -> None:
        self.cb_eval_class = eval_callback_class

    def setup_model(self):
        self.model_kwargs = {"policy": self.model_architecture,
                             "env": self.env,
                             "seed": self.seed,
                             "safe_mult": self.safe_mult,
                             "tensorboard_log": self.log_dir + "runs/" + str(self.run.id) if self.wandb_on else None,
                             "gamma_col_net": self.gamma_col_net,
                             "col_reward": self.collision_reward,
                             "verbose": True,
                             "device": "cuda",
                             }
        if self.model_type == "SAC":
            sac_kwargs = {"policy_kwargs": dict(use_bayes=self.use_bayes, use_beta_dist=self.use_beta_dist)}
            sac_kwargs = join_two_dicts(sac_kwargs, self.get_sac_kwargs())
            return SAC(**self.model_kwargs, **sac_kwargs)
        elif self.model_type == "PPO":
            ppo_kwargs = {"lr_schedule_step": self.lr_schedule_step, "target_kl": self.target_kl,
                          "safe_lagrange": self.safe_lagrange, "clip_range": self.clip_range,
                          "policy_kwargs": dict(use_beta_dist=self.use_beta_dist),
                          "use_bayes": self.use_bayes}
            ppo_kwargs = join_two_dicts(ppo_kwargs, self.get_ppo_kwargs())
            return PPO(**self.model_kwargs, **ppo_kwargs)

        else:
            raise NotImplementedError("Model of name: ", self.model_type, " is not implemented")

    def load_model(self):
        if self.model_type == "SAC":
            return SAC.load(self.load_model_dir, env=self.env, gamma_col_net_init=self.gamma_col_net)
        else:
            return PPO.load(self.load_model_dir, env=self.env)

    def setup_callback(self):
        # Setup paths
        logs = f"{self.log_dir}/runs/{self.run.id}" if self.wandb_on else self.log_dir + "experimental/eval"
        env_eval = self.env if self.env_eval is None else self.env_eval
        model_path = None
        if self.wandb_on and self.save_model:
            model_path = f"{self.log_dir}/models/{self.run.id}"
        cb = [self.cb_eval_class(env_eval, n_eval_episodes=self.n_test_episodes, eval_freq=self.eval_freq,
                                 log_path=logs, render=False, verbose=True, best_model_save_path=model_path)]
        if self.video_freq is not None:
            cb += [self.cb_eval_class(self.env_video, n_eval_episodes=1, eval_freq=self.video_freq, log=False)]
        if self.wandb_on:
            cb += [WandbCallback(gradient_save_freq=100)]
        return cb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Usually don't change
    parser.add_argument('--collision_reward', type=float, default=-100)
    parser.add_argument('--save_model', type=int, default=None)
    parser.add_argument('--model_architecture', type=str, choices=["MlpPolicy"], default="MlpPolicy")
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--normalize_reward', type=int, default=False)
    parser.add_argument('--easy_landing', type=int, default=True)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    # Train Test Params
    parser.add_argument('--eval_freq', type=int, default=50000)
    parser.add_argument('--n_test_episodes', type=int, default=100)
    parser.add_argument('--video_freq', type=int, default=300000)

    # Stability PPO
    parser.add_argument('--kl_target', type=float, default=0.02)
    parser.add_argument('--lr_schedule_step', type=int, default=4)
    parser.add_argument('--clip_range', type=float, default=0.15)

    # Training
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--end_seed', type=int, default=5)
    parser.add_argument('--time_steps', type=int, default=600000)  # 600000
    parser.add_argument('--model', type=str, choices=["SAC", "PPO"], default="PPO")

    # Safe RL Specific
    parser.add_argument('--l_multiplier_init', type=float, default=1.0)
    parser.add_argument('--use_bayes', type=int, default=False)
    parser.add_argument('--use_beta_dist', type=int, default=False)
    parser.add_argument('--gamma_col_net', type=float, default=0.95)
    parser.add_argument('--optimize_gamma_col_net', type=int, default=0)
    parser.add_argument('--gamma_col_net_target', type=float, default=0.95)
    parser.add_argument('--n_lagrange_samples', type=int, default=1)
    parser.add_argument('--n_col_value_samples', type=int, default=1)

    parser.add_argument('--safe_lagrange', type=int, default=True)
    parser.add_argument('--safe_mult', type=int, default=False)
    parser.add_argument('--safe_version', type=str, choices=['Clipped', 'Lagrange', 'Legacy', 'LagrangeOptimized'],
                        default='Clipped')
    # Logging:
    parser.add_argument('--wandb_on', type=int, default=False)
    parser.add_argument('--wandb_name', type=str, default="PPO Beta SafeV1")
    parser.add_argument('--wandb_group', type=str, default="PPO Beta SafeV2")
    parser.add_argument('--wandb_project', type=str, default="report_lunar_lander")
    parser.add_argument('--advantage_mode', type=str, choices=['V1a', 'V1b', 'V2a', 'V2b', 'old'], default='V1a')

    args = parser.parse_args()
    if args.normalize_reward:
        args.collision_reward = -1

    trainer = LunarLanderTrainer(args)
    trainer.train()
