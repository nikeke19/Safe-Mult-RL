from util.callbacks.callback_template import EvalCallbackTemplate
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gym
import torch as th
import numpy as np
from models.common.util import compute_critic_values, update_value_functions_in_env
from gym.spaces.discrete import Discrete


class LunarLanderEvalCallback(EvalCallbackTemplate):
    def __init__(
            self, eval_env: Union[gym.Env, VecEnv], callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5, eval_freq: int = 10000, log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None, deterministic: bool = True, render: bool = False,
            verbose: int = 1, warn: bool = True, log_into_video: bool = True, log: bool = True, max_steps: int = 1005
    ):
        super(LunarLanderEvalCallback, self).__init__(
            eval_env, callback_on_new_best, n_eval_episodes, eval_freq, log_path, best_model_save_path, deterministic,
            render, verbose, warn, log_into_video, log, max_steps)

    def register_debug_action(self, model):
        # Nop, fire left engine, main engine, right engine
        action = np.array([[0, 0], [0, -1], [1, 0], [0, 1]], dtype=np.float32)
        # Put Action to right scale
        action = model.policy.unscale_env_action_to_model_action(action)
        return action

    # def log_value_and_collision_critic_to_env(self, model, observations, actions, env):
    #     # Display key stats on eval video
    #     obs_tensor, _ = model.policy.obs_to_tensor(observations)
    #     c_action, c = None, [None, None, None, None]
    #     with th.no_grad():
    #         if isinstance(model, PPO):
    #             values = model.policy.forward(obs_tensor)[1]
    #             q_action = values.item()
    #             q = [q_action, q_action, q_action, q_action]
    #             # if model.safe_rl_on:
    #             if isinstance(model.action_space, Discrete):
    #                 # Nop, fire left engine, main engine, right engine
    #                 debug_a = np.array([[0], [1], [2], [3]])
    #             else:
    #                 # Nop, fire left engine, main engine, right engine
    #                 debug_a = np.array([[0, 0], [0, -1], [1, 0], [0, 1]])
    #             c_action = compute_critic_values(model.col_net, obs_tensor, actions, minimum=False)
    #             c = [compute_critic_values(model.col_net, obs_tensor, a, minimum=False) for a in debug_a]
    #         elif isinstance(model, DQN):
    #             q = model.q_net(obs_tensor).reshape(-1).cpu().numpy()
    #             q_action = q[actions]
    #             # if model.safe_rl_on:
    #             c = model.col_net(obs_tensor).reshape(-1).cpu().numpy()
    #             c_action = c[actions]
    #         elif isinstance(model, SAC):
    #             # Nop, fire left engine, main engine, right engine
    #             debug_a = np.array([[0, 0], [0, -1], [1, 0], [0, 1]])
    #             q_action = compute_critic_values(model.critic, obs_tensor, actions, minimum=True)
    #             q = [compute_critic_values(model.critic, obs_tensor, a, minimum=True) for a in debug_a]
    #             # if model.safe_rl_on:
    #             c_action = compute_critic_values(model.col_net, obs_tensor, actions, minimum=False)
    #             c = [compute_critic_values(model.col_net, obs_tensor, a, minimum=False) for a in debug_a]
    #
    #         else:
    #             raise NotImplementedError("not implemented model class: ", model.__class__)
    #
    #     update_value_functions_in_env(env, q_action, c_action, q, c)

