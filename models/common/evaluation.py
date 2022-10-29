import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import torch as th
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from models.common.util import compute_critic_values, update_value_functions_in_env
from gym.spaces.discrete import Discrete


def evaluate_ppo(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        log_into_video: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param log_into_video: Whether to display key stats in the video
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn("Evaluation environment is not wrapped with a ``Monitor`` wrapper. This may result in reporting "
                      "modified episode lengths and rewards, if other wrappers happen to modify these. Consider "
                      "wrapping environment first with ``Monitor`` wrapper.", UserWarning, )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    sum_dict_n_envs = [{} for i in range(n_envs)]

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)

        # Display key stats on eval video
        # if log_into_video:  # todo make this work for not only ppo
        obs_tensor, _ = model.policy.obs_to_tensor(observations)
        _, values, _ = model.policy.forward(obs_tensor)
        with th.no_grad():
            c_action, c = None, [None, None, None, None]
            q_action = values.cpu().numpy()[0, 0]
            q = [q_action, q_action, q_action, q_action]
            if model.safe_rl_on:
                if isinstance(model.action_space, Discrete):
                    debug_a = debug_a = np.array([[0], [1], [2], [3]])
                else:
                    debug_a = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float32)  # L R G B
                c_action = compute_critic_values(model.col_net, obs_tensor, actions, minimum=False)
                c = [compute_critic_values(model.col_net, obs_tensor, a, minimum=False) for a in debug_a]
        update_value_functions_in_env(env, q_action, c_action, q, c)

        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                sum_dict = sum_dict_n_envs[i]

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])

                            for key, value in info["episode"].items():
                                if key not in sum_dict.keys():  # First time iteration:
                                    sum_dict[key] = value
                                else:  # Compute sum
                                    sum_dict[key] = sum_dict[key] + value

                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_dict_n_env = []  # list of means for every environment
    for i in range(n_envs):  # Loop over all environments
        mean_dict_1_env = {}
        for key, value in sum_dict_n_envs[i].items():  # For each item compute mean
            # For crash statistics, only account number of crashes, not number of episodes
            if key in ['vels_before_end', 'col_prob_before_end'] and 'crash' in sum_dict_n_envs[i].keys():
                mean_dict_1_env[key] = value / max(1, sum_dict_n_envs[i]['crash'])
            # Dangerous situation, if distance from sideline is less then 1m
            elif key in ['col_probs_dangerous_situations'] and 'n_dangerous_situations' in sum_dict_n_envs[i].keys():
                mean_dict_1_env[key] = value / max(1, sum_dict_n_envs[i]['n_dangerous_situations'])
            else:
                mean_dict_1_env[key] = value / episode_counts[i]
        mean_dict_n_env.append(mean_dict_1_env)

    # Compute overall mean over n environments
    mean_dict = {}
    for key in mean_dict_n_env[0]:
        mean_dict[key] = np.mean([d[key] for d in mean_dict_n_env])

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, mean_dict
    return mean_reward, std_reward, mean_dict
