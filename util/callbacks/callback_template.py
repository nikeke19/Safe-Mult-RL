import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import gym
import warnings
import torch as th
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from models.common.util import compute_critic_values, update_value_functions_in_env
import wandb


class EvalCallbackTemplate(EvalCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    :param log_into_video: Whether to display key stats in the video
    """

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
            log_into_video: bool = True,
            log: bool = True,
            max_steps: int = 5000,
    ):
        super(EvalCallbackTemplate, self).__init__(eval_env, callback_on_new_best, n_eval_episodes, eval_freq, log_path,
                                                   best_model_save_path, deterministic, render, verbose, warn)
        self.log_into_video = log_into_video
        self.log = log
        self.max_steps = max_steps

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            # sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, mean_dict = self.evaluate(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
                callback=self._log_success_callback,
                log_into_video=self.log_into_video
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            # self.logger.record("eval/mean_reward", float(mean_reward))
            if self.log:
                self.logger.record("important/eval_reward", float(mean_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)
                self.logger.record("eval/std_reward", float(std_reward))

                # Log additional metrics:
                for key, value in mean_dict.items():
                    if key in ['r', 'l', 't']:  # Already logged
                        continue
                    self.logger.record("eval/" + key, value)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose > 0:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

            if mean_reward > 0.99 * self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(path)
                    wandb.save(path, base_path=self.best_model_save_path)

                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    def evaluate(
            self,
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
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. This may result in reporting "
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
        sum_dict_n_envs = [{} for i in range(n_envs)]

        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(observations, deterministic=deterministic)
            self.log_value_and_collision_critic_to_env(model, observations, actions, env)

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

                    if done or current_lengths[i] > self.max_steps:
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
                            # Early stopping due to max steps
                            else:
                                episode_rewards.append(current_rewards[i])
                                episode_lengths.append(current_lengths[i])
                                if 'early_stop' not in sum_dict.keys():  # First time iteration:
                                    sum_dict['early_stop'] = 1
                                else:  # Compute sum
                                    sum_dict['early_stop'] = sum_dict['early_stop'] + 1

                            episode_counts[i] += 1
                        else:
                            episode_rewards.append(current_rewards[i])
                            episode_lengths.append(current_lengths[i])
                            episode_counts[i] += 1

                        if current_lengths[i] > self.max_steps:
                            print("Reset due to reaching max steps in eval. Episode count: ", episode_counts[i])
                            env.reset()

                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        if states is not None:
                            states[i] *= 0

            if render:
                env.render()

        # Custom logging
        mean_dict = self.sum_dict_to_custom_mean_dict(sum_dict_n_envs, n_envs, episode_counts)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if reward_threshold is not None:
            assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        if return_episode_rewards:
            return episode_rewards, episode_lengths, mean_dict
        return mean_reward, std_reward, mean_dict

    def log_value_and_collision_critic_to_env(self, model, observations, actions, env):
        """
        Logging Value and Collision probability of chosen action and debug action to environment
        @param model: Either PPO or SAC
        @param observations:
        @param actions: Action in range [low, high] of action space
        @param env: Environment to log into
        @return:
        """
        # Bring obs and action into right shape
        obs_tensor, _ = model.policy.obs_to_tensor(observations)
        actions = model.policy.unscale_env_action_to_model_action(actions)

        with th.no_grad():
            debug_a = self.register_debug_action(model)

            # Calculate Value Functions
            if model.__class__.__name__ == "PPO":
                q_action = model.policy.forward(obs_tensor)[1].cpu().numpy()
                q = [q_action, q_action, q_action, q_action]

            elif model.__class__.__name__ == "SAC":
                q_action = compute_critic_values(model.critic, obs_tensor, actions, minimum=True)
                q = [q_action, q_action, q_action, q_action]
                if debug_a is not None:
                    q = [compute_critic_values(model.critic, obs_tensor, a, minimum=True) for a in debug_a]
            else:
                raise NotImplementedError("not implemented model class: ", model.__class__)

            # Calculate Collision Probabilities
            c_action = compute_critic_values(model.col_net, obs_tensor, actions, minimum=False)
            c = [c_action, c_action, c_action, c_action]
            if debug_a is not None:
                c = [compute_critic_values(model.col_net, obs_tensor, a, minimum=False) for a in debug_a]

        update_value_functions_in_env(env, q_action, c_action, q, c)

    def register_debug_action(self, model) -> np.ndarray:
        """
        Function to register 4 debug actions which are logged to environment
        @return: Debug actions to execute
        """
        # Here return debug action for car racing # L R G B
        action = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float32)
        # Put Action to right scale
        action = model.policy.unscale_env_action_to_model_action(action)
        return action

    def sum_dict_to_custom_mean_dict(self, sum_dict_n_envs, n_envs, episode_counts) -> Dict:
        mean_dict_n_env = []  # list of means for every environment
        for i in range(n_envs):  # Loop over all environments
            mean_dict_1_env = {}
            for key, value in sum_dict_n_envs[i].items():  # For each item compute mean
                # For crash statistics, only account number of crashes, not number of episodes
                if key in ['vels_before_end', 'col_prob_before_end'] and 'crash' in sum_dict_n_envs[i].keys():
                    mean_dict_1_env[key] = value / max(1, sum_dict_n_envs[i]['crash'])
                # Dangerous situation, if distance from sideline is less then 1m
                elif (key in ['col_probs_dangerous_situations']
                      and 'n_dangerous_situations' in sum_dict_n_envs[i].keys()):
                    mean_dict_1_env[key] = value / max(1, sum_dict_n_envs[i]['n_dangerous_situations'])
                else:
                    mean_dict_1_env[key] = value / episode_counts[i]
            mean_dict_n_env.append(mean_dict_1_env)

        # Compute overall mean over n environments
        mean_dict = {}
        for key in mean_dict_n_env[0]:
            mean_dict[key] = np.mean([d[key] for d in mean_dict_n_env])
        return mean_dict
