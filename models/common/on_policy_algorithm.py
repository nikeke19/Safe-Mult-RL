import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.common.buffers import CollisionRolloutBuffer, CollisionDictRolloutBuffer
from models.common.policies import ContinuousCritic, BasePolicy
from models.common.util import compute_critic_values, update_value_functions_in_env, find_variable_in_env
from gym.spaces.discrete import Discrete


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.

    # Safe Rl specific
    :param safe_mult: Whether to use multiplicative advantage
    :param safe_lagrange: Whether to constraint the collision probability and relax it with a lagrange multiplier
    :param col_reward: Reward associated with a collision/constraint violation
    :param td_lambda_col_target: Factor for trade-off of bias vs variance in collision prob estimation
    :param gamma_col_net: Discount factor for collision net
    :param use_bayes: Whether to use Monte Carlo Dropout as Bayes Collision Net
    :param advantage_mode: Mode of how to estimate the multiplicative advantage. Choices are: V1a, V1b, V2a, V2b, old
    :param n_col_value_samples: How many samples of a with P_c(x,a) to take to approximate P_c(x).
        Less means more noise -> more exploration
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            policy_base: Type[BasePolicy] = ActorCriticPolicy,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
            # Safe RL specific
            safe_mult: bool = False,
            safe_lagrange: bool = False,
            legacy_advantage: bool = False,
            col_reward: float = -10.0,
            td_lambda_col_target: float = 1,
            gamma_col_net: float = 0.6,
            use_bayes: bool = False,
            advantage_mode: str = "V1a",
            n_col_value_samples: int = 20,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        # Safe RL specific setup
        self.safe_mult = safe_mult
        self.safe_lagrange = safe_lagrange
        self.legacy_advantage = legacy_advantage

        # Parameters
        self.col_reward = col_reward
        self.td_lambda_col_target = td_lambda_col_target
        self.gamma_col_net = gamma_col_net
        self.advantage_mode = advantage_mode
        self.n_col_value_samples = n_col_value_samples

        # For Bayesian version
        self.use_bayes = use_bayes
        self.min_std = None
        self.epistemic_uncertainties = []
        self.epistemic_stds = []
        self.last_std = 1

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if isinstance(self.observation_space, gym.spaces.Dict):
            buffer_cls = CollisionDictRolloutBuffer
        else:
            buffer_cls = CollisionRolloutBuffer

        kwargs = {"col_reward": self.col_reward, "td_lambda_col_target": self.td_lambda_col_target,
                  "gamma_col_net": self.gamma_col_net, "legacy_advantage": self.legacy_advantage,
                  "safe_mult": self.safe_mult, "mode": self.advantage_mode}

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **kwargs
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.setup_separate_optimizers_value_policy_net()

        self.col_net_kwargs = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": [256, 256],
            "activation_fn": th.nn.ReLU,
            "share_features_extractor": True,
            "softmax": True,
            "use_bayes": self.use_bayes,
        }

        self.col_net = self.make_col_net(features_extractor=self.policy.features_extractor)
        col_net_parameters = [param for name, param in self.col_net.named_parameters() if
                              "features_extractor" not in name]
        self.col_net.optimizer = th.optim.Adam(col_net_parameters, lr=self.lr_schedule(1))

    def setup_separate_optimizers_value_policy_net(self) -> None:
        params_feature_extractor = [param for _, param in self.policy.features_extractor.named_parameters()]
        params_feature_extractor += [param for _, param in self.policy.mlp_extractor.shared_net.named_parameters()]

        params_actor_net = [param for _, param in self.policy.mlp_extractor.policy_net.named_parameters()]
        params_actor_net += [param for _, param in self.policy.action_net.named_parameters()]
        if self.policy.state_dependent_std:
            params_actor_net += [param for _, param in self.policy.std_net.named_parameters()]
        else:
            params_actor_net.append(self.policy.log_std)
        params_actor_net += params_feature_extractor  # Design choice

        params_critic_net = [param for _, param in self.policy.mlp_extractor.value_net.named_parameters()]
        params_critic_net += [param for _, param in self.policy.value_net.named_parameters()]

        self.policy.action_net.params = params_actor_net
        self.policy.value_net.params = params_critic_net
        self.policy.value_net.optimizer = th.optim.Adam(params_critic_net, lr=0.0003)
        self.policy.action_net.optimizer = th.optim.Adam(params_actor_net, lr=self.lr_schedule(1))

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer or CollisionRolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        # Resetting logs
        self.epistemic_uncertainties = []
        self.policy.std_logging = []
        self.policy.mean_logging = []
        self.policy.alpha_logging = []
        self.policy.beta_logging = []

        callback.on_rollout_start()
        # For Gazebo Sim:
        if self.env.__class__.__name__ == "GazeboVecEnv":
            find_variable_in_env(self.env, "continue_simulation")()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)

                # Bayesian Model
                if self.use_bayes:
                    mean, std = compute_critic_values(self.col_net, obs_tensor, actions, use_bayes=True)
                    epistemic_certainty = 1
                    # After Burn in Period
                    if self.num_timesteps > 20000:
                        self.epistemic_stds.append(std.item())

                        # When buffer has enough elements to have significant statistics
                        if len(self.epistemic_stds) > 1000:
                            self.epistemic_stds.pop(0)
                            # Min std is only allowed to grow. This allows epistemic uncertainty to shrink over time
                            # as model gets more confident
                            target = max(0.01 * np.mean(self.epistemic_stds),
                                         np.mean(self.epistemic_stds) - np.var(self.epistemic_stds))
                            if self.min_std is None:
                                self.min_std = target
                            # Allow min_std to slowly grow
                            elif self.min_std < target:
                                self.min_std += min(0.1 * (target - self.min_std), 0.1 * self.min_std)
                            epistemic_certainty = np.clip(self.min_std / std.item(), a_min=0.2, a_max=1)
                            self.epistemic_uncertainties.append(1 - epistemic_certainty)

                    col_prob = (epistemic_certainty * (mean + std)).item()

                # Deterministic Model
                else:
                    col_prob = compute_critic_values(self.col_net, obs_tensor, actions, minimum=False)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = self.policy.scale_action_to_env(actions)
                # clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # Update critic values in env
            self.log_to_env(values.cpu().numpy(), col_prob)

            # Step in the environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Add collision probability of sampled actions to get V(x) = E_a[ Q(x,a) ]
            with th.no_grad():
                col_prob_pi, _ = self.forward_sample_col_prob(obs_tensor, self.n_col_value_samples, reduction="mean")

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                               col_prob, col_prob_pi, new_obs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            next_actions, values, _ = self.policy.forward(obs_tensor)
            next_col_prob = compute_critic_values(self.col_net, obs_tensor, next_actions, minimum=False)
            next_col_prob_pi, _ = self.forward_sample_col_prob(obs_tensor, self.n_col_value_samples, reduction="mean")

        # For Gazebo Sim:
        if self.env.__class__.__name__ == "GazeboVecEnv":
            find_variable_in_env(self.env, "pause_simulation")()

        rollout_buffer.compute_returns_and_advantage(last_values=values, last_col_net_values=next_col_prob,
                                                     dones=dones, last_col_prob_pi=next_col_prob_pi)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    # self.logger.record("rollout/ep_rew_mean",
                    #                    safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("important/train_rew",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_std", np.std([ep_info["r"] for ep_info in self.ep_info_buffer]))

                    # Record additional statistics:
                    self.custom_log()

                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def custom_log(self):
        sum_dict = {}
        for episode_dict in self.ep_info_buffer:  # Iterate through episodes
            for key, value in episode_dict.items():
                if key not in sum_dict.keys():  # First time iteration:
                    sum_dict[key] = value
                else:  # Compute sum
                    sum_dict[key] = sum_dict[key] + value
        mean_dict = {}
        for key, value in sum_dict.items():
            # For crash statistics, only account number of crashes, not number of episodes
            if key in ['vels_before_end', 'col_prob_before_end'] and 'crash' in sum_dict.keys():
                mean_dict[key] = value / max(1, sum_dict['crash'])
            # Dangerous situation, if distance from sideline is less then 1m
            elif key in ['col_probs_dangerous_situations'] and 'n_dangerous_situations' in sum_dict.keys():
                mean_dict[key] = value / max(1, sum_dict['n_dangerous_situations'])
            else:
                mean_dict[key] = value / len(self.ep_info_buffer)
        # Logging:
        for key, value in mean_dict.items():
            if key in ['r', 'l', 't']:  # Already logged
                continue
            self.logger.record("rollout/" + key, value)

        # Bayes Log
        if self.use_bayes and len(self.epistemic_uncertainties) > 0:
            self.logger.record("train/epistemic_uncertainty", np.mean(self.epistemic_uncertainties))
            self.logger.record("train/max_epistemic_uncertainty", np.max(self.epistemic_uncertainties))
            self.logger.record("train/epistemic_normalizer", self.min_std)
            self.logger.record("train/epistemic_stds", np.mean(self.epistemic_stds))

        # Logging distribution statistics
        if self.policy.use_beta_dist:
            self.logger.record("rollout/alpha", np.mean(self.policy.alpha_logging))
            self.logger.record("rollout/beta", np.mean(self.policy.beta_logging))
        else:
            self.logger.record("rollout/mean", np.mean(self.policy.mean_logging))
            self.logger.record("rollout/mean_variance", np.var(self.policy.mean_logging))
            # self.logger.record("rollout/std", np.mean(self.policy.std_logging))

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def make_col_net(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        col_net_kwargs = self._update_features_extractor(self.col_net_kwargs, features_extractor)
        return ContinuousCritic(**col_net_kwargs).to(self.device)

    def _update_features_extractor(
            self,
            net_kwargs: Dict[str, Any],
            features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            raise NotImplementedError("Can only be used with shared feature extractor so far")
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def log_to_env(self, q_action, c_action) -> None:
        """
        Informing simulation environment about values and collision probability of current action
        @param q_action:
        @param c_action:
        @return:
        """
        with th.no_grad():
            c = c_action[0]
            q = q_action[0]
            c = [c, c, c, c]
            q = [q, q, q, q]
        # Update log
        update_value_functions_in_env(self.env, q_action, c_action, q, c)

    def forward_sample_col_prob(self, obs: Union[th.Tensor, dict], n_samples=10, reduction=None):
        """
        Samples an action n_samples times and calculate the collision probability of those
        @param obs: Observation
        @param n_samples: Number of actions to sample
        @param reduction: None: return all collision probabilities, Mean: Reduce to mean collision probability
        @return: Collision probabilities of sampled actions and sampled actions
        """
        features = self.policy.extract_features(obs)
        original_batch_size = features.shape[0]
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
        latent_sde = latent_pi
        if self.policy.sde_features_extractor is not None:
            latent_sde = self.policy.sde_features_extractor(features)

        # If in has shape [batch_size, ...] out has shape [batch_size * n_samples, ...]
        x_actor = latent_pi.repeat((n_samples, *tuple(1 for d in features.shape[1:])))
        x_col_net = features.repeat((n_samples, *tuple(1 for d in features.shape[1:])))

        distribution = self.policy._get_action_dist_from_latent(x_actor, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=False)

        qvalue_input = th.cat([x_col_net, actions], dim=1)
        predictions = None
        for col_net in self.col_net.q_networks:
            out = col_net(qvalue_input)
            # From [batch_size * n_passes, ...] to [n_passes, batch_size, ...]
            out = out.reshape((n_samples, original_batch_size, *tuple(out.shape[1:])))
            predictions = out if predictions is None else th.cat((predictions, out), dim=0)

        if reduction == "mean" or reduction == "Mean":
            predictions = predictions.mean(dim=0)
        return predictions, actions
