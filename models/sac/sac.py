from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update

from models.sac.policies import SACPolicy
from models.common.off_policy_algorithm import OffPolicyAlgorithm

from models.common.util import weighted_bce
# from util.oc_grids import get_local_oc_grid, idx_to_position, position_to_idx
from envs.util.point_robot_navigation.oc_grid import get_local_oc_grid  # TODO: Remove

from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
import io
import pathlib


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    # Safe RL specific
    :param safe_mult: Whether to use multiplicative value function F
    :param safe_lagrange_baseline: Whether to use the SAC Lagrange baseline w/o F.
    :tau_col_net the soft update coefficient ("Polyak update", between 0 and 1) for collision pred network
    :log_into_rendering Whether to pass Q and P_c to rendering of environment
    :param col_reward: The reward that is received for a collision/constraint violation
    :param gamma_col_net: The discount factor for the safety critic
    :param auxiliary_tasks: Additional tasks to help to extract better features.
    :param safe_version: Options: Legacy, Clipped, Lagrange, LagrangeOptimized
        Defines how to deal with (Q-Qmin) * P_c. Lagrange is with fixed Lagrange multiplier. LagrangeOptimized
        implements primal dual optimization
    :param l_multiplier: If Clipped, Value to clip to. If Lagrange or LagrangeOptimized then initial value of multiplier
    :param optimize_gamma_col_net: Whether to optimize gamma of the safety critic, when collision probability decreases
    :param gamma_col_net_target: Maximum value of gamma col net
    """

    def __init__(
            self,
            policy: Union[str, Type[SACPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1000000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,

            # Safe RL
            safe_mult: bool = False,
            safe_lagrange_baseline: bool = False,
            tau_col_net: float = 0.1,
            col_reward: float = -1.0,
            gamma_col_net: float = 1.0,
            auxiliary_tasks: Tuple[str, ...] = None,
            safe_version: str = "Clipped",
            l_multiplier: float = 5.0,
            optimize_gamma_col_net: bool = False,
            gamma_col_net_target: float = 0.8,
            action_penalty: bool = False
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            safe_mult=safe_mult,
        )

        # Only one of both can be active
        assert sum([safe_mult, safe_lagrange_baseline]) < 2

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        # Safe RL
        self.type = "SAC"
        self.q_min_total = th.tensor(100)
        self.r_min = 0
        self.tau_col_net = tau_col_net
        self.safe_mult = safe_mult
        self.safe_lagrange = safe_lagrange_baseline
        self.col_reward = col_reward
        self.gamma_col_net = gamma_col_net
        self.gamma_col_net_init = gamma_col_net
        self.safe_version = safe_version
        self.l_multiplier = l_multiplier

        # Bayesian version
        self.use_bayes = False
        self.min_std = None
        self.epistemic_stds = []
        if "use_bayes" in self.policy_kwargs.keys():
            self.use_bayes = self.policy_kwargs["use_bayes"]
        self.auxiliary_tasks = [] if auxiliary_tasks is None else auxiliary_tasks

        # Gamma col net optimization
        self.optimize_gamma_col_net = optimize_gamma_col_net
        self.gamma_col_net_target = gamma_col_net_target

        self.action_penalty = action_penalty

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        # Add two collision critics approximating collision probability
        self.col_net = self.policy.col_net
        self.col_net_target = self.policy.col_net_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.col_net.optimizer]

        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, col_net_losses = [], [], []
        epistemic_uncertainties = []
        auxiliary_losses = []
        action_penalties = []
        q_multipliers = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            '''
            Optimizing Entropy
            '''

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            '''
            Optimizing Critic
            '''

            # Find q targets
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # Normal learning
                dones = replay_data.dones
                rewards = replay_data.rewards
                # Safe Q-learning: Disregard collisions
                if self.safe_mult:
                    self.r_min = min(self.r_min, rewards[rewards > self.col_reward].min().item())
                    rewards = th.clamp(replay_data.rewards, min=self.r_min)

                # td error + entropy term
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            '''
            Optimizing Safety Critic
            '''
            with th.no_grad():
                next_col_prob = th.cat(self.col_net_target(replay_data.next_observations, next_actions), dim=1)
                next_col_prob, _ = th.max(next_col_prob, dim=1, keepdim=True)
                target_col_prob = (replay_data.rewards < 0.99 * self.col_reward).type(th.int32) + \
                                  (1 - replay_data.dones) * self.gamma_col_net * next_col_prob

            current_col_prob = self.col_net(replay_data.observations, replay_data.actions)
            col_net_loss = 0.5 * sum(
                [weighted_bce(pred, target_col_prob, p_threshold=0.6) for pred in current_col_prob])

            col_net_losses.append(col_net_loss.item())

            self.col_net.optimizer.zero_grad()
            col_net_loss.backward()
            self.col_net.optimizer.step()

            '''
            Optimize the Actor
            '''

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            # Calculate Collision Probability of Sampled Action:
            if self.safe_mult:
                if self.use_bayes:
                    col_prob, epistemic_uncertainty = self.get_bayesian_collision_probability(replay_data, actions_pi)
                    epistemic_uncertainties.append(epistemic_uncertainty)
                else:
                    col_prob = th.cat(self.col_net.forward(replay_data.observations, actions_pi), dim=1)
                    col_prob, _ = th.max(col_prob, dim=1, keepdim=True)

                self.q_min_total = min(self.q_min_total, min_qf_pi.min()).detach()
                # Different Version how to carry out multiplicative Version
                if self.safe_version == "Clipped":
                    q_multiplier = th.clamp(min_qf_pi.detach() - self.q_min_total, min=0, max=self.l_multiplier)
                    f = (1 - col_prob.detach()) * min_qf_pi - q_multiplier * col_prob
                elif self.safe_version in ["Lagrange", "LagrangeOptimized"]:
                    f = (1 - col_prob.detach()) * min_qf_pi - self.l_multiplier * col_prob
                elif self.safe_version == "Legacy":
                    q_multiplier = min_qf_pi.detach() - self.q_min_total
                    f = (min_qf_pi - self.q_min_total) * (1 - col_prob)
                else:
                    raise NotImplemented("Your version: " + self.safe_version + " is not implemented")
            # Lagrange baseline without multiplicative value function
            elif self.safe_lagrange:
                col_prob = th.cat(self.col_net.forward(replay_data.observations, actions_pi), dim=1)
                col_prob, _ = th.max(col_prob, dim=1, keepdim=True)
                f = min_qf_pi - self.l_multiplier * col_prob
            # Otherwise, f = q critic
            else:
                f = min_qf_pi

            # Auxiliary Tasks
            auxiliary_loss = self.get_auxiliary_loss(replay_data)
            if len(self.auxiliary_tasks) > 0:
                auxiliary_losses.append(auxiliary_loss.item())

            action_penalty = 0
            if self.action_penalty:
                action_penalty = 0.1 * (actions_pi[:, 1] ** 2).mean()
                action_penalties.append(action_penalty.item())

            # max( Q(x,a) - alpha * log_prob(a|x) )
            actor_loss = (ent_coef * log_prob - f).mean() + auxiliary_loss + action_penalty
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.col_net.parameters(), self.col_net_target.parameters(), self.tau_col_net)

            # Optimize Lagrange multiplier
            if (self.safe_version == "LagrangeOptimized" and self.safe_mult) or self.safe_lagrange:
                self.l_multiplier = max(0.05, self.l_multiplier + 1e-4 * (col_prob.mean().item() - 0.1))

            # Optimize Gamma col net
            if self.optimize_gamma_col_net and (self.safe_mult or self.safe_lagrange):
                target = 0.1 - col_prob.mean().item()
                alpha = 0.005 if target > 0 else 0.001
                self.gamma_col_net = min(self.gamma_col_net_target, self.gamma_col_net + alpha * target)
                self.gamma_col_net = max(self.gamma_col_net, self.gamma_col_net_init)

            # Log Q-Multiplier
            if self.safe_mult and self.safe_version in ["Clipped", "Legacy"]:
                q_multipliers.append(q_multiplier.mean().item())

        self._n_updates += gradient_steps

        '''
        Logging
        '''

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/gamma_col_net", self.gamma_col_net)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/col_net_loss", np.mean(col_net_losses))

        if len(self.auxiliary_tasks) > 0:
            self.logger.record("train/auxiliary_loss", np.mean(auxiliary_losses))

        if self.action_penalty:
            self.logger.record("train/action_penalty", np.mean(action_penalties))

        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # Log multiplier
        if self.safe_mult:
            if self.safe_version in ["Lagrange", "LagrangeOptimized"]:
                self.logger.record("train/multiplier", self.l_multiplier)
            else:
                self.logger.record("train/multiplier", np.mean(q_multipliers))
        if self.safe_lagrange:
            self.logger.record("train/multiplier", self.l_multiplier)

        if self.safe_mult and self.use_bayes and len(epistemic_uncertainties) > 0:
            self.logger.record("train/epistemic_uncertainty", np.mean(epistemic_uncertainties))
            self.logger.record("train/max_epistemic_uncertainty", np.max(epistemic_uncertainties))
            self.logger.record("train/epistemic_normalizer", self.min_std)
            self.logger.record("train/epistemic_stds", np.mean(self.epistemic_stds))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "SAC",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def get_auxiliary_loss(self, replay_data) -> th.Tensor:
        """
        Collection of auxiliary losses. Append your custom auxiliary loss here
        @param replay_data: Data buffer that stores the transitions
        @return: Mean Auxiliary loss value with grad_fn
        """
        auxiliary_loss = 0
        for task in self.auxiliary_tasks:
            if task == "autoencoder":
                w = 10
                x = replay_data.observations["img"] if isinstance(replay_data.observations, dict) \
                    else replay_data.observations
                x_hat = self.actor.features_extractor.forward_autoencoder(x)
                x = x / 255  # divide by 255 since models outputs number between 0 and 1
                obstacle_loss = (x[:, 0] - x_hat[:, 0]) ** 2
                state_loss = (x[:, 1] - x_hat[:, 1]) ** 2
                weights = th.ones_like(state_loss) * 0.5
                weights[x[:, 1] > 0.35] = 10
                auxiliary_loss = auxiliary_loss + w * (weights * state_loss).mean()
                # auxiliary_loss = auxiliary_loss + w * (0.01 * obstacle_loss + weights * state_loss).mean()

            if task == "local_oc_grid":
                w = 50
                x = replay_data.observations["img"] if isinstance(replay_data.observations,
                                                                  dict) else replay_data.observations
                x_local_hat = self.actor.features_extractor.forward_local_oc_grid(x)
                x_local = get_local_oc_grid(x / 255)
                weights = th.ones_like(x_local) * 0.5
                weights[x_local > 0.1] = 10
                # Cells outside oc grid are marked with -1 in local oc_grid. Disregard those for loss
                weights[x_local < 0] = 0
                reconstruction_loss = weights * (x_local - x_local_hat) ** 2
                auxiliary_loss = auxiliary_loss + w * reconstruction_loss.mean()

            if task == "autoencoder_local":
                w = 10
                x = replay_data.observations["img"] if isinstance(replay_data.observations, dict) \
                    else replay_data.observations
                x_hat = self.actor.features_extractor.forward_autoencoder(x)
                x = x[:, [-1]] / 255  # divide by 255 since models outputs number between 0 and 1
                obstacle_loss = weighted_bce(x_hat, x)
                auxiliary_loss = auxiliary_loss + w * obstacle_loss

            if task == "depth":
                w = 10
                x = replay_data.observations["img"] if isinstance(replay_data.observations, dict) \
                    else replay_data.observations

                batch_size = x.shape[0]
                idx = th.where(x > 250)
                idx = th.stack(idx, dim=0).cpu().numpy()[[0, 2, 3]]

                # find the min distances
                min_distances = []
                start = 0
                end = 0
                min_distances = th.zeros(batch_size, dtype=float, device="cuda")
                for i in range(batch_size):  # First row, last element is last batch
                    # Find how many elements correspond to batch
                    for k in range(start, idx.shape[1]):
                        if idx[0, k] != i:
                            end = k
                            break
                        elif k == idx.shape[1] - 1:  # For last element
                            end = k
                    # Distance calculation
                    distances = np.linalg.norm(idx[1:, start:end].T - np.array([5, 5]), axis=1)
                    if distances.shape[0] == 0:  # No obstacle
                        min_distances[i] = 1
                    else:
                        min_distances[i] = np.min(distances) / 7  # 7 is max distance for 11x11 grid
                    # Update search space
                    start = end

                # Loss calculation
                depth_pred = self.actor.features_extractor.forward_depth(x)
                auxiliary_loss = auxiliary_loss + w * ((depth_pred - min_distances.unsqueeze(1)) ** 2).mean()
        return auxiliary_loss

    def get_bayesian_collision_probability(self, replay_data, actions_pi: th.Tensor):
        """
        Use Monte Carlo Dropout to yield a bayesian estimate of the epistemic uncertainty. Then use a heuristic to
        yield an epistemic discount factor. This factor discounts the safety critic.
        @param replay_data:
        @param actions_pi: Suggested action at current state
        @return: Probability of constraint violation discounted with epistemic uncertainty
        """

        col_probs = self.col_net.forward_bayes(replay_data.observations, actions_pi)
        mean, std = col_probs.mean(), col_probs.std()
        epistemic_certainty = 1
        # After Burn in Period
        if self.num_timesteps - self.learning_starts > 5000:
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
                # Allow min_std to shrink more slowly
                elif self.min_std > target:
                    self.min_std -= min(0.001 * (self.min_std - target), 0.01 * self.min_std)
                epistemic_certainty = np.clip(self.min_std / std.item(), a_min=0.2, a_max=1)

        col_prob = (mean + std) * epistemic_certainty
        return col_prob, 1 - epistemic_certainty

    @classmethod
    def load(
            cls,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device, custom_objects=custom_objects)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            # check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model
