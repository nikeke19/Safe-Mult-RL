from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from typing import Generator, Optional, Union, NamedTuple, Dict, Tuple
from gym import spaces
import torch as th
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

TensorDict = Dict[Union[str, int], th.Tensor]


# TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
# in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
def compute_safety_critic_targets(buffer_size: int,
                                  dones: np.ndarray,
                                  last_col_net_values: np.ndarray,
                                  episode_starts: np.ndarray,
                                  col_prob_preds: np.ndarray,
                                  col_rewards: np.ndarray,
                                  gamma_col_net: float,
                                  td_lambda: float,
                                  n_envs: int
                                  ) -> np.ndarray:
    col_prob_targets = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_safety_lam = 0

    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_col_prob = last_col_net_values.flatten()
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_col_prob = col_prob_preds[step + 1]

        # TD(Lambda) Safety Critic
        q_safety = col_rewards[step] + gamma_col_net * next_col_prob * next_non_terminal
        delta_safety = q_safety - col_prob_preds[step]
        last_safety_lam = delta_safety + gamma_col_net * td_lambda * next_non_terminal * last_safety_lam
        col_prob_targets[step] = last_safety_lam + col_prob_preds[step]

    return col_prob_targets


def compute_critic_targets(last_values: th.Tensor,
                           dones: np.ndarray,
                           buffer_size: int,
                           values: np.ndarray,
                           episode_starts: np.ndarray,
                           safe_mult: bool,
                           rewards: np.ndarray,
                           gamma: float,
                           n_envs: int,
                           gae_lambda: float,
                           min_reward: float,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    returns = np.zeros((buffer_size, n_envs), dtype=np.float32)
    qs = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_val_lam = 0

    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_values = last_values.clone().cpu().numpy().flatten()
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]

        if safe_mult:
            reward = np.clip(rewards[step], a_min=min_reward, a_max=np.inf)
        else:
            reward = rewards[step]

        # TD Lambda Critic
        qs[step] = reward + gamma * next_values * next_non_terminal
        delta_val = qs[step] - values[step]
        last_val_lam = delta_val + gamma * gae_lambda * next_non_terminal * last_val_lam
        returns[step] = last_val_lam + values[step]

    return returns, qs


def compute_advantage(buffer_size: int,
                      dones: np.ndarray,
                      last_col_net_values: np.ndarray,
                      episode_starts: np.ndarray,
                      safe_mult: bool,
                      qs: np.ndarray,
                      values: np.ndarray,
                      col_prob_preds: np.ndarray,
                      gamma: float,
                      n_envs: int,
                      gae_lambda: float,
                      min_value: float,
                      rewards: np.ndarray,
                      last_values: th.Tensor,
                      min_reward: float,
                      col_prob_pi: np.ndarray,
                      last_col_prob_pi: th.Tensor,
                      col_penalty: float,
                      mode="V1a"
                      ) -> Tuple[np.ndarray, np.ndarray]:
    # Modes:
    # V1a:  A = [r + y * (V(x') - Vmin) * P(x'))] - (V(x) - Vmin) * P(x)
    # V1b:  A = [r * next_non_terminal + y * (V(x') - Vmin) * P(x'))] - (V(x) - Vmin) * P(x)
    # V2a:  A = [(Q(x,a) - Qmin) * P(x')] - (V(x) - Vmin) * P(x)
    # V2b:  A = Q(x,a) * P(x,a) - V(x) * P(x)
    # old:  A = [(Q(x,a) - Qmin) * P(x',a')] - (V(x) - Vmin) * P(x,a)

    advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
    diff_col_now_next = np.zeros((buffer_size, n_envs), dtype=np.float32)
    last_gae_lam = 0

    for step in reversed(range(buffer_size)):
        if step == buffer_size - 1:
            next_non_terminal = 1.0 - dones
            next_col_prob = last_col_net_values.flatten()
            next_col_prob_pi = last_col_prob_pi.clone().cpu().numpy().flatten()
            next_values = last_values.clone().cpu().numpy().flatten()
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_col_prob = col_prob_preds[step + 1]
            next_col_prob_pi = col_prob_pi[step + 1]
            next_values = values[step + 1]

        # Infer if collision occurred
        next_non_col = np.abs(rewards[step] - col_penalty) > 0.001

        # Reward Clipping:
        if safe_mult:
            reward = np.clip(rewards[step], a_min=min_reward, a_max=np.inf)
        else:
            reward = rewards[step]

        # GAE(Lambda) Safety:
        if not safe_mult:  # Then normal advantage estimation
            delta = qs[step] - values[step]
        else:
            if mode == "old":
                f = (values[step] - min_value) * (1 - col_prob_preds[step])
                delta = (qs[step] - min_value) * (1 - next_col_prob) - f
            elif mode == "V1a":
                f_next = next_non_terminal * (next_values - min_value) * (1 - next_col_prob_pi)
                f = (values[step] - min_value) * (1 - col_prob_pi[step])
                delta = (reward + gamma * f_next) - f
            elif mode == "V1b":
                f_next = next_non_terminal * (next_values - min_value) * (1 - next_col_prob_pi)
                f = (values[step] - min_value) * (1 - col_prob_pi[step])
                delta = (reward * next_non_col + gamma * f_next) * next_non_terminal - f
            elif mode == "V2a":
                f = (values[step] - min_value) * (1 - col_prob_pi[step])
                q = (qs[step] - min_value) * (1 - next_col_prob_pi)
                delta = q - f
            elif mode == "V2b":
                f = (values[step] - min_value) * (1 - col_prob_pi[step])
                q = (qs[step] - min_value) * (1 - col_prob_preds[step])
                delta = q - f

                # If next state is terminal then set q to minimum value which is q. Else, use col critic
                # q = qs[step].copy()
                # idx_non_terminal = np.where(next_non_terminal > 0.1)
                # q[idx_non_terminal] = (q[idx_non_terminal] - min_value) * (1 - col_prob_preds[step][idx_non_terminal])
                # delta = q - (values[step] - min_value) * (1 - col_prob_pi[step])

                # If next state is terminal then set q to 0. Else, use col critic
                # q = qs[step].copy()
                # idx_non_terminal = np.where(next_non_terminal > 0.1)
                # q[idx_non_terminal] = (q[idx_non_terminal] - min_value) * (1 - col_prob_preds[step][idx_non_terminal])
                # idx_terminal = np.where(next_non_terminal < 0.01)
                # q[idx_terminal] = 0
                # delta = q - (values[step] - min_value) * (1 - col_prob_pi[step])

            else:
                raise NotImplementedError("Your mode: ", mode, " is not implemented. Choose <old>, <q_split_out>, ",
                                          "<q_complete>", "instead")

        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam

        # Log difference in safety to see what model is learning
        if mode == "old":  # A = [Q * P(x',a')] - V(x) * P(x,a)
            diff_col_now_next[step] = col_prob_preds[step] - next_col_prob
        elif mode == "V2b":  # A = Q(x,a) * P(x,a) - V(x) * P(x)
            diff_col_now_next[step] = col_prob_pi[step] - col_prob_preds[step]
        else:  # A = [r + y * (V(x') * P(x'))] - V(x) * P(x)
            diff_col_now_next[step] = col_prob_pi[step] - next_col_prob_pi

    return advantages, diff_col_now_next


class CollisionRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_col_probs: th.Tensor
    col_prob_targets: th.Tensor
    col_rewards: th.Tensor
    diff_col_now_next: th.Tensor
    next_observations: th.Tensor


class CollisionDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_col_probs: th.Tensor
    col_prob_targets: th.Tensor
    col_rewards: th.Tensor
    diff_col_now_next: th.Tensor
    next_observations: TensorDict


class CollisionRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    :@param td_lambda_col_target: TD Lambda factor for collision probability target
    :@param gamma_col_net: Discount factor for the safety critic
    :@param col_reward: Reward associated with a collision/constraint violation
    :@param safe_mult: Whether to use the multiplicative advantage
    :@param legacy_advantage: Wheter to use Jens Version of Advantage
    :@param mode: How to estimate advantage in mult setting: Choices are: <old>, <q_split_out>, <q_complete>
        mapping from report is: old=V2a, q_split_out=V1, q_complete=V2b
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            td_lambda_col_target: float = 1,
            gamma_col_net: float = 0.6,
            col_reward: float = -10,
            safe_mult: bool = False,
            legacy_advantage: bool = False,
            mode: str = "V1a"
    ):
        super(CollisionRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, gae_lambda,
                                                     gamma, n_envs)
        self.td_lambda = td_lambda_col_target
        self.gamma_col_net = gamma_col_net
        self.col_penalty = col_reward
        self.legacy_advantage = legacy_advantage
        self.safe_mult = False if legacy_advantage else safe_mult
        self.col_prob_targets, self.col_prob_preds, self.col_rewards, self.advantages_original = None, None, None, None
        self.diff_col_now_next, self.qs = None, None
        self.next_observations = None
        self.min_value = 0
        self.min_reward = 0

        self.col_prob_pi = None
        self.mode = mode

        self.reset()

    def reset(self) -> None:
        super(CollisionRolloutBuffer, self).reset()

        self.col_prob_targets = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_prob_preds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages_original = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.diff_col_now_next = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.qs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_prob_pi = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)

    def compute_returns_and_advantage(self, last_values: th.Tensor, last_col_net_values: np.ndarray,
                                      dones: np.ndarray, last_col_prob_pi: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        :param last_col_net_values: safety critic estimation P(x,a) for the last step (one for each env)
        :param last_col_prob_pi: safety critic estimation P(x) for the last step (one for each env)
        """
        # For clamping, take the minimum value, which is not a collision
        self.min_reward = min(self.min_reward, np.min(self.rewards[self.rewards > self.col_penalty]))

        self.col_prob_targets = compute_safety_critic_targets(
            self.buffer_size, dones, last_col_net_values, self.episode_starts, self.col_prob_preds, self.col_rewards,
            self.gamma_col_net, self.td_lambda, self.n_envs)

        self.returns, self.qs = compute_critic_targets(
            last_values, dones, self.buffer_size, self.values, self.episode_starts, self.safe_mult, self.rewards,
            self.gamma, self.n_envs, self.gae_lambda, self.min_reward)

        if self.mode == "q_split_out":
            self.min_value = min(self.min_value, np.min(self.values))
        else:
            batch_min_value = min(np.min(self.qs), np.min(self.values))
            self.min_value = min(self.min_value, batch_min_value)

        self.advantages, self.diff_col_now_next = compute_advantage(
            self.buffer_size,
            dones,
            last_col_net_values,
            self.episode_starts,
            self.safe_mult,
            self.qs,
            self.values,
            self.col_prob_preds,
            self.gamma,
            self.n_envs,
            self.gae_lambda,
            self.min_value,
            self.rewards,
            last_values,
            self.min_reward,
            self.col_prob_pi,
            last_col_prob_pi,
            self.col_penalty,
            mode=self.mode
        )

        if self.legacy_advantage:
            self.advantages = (self.advantages - self.advantages.min()) * (1 - self.col_prob_preds)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            col_prob: np.ndarray,
            col_prob_pi: th.Tensor,
            next_obs: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param col_prob: estimated probability of collision P(x,a)
        :param col_prob_pi: estimated probability of collisions by sampling from policy P(x)
        :param next_obs: Next Observation
        """
        super(CollisionRolloutBuffer, self).add(obs, action, reward, episode_start, value, log_prob)
        # self.pos - 1 because pos was already incremented in super.add(...)
        self.col_rewards[self.pos - 1] = np.array(reward < 0.99 * self.col_penalty, dtype=np.int32).copy()
        self.col_prob_preds[self.pos - 1] = col_prob.flatten()
        self.col_prob_pi[self.pos - 1] = col_prob_pi.clone().cpu().numpy().flatten()

        if isinstance(self.observation_space, spaces.Discrete):
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)
        self.next_observations[self.pos - 1] = np.array(next_obs).copy()

    def get(self, batch_size: Optional[int] = None) -> Generator[CollisionRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            old_col_probs: th.Tensor
            col_prob_targets: th.Tensor
            col_reward: th.Tensor
            advantages_original: th.Tensor

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                # Safe Rl specific
                "col_prob_preds",
                "col_prob_targets",
                "col_rewards",
                "diff_col_now_next",
                "next_observations",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> CollisionRolloutBufferSamples:
        # next_idx = np.clip(batch_inds + 1, a_min=0, a_max=self.buffer_size - 1)
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            # Safe rl specific
            self.col_prob_preds[batch_inds].flatten(),
            self.col_prob_targets[batch_inds].flatten(),
            self.col_rewards[batch_inds].flatten(),
            self.diff_col_now_next[batch_inds].flatten(),
            # Auxiliary task prediction
            self.next_observations[batch_inds],
        )
        return CollisionRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CollisionDictRolloutBuffer(DictRolloutBuffer):
    """
        Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
        Extends the RolloutBuffer to use dictionary observations

        It corresponds to ``buffer_size`` transitions collected
        using the current policy.
        This experience will be discarded after the policy update.
        In order to use PPO objective, we also store the current value of each state
        and the log probability of each taken action.

        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        Hence, it is only involved in policy and value function training but not action selection.

        :param buffer_size: Max number of element in the buffer
        :param observation_space: Observation space
        :param action_space: Action space
        :param device:
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Equivalent to classic advantage when set to 1.
        :param gamma: Discount factor
        :param n_envs: Number of parallel environments
        :@param td_lambda_col_target: TD Lambda factor for collision probability target
        :@param gamma_col_net: Discount factor for the collision critic
        :@param col_reward: Reward associated with a collision
        :@param safe_mult: Whether to use Collision Prob Constraint as Lagrangian or in the multiplicative way
        :@param mode: How to estimate advantage in mult setting: Choices are: <old>, <q_split_out>, <q_complete>
        """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            td_lambda_col_target: float = 1,
            gamma_col_net: float = 0.6,
            col_reward: float = -10,
            safe_mult: bool = False,
            legacy_advantage: bool = False,
            mode: str = "V1a"
    ):
        super(CollisionDictRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device,
                                                         gae_lambda, gamma, n_envs)
        self.td_lambda = td_lambda_col_target
        self.gamma_col_net = gamma_col_net
        self.col_penalty = col_reward
        self.legacy_advantage = legacy_advantage
        self.safe_mult = False if legacy_advantage else safe_mult
        self.col_prob_targets, self.col_prob_preds, self.col_rewards, self.advantages_original = None, None, None, None
        self.diff_col_now_next, self.qs = None, None
        self.next_observations = None
        self.min_value = 0
        self.min_reward = 0

        self.col_prob_pi = None
        self.mode = mode

        self.reset()

    def reset(self) -> None:
        super(CollisionDictRolloutBuffer, self).reset()

        self.col_prob_targets = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_prob_preds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages_original = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.diff_col_now_next = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.qs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.col_prob_pi = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.next_observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.next_observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)

    def add(
            self,
            obs: Dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            col_prob: np.ndarray,
            col_prob_pi: th.Tensor,
            next_obs: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param col_prob: estimated probability of collision P(x,a)
        :param col_prob_pi: estimated probability of collisions by sampling from policy P(x)
        :param next_obs: Next Observation
        """
        super(CollisionDictRolloutBuffer, self).add(obs, action, reward, episode_start, value, log_prob)
        # self.pos - 1 because pos was already incremented in super.add(...)
        self.col_rewards[self.pos - 1] = np.array(reward < 0.99 * self.col_penalty, dtype=np.int32).copy()
        self.col_prob_preds[self.pos - 1] = col_prob.flatten()
        self.col_prob_pi[self.pos - 1] = col_prob_pi.clone().cpu().numpy().flatten()

        for key in self.observations.keys():
            next_obs_ = np.array(next_obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs_ = next_obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos - 1] = next_obs_

    def get(self, batch_size: Optional[int] = None) -> Generator[CollisionDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for key, obs in self.next_observations.items():
                self.next_observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                # Safe Rl specific
                "col_prob_preds",
                "col_prob_targets",
                "col_rewards",
                "advantages_original",
                "diff_col_now_next",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> CollisionDictRolloutBufferSamples:

        # next_idx = np.clip(batch_inds + 1, a_min=0, a_max=self.buffer_size - 1)
        return CollisionDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            old_col_probs=self.to_torch(self.col_prob_preds[batch_inds].flatten()),
            col_prob_targets=self.to_torch(self.col_prob_targets[batch_inds].flatten()),
            col_rewards=self.to_torch(self.col_rewards[batch_inds].flatten()),
            diff_col_now_next=self.to_torch(self.diff_col_now_next[batch_inds].flatten()),
            next_observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.next_observations.items()},
        )

    def compute_returns_and_advantage(self, last_values: th.Tensor, last_col_net_values: np.ndarray,
                                      dones: np.ndarray, last_col_prob_pi: th.Tensor) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        :param last_col_net_values: safety critic estimation P(x,a) for the last step (one for each env)
        :param last_col_prob_pi: safety critic estimation P(x) for the last step (one for each env)
        """
        # For clamping, take the minimum value, which is not a collision
        self.min_reward = min(self.min_reward, np.min(self.rewards[self.rewards > self.col_penalty]))

        self.col_prob_targets = compute_safety_critic_targets(
            self.buffer_size, dones, last_col_net_values, self.episode_starts, self.col_prob_preds, self.col_rewards,
            self.gamma_col_net, self.td_lambda, self.n_envs)

        self.returns, self.qs = compute_critic_targets(
            last_values, dones, self.buffer_size, self.values, self.episode_starts, self.safe_mult, self.rewards,
            self.gamma, self.n_envs, self.gae_lambda, self.min_reward)

        if self.mode == "q_split_out":
            self.min_value = min(self.min_value, np.min(self.values))
        else:
            batch_min_value = min(np.min(self.qs), np.min(self.values))
            self.min_value = min(self.min_value, batch_min_value)

        self.advantages, self.diff_col_now_next = compute_advantage(
            self.buffer_size,
            dones,
            last_col_net_values,
            self.episode_starts,
            self.safe_mult,
            self.qs,
            self.values,
            self.col_prob_preds,
            self.gamma,
            self.n_envs,
            self.gae_lambda,
            self.min_value,
            self.rewards,
            last_values,
            self.min_reward,
            self.col_prob_pi,
            last_col_prob_pi,
            self.col_penalty,
            mode=self.mode
        )

        if self.legacy_advantage:
            self.advantages = (self.advantages - self.advantages.min()) * (1 - self.col_prob_preds)
