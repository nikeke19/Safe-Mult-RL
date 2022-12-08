"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    # create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
# from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.policies import BaseModel, BasePolicy, ActorCriticPolicy, create_sde_features_extractor

from models.common.torch_layers import create_mlp
from models.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    BetaDistribution,
    make_proba_distribution,
)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :@param use_beta_dist: Whether to use Beta distribution instead of diagonal Gaussian
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            use_beta_dist: bool = False,
            state_dependent_std: bool = False,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init,
            full_std, sde_net_arch, use_expln, squash_output, features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs)

        self.use_beta_dist = use_beta_dist
        self.state_dependent_std = state_dependent_std
        self.std_net = None
        self.std_logging = []
        self.mean_logging = []
        self.alpha_logging = []
        self.beta_logging = []

        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {"full_std": full_std, "squash_output": squash_output, "use_expln": use_expln,
                           "learn_features": sde_net_arch is not None, }
        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde, dist_kwargs, use_beta_dist,
                                                   state_dependent_std)
        self.build_new(lr_schedule)

    def init_weights_and_biases(self, module: nn.Module, gain: float = 1, bias: float = 0) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(bias)

    def build_new(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            # Network can both set mean and std
            if self.state_dependent_std:
                self.action_net, self.std_net = self.action_dist.proba_distribution_net(latent_dim_pi)
            # Std as torch param
            else:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init)

        elif isinstance(self.action_dist, BetaDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init)
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines, features_extractor/mlp
            # values are originally from openai/baselines (default gains/init_scales).
            module_gains = {self.features_extractor: [np.sqrt(2), 0],  # [weight, bias]
                            self.mlp_extractor: [np.sqrt(2), 0],
                            self.action_net: [0.01, 0],
                            self.value_net: [1, 0]}
            # For state dependent std set bias such that std is std_init
            if self.state_dependent_std:
                module_gains[self.std_net] = [0.01, self.log_std_init]

            for module, gain in module_gains.items():
                weight, bias = gain
                module.apply(partial(self.init_weights_and_biases, gain=weight, bias=bias))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor,
                                     latent_sde: Optional[th.Tensor] = None) -> Distribution:
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            # Std as part of the network
            if self.state_dependent_std:
                log_std = self.std_net(latent_pi)
                log_std = th.clamp(log_std, min=-20, max=1)
                return self.action_dist.proba_distribution(mean_actions, log_std)
            # Std as torch param
            return self.action_dist.proba_distribution(mean_actions, self.log_std)

        elif isinstance(self.action_dist, BetaDistribution):
            alpha = mean_actions[:, 0:self.action_dist.action_dim]
            beta = mean_actions[:, self.action_dist.action_dim:]
            return self.action_dist.proba_distribution(alpha, beta)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Saving Distribution params deviations for logging
        if self.use_beta_dist:
            self.alpha_logging.append(distribution.distribution.concentration1.detach().mean().item())
            self.beta_logging.append(distribution.distribution.concentration0.detach().mean().item())
        else:
            self.mean_logging.append(distribution.distribution.mean.detach().mean().item())
            self.std_logging.append(distribution.distribution.stddev.detach().mean().item())

        return actions, values, log_prob

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic).cpu().numpy()

        # Scale the action to action space of the environment
        if isinstance(self.action_space, gym.spaces.Box):
            actions = self.scale_action_to_env(actions)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def scale_action_to_env(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-inf, inf] or [0, 1] to [low, high]
        :param action: Action for Gauss [-inf, inf] and for Beta in [0,1]
        :return: action in range [low, high]
        """
        low, high = self.action_space.low, self.action_space.high
        if self.use_beta_dist:  # action in [0,1]
            return low + (high - low) * action
        # For Gaussian Policy
        return np.clip(action, low, high)

    def unscale_env_action_to_model_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from env: [low, high] to [0, 1] of beta
        @param action: action from env in range [low, high]
        @return: action in range [0, 1]
        """
        # If not using beta, can't make clipping undone, so just return value
        if not self.use_beta_dist:
            return action
        low, high = self.action_space.low, self.action_space.high
        return (action - low) / (high - low)


class ActorCriticCnnPolicy(CustomActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :@param use_beta_dist: Whether to use Beta distribution instead of diagonal Gaussian
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            use_beta_dist: bool = False,
            state_dependent_std: bool = False,
    ):
        super(ActorCriticCnnPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init,
            full_std, sde_net_arch, use_expln, squash_output, features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs, use_beta_dist, state_dependent_std)


class MultiInputActorCriticPolicy(CustomActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space (Tuple)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Uses the CombinedExtractor
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :@param use_beta_dist: Whether to use Beta distribution instead of diagonal Gaussian
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            use_beta_dist: bool = False,
            state_dependent_std: bool = False,
    ):
        super(MultiInputActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init,
            full_std, sde_net_arch, use_expln, squash_output, features_extractor_class, features_extractor_kwargs,
            normalize_images, optimizer_class, optimizer_kwargs, use_beta_dist, state_dependent_std)


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    :param softmax: Use softmax for safe rl collision prediction
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            softmax: bool = False,
            use_bayes: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.use_bayes = use_bayes
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn, softmax=softmax,
                               dropout=use_bayes)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def forward_bayes(self, obs: th.Tensor, actions: th.Tensor, n_passes=10) -> th.Tensor:
        """
        Performs n forward passes to get a distribution of predictions. Then also calculates the epistemic uncertainty
        @param obs:
        @param actions:
        @param n_passes: Amount of forward passes for the Monte Carlo Posterior Estimation
        @return: Tensor with predictions. Size is: [n_passes * n_q_networks, batch_size, ...]
        """

        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        original_batch_size = features.shape[0]

        # Check if model is in eval mode. If yes: then go to training to enable dropout
        training = self.training
        if not training:
            self.train()
        self.train()  # Important to enable dropout

        # If in has shape [batch_size, ...] out has shape [batch_size * n_passes, ...]
        x = features.repeat((n_passes, *tuple(1 for d in features.shape[1:])))
        a = actions.repeat((n_passes, *tuple(1 for d in actions.shape[1:])))
        qvalue_input = th.cat([x, a], dim=1)
        predictions = None

        for q_net in self.q_networks:
            out = q_net(qvalue_input)
            # From [batch_size * n_passes, ...] to [n_passes, batch_size, ...]
            out = out.reshape((n_passes, original_batch_size, *tuple(out.shape[1:])))
            if predictions is None:
                predictions = out
            else:
                predictions = th.cat((predictions, out), dim=0)

        # If model was originally in eval, then set back to eval
        if not training:
            self.eval()

        return predictions

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


_policy_registry = dict()  # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]
