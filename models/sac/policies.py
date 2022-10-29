import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import (
    BasePolicy,
    # ContinuousCritic,
    register_policy)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from models.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import Actor
from models.common.distributions import BetaDistribution
import numpy as np

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class CustomActor(Actor):
    """
        Actor network (policy) for SAC.

        :param observation_space: Obervation space
        :param action_space: Action space
        :param net_arch: Network architecture
        :param features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        :param features_dim: Number of features
        :param activation_fn: Activation function
        :param use_sde: Whether to use State Dependent Exploration or not
        :param log_std_init: Initial value for the log standard deviation
        :param full_std: Whether to use (n_features x n_actions) parameters
            for the std instead of only (n_features,) when using gSDE.
        :param sde_net_arch: Network architecture for extracting features
            when using gSDE. If None, the latent features from the policy will be used.
            Pass an empty list to use the states as features.
        :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param use_beta_dist: Whether to use Beta distribution instead of squashed Gaussian
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
            use_beta_dist: bool = False,
    ):
        super(CustomActor, self).__init__(
            observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn, use_sde,
            log_std_init, full_std, sde_net_arch, use_expln, clip_mean, normalize_images)

        self.use_beta_dist = use_beta_dist
        self.std_logging = []
        self.mean_logging = []
        self.alpha_logging = []
        self.beta_logging = []

        if self.use_beta_dist:
            action_dim = get_action_dim(self.action_space)
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
            self.action_dist = BetaDistribution(action_dim)
            self.action_net = self.action_dist.proba_distribution_net(last_layer_dim)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        elif self.use_beta_dist:
            action_dim = get_action_dim(self.action_space)
            alpha_beta = self.action_net(latent_pi)
            alpha = alpha_beta[:, 0:action_dim]
            beta = alpha_beta[:, action_dim:]
            self.alpha_logging.append(alpha.detach().mean().item())
            self.beta_logging.append(beta.detach().mean().item())
            return alpha, beta, {}

        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std_logging.append(th.exp(log_std.detach()).mean().item())
        self.mean_logging.append(mean_actions.detach().mean().item())
        return mean_actions, log_std, {}


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param use_beta_dist: Whether to use Beta distribution instead of squashed Gaussian
    :@param use_bayes: Whether to use a MC Dropout approach for the collision critic
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            use_beta_dist: bool = False,
            use_bayes: bool = False,
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs['use_beta_dist'] = use_beta_dist
        self.use_beta_dist = use_beta_dist

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        # Safe Q-learning: Define col networks
        self.col_net_kwargs = self.net_args.copy()
        self.col_net_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "softmax": True,
                "use_bayes": use_bayes,
            }
        )
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.col_net, self.col_net_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.col_net = self.make_col_net(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
            col_net_parameters = [param for name, param in self.col_net.named_parameters() if
                                  "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            self.col_net = self.make_col_net(features_extractor=None)
            critic_parameters = self.critic.parameters()
            col_net_parameters = self.col_net.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.col_net_target = self.make_col_net(features_extractor=None)
        self.col_net_target.load_state_dict(self.col_net.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.col_net.optimizer = self.optimizer_class(col_net_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)
        self.col_net_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def make_col_net(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        col_net_kwargs = self._update_features_extractor(self.col_net_kwargs, features_extractor)
        return ContinuousCritic(**col_net_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.col_net.set_training_mode(mode)
        self.training = mode

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        if self.use_beta_dist:  # action in [low, high]
            return (action - low) / (high - low)

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        if self.use_beta_dist:  # action in [0,1]
            return low + (high - low) * scaled_action

        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def unscale_env_action_to_model_action(self, action: np.ndarray) -> np.ndarray:
        """
        Same as scale_action but more understandable
        Rescale the action from env: [low, high] to [0, 1] of beta or [-1, 1] of Gauss
        @param action: action from env in range [low, high]
        @return: action in range [0, 1] for Beta or [-1,1] for Gauss
        """
        low, high = self.action_space.low, self.action_space.high
        if self.use_beta_dist:  # action in [low, high]
            return (action - low) / (high - low)

        return 2.0 * ((action - low) / (high - low)) - 1.0

    def scale_action_to_env(self, action: np.ndarray) -> np.ndarray:
        """
        Same as unscale_action but more understandable
        Rescale the action from [-inf, inf] or [0, 1] to [low, high]
        :param action: Action for Gauss [-inf, inf] and for Beta in [0,1]
        :return: action in range [low, high]
        """
        low, high = self.action_space.low, self.action_space.high
        if self.use_beta_dist:  # action in [0,1]
            return low + (high - low) * action
        # For Gaussian Policy
        return low + (0.5 * (action + 1.0) * (high - low))


MlpPolicy = SACPolicy


class CnnPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            use_beta_dist: bool = False,
            use_bayes: bool = False,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_beta_dist,
            use_bayes
        )


class MultiInputPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            use_beta_dist: bool = False,
            use_bayes: bool = False,
    ):
        super(MultiInputPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            use_beta_dist,
            use_bayes,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
