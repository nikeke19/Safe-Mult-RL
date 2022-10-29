from torch.distributions.beta import Beta
from torch import nn
import torch as th
from stable_baselines3.common.preprocessing import get_action_dim
from typing import Any, Dict, List, Optional, Tuple, Union
from gym import spaces
from stable_baselines3.common.distributions import (
    Distribution,
    StateDependentNoiseDistribution,
    DiagGaussianDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    BernoulliDistribution,
    sum_independent_dims)


class BetaDistribution(Distribution):
    '''
    Beta distribution for continuous actions

    :param action_dim: Dimension of the action space
    '''

    def __init__(self, action_dim: int):
        super(BetaDistribution, self).__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """
        Create the layers and parameter that represent the distribution:
        alpha and beta

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :return:
        """
        alpha_beta = nn.Sequential(nn.Linear(latent_dim, 2 * self.action_dim), nn.Softplus())
        return alpha_beta

    def proba_distribution(self, alphas: th.Tensor, betas: th.Tensor) -> "Distribution":
        """
        Create the distribution given its parameters (alpha, beta)

        :param alphas:
        :param betas:
        :return:
        """
        self.distribution = Beta(alphas, betas)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """
        Mode is only defined for alpha > 1 and beta > 1. If not defined, sample instead
        @return:
        """
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0

        # Finding mode and sample
        with th.no_grad():
            sample = self.distribution.sample()
        mode = ((alpha - 1) / (alpha + beta - 2))
        mode = th.nan_to_num(mode, nan=0.0)  # If mode not defined, when a=b=1, then nan can occur

        # Determine where mode is defined
        deterministic = th.zeros_like(alpha, dtype=th.int)
        deterministic[th.logical_and(alpha > 1, beta > 1)] = 1

        action = deterministic * mode + (1 - deterministic) * sample

        return action
        # return self.distribution.mean

    def actions_from_params(self, alphas: th.Tensor, betas: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the probability distribution
        self.proba_distribution(alphas, betas)
        # Either sample or return mode in case of deterministic=True
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, alphas: th.Tensor, betas: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param alphas:
        :param betas:
        :return:
        """
        actions = self.actions_from_params(alphas, betas)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class StateDependentStdDiagGaussDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix and state dependent std
    @param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(StateDependentStdDiagGaussDistribution, self).__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Module]:
        mean_net = nn.Linear(latent_dim, self.action_dim)
        log_std_net = nn.Linear(latent_dim, self.action_dim)
        return mean_net, log_std_net


def make_proba_distribution(
        action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None,
        use_beta_dist: bool = False, state_dependent_std: bool = False) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    @param action_space: the input action space
    @param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    @param dist_kwargs: Keyword arguments to pass to the probability distribution
    @param use_beta_dist: Whether to use beta distribution
    @param state_dependent_std: Whether to use state dependent std
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, "Error: the action space must be a vector"
        if use_beta_dist:
            cls = BetaDistribution
        elif use_sde:
            cls = StateDependentNoiseDistribution
        elif state_dependent_std:
            cls = StateDependentStdDiagGaussDistribution
        else:
            cls = DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )
