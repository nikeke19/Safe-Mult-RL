import torch as th
import numpy as np
from torch.nn.functional import binary_cross_entropy
from typing import Any, Dict, Optional, Type, Union, Tuple, List
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit
import warnings


def transpose_image(image: np.ndarray) -> np.ndarray:
    # Convert state from [96, 96, 3] to [1, 3, 96, 96]
    image = np.transpose(image.squeeze(0), (2, 0, 1))
    if len(image.shape) != 4:
        image = np.expand_dims(image, 0)
    return image.copy()


def compute_critic_values(model, observations, action, minimum=True, use_bayes=False) \
        -> Tuple[float, Optional[float]]:
    """
    Computes the output of the critic with minimum or maximum to avoid bias
    @param model: critic model
    @param observations: observation to evaluate
    @param action: action to evaluate
    @param minimum: Whether to take the min or max of the n critics
    @param use_bayes: If True, Use MC Dropout Bayesian approach
    @return: returns the critic prediction
    """
    # Sometimes second element is not a tensor, f.e. for SAC MlP. Kick this out
    if type(observations) is tuple:
        observations = observations[0]

    # Observation dict is already tensor
    if type(observations) is not dict:
        if not isinstance(observations, th.Tensor):
            observations = th.from_numpy(observations).to("cuda")

    # Action is sometimes 1d array, but 2d needed
    if len(action.shape) == 1:
        action = action.reshape(1, -1)
    if not isinstance(action, th.Tensor):
        action = th.from_numpy(action).to("cuda")

    if use_bayes:
        out = model.forward_bayes(observations, action)
        return out.mean(), out.std()

    with th.no_grad():
        out = model(observations, action)
    out = th.cat(out, dim=1)
    out, _ = th.min(out, dim=1, keepdim=True) if minimum else th.max(out, dim=1, keepdim=True)
    # if len(out.shape) > 1 and out.shape[0] > 1:
    #     return out.cpu().numpy()
    # return out.item()
    return out.cpu().numpy()


def weighted_bce(pred, target, p_threshold=0.5):
    bce = binary_cross_entropy(pred, target, reduction='none')
    n = 1
    for n_elements in target.shape:
        n *= n_elements
    # n = target.shape[0]
    n_positives = target[target >= p_threshold].shape[0]
    n_negatives = n - n_positives

    if n_positives == 0:  # No collision in targets
        positive_weight = 1
        negative_weight = 0.1
    elif n_negatives == 0:
        positive_weight = 0.1
        negative_weight = 1
    else:  # taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        positive_weight = 1 / n_positives * n / 2
        negative_weight = 1 / n_negatives * n / 2

    weight = (target >= p_threshold).type(th.int32) * positive_weight \
             + (target < p_threshold).type(th.int32) * negative_weight

    return th.mean(bce * weight)


def update_value_functions_in_env(env, q_action, c_action, q, c):
    # Update log
    env_log = env  # Need to call st like: env.venv.venv.envs[0].env.env.update_critic_values
    # Go through all vectorized environments "venv"
    for i in range(20):
        if hasattr(env_log, 'venv'):
            env_log = env_log.venv
        else:
            break
    # Now go through all sub environments. Only 1 env if no multi environment setup
    for idx, environment in enumerate(env_log.envs):
        for i in range(20):
            if hasattr(environment.__class__, 'update_critic_values'):
                environment.update_critic_values(q_action[idx].item(), c_action[idx].item(), q, c)
                break
            else:
                environment = environment.env


def find_variable_in_env(env, variable_name, env_number=0):
    env_find = env  # Need to call st like: env.venv.venv.envs[0].env.env.variable_name
    # Go through all vectorized environments "venv"
    for i in range(20):
        if hasattr(env_find, 'venv'):
            env_find = env_find.venv
        else:
            break

    # No go through the environment of interest
    environment = env_find.envs[env_number]
    for i in range(20):
        if hasattr(environment, variable_name):
            return getattr(environment, variable_name)
        elif hasattr(environment, 'env'):
            environment = environment.env
        else:
            warnings.warn("Environment does not have attribute: " + variable_name)
            return None


def get_linear_fn(start: float, end: float, start_fraction: float = 0, end_fraction: float = 1) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :@param start_fraction: Value at which to start to decrease
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress > end_fraction:
            return end
        elif progress < start_fraction:
            return start
        else:
            return start + (progress - start_fraction) / (end_fraction - start_fraction) * (end - start)

    return func


def get_piecewise_fn(progress_steps: List, magnitudes: List):
    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress == 0:
            return magnitudes[0]
        elif progress >= 1:
            return magnitudes[-1]
        # Iterate through progress list and find right step
        for i in range(1, len(progress_steps)):
            if progress < progress_steps[i]:
                return magnitudes[i - 1]
        # Progress can be greater 1 at the end:
        return magnitudes[-1]

    return func


def weighted_mse_logits(input: th.Tensor,
                        target: th.Tensor,
                        rewards: th.Tensor,
                        col_reward: float,
                        weight: float = 5) -> th.Tensor:
    """
    Collisions are under represented class: weight them more
    loss = mean( (y - y_true)² * scaling), * = elementwise product
    @param weight: Weight with which collisions pred shall be scaled
    @param col_reward: Reward which is associated with a collision
    @param input: input data tensor of arbitrary shape without sigmoid
    @param target: the target tensor with shape matching input and range [0,1]
    @param rewards:
    """
    input = th.sigmoid(input)
    squared_difference = th.nn.functional.mse_loss(input, target, reduction='none')
    scaling = th.ones_like(input, device='cuda') + (rewards < col_reward + 0.01).type(th.int32) * weight
    return th.mean(squared_difference * scaling) * 10


# Taken from: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
def binary_focal_loss(
        input: th.Tensor,
        target: th.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        eps: Optional[float] = None,
) -> th.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
    Returns:
        the computed loss.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = th.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = th.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """

    if eps is not None and not th.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, th.Tensor):
        raise TypeError(f"Input type is not a th.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    probs_pos = th.sigmoid(input)
    probs_neg = th.sigmoid(-input)
    loss_tmp = -alpha * th.pow(probs_neg, gamma) * target * th.nn.functional.logsigmoid(input) - (
            1 - alpha
    ) * th.pow(probs_pos, gamma) * (1.0 - target) * th.nn.functional.logsigmoid(-input)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = th.mean(loss_tmp)
    elif reduction == 'sum':
        loss = th.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


def none_or_value(value):
    if value == 'None':
        return None
    return float(value)


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


def none_or_str(value):
    if value in ['None', 'none']:
        return None
    return str(value)
