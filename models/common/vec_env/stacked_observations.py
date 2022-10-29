import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces

from stable_baselines3.common.vec_env.stacked_observations import StackedObservations, StackedDictObservations


class CustomStackedObservations(StackedObservations):
    """
    For full documentation see parent class. This is reimplemented since class bugs if only 1 environment
    with channel first.
    Bug is in common/vec_frame_stack line 126
    """

    def __init__(
            self,
            num_envs: int,
            n_stack: int,
            observation_space: spaces.Space,
            channels_order: Optional[str] = None,
    ):

        super(CustomStackedObservations, self).__init__(num_envs, n_stack, observation_space, channels_order)

    def update(
            self,
            observations: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Adds the observations to the stack and uses the dones to update the infos.

        :param observations: numpy array of observations
        :param dones: numpy array of done info
        :param infos: numpy array of info dicts
        :return: tuple of the stacked observations and the updated infos
        """
        stack_ax_size = observations.shape[self.stack_dimension]
        self.stackedobs = np.roll(self.stackedobs, shift=-stack_ax_size, axis=self.stack_dimension)
        for i, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[i]:
                    old_terminal = infos[i]["terminal_observation"]
                    if self.channels_first:
                        new_terminal = np.concatenate((self.stackedobs[[i], :-stack_ax_size, ...], old_terminal),
                                                      axis=self.stack_dimension, )
                    else:
                        new_terminal = np.concatenate((self.stackedobs[[i], ..., :-stack_ax_size], old_terminal),
                                                      axis=self.stack_dimension, )
                    infos[i]["terminal_observation"] = new_terminal
                else:
                    warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        if self.channels_first:
            self.stackedobs[:, -observations.shape[self.stack_dimension]:, ...] = observations
        else:
            self.stackedobs[..., -observations.shape[self.stack_dimension]:] = observations
        return self.stackedobs, infos


class CustomStackedDictObservations(StackedDictObservations):
    """
    For full documentation see parent class. This is reimplemented since class bugs if only 1 environment
    with channel first.
    Bug is in common/vec_frame_stack line 243
    """

    def __init__(
            self,
            num_envs: int,
            n_stack: int,
            observation_space: spaces.Dict,
            channels_order: Optional[Union[str, Dict[str, str]]] = None,
    ):
        super(CustomStackedDictObservations, self).__init__(num_envs, n_stack, observation_space, channels_order)

    def update(
            self,
            observations: Dict[str, np.ndarray],
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """
        Adds the observations to the stack and uses the dones to update the infos.

        :param observations: Dict of numpy arrays of observations
        :param dones: numpy array of dones
        :param infos: dict of infos
        :return: tuple of the stacked observations and the updated infos
        """
        for key in self.stackedobs.keys():
            stack_ax_size = observations[key].shape[self.stack_dimension[key]]
            self.stackedobs[key] = np.roll(
                self.stackedobs[key],
                shift=-stack_ax_size,
                axis=self.stack_dimension[key],
            )

            for i, done in enumerate(dones):
                if done:
                    if "terminal_observation" in infos[i]:
                        old_terminal = infos[i]["terminal_observation"][key]
                        if self.channels_first[key]:
                            new_terminal = np.concatenate(
                                (self.stackedobs[key][[i], :-stack_ax_size, ...], old_terminal),
                                axis=self.stack_dimension[key])
                        else:
                            # todo assert, that stackedobs and old_terminal have same shape
                            new_terminal = np.concatenate(
                                (self.stackedobs[key][i, ..., :-stack_ax_size], old_terminal,),
                                axis=self.stack_dimension[key], )

                        infos[i]["terminal_observation"][key] = new_terminal
                    else:
                        warnings.warn("VecFrameStack wrapping a VecEnv without terminal_observation info")
                    self.stackedobs[key][i] = 0
            if self.channels_first[key]:
                self.stackedobs[key][:, -stack_ax_size:, ...] = observations[key]
            else:
                self.stackedobs[key][..., -stack_ax_size:] = observations[key]
        return self.stackedobs, infos
