from typing import Any, Dict, List, Optional, Tuple, Union
from gym import spaces

from models.common.vec_env.stacked_observations import CustomStackedDictObservations, CustomStackedObservations
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class CustomVecFrameStack(VecFrameStack):
    """
    For full documentation see parent class. This is reimplemented since class bugs if only 1 environment
    with channel first.
    Bug is in common/vec_frame_stack line 126 or for dict in 243
    """

    def __init__(self, venv: VecEnv, n_stack: int, channels_order: Optional[Union[str, Dict[str, str]]] = None):
        super(CustomVecFrameStack, self).__init__(venv, n_stack, channels_order)
        wrapped_obs_space = venv.observation_space

        if isinstance(venv.observation_space, spaces.Box):
            self.stackedobs = CustomStackedObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order)
        elif isinstance(wrapped_obs_space, spaces.Dict):
            self.stackedobs = CustomStackedDictObservations(venv.num_envs, n_stack, wrapped_obs_space, channels_order)
