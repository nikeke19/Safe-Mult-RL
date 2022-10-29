import numpy as np
from gym.spaces import Box, Dict
from gym import ObservationWrapper


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale."""

    def __init__(self, env, keep_dim=False, binary_threshold=None):
        super(GrayScaleObservation, self).__init__(env)
        self.keep_dim = keep_dim
        self.env_is_dict = False
        self.threshold = None if binary_threshold is None else binary_threshold * 255

        if isinstance(self.observation_space, Dict):
            self.env_is_dict = True
            assert len(env.observation_space['img'].shape) == 3 and env.observation_space['img'].shape[-1] == 3
            obs_shape = self.observation_space['img'].shape[:2]
            if self.keep_dim:
                self.observation_space['img'] = Box(0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
            else:
                self.observation_space['img'] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        else:
            assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3
            obs_shape = self.observation_space.shape[:2]
            if self.keep_dim:
                self.observation_space = Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
            else:
                self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2

        if self.env_is_dict:
            vec = observation['vec']
            observation = observation['img']

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.threshold is not None:
            observation = self.make_image_binary(observation)

        if self.keep_dim:
            observation = np.expand_dims(observation, -1)

        if self.env_is_dict:
            return {'img': observation, 'vec': vec}

        return observation

    def make_image_binary(self, x):
        x[x > self.threshold] = 255
        x[x <= self.threshold] = 0
        return x


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)
        self.env_is_dict = False

        if isinstance(self.observation_space, Dict):
            obs_shape = self.shape + self.observation_space["img"].shape[2:]
            self.observation_space["img"] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
            self.env_is_dict = True

        else:
            obs_shape = self.shape + self.observation_space.shape[2:]
            self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2

        if self.env_is_dict:
            vec = observation["vec"]
            observation = observation["img"]

        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)

        if self.env_is_dict:
            return {"img": observation, "vec": vec}

        return observation
