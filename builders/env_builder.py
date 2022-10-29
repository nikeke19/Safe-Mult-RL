from gym import Env
from typing import Union, List, Tuple
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecFrameStack
from util.monitors.lunar_lander_monitor import LunarLanderMonitor
from envs import LunarLander
import os


def _build_one_env(env_func: Env, env_kwargs: dict, wrappers: List[dict], vec_wrappers: List[dict], seed: int) -> Env:
    """
    Building the environment according to the cfg.
    :param env_func: Function of the base environment
    :param env_kwargs: Base environment kwargs
    :param wrappers: List[dict{"wrapper", "kwargs"}] of non-vectorized wrappers with kwargs
    :param vec_wrappers: List[dict{"wrapper", "wrapper_kwargs"}] of vectorized wrappers with kwargs
    :param seed: Seed to seed the env with
    :return: A Gym environment
    """

    def environment():
        """Function to create a callable environment"""
        env = env_func(**env_kwargs)
        env.seed(seed)
        for wrapper in wrappers:
            env = wrapper["wrapper"](env, **wrapper["kwargs"])
        return env

    env = DummyVecEnv([environment])
    for vec_wrapper in vec_wrappers:
        env = vec_wrapper["wrapper"](env, **vec_wrapper["kwargs"])

    return env


def build_env(env_name: str, log_dir: str, seed: int = 0) -> Tuple[Env, Env, Union[Env, None]]:
    """
    Building the environment according to the cfg.
    :param env_name: Name of the environment to build
    :param log_dir: Directory where Monitor output is saved
    :param seed: Seed to seed the env with
    :return: Train, Eval and Video Gym environments
    """
    # Common training and eval env args
    monitor_log_dir = os.path.join(log_dir, "monitor.csv")
    env_kwargs = {}

    # Specific training and eval env args
    if env_name == "lunar_lander":
        env_func = LunarLander
        wrapper = [{"wrapper": LunarLanderMonitor, "kwargs": dict(filename=monitor_log_dir, col_reward=-100)}]
        vec_wrapper = [{"wrapper": VecFrameStack, "kwargs": dict(n_stack=4)}]
    else:
        raise NotImplemented(f"Env {env_name} is not implemented. Choose [lunar_lander, car_racing, point_navigation]")

    # Build Environments
    train_env = _build_one_env(env_func, env_kwargs, wrapper, vec_wrapper, seed)
    eval_env = _build_one_env(env_func, env_kwargs, wrapper, vec_wrapper, seed)

    # Building Video Env
    video_env = None
    if env_name == "lunar_lander":
        video_folder = os.path.join(log_dir, "video")
        if not os.path.isdir(video_folder):
            os.mkdir(video_folder)
        video_kwargs = dict(video_folder=video_folder, record_video_trigger=lambda x: x % 1000 == 0, video_length=1000)
        vec_wrapper += [{"wrapper": VecVideoRecorder, "kwargs": video_kwargs}]
        video_env = _build_one_env(env_func, env_kwargs, wrapper, vec_wrapper, seed)

    return train_env, eval_env, video_env


if __name__ == '__main__':
    envs = build_env("lunar_lander", log_dir="logs")
    print("finished")
