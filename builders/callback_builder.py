from gym import Env
from stable_baselines3.common.callbacks import BaseCallback
from util.callbacks import LunarLanderEvalCallback
from typing import List
import os
from omegaconf import DictConfig


def build_callback(
        env_name: str,
        log_dir: str,
        eval_env: Env,
        cfg: DictConfig,
        video_env: Env = None,
) -> List[BaseCallback]:
    """
    Building the eval callback and optionally the video eval
    :param env_name: Name of the environment to build
    :param log_dir: Where to log evaluation results and optionally the model
    :param eval_env: Evaluation environment
    :param cfg: Kwargs for the evaluation environment
    :param video_env: Video environment
    :return: List of callbacks
    """
    eval_dir = os.path.join(log_dir, "eval")
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    # Common kwargs
    model_path = None
    if cfg.evaluation_cb.save_model:
        model_path = eval_dir

    # Env specific kwargs
    if env_name == "lunar_lander":
        cb_func = LunarLanderEvalCallback
    else:
        raise NotImplemented(f"Env {env_name} is not implemented. Choose [lunar_lander, car_racing, point_navigation]")

    # Building Eval Callback
    cb = [cb_func(eval_env, log_path=eval_dir, best_model_save_path=model_path, **cfg.evaluation_cb.kwargs)]

    # Building video callback
    if cfg.video_cb.video_freq is not None:
        assert video_env is not None, "video frequency was specified but eval env is None"
        cb += [cb_func(video_env, **cfg.video_cb.kwargs)]

    return cb
