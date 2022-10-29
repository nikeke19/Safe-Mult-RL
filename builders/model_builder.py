from omegaconf import DictConfig
from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from models import PPO, SAC
import os
from models.common.util import get_linear_fn


def _load_model(path):
    return PPO


def build_model(env_name: str, train_env: Env, cfg: DictConfig, log_dir: str, seed: int = 0) -> BaseAlgorithm:
    """
    Building either a SAC or PPO model
    :param env_name: Name of the environment to build model of
    :param train_env: Training environment
    :param cfg: Model config
    :param log_dir: Dir to store the tensorboard logs
    :param seed: Seed to seed the model
    :return: Either SAC or PPO algorithm
    """
    # Loading the model if it exists
    if cfg.load_model_dir is not None:
        return _load_model(cfg.load_model_dir)

    # Setting up base kwargs
    cfg.base.tensorboard_log = os.path.join(log_dir, "tensorboard")
    if not os.path.isdir(cfg.base.tensorboard_log):
        os.mkdir(cfg.base.tensorboard_log)

    # Setting up env specific kwargs
    if env_name == "lunar_lander":
        policy = "MlpPolicy"
        if cfg.model.name == "PPO":
            cfg.base.ent_coef = get_linear_fn(0.02, 0, start_fraction=0.5, end_fraction=1)
        elif cfg.model.name == "SAC":
            cfg.base.learning_rate = get_linear_fn(3e-4, 1e-8)

    else:
        raise NotImplemented(f"Env {env_name} is not implemented. Choose [lunar_lander, car_racing, point_navigation]")

    # Choosing correct model
    if cfg.model.name == "PPO":
        model_func = PPO
        policy = "Custom" + policy
    elif cfg.model.name == "SAC":
        model_func = SAC
    else:
        raise NotImplemented(f"Model {cfg.model.name} is not implemented. Choose from [SAC, PPO]")

    return model_func(policy, train_env, seed=seed, **cfg.base, **cfg.safe_rl)
