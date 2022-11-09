from omegaconf import DictConfig, OmegaConf
from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from models import PPO, SAC
import os
from models.common.util import get_linear_fn
import warnings


def _load_model(path: str, model_type: str, training_env: Env) -> BaseAlgorithm:
    """Loading the model"""
    if model_type == "SAC":
        return SAC.load(path, env=training_env)
    else:
        return PPO.load(path, env=training_env)


def _omegaconf_to_dict(conf: DictConfig) -> dict:
    """Cast the conf to dict and evaluate all expressions"""
    target_dict = OmegaConf.to_container(conf)
    for key, value in target_dict.items():
        if isinstance(value, str) and "$" in value:
            target_dict[key] = getattr(conf, key)
    return target_dict


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
        return _load_model(cfg.load_model_dir, cfg.name, train_env)

    # Setting up base kwargs
    cfg.base.tensorboard_log = os.path.join(log_dir, "tensorboard")
    if not os.path.isdir(cfg.base.tensorboard_log):
        os.mkdir(cfg.base.tensorboard_log)

    # Casting to dict container
    base_cfg = _omegaconf_to_dict(cfg.base)
    safe_rl_cfg = _omegaconf_to_dict(cfg.safe_rl)

    # Setting up env specific kwargs
    if env_name == "lunar_lander":
        policy = "MlpPolicy"
        if cfg.name == "PPO":
            base_cfg["ent_coef"] = get_linear_fn(0.02, 0, start_fraction=0.5, end_fraction=1)
        elif cfg.name == "SAC":
            base_cfg["learning_rate"] = get_linear_fn(3e-4, 1e-8)
    elif env_name == "car_racing":
        if cfg.name == "PPO":
            warnings.warn("For Car Racing only PPO runs succesfully")
        policy = "MultiInputPolicy"
    else:
        raise NotImplemented(f"Env {env_name} is not implemented. Choose [lunar_lander, car_racing, point_navigation]")

    # Choosing correct model
    if cfg.name == "PPO":
        model_func = PPO
        policy = "Custom" + policy
    elif cfg.name == "SAC":
        model_func = SAC
    else:
        raise NotImplemented(f"Model {cfg.name} is not implemented. Choose from [SAC, PPO]")

    return model_func(policy, train_env, **base_cfg, **safe_rl_cfg)
