import os
from omegaconf import DictConfig
from pathlib import Path


def _create_dir(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    return directory


def _resolve_model_name(cfg: DictConfig):
    if cfg.model.name == "PPO":
        if not cfg.model.safe_rl.safe_mult and not cfg.model.safe_rl.safe_lagrange:
            return "baseline"
        if not cfg.model.safe_rl.safe_mult and cfg.model.safe_rl.safe_lagrange:
            return "lagrange"
        else:
            model_conversion = {"V1a": "V1", "V2b": "V2", "V2a": "V3"}
            return model_conversion[cfg.model.safe_rl.advantage_mode]
    if cfg.model.name == "SAC":
        if not cfg.model.safe_rl.safe_mult and not cfg.model.safe_rl.safe_lagrange_baseline:
            return "baseline"
        if not cfg.model.safe_rl.safe_mult and cfg.model.safe_rl.safe_lagrange_baseline:
            return "lagrange"
        else:
            return f"mult_{cfg.model.safe_rl.safe_version.lower()}"


def build_logdir(cfg: DictConfig):
    """
    Building a new logdir for each experiment
    :param cfg: Config describing the experiment
    :return: logdir of form logs/environment/algorithm/safe_rl_type/idx, e.g. logs/lunar_lander/ppo/baseline/0
    """
    # Target logs/lunar_lander/ppo/baseline/0/logs
    base_dir = Path(f"logs/{cfg.env_name}/{cfg.model.name.lower()}/{_resolve_model_name(cfg)}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Creating an idx subfolder and write to the highest index
    idx = -1
    for file in os.listdir(base_dir):
        idx = max(idx, int(file))

    log_dir = os.path.join(base_dir, str(idx + 1))
    assert not os.path.isdir(log_dir)
    os.mkdir(log_dir)

    return log_dir

