from omegaconf import DictConfig, OmegaConf
from builders import build_env, build_callback
import hydra
import os


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_env, self.eval_env, self.video_env = None, None, None
        self.log_dir = None

        self._setup()

    def _setup(self):
        self._setup_log_dir()
        self.train_env, self.eval_env, self.video_env = build_env(self.cfg.env_name, self.log_dir, self.cfg.seed)
        self.cb = build_callback(self.cfg.env_name, self.log_dir, self.eval_env, self.cfg.eval, self.video_env)

    def _setup_log_dir(self):
        if self.cfg.safe_mult:
            self.log_dir = f"logs/safe_rl/{self.cfg.env_name}/"
        else:
            self.log_dir = f"logs/baseline/{self.cfg.env_name}/"

    def _setup_callbacks(self):
        pass

    def _setup_model(self):
        pass

    def train(self):
        pass


@hydra.main(version_base=None, config_path="hydra_config", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    print(os.getcwd())
    main()
