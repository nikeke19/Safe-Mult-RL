from omegaconf import DictConfig
from builders import build_logdir, build_env, build_callback, build_model
import hydra
import os


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        self.log_dir = build_logdir(self.cfg)
        self.train_env, self.eval_env, self.video_env = build_env(self.cfg.env_name, self.log_dir, self.cfg.seed)
        self.cb = build_callback(self.cfg.env_name, self.log_dir, self.eval_env, self.cfg.eval, self.video_env)
        self.model = build_model(self.cfg.env_name, self.train_env, self.cfg.model, self.log_dir, self.cfg.seed)

    def train(self):
        self.model.learn(total_timesteps=self.cfg.train.time_steps, log_interval=5, callback=self.cb)
        self.train_env.close()
        self.eval_env.close()
        self.video_env.close()


@hydra.main(version_base=None, config_path="hydra_config", config_name="default")
def main(cfg: DictConfig):
    print(f"Training model {cfg.model.name} in {cfg.env_name} env")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
