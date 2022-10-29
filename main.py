from omegaconf import DictConfig, OmegaConf
import hydra
import os


@hydra.main(version_base=None, config_path="hydra_config", config_name="default")
def my_app(cfg: DictConfig):
    # print(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(OmegaConf.create(cfg))


if __name__ == "__main__":
    print(os.getcwd())
    my_app()
