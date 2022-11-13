from os.path import join
from omegaconf import OmegaConf
# from hydra.core.hydra_config import HydraConfig


def get_original_cfg(cfg):
    if cfg.restore is None:
        raise ValueError("Cannot restore the original config file, no restore path is specified")
    original_cfg = OmegaConf.load(join(cfg.restore, '../.hydra/config.yaml'))
    return original_cfg


# def get_cfg_overrides_only():
#     overrides = HydraConfig.get().overrides
#     overrides.pop("hydra")
#     return overrides