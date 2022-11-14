from os.path import join
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path


def get_original(restore, dot_path):
    if restore is None:
        return None
    original_cfg = OmegaConf.load(join(to_absolute_path(restore), '../.hydra/config.yaml'))
    return OmegaConf.select(original_cfg, dot_path)

OmegaConf.register_new_resolver("get_original", get_original)