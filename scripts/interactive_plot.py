import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import hydra
from src.trainer import Trainer
import logging
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import src.utils
from src.utils import get_original_cfg
from hydra.utils import to_absolute_path


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../conf/", config_name="interactive_plot")
def main(cfg):
    plt.rcParams['keymap.back'].remove('left')
    log.info(f'Current directory: {os.getcwd()}')
    trainer = Trainer(cfg)
    trainer.restore(to_absolute_path(cfg.restore))
    fig = plt.figure()
    trainer.plot_interactive(fig, cfg.n_nearest)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()