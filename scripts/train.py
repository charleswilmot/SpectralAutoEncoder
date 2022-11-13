import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import hydra
from src.trainer import Trainer
from jax import numpy as jnp
import logging
import jax
from omegaconf import OmegaConf
from src.utils import get_original_cfg
from hydra.utils import to_absolute_path


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../conf/", config_name="train")
def main(cfg):
    if cfg.restore and cfg.restore_cfg:
        original_cfg = get_original_cfg(cfg)
        cfg = OmegaConf.merge(original_cfg, cfg)
    log.info(f'Current directory: {os.getcwd()}')

    key = jax.random.PRNGKey(cfg.seed)
    trainer = Trainer(cfg)
    if cfg.restore:
        trainer.restore(to_absolute_path(cfg.restore))
    else:
        trainer.init(key, jnp.zeros(shape=(1,) + tuple(cfg.image_shape)))

    for iteration in range(cfg.n_training):
        trainer.plot(iteration)
        trainer.log(iteration, n_nearest=cfg.n_nearest)
        trainer.checkpoint(f"{iteration:03d}")
        trainer.train(cfg.n_train_batches, regularizer_coef=cfg.regularizer_coef, desired_dim=cfg.desired_dim, n_nearest=cfg.n_nearest)

    trainer.plot(iteration + 1)
    trainer.log(iteration + 1, n_nearest=cfg.n_nearest)
    trainer.checkpoint(f"{iteration + 1:03d}")

if __name__ == '__main__':
    main()