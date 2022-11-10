from tensorboardX import SummaryWriter
from .autoencoder import AutoEncoder
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from functools import partial
import tensorflow as tf
import jax.numpy as jnp
import haiku as hk
import logging
import pickle
import optax
import jax
import os


log = logging.getLogger(__name__)


def normalize(images, labels):
    images = tf.cast(images, tf.float32) / (255. / 2) - 1.
    return tf.reshape(images, (28, 28, 1))


class Trainer:
    def __init__(self, cfg):
        self.path = os.getcwd()
        self.tensorboard = SummaryWriter(logdir=f'{self.path}/tensorboard/')
        self.autoencoder = hk.without_apply_rng(hk.transform(lambda x: AutoEncoder(cfg.autoencoder)(x)))
        self.encoder = hk.without_apply_rng(hk.transform(lambda x: AutoEncoder(cfg.autoencoder).encoder(x)))
        self.decoder = hk.without_apply_rng(hk.transform(lambda x: AutoEncoder(cfg.autoencoder).decoder(x)))
        self.optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.exponential_decay(
                init_value=cfg.optimizer.lr.init_value,
                transition_steps=cfg.optimizer.lr.transition_steps,
                decay_rate=cfg.optimizer.lr.decay_rate,
                transition_begin=cfg.optimizer.lr.transition_begin,
                staircase=cfg.optimizer.lr.staircase,
                end_value=cfg.optimizer.lr.end_value,
            )
        )
        dataset, = tfds.load(
            'mnist',
            split=['train'],
            as_supervised=True,
            with_info=False,
        )
        dataset = dataset.map(normalize)
        self.log_data = next(dataset.take(4096).batch(4096).as_numpy_iterator())
        self.plot_data = next(dataset.take(16).batch(16).as_numpy_iterator())
        dataset = dataset.shuffle(10000, seed=cfg.seed)
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(cfg.trainer.batch_size)
        self.dataset = dataset.as_numpy_iterator()
        ### define all differentiable functions

        @jax.jit
        def reconstruction_loss(autoencoder_params, x):
            reconstructions = self.autoencoder.apply(autoencoder_params, x)
            loss = jnp.sum(jnp.mean((x - reconstructions) ** 2, axis=(1,2,3)), axis=0)
            return loss

        @partial(jax.jit, static_argnames=('n_nearest'))
        def get_local_singular_values(n_nearest, latent):
            batch_dim = latent.shape[0]
            squared_distance_matrix = jnp.sum((latent[:, None] - latent[None, :]) ** 2, axis=-1)   # shape [N, N]
            neighbors_indices = jnp.argsort(squared_distance_matrix, axis=-1)                      # shape [N, N]
            neighbors = jnp.take(latent, neighbors_indices, axis=0)                                # shape [N, N, S]
            close_neighbors = neighbors[:batch_dim // n_nearest, :n_nearest + 1]                   # shape [M, S + 1, S]
            close_neighbors_relative = close_neighbors[:, 1:] - close_neighbors[:, :1]             # shape [M, S, S]
            return jnp.linalg.svd(close_neighbors_relative, compute_uv=False)                      # shape [M, S]

        @partial(jax.jit, static_argnames=('desired_dim', 'n_nearest'))
        def skeletonize_loss(autoencoder_params, desired_dim, n_nearest, x):
            latent = self.encoder.apply(autoencoder_params, x)                                     # shape [N, S]
            batch_dim = latent.shape[0]
            singular_values = get_local_singular_values(n_nearest, latent)                         # shape [M, S]
            return jnp.sum(singular_values[:, desired_dim:]) / batch_dim                           # shape []

        def total_loss(autoencoder_params, regularizer_coef, desired_dim, n_nearest, x):
            reconstruction_term = reconstruction_loss(autoencoder_params, x)
            skeletonize_term = skeletonize_loss(autoencoder_params, desired_dim, n_nearest, x)
            return reconstruction_term + regularizer_coef * skeletonize_term

        self.get_local_singular_values = get_local_singular_values
        self.reconstruction_grad = jax.grad(reconstruction_loss)
        self.skeletonize_grad = jax.grad(skeletonize_loss)
        self.total_grad = jax.grad(total_loss)

    def init(self, key, dummy):
        self.autoencoder_params = self.autoencoder.init(key, dummy)
        self.learner_state = self.optimizer.init(self.autoencoder_params)

    def train(self, n_batches, regularizer_coef=0.0, desired_dim=2, n_nearest=128):
        for _, x in zip(range(n_batches), self.dataset):
            if regularizer_coef == 0.0:
                dloss_dtheta = self.reconstruction_grad(self.autoencoder_params, x)
            else:
                dloss_dtheta = self.total_grad(self.autoencoder_params, regularizer_coef, desired_dim, n_nearest, x)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.autoencoder_params = optax.apply_updates(self.autoencoder_params, updates)

    def skeletonize(self, n_batches, desired_dim=2, n_nearest=128):
        for _, x in zip(range(n_batches), self.dataset):
            dloss_dtheta = self.skeletonize_grad(self.autoencoder_params, desired_dim, n_nearest, x)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.autoencoder_params = optax.apply_updates(self.autoencoder_params, updates)

    def log(self, iteration, n_nearest=128):
        latent = self.encoder.apply(self.autoencoder_params, self.log_data)
        reconstructions = self.decoder.apply(self.autoencoder_params, latent)
        rmse = jnp.mean(jnp.abs(reconstructions - self.log_data))
        self.tensorboard.add_scalar('rmse', rmse, iteration)
        log.info(f'rmse: {rmse:04f}')
        singular_values = self.get_local_singular_values(n_nearest, latent)
        mean_singular_values = jnp.mean(singular_values, axis=0)
        log.info(f'mean_singular_values: {mean_singular_values}')
        for i in range(latent.shape[-1]):
            self.tensorboard.add_histogram(f"svd_{i}", singular_values[:, i], iteration)
            self.tensorboard.add_scalar(f"mean_svd_{i}", mean_singular_values[i], iteration)
        for i in range(1, latent.shape[-1]):
            self.tensorboard.add_histogram(f"normalized_svd_{i}", singular_values[:, i] / singular_values[:, 0], iteration)
            self.tensorboard.add_scalar(f"normalized_mean_svd_{i}", mean_singular_values[i] / mean_singular_values[0], iteration)

    def plot_reconstructions(self, fig):
        reconstructions = self.autoencoder.apply(self.autoencoder_params, self.plot_data)
        for i, (a, b) in enumerate(zip(self.plot_data, reconstructions)):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.imshow(jnp.concatenate([a, b], axis=1))

    def plot_latent(self, fig):
        ax = fig.add_subplot(111, projection='3d')
        latent = self.encoder.apply(self.autoencoder_params, self.log_data)
        ax.scatter(*latent.T, s=0.4)

    def plot(self, iteration):
        fig = plt.figure()
        #
        self.plot_reconstructions(fig)
        fig.savefig(os.path.join(self.path, f'reconstruction_{iteration:06d}.png'), dpi=300)
        fig.clear()
        #
        self.plot_latent(fig)
        fig.savefig(os.path.join(self.path, f'latent_{iteration:06d}.png'), dpi=300)
        plt.close(fig)

    def checkpoint(self, name):
        path = os.path.join(self.path, name)
        os.makedirs(path)
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump(self.autoencoder_params, f)
        with open(os.path.join(path, "learner_state.pkl"), "wb") as f:
            pickle.dump(self.learner_state, f)

    def restore(self, path):
        with open(os.path.join(path, "params.pkl"), "rb") as f:
            self.autoencoder_params = pickle.load(f)
        with open(os.path.join(path, "learner_state.pkl"), "rb") as f:
            self.learner_state = pickle.load(f)
