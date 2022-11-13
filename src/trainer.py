from tensorboardX import SummaryWriter
from .autoencoder import AutoEncoder
import tensorflow_datasets as tfds
from matplotlib.patches import Patch
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


color = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
    3: (1.0, 1.0, 0.0),
    4: (1.0, 0.0, 1.0),
    5: (0.0, 1.0, 1.0),
    6: (0.3, 0.3, 0.7),
    7: (0.3, 0.7, 0.3),
    8: (0.7, 0.3, 0.3),
    9: (0.7, 0.7, 0.3),
}

def normalize(images, labels):
    images = tf.cast(images, tf.float32) / (255. / 2) - 1.
    return tf.reshape(images, (28, 28, 1)), labels


def strip_off_labels(images, labels): return images


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
        self.log_data_images, self.log_data_labels = next(dataset.take(cfg.log_data_size).batch(cfg.log_data_size).as_numpy_iterator())
        self.plot_data_images, self.plot_data_labels = next(dataset.take(16).batch(16).as_numpy_iterator())
        dataset = dataset.map(strip_off_labels)
        dataset = dataset.shuffle(10000, seed=cfg.seed)
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.batch(cfg.trainer.batch_size)
        self.dataset = dataset.as_numpy_iterator()
        self.latent_size = cfg.autoencoder.encoder.mlp.output_sizes[-1]
        ### define all differentiable functions

        @jax.jit
        def reconstruction_loss(autoencoder_params, x):
            reconstructions = self.autoencoder.apply(autoencoder_params, x)
            loss = jnp.sum(jnp.mean((x - reconstructions) ** 2, axis=(1,2,3)), axis=0)
            return loss

        @partial(jax.jit, static_argnames=('n_nearest', 'n_firsts', 'center', 'scale'))
        def get_close_neighbors(latent, n_nearest, n_firsts=None, center=True, scale=True):
            squared_distance_matrix = jnp.sum((latent[:, None] - latent[None, :]) ** 2, axis=-1)   # shape [N, N]
            if n_firsts is not None:
                squared_distance_matrix = squared_distance_matrix[:n_firsts]                       # shape [N_FIRSTS, N]
            neighbors_indices = jnp.argsort(squared_distance_matrix, axis=-1)                      # shape [N_FIRSTS, N]
            neighbors = jnp.take(latent, neighbors_indices, axis=0)                                # shape [N_FIRSTS, N, S]
            close_neighbors = neighbors[:, :n_nearest]                                             # shape [N_FIRSTS, N_NEAREST, S]
            if not center and scale:
                raise ValueError("Scaling without centering makes no sense")
            if center:
                mean = jnp.mean(close_neighbors, axis=-2, keepdims=True)                           # shape [N_FIRSTS, 1, S]
                close_neighbors = close_neighbors - mean                                  # shape [N_FIRSTS, N_NEAREST, S]
            if scale:
                lengths = jnp.sqrt(jnp.sum(close_neighbors ** 2, axis=-1, keepdims=True)) # shape [N_FIRSTS, N_NEAREST, 1]
                scale = jnp.mean(lengths, axis=-2, keepdims=True)                                  # shape [N_FIRSTS, 1, 1]
                close_neighbors = close_neighbors / scale                          # shape [N_FIRSTS, N_NEAREST, S]
            return close_neighbors

        @partial(jax.jit, static_argnames=('n_nearest', 'n_firsts', 'scale', 'compute_uv'))
        def get_local_singular_values(latent, n_nearest, n_firsts=None, scale=True, compute_uv=False):
            close_neighbors = get_close_neighbors(latent, n_nearest, n_firsts, center=True, scale=scale) # shape [N_FIRSTS, N_NEAREST, S]
            return jnp.linalg.svd(close_neighbors, compute_uv=compute_uv)                          # shape [N_FIRSTS, S]

        @partial(jax.jit, static_argnames=('desired_dim', 'n_nearest', 'n_firsts', 'scale'))
        def skeletonize_loss(autoencoder_params, x, desired_dim, n_nearest, n_firsts=None, scale=True):
            latent = self.encoder.apply(autoencoder_params, x)                                     # shape [N, S]
            batch_dim = latent.shape[0]
            singular_values = get_local_singular_values(latent, n_nearest, n_firsts, scale, False) # shape [M, S]
            return jnp.sum(singular_values[:, desired_dim:]) / batch_dim                           # shape []

        def total_loss(autoencoder_params, x, regularizer_coef, desired_dim, n_nearest, n_firsts=None, scale=True):
            reconstruction_term = reconstruction_loss(autoencoder_params, x)
            skeletonize_term = skeletonize_loss(autoencoder_params, x, desired_dim, n_nearest, n_firsts, scale)
            return reconstruction_term + regularizer_coef * skeletonize_term

        self.get_close_neighbors = get_close_neighbors
        self.get_local_singular_values = get_local_singular_values
        self.reconstruction_grad = jax.grad(reconstruction_loss)
        self.skeletonize_grad = jax.grad(skeletonize_loss)
        self.total_grad = jax.grad(total_loss)

    def init(self, key, dummy):
        self.autoencoder_params = self.autoencoder.init(key, dummy)
        self.learner_state = self.optimizer.init(self.autoencoder_params)

    def train(self, n_batches, regularizer_coef=0.0, desired_dim=2, n_nearest=128, scale=True):
        for _, x in zip(range(n_batches), self.dataset):
            if regularizer_coef == 0.0:
                dloss_dtheta = self.reconstruction_grad(self.autoencoder_params, x)
            else:
                n_firsts = x.shape[0] // n_nearest
                dloss_dtheta = self.total_grad(self.autoencoder_params, x, regularizer_coef, desired_dim, n_nearest, n_firsts, scale)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.autoencoder_params = optax.apply_updates(self.autoencoder_params, updates)

    def skeletonize(self, n_batches, desired_dim=2, n_nearest=128, scale=True):
        for _, x in zip(range(n_batches), self.dataset):
            n_firsts = x.shape[0] // n_nearest
            dloss_dtheta = self.skeletonize_grad(self.autoencoder_params, x, desired_dim, n_nearest, n_firsts, scale)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.autoencoder_params = optax.apply_updates(self.autoencoder_params, updates)

    def log(self, iteration, n_nearest=128, scale=True):
        latent = self.encoder.apply(self.autoencoder_params, self.log_data_images)
        reconstructions = self.decoder.apply(self.autoencoder_params, latent)
        rmse = jnp.mean(jnp.abs(reconstructions - self.log_data_images))
        self.tensorboard.add_scalar('rmse', rmse, iteration)
        log.info(f'rmse: {rmse:04f}')
        n_firsts = self.log_data_images.shape[0] // n_nearest
        singular_values = self.get_local_singular_values(latent, n_nearest, n_firsts, scale, compute_uv=False)
        mean_singular_values = jnp.mean(singular_values, axis=0)
        log.info(f'mean_singular_values: {mean_singular_values}')
        for i in range(latent.shape[-1]):
            self.tensorboard.add_histogram(f"svd_{i}", singular_values[:, i], iteration)
            self.tensorboard.add_scalar(f"mean_svd_{i}", mean_singular_values[i], iteration)
        for i in range(1, latent.shape[-1]):
            self.tensorboard.add_histogram(f"normalized_svd_{i}", singular_values[:, i] / singular_values[:, 0], iteration)
            self.tensorboard.add_scalar(f"normalized_mean_svd_{i}", mean_singular_values[i] / mean_singular_values[0], iteration)

    def plot_reconstructions(self, fig):
        reconstructions = self.autoencoder.apply(self.autoencoder_params, self.plot_data_images)
        for i, (a, b) in enumerate(zip(self.plot_data_images, reconstructions)):
            ax = fig.add_subplot(4, 4, i + 1)
            ax.imshow(jnp.concatenate([a, b], axis=1))

    def plot_latent(self, fig):
        ax = fig.add_subplot(111, projection='3d')
        latent = self.encoder.apply(self.autoencoder_params, self.log_data_images)
        colors = [color[num] for num in self.log_data_labels]
        ax.scatter(*latent.T, s=1.3, c=colors)
        patches = [Patch(facecolor=color[i], label=f"{i}") for i in sorted(set(self.log_data_labels))]
        ax.legend(handles=patches)

    def plot_t_sne(self, fig, dimension_reduction):
        if dimension_reduction == 3:
            ax = fig.add_subplot(111, projection='3d')
        elif dimension_reduction == 2:
            ax = fig.add_subplot(111)
        latent = self.encoder.apply(self.autoencoder_params, self.log_data_images)
        from sklearn.manifold import TSNE
        tsne = TSNE(
            dimension_reduction,
            perplexity=20,
        )
        projected = tsne.fit_transform(latent)
        colors = [color[num] for num in self.log_data_labels]
        ax.scatter(*projected.T, s=1.3, c=colors)
        patches = [Patch(facecolor=color[i], label=f"{i}") for i in sorted(set(self.log_data_labels))]
        ax.legend(handles=patches)

    def plot_interactive(self, fig, n_nearest):
        if self.latent_size > 3:
            raise ValueError("Latent size must be <= 3")
        gs = fig.add_gridspec(2, 2)
        ax_3D = fig.add_subplot(gs[:, 0], projection='3d')
        ax_singular_values = fig.add_subplot(gs[0, 1])
        ax_reconstructions = fig.add_subplot(gs[1, 1])
        latent = self.encoder.apply(self.autoencoder_params, self.log_data_images)

        # 3D plot
        colors = [color[num] for num in self.log_data_labels]
        ax_3D.scatter(*latent.T, s=2, c=colors)
        patches = [Patch(facecolor=color[i], label=f"{i}") for i in sorted(set(self.log_data_labels))]
        ax_3D.legend(handles=patches)
        mean = jnp.mean(latent, axis=0)
        mean = [2.8873544, -1.3857353, -0.7125883]
        cursor, = ax_3D.plot(
            xs=(mean[0],),
            ys=(mean[1],),
            zs=(mean[2],),
            marker='o',
            color='r',
            alpha=1,
        )

        def get_nearest_index():
            cursor_coord = jnp.array(cursor.get_data_3d()).flatten()
            return jnp.argmin(jnp.sum((latent - jnp.array(cursor_coord)) ** 2, axis=-1))

        nearest_index = get_nearest_index()
        nearest, = ax_3D.plot(
            xs=(latent[nearest_index, 0],),
            ys=(latent[nearest_index, 1],),
            zs=(latent[nearest_index, 2],),
            marker='o',
            color='b'
        )

        # singular values
        _, singular_values, v = self.get_local_singular_values(
            latent,
            n_nearest,
            n_firsts=None,
            scale=True,
            compute_uv=True
        ) # shape [N, N_NEAREST, S]
        close_neighbors = self.get_close_neighbors(latent, n_nearest, n_firsts=None, center=False, scale=False)

        neighbors_scatter = ax_3D.scatter(*close_neighbors[nearest_index, :n_nearest].T, c='k', s=10, zorder=3)

        bars = ax_singular_values.bar(x=tuple(range(latent.shape[-1])), height=singular_values[nearest_index])

        # arrows
        quiver = ax_3D.quiver(
            (latent[nearest_index, 0],) * 3,
            (latent[nearest_index, 1],) * 3,
            (latent[nearest_index, 2],) * 3,
            v[nearest_index, :, 0] * singular_values[nearest_index],
            v[nearest_index, :, 1] * singular_values[nearest_index],
            v[nearest_index, :, 2] * singular_values[nearest_index],
        )

        # reconstructions
        N = 15

        grid_x, grid_y = jnp.mgrid[-1:1:N*1j, -1:1:N*1j]
        coord = jnp.stack((grid_x, grid_y), axis=-1) # [N, N, 2]
        reshaped_coord = jnp.reshape(coord, (N * N,) + coord.shape[2:]) # [N * N, 2]
        def get_picture_coords(nearest_index):
            return (reshaped_coord @ v[nearest_index, :2]) + latent[nearest_index]


        coords = get_picture_coords(nearest_index)
        minigrid_scatter = ax_3D.scatter(*coords.T, alpha=0.1)


        def get_picture(coords):
            reconstructions = self.decoder.apply(self.autoencoder_params, coords) # [N * N, 28, 28, 1]
            picture = jnp.reshape(reconstructions, (N, N) + reconstructions.shape[1:])
            picture = jnp.concatenate(picture, axis=1) # [N, 28 * N, 28, 1]
            picture = jnp.concatenate(picture, axis=1) # [28 * N, 28 * N, 1]
            return picture

        picture = get_picture(coords)
        image = ax_reconstructions.imshow(picture)

        # connecting people
        def on_press(event):
            log.info(f"key pressed: {event.key}")
            update_cursor = False
            update_mini_grid = False
            delta = 0.2
            if event.key == 'left':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((-delta, 0, 0))[:, None])
                update_cursor = True
            if event.key == 'right':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((delta, 0, 0))[:, None])
                update_cursor = True
            if event.key == 'up':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((0, delta, 0))[:, None])
                update_cursor = True
            if event.key == 'down':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((0, -delta, 0))[:, None])
                update_cursor = True
            if event.key == 'pageup':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((0, 0, delta))[:, None])
                update_cursor = True
            if event.key == 'pagedown':
                data = jnp.array(cursor.get_data_3d())
                cursor.set_data_3d(data + jnp.array((0, 0, -delta))[:, None])
                update_cursor = True
            if event.key == 'enter':
                update_mini_grid = True
            if update_cursor:
                log.info(f"cursor position: {cursor.get_data_3d().flatten()}")
                nearest_index = get_nearest_index()
                nearest.set_data_3d(latent[nearest_index][:, None])
                for rect, val in zip(bars, singular_values[nearest_index]): rect.set_height(val)
                xyz = jnp.repeat(latent[nearest_index][None], 3, axis=0)
                uvw = xyz + v[nearest_index] * singular_values[nearest_index, :, None]
                segments = jnp.stack([xyz, uvw], axis=1)
                quiver.set_segments(segments)
                neighbors_scatter._offsets3d = close_neighbors[nearest_index, :n_nearest].T
            if update_mini_grid:
                nearest_index = get_nearest_index()
                coords = get_picture_coords(nearest_index)
                minigrid_scatter._offsets3d = coords.T
                picture = get_picture(coords)
                image.set_data(picture)
            if update_cursor or update_mini_grid:
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_press)

    def plot(self, iteration):
        fig = plt.figure()
        #
        self.plot_reconstructions(fig)
        fig.savefig(os.path.join(self.path, f'reconstruction_{iteration:06d}.png'), dpi=300)
        fig.clear()
        #
        if self.latent_size > 3:
            return
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
