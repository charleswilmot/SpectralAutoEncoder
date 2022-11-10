import jax.numpy as jnp
import haiku as hk
import logging


log = logging.getLogger(__name__)


class Encoder(hk.Module):
    def __init__(self, cfg):
        super().__init__(name='encoder')
        self.cfg = cfg
        self.conv1 = hk.Conv2D(
            output_channels=cfg.conv1.output_channels,
            kernel_shape=cfg.conv1.kernel_shape,
            stride=cfg.conv1.stride,
            name="conv1",
        )
        self.conv2 = hk.Conv2D(
            output_channels=cfg.conv2.output_channels,
            kernel_shape=cfg.conv2.kernel_shape,
            stride=cfg.conv2.stride,
            name="conv2",
        )
        self.mlp = hk.nets.MLP(
            output_sizes=cfg.mlp.output_sizes,
            activation=jnp.tanh,
            name="fully_connected_part",
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = jnp.tanh(x)
        x = self.conv2(x)
        x = jnp.tanh(x)
        x = hk.Flatten(preserve_dims=-3, name='flatten')(x)
        x = self.mlp(x)
        return x


class Decoder(hk.Module):
    def __init__(self, cfg):
        super().__init__(name='decoder')
        self.cfg = cfg
        self.mlp = hk.nets.MLP(
            output_sizes=cfg.mlp.output_sizes,
            activation=jnp.tanh,
            activate_final=True,
            name="fully_connected_part",
        )
        self.reshape = hk.Reshape(
            output_shape=cfg.reshape.output_shape,
            preserve_dims=-1,
            name='reshape'
        )
        self.conv1 = hk.Conv2DTranspose(
            output_channels=cfg.conv1.output_channels,
            kernel_shape=cfg.conv1.kernel_shape,
            stride=cfg.conv1.stride,
            output_shape=cfg.conv1.output_shape,
            name="conv1",
        )
        self.conv2 = hk.Conv2DTranspose(
            output_channels=cfg.conv2.output_channels,
            kernel_shape=cfg.conv2.kernel_shape,
            stride=cfg.conv2.stride,
            output_shape=cfg.conv2.output_shape,
            name="conv2",
        )

    def __call__(self, x):
        x = self.mlp(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = jnp.tanh(x)
        x = self.conv2(x)
        x = jnp.tanh(x)
        return x


class AutoEncoder(hk.Module):
    def __init__(self, cfg):
        super().__init__(name='autoencoder')
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
