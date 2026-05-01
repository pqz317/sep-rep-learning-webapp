from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .nn_blocks import MLP


class ZInferenceHead(nn.Module):
    """
    Produces a posterior q(z | h_t) over the convention latent from the LSTM hidden
    state. Mirrors the VIB pattern: MLP trunk -> (mu, logvar), optional reparameterized
    sample, plus KL(q || N(0, I)). The head is intended to read directly off the
    recurrent feature (h_t), before any VIB/readout bottleneck.
    """
    latent_size: int
    hidden_sizes: Sequence[int] = (256, 128)
    activation_type: str = 'relu'
    kernel_init_scale: float | None = None

    def setup(self) -> None:
        self.trunk = MLP(
            hidden_size_lst=tuple(self.hidden_sizes),
            activation_type=self.activation_type,
            kernel_init_scale=self.kernel_init_scale,
        )
        self.mu_layer = nn.Dense(self.latent_size)
        self.logvar_layer = nn.Dense(self.latent_size)

    def __call__(
        self,
        feature: jax.Array,
        return_info: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        h = self.trunk(feature)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        if self.has_rng('z_infer'):
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(self.make_rng('z_infer'), shape=std.shape)
            z = mu + eps * std
        else:
            z = mu

        if return_info:
            kl_per_dim = -0.5 * (1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))
            kl = jnp.sum(kl_per_dim, axis=-1)
            return z, {
                'z_infer_mu': mu,
                'z_infer_logvar': logvar,
                'z_infer_sample': z,
                'z_infer_kl': kl,
            }
        return z
