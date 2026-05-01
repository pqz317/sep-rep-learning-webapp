import flax.linen as nn
import jax
import jax.numpy as jnp


class VariationalInformationBottleneck(nn.Module):
    latent_size: int
    free_bits: float = 0.0

    def setup(self) -> None:
        self.mu_layer = nn.Dense(self.latent_size)
        self.logvar_layer = nn.Dense(self.latent_size)

    def __call__(
        self,
        feature: jax.Array,
        return_info: bool = False
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        mu = self.mu_layer(feature)
        logvar = self.logvar_layer(feature)
        if self.has_rng('vib'):
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(self.make_rng('vib'), shape=std.shape)
            z = mu + eps * std
        else:
            z = mu

        if return_info:
            kl_per_dim = -0.5 * (1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))
            if self.free_bits > 0:
                kl_per_dim = jnp.maximum(kl_per_dim, self.free_bits)
            kl = jnp.sum(kl_per_dim, axis=-1)
            return z, {
                'vib_kl': kl,
                'vib_mu': mu,
                'vib_logvar': logvar,
            }
        return z
