from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode

from ..env.spaces import Observation, ObservationSpace, ContinuousSpace
from .nn_blocks import Activation, MLP, RNN
from .nn_configs import NNConfig, CNNConfig, MLPConfig


class EmbeddingExtractor(nn.Module):
    """
    Extracts per-modality embeddings from a structured observation.

    This module applies a set of feature extractors (CNN / MLP / etc.)
    to different keys in the observation dictionary independently.

    Each observation key corresponds to one neural network defined
    by `nn_configs[key]`. The outputs are concatenated later by the
    downstream module.

    Architecture:
        obs (dict[str, Array])
            ├─ key_1 → FeatureNet_1 → embedding_1
            ├─ key_2 → FeatureNet_2 → embedding_2
            └─ ...
        (optional) LayerNorm is applied per embedding.
    """
    nn_configs: dict[str, NNConfig]
    use_embedding_layer_norm: bool = True

    def setup(self) -> None:
        self.feature_extractors = {
            k: self.nn_configs[k].create()
            for k in sorted(self.nn_configs.keys())
        }
        if self.use_embedding_layer_norm:
            self.layer_norms = {
                k: nn.LayerNorm() for k in sorted(self.nn_configs.keys())
            }
    
    def __call__(self, obs: Observation) -> dict[str, jax.Array]:
        embedding_dict = {
            k: self.feature_extractors[k](obs[k])
            for k in sorted(set(obs.keys()) & set(self.feature_extractors.keys()))
        }
        if self.use_embedding_layer_norm:
            embedding_dict = {k: self.layer_norms[k](v) for k, v in embedding_dict.items()}
        return embedding_dict

class RecurrentFeatureExtractor(nn.Module):
    """
    Extracts a unified feature representation from structured observations,
    optionally with temporal modeling via an RNN.

    This module is composed of four logical stages:

    1. Embedding extraction (per observation key)
       - Each observation component is encoded independently by
         `EmbeddingExtractor`.

    2. Feature aggregation
       - All embeddings are concatenated along the feature dimension.
       - An optional MLP is applied to mix information across modalities.

    3. Temporal modeling (optional)
       - An RNN processes the aggregated features over time, maintaining
         a recurrent hidden state.

    4. Output normalization (optional)
       - A final LayerNorm can be applied to the output features.
    """
    embedding_extractor_config: dict[str, NNConfig]
    use_embedding_layer_norm: bool = True
    aggr_mlp_layers: int = 0
    aggr_mlp_hidden_size: int = 64
    aggr_mlp_activation_type: Activation.Type = 'relu'
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru'
    use_rnn_layer_norm: bool = True
    use_final_layer_norm: bool = False
    
    def setup(self) -> None:
        self.embedding_extractor = EmbeddingExtractor(
            nn_configs=self.embedding_extractor_config,
            use_embedding_layer_norm=self.use_embedding_layer_norm
        )
        if self.aggr_mlp_layers > 0:
            self.aggr_mlp = MLP(
                hidden_size_lst=[self.aggr_mlp_hidden_size for i in range(self.aggr_mlp_layers)],
                activation_type=self.aggr_mlp_activation_type
            )
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm,
                cell_type=self.rnn_cell_type,
            )
        if self.use_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm()

    def __call__(self, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        """
        [Input]
            - obs[key]: [batch_size, sequence_length, ...].
            - rnn_state: PyTree compatible with the RNN implementation.

        [Output]
            - feature: [batch_size, sequence_length, feature_dim].
            - rnn_state: updated recurrent state.
        """
        embedding_dict = self.embedding_extractor(obs)
        feature = jnp.concatenate([embedding_dict[k] for k in sorted(embedding_dict.keys())], axis=-1)
        if self.aggr_mlp_layers > 0:
            feature = self.aggr_mlp(feature)
        if self.use_rnn:
            rnn_state, feature = self.rnn_model(rnn_state, feature)
        if self.use_final_layer_norm:
            feature = self.final_layer_norm(feature)
        return rnn_state, feature

    def encode(self, obs: Observation) -> jax.Array:
        """Feedforward-only encoding: embeddings + optional aggregation MLP,
        without advancing any RNN. Intended for branches that want a stateless
        observation feature alongside a recurrent pass elsewhere in the model.
        """
        embedding_dict = self.embedding_extractor(obs)
        feature = jnp.concatenate([embedding_dict[k] for k in sorted(embedding_dict.keys())], axis=-1)
        if self.aggr_mlp_layers > 0:
            feature = self.aggr_mlp(feature)
        return feature
    
    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        if self.use_rnn:
            rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=self.rnn_hidden_size,
                use_layer_norm=self.use_rnn_layer_norm,
                cell_type=self.rnn_cell_type,
            )
            return rnn_model.initialize_carry(rng, batch_size)
        else:
            return jnp.zeros((batch_size,))

def build_consistent_embedding_extractor_config(
    obs_space: ObservationSpace,
    hidden_size: int = 64,
    mlp_hidden_layer: int = 2,
    cnn_feature_lst: Sequence[int] = [64, 64, 64],
    cnn_kernel_lst: Sequence[Sequence[int]] = [(3, 3), (3, 3), (3, 3)],
    verbose_display: bool = False
) -> dict[str, NNConfig]:
    model_config: dict[str, NNConfig] = {}

    for k, space in obs_space.items():
        if not isinstance(space, ContinuousSpace):
            raise NotImplementedError(
                f"Expected symbolic or pixel observations, but got obs[{k}] = {space}"
            )

        if len(space.shape) == 1:
            if verbose_display:
                print(f"Creating MLPConfig for symbolic input: {k}")
            model_config[k] = MLPConfig(hidden_layers=mlp_hidden_layer, hidden_size=hidden_size)
        elif len(space.shape) == 3:
            if verbose_display:
                print(f"Creating CNNConfig for pixel input: {k}")
            model_config[k] = CNNConfig(
                feature_lst=cnn_feature_lst,
                kernel_lst=cnn_kernel_lst,
                post_mlp_hidden_size=hidden_size
            )
        else:
            raise NotImplementedError(f"Unsupported observation shape {space.shape} for obs[{k}]")

    return model_config

def build_feature_extractor_with_consistent_layers(
    obs_space: ObservationSpace, 
    hidden_size: int = 64,
    mlp_hidden_layer: int = 2,
    cnn_feature_lst: Sequence[int] = [64, 64, 64], 
    cnn_kernel_lst: Sequence[Sequence[int]] = [(3, 3), (3, 3), (3, 3)], 
    aggr_mlp_layers: int = 0,
    rnn_hidden_layers: int = 1,
    rnn_hidden_size: int = 64,
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
    use_rnn: bool = False,
    verbose_display: bool = False 
) -> RecurrentFeatureExtractor:
    """
    Loads a feature extractor while ensuring consistent hidden layers 
    across embbedings/feature extraction/aggregation. Uses default hyperparameters where applicable.

    [Input]
        - obs_space: A dictionary mapping observation keys to their respective spaces.
        - hidden_size: Hidden size used for MLP layers, RNN hidden states, and aggregation.
        - mlp_hidden_layer: Number of hidden layers in the MLP configuration.
        - cnn_feature_lst: List of feature sizes for CNN layers.
        - cnn_kernel_lst: List of kernel sizes for CNN layers.
        - aggr_mlp_layers: Number of hidden layers in the aggregation MLP.
        - rnn_hidden_layers: Number of layers in the RNN model.
        - rnn_hidden_size: Hidden size for RNN states.
        - use_rnn: Whether to use an RNN-based feature extractor.
        - verbose_display: If True, prints readable messages for users.

    [Output]
        - RecurrentFeatureExtractor: A feature extractor with consistent hidden layers.
    """
    model_config = build_consistent_embedding_extractor_config(
        obs_space=obs_space,
        hidden_size=hidden_size,
        mlp_hidden_layer=mlp_hidden_layer,
        cnn_feature_lst=cnn_feature_lst,
        cnn_kernel_lst=cnn_kernel_lst,
        verbose_display=verbose_display
    )

    if verbose_display:
        print("Using RecurrentFeatureExtractor with consistent hidden sizes.")

    return RecurrentFeatureExtractor(
        embedding_extractor_config=model_config,
        use_rnn=use_rnn,
        rnn_hidden_layers=rnn_hidden_layers, 
        rnn_hidden_size=rnn_hidden_size,
        rnn_cell_type=rnn_cell_type,
        aggr_mlp_layers=aggr_mlp_layers,
        aggr_mlp_hidden_size=hidden_size
    )


# def build_predictive_feature_extractor_with_consistent_layers(
#     obs_space: ObservationSpace,
#     action_space: ActionSpace | None = None,
#     hidden_size: int = 64,
#     mlp_hidden_layer: int = 2,
#     cnn_feature_lst: Sequence[int] = [64, 64, 64],
#     cnn_kernel_lst: Sequence[Sequence[int]] = [(3, 3), (3, 3), (3, 3)],
#     aggr_mlp_layers: int = 0,
#     rnn_hidden_layers: int = 1,
#     rnn_hidden_size: int = 64,
#     use_rnn: bool = False,
#     verbose_display: bool = False
# ) -> PredictiveRecurrentFeatureExtractor:
#     model_config = build_consistent_embedding_extractor_config(
#         obs_space=obs_space,
#         hidden_size=hidden_size,
#         mlp_hidden_layer=mlp_hidden_layer,
#         cnn_feature_lst=cnn_feature_lst,
#         cnn_kernel_lst=cnn_kernel_lst,
#         verbose_display=verbose_display
#     )

#     if verbose_display:
#         print("Using PredictiveRecurrentFeatureExtractor with consistent hidden sizes.")

#     return PredictiveRecurrentFeatureExtractor(
#         obs_space=obs_space,
#         action_space=action_space,
#         embedding_extractor_config=model_config,
#         use_rnn=use_rnn,
#         rnn_hidden_layers=rnn_hidden_layers,
#         rnn_hidden_size=rnn_hidden_size,
#         aggr_mlp_layers=aggr_mlp_layers,
#         aggr_mlp_hidden_size=hidden_size
#     )
