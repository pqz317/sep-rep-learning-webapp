from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import flax.linen as nn
import flax.struct as struct
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from ..env import Observation, Action, ObservationSpace, ActionSpace
from ..env.spaces import ContinuousSpace
from ..model.action_distribution import ActionDistribution, get_distribution_info, build_action_distribution_layer
from ..model.feature_extractors import RecurrentFeatureExtractor, build_feature_extractor_with_consistent_layers
from ..model.nn_blocks import Activation, MLP
from ..model.nn_configs import CNNConfig, MLPConfig, NNConfig
from ..model.next_state_prediction import ObservationPredictor, ActionFeatureEncoder
from ..model.vib import VariationalInformationBottleneck
from ..model.latent_convention import ZInferenceHead
from ..trainer.bc import BCTrainer
from ..trainer.ppo import PPOTrainer, PPOTransition
from ..utils import global_norm
from .agent_wrapper import RecurrentAgent, RecurrentPPOAgent


FeatureExtractorConfig = TypeVar('FeatureExtractorConfig', bound='BaseFeatureExtractorConfig')

class PolicyLayerConfig(PyTreeNode):
    action_space: ActionSpace
    output_kernel_init_scale: float | None = None
    def build(self) -> dict[str, nn.Module]:
        return build_action_distribution_layer(
            self.action_space,
            output_kernel_init_scale=self.output_kernel_init_scale,
        )

class BaseFeatureExtractorConfig(PyTreeNode):
    # A pure configuration object for building feature extractor modules.
    obs_space: ObservationSpace
    def build(self) -> RecurrentFeatureExtractor:
        return build_feature_extractor_with_consistent_layers(self.obs_space)

class SimpleFeatureExtractorConfig(BaseFeatureExtractorConfig):
    common_hidden_size: int = 64
    cnn_kernel_size: int = 3
    cnn_num_conv_layers: int = 3
    use_rnn: bool = False
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru'
    def build(self) -> RecurrentFeatureExtractor:
        feature_extractor = build_feature_extractor_with_consistent_layers(
            self.obs_space,
            hidden_size=self.common_hidden_size,
            cnn_feature_lst=[self.common_hidden_size for _ in range(self.cnn_num_conv_layers)],
            cnn_kernel_lst=[(self.cnn_kernel_size, self.cnn_kernel_size) for _ in range(self.cnn_num_conv_layers)],
            use_rnn=self.use_rnn,
            rnn_hidden_layers=1,
            rnn_hidden_size=self.common_hidden_size,
            rnn_cell_type=self.rnn_cell_type,
        )
        return feature_extractor


class ConfigurableFeatureExtractorConfig(BaseFeatureExtractorConfig):
    cnn_feature_lst: tuple[int, ...] = struct.field(default_factory=lambda: (64, 64, 64))
    cnn_kernel_lst: tuple[tuple[int, int], ...] = struct.field(default_factory=lambda: ((3, 3), (3, 3), (3, 3)))
    cnn_stride_lst: tuple[int, ...] = struct.field(default_factory=lambda: (1, 1, 1))
    encoder_fc_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=lambda: (64,))
    activation_type: Activation.Type = 'relu'
    encoder_kernel_init_scale: float | None = None
    use_rnn: bool = False
    rnn_hidden_layers: int = 1
    rnn_hidden_size: int = 64
    rnn_cell_type: Literal['gru', 'lstm'] = 'gru'
    use_rnn_layer_norm: bool = False

    def build(self) -> RecurrentFeatureExtractor:
        if len(self.cnn_feature_lst) != len(self.cnn_kernel_lst) or len(self.cnn_feature_lst) != len(self.cnn_stride_lst):
            raise ValueError(
                "cnn_feature_lst, cnn_kernel_lst, and cnn_stride_lst must have the same length "
                f"(got {len(self.cnn_feature_lst)}, {len(self.cnn_kernel_lst)}, {len(self.cnn_stride_lst)})"
            )

        embedding_extractor_config: dict[str, NNConfig] = {}
        for key, space in self.obs_space.items():
            if not isinstance(space, ContinuousSpace):
                raise NotImplementedError(
                    f"Expected continuous observation space for key '{key}', but got {type(space)}"
                )
            if len(space.shape) == 1:
                embedding_extractor_config[key] = MLPConfig(
                    hidden_size_lst=list(self.encoder_fc_hidden_size_lst),
                    activation_type=self.activation_type,
                    kernel_init_scale=self.encoder_kernel_init_scale,
                )
            elif len(space.shape) == 3:
                embedding_extractor_config[key] = CNNConfig(
                    feature_lst=self.cnn_feature_lst,
                    kernel_lst=self.cnn_kernel_lst,
                    stride_lst=self.cnn_stride_lst,
                    activation_type=self.activation_type,
                    post_mlp_hidden_size_lst=self.encoder_fc_hidden_size_lst,
                    post_mlp_activation_type=self.activation_type,
                    kernel_init_scale=self.encoder_kernel_init_scale,
                    post_mlp_kernel_init_scale=self.encoder_kernel_init_scale,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported observation shape {space.shape} for key '{key}'. "
                    "Only 1D and 3D continuous observations are supported."
                )

        return RecurrentFeatureExtractor(
            embedding_extractor_config=embedding_extractor_config,
            use_rnn=self.use_rnn,
            rnn_hidden_layers=self.rnn_hidden_layers,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_cell_type=self.rnn_cell_type,
            use_rnn_layer_norm=self.use_rnn_layer_norm,
            aggr_mlp_layers=0,
        )

class ActorNet(nn.Module, Generic[FeatureExtractorConfig]):
    feature_extractor_config: FeatureExtractorConfig
    policy_config: PolicyLayerConfig
    def setup(self) -> None:
        self.feature_extractor = self.feature_extractor_config.build()
        self.policy_layers = self.policy_config.build()

    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        rnn_states, feature = self.feature_extractor(rnn_states, obs)
        pi = ActionDistribution({
            action_name: policy_layer(feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        return rnn_states, pi
    
    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        feature_extractor = self.feature_extractor_config.build()
        return feature_extractor.init_rnn_state(rng, batch_size)

class CriticNet(nn.Module, Generic[FeatureExtractorConfig]):
    feature_extractor_config: FeatureExtractorConfig
    def setup(self) -> None:
        self.feature_extractor = self.feature_extractor_config.build()
        self.value_layer = nn.Dense(1)

    def __call__(self, rnn_states: jax.Array, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        rnn_states, feature = self.feature_extractor(rnn_states, obs)
        val = self.value_layer(feature).squeeze(-1)
        return rnn_states, val
    
    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        feature_extractor = self.feature_extractor_config.build()
        return feature_extractor.init_rnn_state(rng, batch_size)

class SharedActorCriticNet(nn.Module, Generic[FeatureExtractorConfig]):
    feature_extractor_config: FeatureExtractorConfig
    policy_config: PolicyLayerConfig
    use_predictive_head: bool = False
    use_vib: bool = False
    use_readout_layer: bool = False
    readout_size: int = 256
    vib_free_bits: float = 0.0
    vib_noise_mode: str = 'all'  # 'all' | 'predictive_only'
    vib_stop_ppo_grads: bool = False
    obs_space: ObservationSpace | None = None
    action_space: ActionSpace | None = None
    actor_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    critic_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    head_activation_type: Activation.Type = 'relu'
    head_kernel_init_scale: float | None = None
    critic_output_kernel_init_scale: float | None = None
    latent_dim: int = 0
    use_z_inference: bool = False
    z_inference_hidden_sizes: tuple[int, ...] = struct.field(default_factory=lambda: (256, 128))

    def setup(self) -> None:
        self.feature_extractor = self.feature_extractor_config.build()
        if self.use_vib:
            self.vib = VariationalInformationBottleneck(latent_size=self.readout_size, free_bits=self.vib_free_bits)
        elif self.use_readout_layer:
            self.readout_layer = nn.Dense(self.readout_size)
        self.policy_layers = self.policy_config.build()
        if len(self.actor_head_hidden_size_lst) > 0:
            self.actor_head = MLP(
                hidden_size_lst=self.actor_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        if len(self.critic_head_hidden_size_lst) > 0:
            self.critic_head = MLP(
                hidden_size_lst=self.critic_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        value_layer_kernel_init = (
            nn.initializers.orthogonal(self.critic_output_kernel_init_scale)
            if self.critic_output_kernel_init_scale is not None
            else nn.initializers.lecun_normal()
        )
        self.value_layer = nn.Dense(1, kernel_init=value_layer_kernel_init)
        if self.use_predictive_head:
            if self.obs_space is None or self.action_space is None:
                raise ValueError("obs_space and action_space must be provided when use_predictive_head=True")
            self.observation_predictor = ObservationPredictor(
                obs_space=self.obs_space,
                embedding_extractor_config=self.feature_extractor.embedding_extractor_config
            )
            self.action_feature_encoder = ActionFeatureEncoder(action_space=self.action_space)
        if self.use_z_inference:
            if self.latent_dim <= 0:
                raise ValueError("latent_dim must be > 0 when use_z_inference=True")
            self.z_inference = ZInferenceHead(
                latent_size=self.latent_dim,
                hidden_sizes=tuple(self.z_inference_hidden_sizes),
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )

    def _apply_heads(
        self,
        policy_feature: jax.Array,
    ) -> tuple[ActionDistribution, jax.Array]:
        actor_feature = self.actor_head(policy_feature) if hasattr(self, 'actor_head') else policy_feature
        critic_feature = self.critic_head(policy_feature) if hasattr(self, 'critic_head') else policy_feature
        pi = ActionDistribution({
            action_name: policy_layer(actor_feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        val = self.value_layer(critic_feature).squeeze(-1)
        return pi, val

    def __call__(
        self,
        rnn_states: jax.Array,
        obs: Observation,
        action: Action | None = None,
        return_prediction: bool = False,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]] | tuple[PyTreeNode, ActionDistribution, jax.Array, Observation, dict[str, jax.Array]]:
        rnn_states, feature = self.feature_extractor(rnn_states, obs)
        info = {'feature': feature}
        # Z-inference reads directly off the LSTM output `feature` (h_t), before any
        # VIB/readout bottleneck. The posterior supervises P0's convention beliefs.
        if self.use_z_inference:
            z_inferred, z_info = self.z_inference(feature, return_info=True)
            info.update(z_info)
            if z is None:
                z_effective = z_inferred
            else:
                if is_p1 is not None:
                    mask = is_p1.astype(z.dtype)
                    mask = mask.reshape(mask.shape + (1,) * (z.ndim - mask.ndim))
                    z_effective = mask * z + (1.0 - mask) * z_inferred
                else:
                    z_effective = z
        else:
            z_effective = z
        if self.use_vib:
            readout, vib_info = self.vib(feature, return_info=True)
            info.update(vib_info)
            if self.vib_noise_mode == 'predictive_only':
                policy_feature = vib_info['vib_mu']
                pred_feature = readout
            else:
                policy_feature = readout
                pred_feature = readout
            if self.vib_stop_ppo_grads:
                policy_feature = jax.lax.stop_gradient(policy_feature)
        elif self.use_readout_layer:
            readout = self.readout_layer(feature)
            info['readout'] = readout
            policy_feature = readout
            pred_feature = readout
        else:
            policy_feature = feature
            pred_feature = feature
        if self.latent_dim > 0 and z_effective is not None:
            policy_feature = jnp.concatenate([policy_feature, z_effective], axis=-1)
        pi, val = self._apply_heads(policy_feature)
        if self.use_predictive_head and return_prediction:
            action = pi.mode() if action is None else action
            action_feature = self.action_feature_encoder(action)
            predicted_obs = self.observation_predictor(pred_feature, action_feature)
            return rnn_states, pi, val, predicted_obs, info
        return rnn_states, pi, val, info

    def policy_from_features(
        self,
        feature: jax.Array,
        z: jax.Array | None = None,
    ) -> ActionDistribution:
        """
        Compute the action distribution given a pre-computed post-LSTM feature and a
        latent `z`. Bypasses the feature extractor (CNN -> LSTM) and the z-inference
        head. Used for the B^2 population-entropy evaluation where the same state
        feature is evaluated under many different z samples.
        """
        if self.use_vib:
            mu = self.vib.mu_layer(feature)
            if self.vib_noise_mode == 'predictive_only':
                policy_feature = mu
            else:
                # deterministic: reparameterization RNG is absent in this pure path.
                policy_feature = mu
            if self.vib_stop_ppo_grads:
                policy_feature = jax.lax.stop_gradient(policy_feature)
        elif self.use_readout_layer:
            policy_feature = self.readout_layer(feature)
        else:
            policy_feature = feature
        if self.latent_dim > 0 and z is not None:
            policy_feature = jnp.concatenate([policy_feature, z], axis=-1)
        pi, _ = self._apply_heads(policy_feature)
        return pi

    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        feature_extractor = self.feature_extractor_config.build()
        return feature_extractor.init_rnn_state(rng, batch_size)


class PredictiveSharedActorCriticNet(SharedActorCriticNet[FeatureExtractorConfig]):
    use_predictive_head: bool = True


class LatentConventionActorCriticNet(nn.Module, Generic[FeatureExtractorConfig]):
    """Dual-extractor actor-critic for latent-convention self-play.

    Two feature extractors with independent parameters:
      - `inference_feature_extractor` (CNN/MLP + RNN): carries convention-inference
        signal over time; its output h_t feeds the z-inference head that
        produces the P0 posterior q(z | h_t). Also owns the RNN state.
      - `policy_feature_extractor` (CNN/MLP, no RNN): produces a stateless
        per-timestep feature fed (concatenated with z) to the actor/critic heads.

    Exposes the same Flax call contract as `SharedActorCriticNet` so the same
    `UnifiedSharedActorCriticModel` wrapper can drive it:
      __call__(rnn_states, obs, action=None, return_prediction=False,
               z=None, is_p1=None) -> (rnn_states, pi, val, info)

    `info` contains z-inference stats (`z_infer_mu`, `z_infer_logvar`,
    `z_infer_sample`, `z_infer_kl`) plus the policy-branch `cnn_feature` used
    by the structured-PE reward kernel.
    """
    # Named `feature_extractor_config` so UnifiedSharedActorCriticModel's
    # introspection (`feature_extractor_config.obs_space.example(...)`) works.
    feature_extractor_config: FeatureExtractorConfig  # inference branch (RNN)
    policy_feature_extractor_config: FeatureExtractorConfig  # policy branch (no RNN)
    policy_config: PolicyLayerConfig
    actor_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    critic_head_hidden_size_lst: tuple[int, ...] = struct.field(default_factory=tuple)
    head_activation_type: Activation.Type = 'relu'
    head_kernel_init_scale: float | None = None
    critic_output_kernel_init_scale: float | None = None
    latent_dim: int = 4
    z_inference_hidden_sizes: tuple[int, ...] = struct.field(default_factory=lambda: (256, 128))
    # Flags consumed by the UnifiedSharedActorCriticModel wrapper.
    use_predictive_head: bool = False
    use_vib: bool = False
    use_z_inference: bool = True

    def setup(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0 for LatentConventionActorCriticNet")
        self.inference_feature_extractor = self.feature_extractor_config.build()
        self.policy_feature_extractor = self.policy_feature_extractor_config.build()
        self.policy_layers = self.policy_config.build()
        if len(self.actor_head_hidden_size_lst) > 0:
            self.actor_head = MLP(
                hidden_size_lst=self.actor_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        if len(self.critic_head_hidden_size_lst) > 0:
            self.critic_head = MLP(
                hidden_size_lst=self.critic_head_hidden_size_lst,
                activation_type=self.head_activation_type,
                kernel_init_scale=self.head_kernel_init_scale,
            )
        value_layer_kernel_init = (
            nn.initializers.orthogonal(self.critic_output_kernel_init_scale)
            if self.critic_output_kernel_init_scale is not None
            else nn.initializers.lecun_normal()
        )
        self.value_layer = nn.Dense(1, kernel_init=value_layer_kernel_init)
        self.z_inference = ZInferenceHead(
            latent_size=self.latent_dim,
            hidden_sizes=tuple(self.z_inference_hidden_sizes),
            activation_type=self.head_activation_type,
            kernel_init_scale=self.head_kernel_init_scale,
        )

    def _apply_heads(
        self,
        policy_feature: jax.Array,
    ) -> tuple[ActionDistribution, jax.Array]:
        actor_feature = self.actor_head(policy_feature) if hasattr(self, 'actor_head') else policy_feature
        critic_feature = self.critic_head(policy_feature) if hasattr(self, 'critic_head') else policy_feature
        pi = ActionDistribution({
            action_name: policy_layer(actor_feature)
            for action_name, policy_layer in self.policy_layers.items()
        })
        val = self.value_layer(critic_feature).squeeze(-1)
        return pi, val

    def __call__(
        self,
        rnn_states: jax.Array,
        obs: Observation,
        action: Action | None = None,
        return_prediction: bool = False,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]]:
        # Inference branch: CNN/MLP + RNN → h_t (owns rnn state).
        rnn_states, h_t = self.inference_feature_extractor(rnn_states, obs)
        z_inferred, z_info = self.z_inference(h_t, return_info=True)
        if z is None:
            z_effective = z_inferred
        else:
            if is_p1 is not None:
                mask = is_p1.astype(z.dtype)
                mask = mask.reshape(mask.shape + (1,) * (z.ndim - mask.ndim))
                z_effective = mask * z + (1.0 - mask) * z_inferred
            else:
                z_effective = z
        # Policy branch: feedforward-only encoder → feat_pol.
        feat_pol = self.policy_feature_extractor.encode(obs)
        policy_feature = jnp.concatenate([feat_pol, z_effective], axis=-1)
        pi, val = self._apply_heads(policy_feature)
        info: dict[str, jax.Array] = {'cnn_feature': feat_pol, 'h_t': h_t}
        info.update(z_info)
        return rnn_states, pi, val, info

    def policy_from_features(
        self,
        feature: jax.Array,
        z: jax.Array | None = None,
    ) -> ActionDistribution:
        """Evaluate pi given a pre-computed policy-branch CNN feature and a latent z.

        Bypasses both feature extractors and the z-inference head. Used by the
        B^2 population-entropy kernel where the same state feature is evaluated
        under many z samples.
        """
        if z is not None:
            policy_feature = jnp.concatenate([feature, z], axis=-1)
        else:
            policy_feature = feature
        pi, _ = self._apply_heads(policy_feature)
        return pi

    @nn.nowrap
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        feature_extractor = self.feature_extractor_config.build()
        return feature_extractor.init_rnn_state(rng, batch_size)


class BaseActorCriticModel(ABC):
    @abstractmethod
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        ...
    
    @abstractmethod
    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        ...
    
    def init_actor_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.init_rnn_state(rng, batch_size)

    def has_predictive_head(self) -> bool:
        return False

    def has_vib(self) -> bool:
        return False

    @abstractmethod
    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        ...

    def forward_with_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]]:
        next_rnn_state, pi, val = self.forward(rng, model_state, rnn_state, obs)
        return next_rnn_state, pi, val, {}

    @abstractmethod
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        ...

    @abstractmethod
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        ...
    
    @classmethod
    def output_nn(self, fn: ActorNet | CriticNet | SharedActorCriticNet | PredictiveSharedActorCriticNet, obs_space: ObservationSpace, depth: int | None = -1) -> None:
        if depth != -1:
            rnn_state = fn.init_rnn_state(jax.random.key(0), batch_size=1)
            obs_example = obs_space.example(batch_shape=(1,))
            if getattr(fn, 'use_predictive_head', False):
                action_space = getattr(fn, 'action_space', None)
                if action_space is None:
                    raise ValueError("action_space must be set when use_predictive_head=True")
                action_example = action_space.example(batch_shape=(1,))
                print(
                    fn.tabulate(
                        jax.random.key(0),
                        rnn_state,
                        obs_example,
                        action=action_example,
                        return_prediction=True,
                        depth=depth,
                    )
                )
            else:
                print(fn.tabulate(jax.random.key(0), rnn_state, obs_example, depth=depth))

class SeparatedActorCriticModel(BaseActorCriticModel):
    @classmethod
    def build_base_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, output_nn_depth: int | None = -1) -> tuple[ActorNet[BaseFeatureExtractorConfig], CriticNet[BaseFeatureExtractorConfig]]:
        actor_fn = ActorNet(
            feature_extractor_config=BaseFeatureExtractorConfig(obs_space=obs_space),
            policy_config=PolicyLayerConfig(action_space=action_space)
        )
        critic_fn = CriticNet(
            feature_extractor_config=BaseFeatureExtractorConfig(obs_space=obs_space),
        )
        cls.output_nn(actor_fn, obs_space, output_nn_depth)
        cls.output_nn(critic_fn, obs_space, output_nn_depth)
        return actor_fn, critic_fn
    
    @classmethod
    def build_simple_actor_critic(cls, obs_space: ObservationSpace, action_space: ActionSpace, common_hidden_size: int = 64, cnn_kernel_size: int = 3, conv_layers: int = 3, use_rnn: bool = False, output_nn_depth: int | None = -1) -> tuple[ActorNet[SimpleFeatureExtractorConfig], CriticNet[SimpleFeatureExtractorConfig]]:
        actor_fn = ActorNet(
            feature_extractor_config=SimpleFeatureExtractorConfig(
                obs_space=obs_space,
                common_hidden_size=common_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                cnn_num_conv_layers=conv_layers,
                use_rnn=use_rnn
            ),
            policy_config=PolicyLayerConfig(action_space=action_space)
        )
        critic_fn = CriticNet(
            feature_extractor_config=SimpleFeatureExtractorConfig(
                obs_space=obs_space,
                common_hidden_size=common_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                cnn_num_conv_layers=conv_layers,
                use_rnn=use_rnn
            )
        )
        cls.output_nn(actor_fn, obs_space, output_nn_depth)
        cls.output_nn(critic_fn, obs_space, output_nn_depth)
        return actor_fn, critic_fn
    
    def __init__(self, actor_fn: ActorNet[FeatureExtractorConfig], critic_fn: ActorNet[FeatureExtractorConfig]) -> None:
        self.actor_fn = actor_fn
        self.critic_fn = critic_fn

    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        actor_rnn_state = self.actor_fn.init_rnn_state(rng, batch_size)
        critic_rnn_state = self.critic_fn.init_rnn_state(rng, batch_size)
        return {'actor_rnn_state': actor_rnn_state, 'critic_rnn_state': critic_rnn_state}
    
    def init_actor_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        actor_rnn_state = self.actor_fn.init_rnn_state(rng, batch_size)
        return {'actor_rnn_state': actor_rnn_state}
    
    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        # An example input (batch_size=1) without the sequence length dimension
        rng, rng_rnn = jax.random.split(rng)
        rnn_state_example = self.init_rnn_state(rng_rnn, batch_size=1)
        actor_obs_example = self.actor_fn.feature_extractor_config.obs_space.example(batch_shape=(1,))
        critic_obs_example = self.critic_fn.feature_extractor_config.obs_space.example(batch_shape=(1,))

        actor_rng, critic_rng = jax.random.split(rng)
        actor_params = self.actor_fn.init(
            actor_rng,
            rnn_state_example['actor_rnn_state'],
            actor_obs_example
        )
        critic_params = self.critic_fn.init(
            critic_rng,
            rnn_state_example['critic_rnn_state'],
            critic_obs_example
        )
        return {'actor_params': actor_params, 'critic_params': critic_params}
    
    def get_action_distribution(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution]:
        """
        Computes the action distribution given the current model parameters, recurrent state, and observation.

        [Input]
            - rng: A random number generator used to control stochastic behavior in the actor model (e.g., dropout).
            - model_state:  A dictionary containing 'actor_params', which are the model parameters for the actor network.
            - rnn_state: A dictionary containing 'actor_rnn_state' with shape [batch_size, *rnn_state_shape],
                representing the hidden state of the recurrent network in the actor model.
            - obs: Input observation with shape [batch_size, length, *observation_shape]

        [Output]
            - next_rnn_state: A dictionary replacing 'actor_rnn_state' with next_actor_rnn_state,
                which represents the updated hidden state after processing the input observation.
            - action_distribution: A batch of action distributions computed from the actor network, with shape [batch_size, length].
        """

        actor_rnn_state = rnn_state['actor_rnn_state']
        actor_params = model_state['actor_params']
        next_actor_rnn_state, pi = self.actor_fn.apply(actor_params, actor_rnn_state, obs)
        return {'actor_rnn_state': next_actor_rnn_state}, pi
    
    def get_value(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, jax.Array]:
        """
        Computes the state value given the current model parameters, recurrent state, and observation.

        [Input]
            - rng: A random number generator used to control stochastic behavior in the critic model (e.g., dropout).
            - model_state:  A dictionary containing 'critic_params', which are the model parameters for the critic network.
            - rnn_state: A dictionary containing 'critic_rnn_state' with shape [batch_size, *rnn_state_shape],
                representing the hidden state of the recurrent network in the actor model.
            - obs: Input observation with shape [batch_size, length, *observation_shape]

        [Output]
            - next_rnn_state: A dictionary replacing 'critic_rnn_state' with next_critic_rnn_state,
                which represents the updated hidden state after processing the input observation.
            - value: A batch of values computed from the critic network, with shape [batch_size, length].
        """

        critic_rnn_state = rnn_state['critic_rnn_state']
        critic_params = model_state['critic_params']
        next_critic_rnn_state, val = self.critic_fn.apply(critic_params, critic_rnn_state, obs)
        return {'critic_rnn_state': next_critic_rnn_state}, val

    def forward(self, rng: jax.Array, model_state: PyTreeNode, rnn_state: PyTreeNode, obs: Observation) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        actor_rng, critic_rng = jax.random.split(rng)
        next_actor_rnn_state, pi = self.get_action_distribution(actor_rng, model_state, rnn_state, obs)
        next_critic_rnn_state, val = self.get_value(critic_rng, model_state, rnn_state, obs)
        return {
            'actor_rnn_state': next_actor_rnn_state['actor_rnn_state'],
            'critic_rnn_state': next_critic_rnn_state['critic_rnn_state']
        }, pi, val

class UnifiedSharedActorCriticModel(BaseActorCriticModel):
    @classmethod
    def build_base_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        use_predictive_head: bool = False,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> SharedActorCriticNet[BaseFeatureExtractorConfig]:
        actor_critic_fn = SharedActorCriticNet(
            feature_extractor_config=BaseFeatureExtractorConfig(obs_space=obs_space),
            policy_config=PolicyLayerConfig(action_space=action_space),
            use_predictive_head=use_predictive_head,
            use_vib=use_vib,
            use_readout_layer=True,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            obs_space=obs_space if use_predictive_head else None,
            action_space=action_space if use_predictive_head else None,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

    @classmethod
    def build_simple_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        common_hidden_size: int = 64,
        cnn_kernel_size: int = 3,
        conv_layers: int = 3,
        use_rnn: bool = False,
        use_predictive_head: bool = False,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> SharedActorCriticNet[SimpleFeatureExtractorConfig]:
        actor_critic_fn = SharedActorCriticNet(
            feature_extractor_config=SimpleFeatureExtractorConfig(
                obs_space=obs_space,
                common_hidden_size=common_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                cnn_num_conv_layers=conv_layers,
                use_rnn=use_rnn
            ),
            policy_config=PolicyLayerConfig(action_space=action_space),
            use_predictive_head=use_predictive_head,
            use_vib=use_vib,
            use_readout_layer=True,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            obs_space=obs_space if use_predictive_head else None,
            action_space=action_space if use_predictive_head else None,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

    @classmethod
    def build_configurable_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        cnn_feature_lst: tuple[int, ...] = (64, 64, 64),
        cnn_kernel_lst: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        cnn_stride_lst: tuple[int, ...] = (1, 1, 1),
        encoder_fc_hidden_size_lst: tuple[int, ...] = (64,),
        use_rnn: bool = True,
        rnn_hidden_layers: int = 1,
        rnn_hidden_size: int = 64,
        rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
        use_rnn_layer_norm: bool = False,
        activation_type: Activation.Type = 'relu',
        actor_head_hidden_size_lst: tuple[int, ...] = (),
        critic_head_hidden_size_lst: tuple[int, ...] = (),
        encoder_kernel_init_scale: float | None = None,
        head_kernel_init_scale: float | None = None,
        actor_output_kernel_init_scale: float | None = None,
        critic_output_kernel_init_scale: float | None = None,
        use_predictive_head: bool = False,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        latent_dim: int = 0,
        use_z_inference: bool = False,
        z_inference_hidden_sizes: tuple[int, ...] = (256, 128),
        output_nn_depth: int | None = -1,
    ) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        actor_critic_fn = SharedActorCriticNet(
            feature_extractor_config=ConfigurableFeatureExtractorConfig(
                obs_space=obs_space,
                cnn_feature_lst=cnn_feature_lst,
                cnn_kernel_lst=cnn_kernel_lst,
                cnn_stride_lst=cnn_stride_lst,
                encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
                activation_type=activation_type,
                encoder_kernel_init_scale=encoder_kernel_init_scale,
                use_rnn=use_rnn,
                rnn_hidden_layers=rnn_hidden_layers,
                rnn_hidden_size=rnn_hidden_size,
                rnn_cell_type=rnn_cell_type,
                use_rnn_layer_norm=use_rnn_layer_norm,
            ),
            policy_config=PolicyLayerConfig(
                action_space=action_space,
                output_kernel_init_scale=actor_output_kernel_init_scale,
            ),
            use_predictive_head=use_predictive_head,
            use_vib=use_vib,
            use_readout_layer=True,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            obs_space=obs_space if use_predictive_head else None,
            action_space=action_space if use_predictive_head else None,
            actor_head_hidden_size_lst=actor_head_hidden_size_lst,
            critic_head_hidden_size_lst=critic_head_hidden_size_lst,
            head_activation_type=activation_type,
            head_kernel_init_scale=head_kernel_init_scale,
            critic_output_kernel_init_scale=critic_output_kernel_init_scale,
            latent_dim=latent_dim,
            use_z_inference=use_z_inference,
            z_inference_hidden_sizes=tuple(z_inference_hidden_sizes),
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

    def __init__(self, actor_critic_fn: SharedActorCriticNet[FeatureExtractorConfig]) -> None:
        self.actor_critic_fn = actor_critic_fn

    def has_predictive_head(self) -> bool:
        return self.actor_critic_fn.use_predictive_head

    def has_vib(self) -> bool:
        return self.actor_critic_fn.use_vib

    def has_z_inference(self) -> bool:
        return self.actor_critic_fn.use_z_inference

    @property
    def latent_dim(self) -> int:
        return self.actor_critic_fn.latent_dim

    def _build_apply_rngs(self, rng: jax.Array) -> dict[str, jax.Array]:
        rng_dict: dict[str, jax.Array] = {}
        if self.has_vib():
            rng, rng_vib = jax.random.split(rng)
            rng_dict['vib'] = rng_vib
        if self.has_z_inference():
            rng, rng_z = jax.random.split(rng)
            rng_dict['z_infer'] = rng_z
        return rng_dict

    def init_rnn_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        rnn_state = self.actor_critic_fn.init_rnn_state(rng, batch_size)
        return {'rnn_state': rnn_state}

    def init_model_state(self, rng: jax.Array) -> PyTreeNode:
        rng, rng_rnn = jax.random.split(rng)
        rnn_state_example = self.init_rnn_state(rng_rnn, batch_size=1)
        obs_example = self.actor_critic_fn.feature_extractor_config.obs_space.example(batch_shape=(1,))
        action_example = self.actor_critic_fn.policy_config.action_space.example(batch_shape=(1,))

        rng, rng_params = jax.random.split(rng)
        rngs: dict[str, jax.Array] = {'params': rng_params}
        if self.has_vib():
            rng_params, rng_vib = jax.random.split(rng_params)
            rngs['params'] = rng_params
            rngs['vib'] = rng_vib
        if self.has_z_inference():
            rng_params, rng_z = jax.random.split(rng_params)
            rngs['params'] = rng_params
            rngs['z_infer'] = rng_z
        latent_kwargs = {}
        if self.latent_dim > 0:
            latent_kwargs['z'] = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
            latent_kwargs['is_p1'] = jnp.zeros((1,), dtype=jnp.bool_)
        params = self.actor_critic_fn.init(
            rngs if len(rngs) > 1 else rngs['params'],
            rnn_state_example['rnn_state'],
            obs_example,
            action=action_example,
            return_prediction=self.has_predictive_head(),
            **latent_kwargs,
        )
        return {'actor_critic_params': params}

    def forward(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array]:
        apply_kwargs: dict = {'return_prediction': False}
        if z is not None:
            apply_kwargs['z'] = z
        if is_p1 is not None:
            apply_kwargs['is_p1'] = is_p1
        rngs = self._build_apply_rngs(rng)
        next_rnn_state, pi, val, _ = self.actor_critic_fn.apply(
            model_state['actor_critic_params'],
            rnn_state['rnn_state'],
            obs,
            **apply_kwargs,
            **({'rngs': rngs} if rngs else {}),
        )
        return {'rnn_state': next_rnn_state}, pi, val

    def forward_with_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, dict[str, jax.Array]]:
        apply_kwargs: dict = {'return_prediction': False}
        if z is not None:
            apply_kwargs['z'] = z
        if is_p1 is not None:
            apply_kwargs['is_p1'] = is_p1
        rngs = self._build_apply_rngs(rng)
        next_rnn_state, pi, val, info = self.actor_critic_fn.apply(
            model_state['actor_critic_params'],
            rnn_state['rnn_state'],
            obs,
            **apply_kwargs,
            **({'rngs': rngs} if rngs else {}),
        )
        return {'rnn_state': next_rnn_state}, pi, val, info

    def forward_with_prediction(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        action: Action | None = None
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, Observation]:
        if not self.has_predictive_head():
            raise ValueError("forward_with_prediction requires a shared actor-critic model with predictive head enabled")
        if self.has_vib():
            rng, rng_vib = jax.random.split(rng)
            next_rnn_state, pi, val, pred_obs, _ = self.actor_critic_fn.apply(
                model_state['actor_critic_params'],
                rnn_state['rnn_state'],
                obs,
                action,
                return_prediction=True,
                rngs={'vib': rng_vib}
            )
        else:
            next_rnn_state, pi, val, pred_obs, _ = self.actor_critic_fn.apply(
                model_state['actor_critic_params'],
                rnn_state['rnn_state'],
                obs,
                action,
                return_prediction=True
            )
        return {'rnn_state': next_rnn_state}, pi, val, pred_obs

    def forward_with_prediction_and_aux(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        action: Action | None = None
    ) -> tuple[PyTreeNode, ActionDistribution, jax.Array, Observation, dict[str, jax.Array]]:
        if not self.has_predictive_head():
            raise ValueError("forward_with_prediction_and_aux requires a shared actor-critic model with predictive head enabled")
        if self.has_vib():
            rng, rng_vib = jax.random.split(rng)
            next_rnn_state, pi, val, pred_obs, info = self.actor_critic_fn.apply(
                model_state['actor_critic_params'],
                rnn_state['rnn_state'],
                obs,
                action,
                return_prediction=True,
                rngs={'vib': rng_vib}
            )
        else:
            next_rnn_state, pi, val, pred_obs, info = self.actor_critic_fn.apply(
                model_state['actor_critic_params'],
                rnn_state['rnn_state'],
                obs,
                action,
                return_prediction=True
            )
        return {'rnn_state': next_rnn_state}, pi, val, pred_obs, info

    def predict_next_obs(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        action: Action | None = None
    ) -> tuple[PyTreeNode, Observation]:
        next_rnn_state, _, _, pred_obs = self.forward_with_prediction(rng, model_state, rnn_state, obs, action)
        return next_rnn_state, pred_obs

    def get_action_distribution(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, ActionDistribution]:
        next_rnn_state, pi, _ = self.forward(rng, model_state, rnn_state, obs, z=z, is_p1=is_p1)
        return next_rnn_state, pi

    def get_value(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        rnn_state: PyTreeNode,
        obs: Observation,
        z: jax.Array | None = None,
        is_p1: jax.Array | None = None,
    ) -> tuple[PyTreeNode, jax.Array]:
        next_rnn_state, _, val = self.forward(rng, model_state, rnn_state, obs, z=z, is_p1=is_p1)
        return next_rnn_state, val


class SharedActorCriticModel(UnifiedSharedActorCriticModel):
    @classmethod
    def build_base_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> SharedActorCriticNet[BaseFeatureExtractorConfig]:
        return super().build_base_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            use_predictive_head=False,
            use_vib=use_vib,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            output_nn_depth=output_nn_depth,
        )

    @classmethod
    def build_simple_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        common_hidden_size: int = 64,
        cnn_kernel_size: int = 3,
        conv_layers: int = 3,
        use_rnn: bool = False,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> SharedActorCriticNet[SimpleFeatureExtractorConfig]:
        return super().build_simple_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            common_hidden_size=common_hidden_size,
            cnn_kernel_size=cnn_kernel_size,
            conv_layers=conv_layers,
            use_rnn=use_rnn,
            use_predictive_head=False,
            use_vib=use_vib,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            output_nn_depth=output_nn_depth,
        )

    @classmethod
    def build_configurable_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        cnn_feature_lst: tuple[int, ...] = (64, 64, 64),
        cnn_kernel_lst: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        cnn_stride_lst: tuple[int, ...] = (1, 1, 1),
        encoder_fc_hidden_size_lst: tuple[int, ...] = (64,),
        use_rnn: bool = True,
        rnn_hidden_layers: int = 1,
        rnn_hidden_size: int = 64,
        rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
        use_rnn_layer_norm: bool = False,
        activation_type: Activation.Type = 'relu',
        actor_head_hidden_size_lst: tuple[int, ...] = (),
        critic_head_hidden_size_lst: tuple[int, ...] = (),
        encoder_kernel_init_scale: float | None = None,
        head_kernel_init_scale: float | None = None,
        actor_output_kernel_init_scale: float | None = None,
        critic_output_kernel_init_scale: float | None = None,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        latent_dim: int = 0,
        use_z_inference: bool = False,
        z_inference_hidden_sizes: tuple[int, ...] = (256, 128),
        output_nn_depth: int | None = -1,
    ) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        return super().build_configurable_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            cnn_feature_lst=cnn_feature_lst,
            cnn_kernel_lst=cnn_kernel_lst,
            cnn_stride_lst=cnn_stride_lst,
            encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
            use_rnn=use_rnn,
            rnn_hidden_layers=rnn_hidden_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_cell_type=rnn_cell_type,
            use_rnn_layer_norm=use_rnn_layer_norm,
            activation_type=activation_type,
            actor_head_hidden_size_lst=actor_head_hidden_size_lst,
            critic_head_hidden_size_lst=critic_head_hidden_size_lst,
            encoder_kernel_init_scale=encoder_kernel_init_scale,
            head_kernel_init_scale=head_kernel_init_scale,
            actor_output_kernel_init_scale=actor_output_kernel_init_scale,
            critic_output_kernel_init_scale=critic_output_kernel_init_scale,
            use_predictive_head=False,
            use_vib=use_vib,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            latent_dim=latent_dim,
            use_z_inference=use_z_inference,
            z_inference_hidden_sizes=z_inference_hidden_sizes,
            output_nn_depth=output_nn_depth,
        )


class LatentConventionActorCriticModel(UnifiedSharedActorCriticModel):
    """Model wrapper for `LatentConventionActorCriticNet`.

    Reuses `UnifiedSharedActorCriticModel`'s `forward`, `forward_with_aux`,
    `init_model_state`, etc. The wrapper introspects
    `actor_critic_fn.feature_extractor_config.obs_space`,
    `actor_critic_fn.latent_dim`, `actor_critic_fn.use_z_inference`,
    `actor_critic_fn.use_vib`, and `actor_critic_fn.use_predictive_head`;
    `LatentConventionActorCriticNet` exposes all of these with matching
    semantics.
    """

    @classmethod
    def build_configurable_latent_convention_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        cnn_feature_lst: tuple[int, ...] = (64, 64, 64),
        cnn_kernel_lst: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        cnn_stride_lst: tuple[int, ...] = (1, 1, 1),
        encoder_fc_hidden_size_lst: tuple[int, ...] = (64,),
        rnn_hidden_layers: int = 1,
        rnn_hidden_size: int = 64,
        rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
        use_rnn_layer_norm: bool = False,
        activation_type: Activation.Type = 'relu',
        actor_head_hidden_size_lst: tuple[int, ...] = (),
        critic_head_hidden_size_lst: tuple[int, ...] = (),
        encoder_kernel_init_scale: float | None = None,
        head_kernel_init_scale: float | None = None,
        actor_output_kernel_init_scale: float | None = None,
        critic_output_kernel_init_scale: float | None = None,
        latent_dim: int = 4,
        z_inference_hidden_sizes: tuple[int, ...] = (256, 128),
        output_nn_depth: int | None = -1,
    ) -> LatentConventionActorCriticNet[ConfigurableFeatureExtractorConfig]:
        # The two feature-extractor configs share CNN/MLP hyperparameters but
        # differ in whether they own an RNN. Flax gives each extractor its own
        # weight scope, so parameters are independent despite shared config.
        inference_cfg = ConfigurableFeatureExtractorConfig(
            obs_space=obs_space,
            cnn_feature_lst=cnn_feature_lst,
            cnn_kernel_lst=cnn_kernel_lst,
            cnn_stride_lst=cnn_stride_lst,
            encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
            activation_type=activation_type,
            encoder_kernel_init_scale=encoder_kernel_init_scale,
            use_rnn=True,
            rnn_hidden_layers=rnn_hidden_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_cell_type=rnn_cell_type,
            use_rnn_layer_norm=use_rnn_layer_norm,
        )
        policy_cfg = ConfigurableFeatureExtractorConfig(
            obs_space=obs_space,
            cnn_feature_lst=cnn_feature_lst,
            cnn_kernel_lst=cnn_kernel_lst,
            cnn_stride_lst=cnn_stride_lst,
            encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
            activation_type=activation_type,
            encoder_kernel_init_scale=encoder_kernel_init_scale,
            use_rnn=False,
        )
        actor_critic_fn = LatentConventionActorCriticNet(
            feature_extractor_config=inference_cfg,
            policy_feature_extractor_config=policy_cfg,
            policy_config=PolicyLayerConfig(
                action_space=action_space,
                output_kernel_init_scale=actor_output_kernel_init_scale,
            ),
            actor_head_hidden_size_lst=actor_head_hidden_size_lst,
            critic_head_hidden_size_lst=critic_head_hidden_size_lst,
            head_activation_type=activation_type,
            head_kernel_init_scale=head_kernel_init_scale,
            critic_output_kernel_init_scale=critic_output_kernel_init_scale,
            latent_dim=latent_dim,
            z_inference_hidden_sizes=tuple(z_inference_hidden_sizes),
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn


class PredictiveSharedActorCriticModel(UnifiedSharedActorCriticModel):
    @classmethod
    def build_base_predictive_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> PredictiveSharedActorCriticNet[BaseFeatureExtractorConfig]:
        actor_critic_fn = PredictiveSharedActorCriticNet(
            feature_extractor_config=BaseFeatureExtractorConfig(obs_space=obs_space),
            policy_config=PolicyLayerConfig(action_space=action_space),
            use_vib=use_vib,
            use_readout_layer=True,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            obs_space=obs_space,
            action_space=action_space,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

    @classmethod
    def build_simple_predictive_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        common_hidden_size: int = 64,
        cnn_kernel_size: int = 3,
        conv_layers: int = 3,
        use_rnn: bool = False,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1
    ) -> PredictiveSharedActorCriticNet[SimpleFeatureExtractorConfig]:
        actor_critic_fn = PredictiveSharedActorCriticNet(
            feature_extractor_config=SimpleFeatureExtractorConfig(
                obs_space=obs_space,
                common_hidden_size=common_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                cnn_num_conv_layers=conv_layers,
                use_rnn=use_rnn,
            ),
            policy_config=PolicyLayerConfig(action_space=action_space),
            use_vib=use_vib,
            use_readout_layer=True,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            obs_space=obs_space,
            action_space=action_space,
        )

    @classmethod
    def build_configurable_predictive_shared_actor_critic(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        cnn_feature_lst: tuple[int, ...] = (64, 64, 64),
        cnn_kernel_lst: tuple[tuple[int, int], ...] = ((3, 3), (3, 3), (3, 3)),
        cnn_stride_lst: tuple[int, ...] = (1, 1, 1),
        encoder_fc_hidden_size_lst: tuple[int, ...] = (64,),
        use_rnn: bool = True,
        rnn_hidden_layers: int = 1,
        rnn_hidden_size: int = 64,
        rnn_cell_type: Literal['gru', 'lstm'] = 'gru',
        use_rnn_layer_norm: bool = False,
        activation_type: Activation.Type = 'relu',
        actor_head_hidden_size_lst: tuple[int, ...] = (),
        critic_head_hidden_size_lst: tuple[int, ...] = (),
        encoder_kernel_init_scale: float | None = None,
        head_kernel_init_scale: float | None = None,
        actor_output_kernel_init_scale: float | None = None,
        critic_output_kernel_init_scale: float | None = None,
        use_vib: bool = False,
        readout_size: int = 256,
        vib_free_bits: float = 0.0,
        vib_noise_mode: str = 'all',
        vib_stop_ppo_grads: bool = False,
        output_nn_depth: int | None = -1,
    ) -> SharedActorCriticNet[ConfigurableFeatureExtractorConfig]:
        return super().build_configurable_shared_actor_critic(
            obs_space=obs_space,
            action_space=action_space,
            cnn_feature_lst=cnn_feature_lst,
            cnn_kernel_lst=cnn_kernel_lst,
            cnn_stride_lst=cnn_stride_lst,
            encoder_fc_hidden_size_lst=encoder_fc_hidden_size_lst,
            use_rnn=use_rnn,
            rnn_hidden_layers=rnn_hidden_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_cell_type=rnn_cell_type,
            use_rnn_layer_norm=use_rnn_layer_norm,
            activation_type=activation_type,
            actor_head_hidden_size_lst=actor_head_hidden_size_lst,
            critic_head_hidden_size_lst=critic_head_hidden_size_lst,
            encoder_kernel_init_scale=encoder_kernel_init_scale,
            head_kernel_init_scale=head_kernel_init_scale,
            actor_output_kernel_init_scale=actor_output_kernel_init_scale,
            critic_output_kernel_init_scale=critic_output_kernel_init_scale,
            use_predictive_head=True,
            use_vib=use_vib,
            readout_size=readout_size,
            vib_free_bits=vib_free_bits,
            vib_noise_mode=vib_noise_mode,
            vib_stop_ppo_grads=vib_stop_ppo_grads,
            output_nn_depth=output_nn_depth,
        )
        cls.output_nn(actor_critic_fn, obs_space, output_nn_depth)
        return actor_critic_fn

class ActorWrapperAgent(RecurrentAgent):
    def __init__(self, actor_critic_model: BaseActorCriticModel) -> None:
        self.actor_critic_model = actor_critic_model
    
    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.actor_critic_model.init_actor_rnn_state(rng, batch_size)
    
    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        # It is expected done has shape [batch_size,] and agent_state has shape [batch_size, *agent_state_shape]
        default_state = self.init_agent_state(rng, batch_size=done.shape[0])
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )
    
    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict]:
        """
        Perform a single step in the environment with recurrent state updates.

        [Input]
            - rng (jax.Array): Random number generator state for stochastic policies.
            - model_state (PyTreeNode): The agent's current model state.
            - agent_state (PyTreeNode): The agent's recurrent state (e.g., RNN hidden states).
            - obs (Observation): The observation received from the environment.

        [Output]
            - next_agent_state (PyTreeNode): Updated agent state (e.g., next RNN hidden state).
            - action (Action): The action selected by the agent.
        """
        rng, rng_pi = jax.random.split(rng)
        next_actor_rnn_state, pi = self.actor_critic_model.get_action_distribution(rng_pi, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        return next_actor_rnn_state, action, {}

class ActorCriticPPOAgent(RecurrentPPOAgent):
    def __init__(self, actor_critic_model: SeparatedActorCriticModel | UnifiedSharedActorCriticModel) -> None:
        self.actor_critic_model = actor_critic_model
    
    def init_agent_state(self, rng: jax.Array, batch_size: int = 1) -> PyTreeNode:
        return self.actor_critic_model.init_rnn_state(rng, batch_size)
    
    def reset_agent_state(self, rng: jax.Array, agent_state: PyTreeNode, done: jax.Array) -> PyTreeNode:
        # It is expected done has shape [batch_size,] and agent_state has shape [batch_size, *agent_state_shape]
        default_state = self.init_agent_state(rng, batch_size=done.shape[0])
        return jax.tree_util.tree_map(
            lambda x, y: x * (1 - done.reshape(done.shape + (1,) * (x.ndim - done.ndim))) + y, agent_state, default_state
        )
    
    def step(
        self,
        rng: jax.Array,
        model_state: PyTreeNode,
        agent_state: PyTreeNode,
        obs: Observation
    ) -> tuple[PyTreeNode, Action, dict[str, jax.Array | Observation]]:
        """
        Perform a single step in the environment with recurrent state updates and additional outputs for policy optimization.

        [Input]
            - rng: Random number generator state for stochastic policies.
            - model_state: The agent's current model state.
            - agent_state: The agent's recurrent state (e.g., RNN hidden states).
            - obs: The observation received from the environment.
        
        The batch shape of agent_state and obs should be the same.

        [Output]
            - next_agent_state: Updated agent state (e.g., next RNN hidden state).
            - action: The action selected by the agent.
            - extra_data: Dict with keys:
                - log_p: Log probability of the selected action.
                - val: Estimated value function output at the current observation.
                - pred_next_obs: Predicted next observation (only for predictive actor-critic model).
        """
        rng, rng_forward = jax.random.split(rng)
        next_agent_state, pi, val = self.actor_critic_model.forward(rng_forward, model_state, agent_state, obs)
        rng, rng_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_sample)
        log_p = pi.log_prob(action)

        if self.actor_critic_model.has_predictive_head():
            rng, rng_pred = jax.random.split(rng)
            _, _, _, pred_next_obs = self.actor_critic_model.forward_with_prediction(
                rng_pred,
                model_state,
                agent_state,
                obs,
                action
            )
            return next_agent_state, action, {'log_p': log_p, 'val': val, 'pred_next_obs': pred_next_obs}

        return next_agent_state, action, {'log_p': log_p, 'val': val}

class ActorBCTrainer(BCTrainer):
    def __init__(
        self,
        actor_critic_model: SeparatedActorCriticModel | SharedActorCriticModel,
        feature_shared: bool = False,
        lr: float = 1e-4,
        grad_clip_norm: float = 10.0,
        num_iterations: int = 1,
        minibatch_seq_len: int = 10,
        num_minibatches: int = 1,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        See super().__init__() for more description.

        [Param (additional)]
            lr: Learning rate for the policy (actor) network to fit target data
            grad_clip_norm: Maximum norm for gradient clipping to prevent exploding gradients and improve stability.
        """
        super().__init__(
            num_iterations=num_iterations,
            minibatch_seq_len=minibatch_seq_len,
            num_minibatches=num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
        self.actor_critic_model = actor_critic_model
        self.feature_shared = feature_shared
        self.bc_config.update({
            'lr': lr,
            'grad_clip_norm': grad_clip_norm
        })
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.bc_config['grad_clip_norm']),
            optax.adam(
                learning_rate=self.bc_config['lr']
            )
        )

    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        if self.feature_shared:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_model.init_model_state(rng_init_model)
            actor_critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_model.actor_fn.apply,
                params=model_state['actor_critic_params'],
                tx=self.tx
            )
            return {'actor_critic_train_state': actor_critic_train_state}
        else:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_model.init_model_state(rng_init_model)
            actor_train_state = TrainState.create(
                apply_fn=self.actor_critic_model.actor_fn.apply,
                params=model_state['actor_params'],
                tx=self.tx
            )
            return {'actor_train_state': actor_train_state}
        
    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        if self.feature_shared:
            return {
                'actor_critic_params': trainer_state['actor_critic_train_state'].params,
            }
        else:
            return {
                'actor_params': trainer_state['actor_train_state'].params,
            }
    
    def evaluate_target_action(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [batch_size, minibatch_seq_len, *observation_shape].
            - target_action: Action to be evaluated with shape [batch_size, minibatch_seq_len, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the log probability and the entropy.
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).
        
        [Output]
            - log_p: Log probability of the taken action with shape [batch_size, minibatch_seq_len].
            - acc: Accuracy of predicting the target_action with shape [batch_size, minibatch_seq_len].
        """
        if self.feature_shared:
            rnn_state = {
                'rnn_state': aux_data['agent_state']['rnn_state'][:, 0],
            }
            next_rnn_state, pi = self.actor_critic_model.get_action_distribution(jax.random.key(0), model_state, rnn_state, obs)
        else:
            rnn_state = {
                'actor_rnn_state': aux_data['agent_state']['rnn_state'][:, 0],
            }
            next_rnn_state, pi = self.actor_critic_model.get_action_distribution(jax.random.key(0), model_state, rnn_state, obs)
        log_p = pi.log_prob(target_action)
        action_mode = pi.mode()
        eq_dict = {
            k: jnp.isclose(action_mode[k], target_action[k]).reshape(*pi[k].batch_shape, -1).all(-1)
            for k in target_action.keys()
        }
        acc = jnp.stack(list(eq_dict.values()), 0).all(0)
        return log_p, acc
    
    def model_gradient_update(
        self,
        trainer_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [batch_size, minibatch_seq_len, *observation_shape].
            - target_action: Target action  with shape [batch_size, minibatch_seq_len, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the policy loss.
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        model_state = self.model_state_from_trainer_state(trainer_state)
        (loss, info), grads = jax.value_and_grad(self.nll_loss, has_aux=True)(
            model_state,
            obs=obs,
            target_action=target_action,
            aux_data=aux_data
        )

        if self.feature_shared:
            info['grad_norm'] = global_norm(grads['actor_critic_params'])
            new_actor_critic_train_state = trainer_state['actor_critic_train_state'].apply_gradients(grads=grads['actor_critic_params'])
            return {'actor_critic_train_state': new_actor_critic_train_state}, info
        else:
            info['actor_grad_norm'] = global_norm(grads['actor_params'])
            new_actor_train_state = trainer_state['actor_train_state'].apply_gradients(grads=grads['actor_params'])
            return {'actor_train_state': new_actor_train_state}, info

class ActorCriticPPOTrainer(PPOTrainer):
    def __init__(
        self,
        actor_critic_model: SeparatedActorCriticModel | UnifiedSharedActorCriticModel,
        feature_shared: bool = False,
        lr: float = None,
        pi_lr: float = None,
        val_lr: float = None,
        lr_schedule_steps: int = None,
        lr_schedule: str = 'linear',
        lr_warmup_ratio: float = 0.0,
        grad_clip_norm: float = 10.0,
        val_loss_coef: float = 1.0,
        pi_loss_coef: float = 1.0,
        pred_loss_coef: float = 0.0,
        vib_kl_coef: float = 0.0,
        vib_beta_schedule: str = 'constant',
        vib_beta_warmup_steps: int = 0,
        vib_beta_cycle_steps: int = 0,
        z_infer_mse_coef: float = 0.0,
        z_infer_kl_coef: float = 0.0,
        entropy_coef: float = 0.0,
        gamma: float = 0.99,
        gae_lam: float = 0.95,
        ppo_epochs: int = 10,
        ratio_clip: float = 0.2,
        value_clip: float = None,
        use_advantage_normalization: bool = True,
        minibatch_seq_len: int = 10,
        num_minibatches: int = 1,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        See super().__init__() for more description.

        [Param (additional)]
            pi_lr: Learning rate for the policy (actor) network.
            val_lr: Learning rate for the value (critic) network.
            grad_clip_norm: Maximum norm for gradient clipping to prevent exploding gradients and improve stability.
            lr_schedule_steps: Total number of optimizer steps for the LR schedule. If None, a constant LR is used.
            lr_schedule: LR schedule type. 'constant' holds lr fixed; 'linear' decays linearly to 0;
                'warmup_cosine' linearly warms up to peak LR then cosine-decays to 0. Ignored when
                lr_schedule_steps is None.
            lr_warmup_ratio: Fraction of lr_schedule_steps used for linear warmup (only used when
                lr_schedule='warmup_cosine'). E.g. 0.05 = 5% warmup.
        """

        super().__init__(
            pi_loss_coef=pi_loss_coef,
            val_loss_coef=val_loss_coef,
            entropy_coef=entropy_coef,
            gamma=gamma,
            gae_lam=gae_lam,
            ppo_epochs=ppo_epochs,
            ratio_clip=ratio_clip,
            value_clip=value_clip,
            use_advantage_normalization=use_advantage_normalization,
            minibatch_seq_len=minibatch_seq_len,
            num_minibatches=num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks
        )
        self.actor_critic_model = actor_critic_model
        self.feature_shared = feature_shared
        if self.feature_shared:
            assert isinstance(self.actor_critic_model, UnifiedSharedActorCriticModel)
        else:
            assert isinstance(self.actor_critic_model, SeparatedActorCriticModel)

        if lr_schedule not in ('constant', 'linear', 'warmup_cosine'):
            raise ValueError(f"lr_schedule must be 'constant', 'linear', or 'warmup_cosine', got '{lr_schedule}'")

        self.ppo_config.update({
            'lr': lr,
            'pi_lr': pi_lr if pi_lr is not None else lr,
            'val_lr': val_lr if val_lr is not None else lr,
            'grad_clip_norm': grad_clip_norm,
            'lr_schedule_steps': lr_schedule_steps,
            'lr_schedule': lr_schedule,
            'lr_warmup_ratio': lr_warmup_ratio,
            'pred_loss_coef': pred_loss_coef,
            'vib_kl_coef': vib_kl_coef,
            'vib_beta_schedule': vib_beta_schedule,
            'vib_beta_warmup_steps': vib_beta_warmup_steps,
            'vib_beta_cycle_steps': vib_beta_cycle_steps,
            'z_infer_mse_coef': z_infer_mse_coef,
            'z_infer_kl_coef': z_infer_kl_coef,
        })
        if self.feature_shared:
            self.actor_critic_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=self._make_lr_schedule(
                        self.ppo_config['lr'],
                        self.ppo_config['lr_schedule_steps'],
                        self.ppo_config['lr_schedule'],
                        self.ppo_config['lr_warmup_ratio'],
                    )
                )
            )
        else:
            self.actor_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=self._make_lr_schedule(
                        self.ppo_config['pi_lr'],
                        self.ppo_config['lr_schedule_steps'],
                        self.ppo_config['lr_schedule'],
                        self.ppo_config['lr_warmup_ratio'],
                    )
                )
            )
            self.critic_tx = optax.chain(
                optax.clip_by_global_norm(self.ppo_config['grad_clip_norm']),
                optax.adam(
                    learning_rate=self._make_lr_schedule(
                        self.ppo_config['val_lr'],
                        self.ppo_config['lr_schedule_steps'],
                        self.ppo_config['lr_schedule'],
                        self.ppo_config['lr_warmup_ratio'],
                    )
                )
            )

    @staticmethod
    def _make_lr_schedule(
        peak_lr: float,
        total_steps: int | None,
        schedule: str,
        warmup_ratio: float,
    ):
        if total_steps is None or schedule == 'constant':
            return peak_lr
        if schedule == 'warmup_cosine':
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=peak_lr,
                warmup_steps=int(warmup_ratio * total_steps),
                decay_steps=total_steps,
                end_value=0.0,
            )
        return optax.linear_schedule(
            init_value=peak_lr,
            end_value=0.0,
            transition_steps=total_steps,
        )

    def init_trainer_state(self, rng: jax.Array) -> PyTreeNode:
        """
        Initialize the trainer state.
        """
        if self.feature_shared:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_model.init_model_state(rng)
            actor_critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_model.actor_critic_fn.apply,
                params=model_state['actor_critic_params'],
                tx=self.actor_critic_tx
            )
            return {'actor_critic_train_state': actor_critic_train_state}
        else:
            rng, rng_init_model = jax.random.split(rng)
            model_state = self.actor_critic_model.init_model_state(rng_init_model)
            actor_train_state = TrainState.create(
                apply_fn=self.actor_critic_model.actor_fn.apply,
                params=model_state['actor_params'],
                tx=self.actor_tx
            )
            critic_train_state = TrainState.create(
                apply_fn=self.actor_critic_model.critic_fn.apply,
                params=model_state['critic_params'],
                tx=self.critic_tx
            )
            return {'actor_train_state': actor_train_state, 'critic_train_state': critic_train_state}

    def model_state_from_trainer_state(self, trainer_state: PyTreeNode) -> PyTreeNode:
        """
        Retreive the model state from the trainer state.
        """
        if self.feature_shared:
            return {'actor_critic_params': trainer_state['actor_critic_train_state'].params}
        else:
            return {
                'actor_params': trainer_state['actor_train_state'].params,
                'critic_params': trainer_state['critic_train_state'].params
            }
    
    def evaluate_action(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        action: Action,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, float | jax.Array]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [batch_size, minibatch_seq_len, *observation_shape].
            - action: Action to be evaluated with shape [batch_size, minibatch_seq_len, *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the log probability and the entropy.
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).
        
        [Output]
            - log_p: Log probability of the taken action with shape [batch_size, minibatch_seq_len].
            - entropy: Entropy of the policy distribution with shape [batch_size, minibatch_seq_len].
            - value: Estimated value function output with shape [batch_size, minibatch_seq_len].
            - info: Extra logging/debugging info; values should be floats or scalar jax.Arrays.
        """
        if self.feature_shared:
            rnn_state = {'rnn_state': jax.tree_util.tree_map(lambda x: x[:, 0], aux_data['agent_state']['rnn_state'])}
            latent_kwargs: dict = {}
            if (
                isinstance(self.actor_critic_model, UnifiedSharedActorCriticModel)
                and self.actor_critic_model.latent_dim > 0
            ):
                z_true = aux_data.get('z_true', None)
                is_p1 = aux_data.get('is_p1', None)
                if z_true is not None:
                    latent_kwargs['z'] = z_true
                if is_p1 is not None:
                    latent_kwargs['is_p1'] = is_p1
            if self.actor_critic_model.has_predictive_head():
                next_rnn_state, pi, val, pred_next_obs, model_info = self.actor_critic_model.forward_with_prediction_and_aux(
                    rng,
                    model_state,
                    rnn_state,
                    obs,
                    action
                )
            else:
                next_rnn_state, pi, val, model_info = self.actor_critic_model.forward_with_aux(
                    rng,
                    model_state,
                    rnn_state,
                    obs,
                    **latent_kwargs,
                )
                pred_next_obs = None
        else:
            rnn_state = {
                'actor_rnn_state': jax.tree_util.tree_map(lambda x: x[:, 0], aux_data['agent_state']['actor_rnn_state']),
                'critic_rnn_state': jax.tree_util.tree_map(lambda x: x[:, 0], aux_data['agent_state']['critic_rnn_state']),
            }
            next_rnn_state, pi, val = self.actor_critic_model.forward(rng, model_state, rnn_state, obs)
            pred_next_obs = None
            model_info = {}
        log_p = pi.log_prob(action)
        entropy = pi.entropy()
        info = get_distribution_info(pi)

        vib_kl = model_info.get('vib_kl', None)
        if vib_kl is not None:
            if 'mask' in aux_data:
                info['vib_kl'] = (vib_kl * aux_data['mask']).mean()
            else:
                info['vib_kl'] = vib_kl.mean()

        z_infer_mu = model_info.get('z_infer_mu', None)
        if z_infer_mu is not None:
            z_true = aux_data.get('z_true', None)
            is_p1 = aux_data.get('is_p1', None)
            if z_true is not None:
                # MSE is applied only where the true z is observable (P0 slice, is_p1=False).
                sq_err = jnp.square(z_infer_mu - z_true).mean(axis=-1)
                weight = jnp.ones_like(sq_err)
                if is_p1 is not None:
                    # is_p1 has shape [B, T]; same as sq_err
                    weight = weight * (1.0 - is_p1.astype(sq_err.dtype))
                if 'mask' in aux_data:
                    weight = weight * aux_data['mask']
                denom = jnp.maximum(weight.sum(), 1.0)
                info['z_infer_mse'] = (sq_err * weight).sum() / denom
            z_infer_kl = model_info.get('z_infer_kl', None)
            if z_infer_kl is not None:
                weight = jnp.ones_like(z_infer_kl)
                if is_p1 is not None:
                    weight = weight * (1.0 - is_p1.astype(z_infer_kl.dtype))
                if 'mask' in aux_data:
                    weight = weight * aux_data['mask']
                denom = jnp.maximum(weight.sum(), 1.0)
                info['z_infer_kl'] = (z_infer_kl * weight).sum() / denom

        target_next_obs = aux_data.get('next_obs', None)
        if pred_next_obs is not None and target_next_obs is not None:
            pred_loss_dict: dict[str, jax.Array] = {}
            for obs_key in pred_next_obs.keys():
                if obs_key not in target_next_obs:
                    continue
                sq_err = jnp.square(pred_next_obs[obs_key] - target_next_obs[obs_key]).reshape(*pred_next_obs[obs_key].shape[:2], -1).mean(-1)
                if 'mask' in aux_data:
                    key_loss = (sq_err * aux_data['mask']).mean()
                else:
                    key_loss = sq_err.mean()
                pred_loss_dict[obs_key] = key_loss
                info[f'pred_loss[{obs_key}]'] = key_loss
            if len(pred_loss_dict) > 0:
                info['pred_loss'] = jnp.mean(jnp.stack(list(pred_loss_dict.values()), axis=0))

        return log_p, entropy, val, info

    def ppo_loss(
        self,
        model_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[float, dict[str, float]]:
        loss, info = super().ppo_loss(model_state, buffer, advantage, target_val, aux_data, rng)
        pred_loss = info.get('pred_loss', None)
        if pred_loss is not None:
            loss = loss + self.ppo_config['pred_loss_coef'] * pred_loss
            info['pred_loss_term'] = self.ppo_config['pred_loss_coef'] * pred_loss
        vib_kl = info.get('vib_kl', None)
        if vib_kl is not None:
            vib_kl_coef = self.ppo_config['vib_kl_coef']
            schedule = self.ppo_config['vib_beta_schedule']
            optimizer_step = aux_data.get('optimizer_step', None)
            if optimizer_step is not None and schedule != 'constant':
                if schedule == 'linear_warmup':
                    warmup_steps = self.ppo_config['vib_beta_warmup_steps']
                    effective_beta = vib_kl_coef * jnp.clip(optimizer_step / jnp.maximum(warmup_steps, 1), 0.0, 1.0)
                elif schedule == 'cyclical':
                    cycle_steps = self.ppo_config['vib_beta_cycle_steps']
                    cycle_pos = (optimizer_step % jnp.maximum(cycle_steps, 1)) / jnp.maximum(cycle_steps, 1)
                    effective_beta = vib_kl_coef * 0.5 * (1 - jnp.cos(jnp.pi * cycle_pos))
                else:
                    effective_beta = vib_kl_coef
            else:
                effective_beta = vib_kl_coef
            loss = loss + effective_beta * vib_kl
            info['vib_loss_term'] = effective_beta * vib_kl
            info['vib_effective_beta'] = effective_beta
        z_infer_mse = info.get('z_infer_mse', None)
        if z_infer_mse is not None:
            z_mse_coef = self.ppo_config['z_infer_mse_coef']
            loss = loss + z_mse_coef * z_infer_mse
            info['z_infer_mse_term'] = z_mse_coef * z_infer_mse
        z_infer_kl = info.get('z_infer_kl', None)
        if z_infer_kl is not None:
            z_kl_coef = self.ppo_config['z_infer_kl_coef']
            loss = loss + z_kl_coef * z_infer_kl
            info['z_infer_kl_term'] = z_kl_coef * z_infer_kl
        info['loss'] = loss
        return loss, info

    def model_gradient_update(
        self,
        trainer_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict[str, PyTreeNode],
        rng: jax.Array
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        [Input]
            - trainer_state: Trainer state (e.g., model parameters, optimizer states, step, etc.).
            - buffer: PPO transition buffer with attributes having shape [batch_size, minibatch_seq_len, *attribute_shape].
            - target_val: Target value function with shape [batch_size, minibatch_seq_len].
            - advantage: Advantage estimates with shape [batch_size, minibatch_seq_len].
            - aux_data: Dictionary containing any additional information.
                - agent_state: The agent's recurrent state (e.g., RNN hidden states with shape [batch_size, minibatch_seq_len, *rnn_state_shape]).
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        model_state = self.model_state_from_trainer_state(trainer_state)
        if self.feature_shared:
            aux_data = {**aux_data, 'optimizer_step': trainer_state['actor_critic_train_state'].step}
        (loss, info), grads = jax.value_and_grad(self.ppo_loss, has_aux=True)(
            model_state,
            buffer=buffer,
            advantage=advantage,
            target_val=target_val,
            aux_data=aux_data,
            rng=rng
        )

        if self.feature_shared:
            info['grad_norm'] = global_norm(grads['actor_critic_params'])
            new_actor_critic_train_state = trainer_state['actor_critic_train_state'].apply_gradients(grads=grads['actor_critic_params'])
            return {'actor_critic_train_state': new_actor_critic_train_state}, info
        else:
            info['actor_grad_norm'] = global_norm(grads['actor_params'])
            info['critic_grad_norm'] = global_norm(grads['critic_params'])
            new_actor_train_state = trainer_state['actor_train_state'].apply_gradients(grads=grads['actor_params'])
            new_critic_train_state = trainer_state['critic_train_state'].apply_gradients(grads=grads['critic_params'])
            return {'actor_train_state': new_actor_train_state, 'critic_train_state': new_critic_train_state}, info
