from abc import abstractmethod

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from ..env.spaces import Observation, Action
from ..utils import split_into_minibatch, split_rng_to_list
from .base_trainer import BaseTrainer


class BCTrainer(BaseTrainer):
    def __init__(
        self,
        num_iterations: int = 1,
        minibatch_seq_len: int = 10,
        num_minibatches: int = None,
        minibatch_num_chunks: int = None
    ) -> None:
        """
        Behavior Cloning Algorithm Trainer

        To make this work for a specific model, inherit from this class and implement the model-specific gradient update method (e.g., model_gradient_update).
        The trainer_state (which is also initialized by a model-specific function init_trainer_state) usually contains the current model parameters along with the optimizer state.

        [Param: BC Config]
            minibatch_seq_len: Length of each data chunk used in training (e.g., episode length or fixed segment length).
            num_minibatches: Number of minibatches to divide the training data into.
            minibatch_num_chunks: Optional number of chunks per minibatch (if None, default behavior is applied).
        """

        self.bc_config = {
            'num_iterations': num_iterations,
            'minibatch_seq_len': minibatch_seq_len,
            'num_minibatches': num_minibatches,
            'minibatch_num_chunks': minibatch_num_chunks
        }

    @abstractmethod
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
        
        [Output]
            - log_p: Log probability of the taken action with shape [batch_size, minibatch_seq_len].
            - acc: Accuracy of predicting the target_action with shape [batch_size, minibatch_seq_len].
        """
        ...

    def nll_loss(
        self,
        model_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[float, dict[str, float]]:
        """
        [Input]
            - model_state: Model state (e.g., model parameters).
            - obs: Observation with shape [batch_size, minibatch_seq_len, *observation_shape].
            - target_action: Target action  with shape [batch_size, minibatch_seq_len *action_shape].
            - aux_data: Dictionary containing any additional information needed for computing the policy loss.
        
        [Output]
            - loss: Scalar PPO loss value.
            - loss_info: Dictionary containing auxiliary loss-related statistics.
        
        [Notation]
            - L: minibatch_seq_len
            - B: batch_size
        """
        info = {}
        log_p, acc = self.evaluate_target_action(model_state, obs, target_action, aux_data)  # [B, L], [B, L]

        if 'mask' in aux_data:
            loss = (-log_p * aux_data['mask']).mean()
            acc = (acc * aux_data['mask']).sum() / (aux_data['mask'].sum() + 1e-5)
        else:
            loss = -log_p.mean()
            acc = acc.mean()
        info.update({
            'nll_loss': loss,
            'acc': acc
        })
        return loss, info
    
    @abstractmethod
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
        
        [Output]
            - new_trainer_state: Updated trainer state after one gradient update using the PPO loss.
            - optim_log: Training logs.
        """
        ...

    def bc_epoch(
        self,
        rng: jax.Array,
        trainer_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        Perform one BC epoch and return the updated trainer state and logs. Note that the osbervation/action is expected to have shape [sequence_length, batch_size, *data_attribute_shape].
        """
        rng, rng_split = jax.random.split(rng)
        obs_batch_stack, target_action_batch_stack, aux_data_batch_stack = split_into_minibatch(
            rng=rng_split,
            data=(obs, target_action, aux_data),
            minibatch_seq_len=self.bc_config['minibatch_seq_len'],
            num_minibatches=self.bc_config['num_minibatches'],
            minibatch_num_chunks=self.bc_config['minibatch_num_chunks']
        )

        def bc_minibatch_update(trainer_state: PyTreeNode, data: tuple[Observation, Action, dict[str, PyTreeNode]]) -> tuple[PyTreeNode, dict[str, jax.Array]]:
            obs, target_action, aux_data = data
            return self.model_gradient_update(trainer_state, obs, target_action, aux_data)
        new_trainer_state, optim_log = jax.lax.scan(bc_minibatch_update, trainer_state, (obs_batch_stack, target_action_batch_stack, aux_data_batch_stack))
        optim_log = jax.tree_util.tree_map(jnp.mean, optim_log)
        return new_trainer_state, optim_log

    def fit(
        self,
        rng: jax.Array,
        trainer_state: PyTreeNode,
        obs: Observation,
        target_action: Action,
        aux_data: dict[str, PyTreeNode]
    ) -> tuple[PyTreeNode, dict[str, jax.Array]]:
        """
        Run full bahavior cloning (multiple epochs) and return the updated trainer state and logs. 
        """
        def bc_epoch_update(trainer_state: PyTreeNode, rng: jax.Array) -> tuple[PyTreeNode, dict[str, jax.Array]]:
            return self.bc_epoch(rng, trainer_state, obs, target_action, aux_data)
        rng, rng_bc_epochs = split_rng_to_list(rng, self.bc_config['num_iterations'])
        rng_bc_epochs = jnp.stack(rng_bc_epochs, axis=0)
        new_trainer_state, optim_log = jax.lax.scan(bc_epoch_update, trainer_state, rng_bc_epochs, self.bc_config['num_iterations'])
        return new_trainer_state, optim_log
