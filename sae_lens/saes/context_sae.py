"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    _disable_hooks,
)

from sae_lens.saes.topk_sae import (
    SparseHookPoint,
    _sparse_matmul_nd,
    _fold_norm_topk,
    _init_weights_topk,
    _calculate_topk_aux_acts,
)

class SplitTopK(nn.Module):
    """
    A TopK activation that zeroes out all but the top K elements along the last dimension,
    and applies ReLU to the top K elements.
    SplitTopK also partitions the activations based on the partition indices. Each partition receives a k budget.
    """

    def __init__(
        self,
        k_budgets: list[int],
        partition_sizes : list[int],
    ):
        if len(k_budgets) != len(partition_sizes):
            raise ValueError("Error: Mismatch size between given k budgets and partition sizes (need n sizes for n budgets)")

        super().__init__()
        self.k_budgets = k_budgets
        self.partition_sizes = partition_sizes
        self.compiled_partitioned_top_k = torch.compile(self._partitioned_top_k)
    
    def _partitioned_top_k(self, 
                           x : torch.Tensor, 
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        x_partitioned = torch.split(x, self.partition_sizes, dim=-1)

        # Collect 
        all_vals = []
        all_indices = []

        current_offset = 0

        for partition, k, sizes in zip(x_partitioned, self.k_budgets, self.partition_sizes):
            vals, indices = torch.topk(partition, k=k, dim=-1, sorted=False)
            
            # Shift indices back to original position in x
            indices = indices + current_offset

            all_vals.append(vals)
            all_indices.append(indices)

            current_offset += sizes

        return torch.cat(all_vals, dim=-1), torch.cat(all_indices, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) Select top K elements along the last dimension.
        2) Apply ReLU.
        3) Zero out all other entries.
        """
        
        topk_values, topk_indices = self.compiled_partitioned_top_k(x)
        # Take top k on each partition
        

        values = topk_values.relu()

        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result

@dataclass
class ContextSAEConfig(SAEConfig):
    """
    Configuration class for ContextSAE inference.

    Args:
        k (int): Number of top features to keep active during inference. Only the top k
            features with the highest pre-activations will be non-zero. Defaults to 100.
        rescale_acts_by_decoder_norm (bool): Whether to treat the decoder as if it was
            already normalized. This affects the topk selection by rescaling pre-activations
            by decoder norms. Requires that the SAE was trained this way. Defaults to False.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name, etc.).
            Inherited from SAEConfig.
    """
    pct_context_features: float = 1/2 # By default, use half of the feature set for context and half for token-specific. 
    k_context: int = 64
    k_token: int = 64

    rescale_acts_by_decoder_norm: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "split-topk"

class ContextSAE(SAE[ContextSAEConfig]):
    """
    An inference-only sparse autoencoder using a "topk" activation function.
    It uses linear encoder and decoder layers, applying the TopK activation
    to the hidden pre-activation in its encode step.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: ContextSAEConfig, use_error_term: bool = False):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the forward pass.
        """
        super().__init__(cfg, use_error_term)

        self.n_context_features = int(self.cfg.d_sae * self.cfg.pct_context_features)
        self.n_token_features = self.cfg.d_sae - self.n_context_features

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_topk(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input x into feature activations.
        Uses topk activation under the hood.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        # The BaseSAE already sets self.activation_fn to TopK(...) if config requests topk.
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))
    
    def encode_partitioned(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same as standard encode function but returns a dictionary containing
         context: context features
          token: token specific features"""
        enc = self.encode(x)

        return {
            "context": enc[0:self.n_context_features],
            "token": enc[0:self.n_context_features:]
        }

    def decode(
        self,
        feature_acts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs the input from topk feature activations.
        Applies optional finetuning scaling, hooking to recons, out normalization,
        and optional head reshaping.
        """
        # Handle sparse tensors using efficient sparse matrix multiplication
        if self.cfg.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts / self.W_dec.norm(dim=-1)
        if feature_acts.is_sparse:
            sae_out_pre = _sparse_matmul_nd(feature_acts, self.W_dec) + self.b_dec
        else:
            sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SplitTopK(
            [self.cfg.k_context, self.cfg.k_token], 
            [self.n_context_features, 
             self.n_token_features])

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for TopKSAEs when rescale_acts_by_decoder_norm is False, as this may change the topk activations"
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)


@dataclass
class ContextTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a ContextTrainingSAE.

    Args:
        k (int): Number of top features to keep active. Only the top k features
            with the highest pre-activations will be non-zero. Defaults to 100.
        use_sparse_activations (bool): Whether to use sparse tensor representations
            for activations during training. This can reduce memory usage and improve
            performance when k is small relative to d_sae, but is only worthwhile if
            using float32 and not using autocast. Defaults to False.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. This loss helps prevent neuron death
            in TopK SAEs by having dead neurons reconstruct the residual error from
            live neurons. Defaults to 1.0.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already normalized.
            This is a good idea since decoder norm can randomly drift during training, and this
            affects what the topk activations will be. Defaults to True.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            0.1 corresponds to the "heuristic" initialization from Anthropic's April update.
            Use None to disable. Inherited from TrainingSAEConfig. Defaults to 0.1.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
            Defaults to "float32".
        device (str): Device to place the SAE on. Inherited from SAEConfig.
            Defaults to "cpu".
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig. Defaults to True.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
            Defaults to "none".
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
            Defaults to "none".
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    pct_context_features: float = 1/2 # By default, use half of the feature set for context and half for token-specific. 
    k_context: int = 64
    k_token: int = 64

    # use_sparse_activations: bool = False # Not implemented for SplitTopK
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "split-topk"


class ContextTrainingSAE(TrainingSAE[ContextTrainingSAEConfig]):
    """
    Context (SplitTopK) variant with training functionality. Calculates a topk-related auxiliary loss, etc.
    For residual stream h_t, we predict context latents and token-specific latents. We reconstruct as normal.
    For residual stream h_{t+k}, we predict only token-specific latents. We reconstruct from its token-specific latents and h_t's context latents.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: ContextTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

        self.n_context_features = int(self.cfg.d_sae * self.cfg.pct_context_features)
        self.n_token_features = self.cfg.d_sae - self.n_context_features

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Similar to the base training method: calculate pre-activations, then apply SplitTopK.
        We additionally must assume that we receive inputs like [batch_size, 2, d_in] as we split it up. 
        """
        sae_in_t = self.process_sae_in(x[:, 0, :])
        sae_in_ts = self.process_sae_in(x[:, 1, :])

        # Only return hidden_pre for token t
        hidden_pre = self.hook_sae_acts_pre(sae_in_t @ self.W_enc + self.b_enc)
        enc_conext_t = hidden_pre[:, :self.n_context_features]


        # Get only token specific component via slicing
        enc_ts_token = sae_in_ts @ self.W_enc[:, self.n_context_features:] + self.b_enc[self.n_context_features:]
        enc_ts = torch.cat([enc_conext_t, enc_ts_token], dim=-1)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
            enc_ts = enc_ts * self.W_dec.norm(dim=-1)
        
        act_t = self.activation_fn(hidden_pre)
        act_ts = self.activation_fn(enc_ts)

        act_all = torch.stack([act_t, act_ts], dim=1)

        # Apply the SplitTopK activation function
        feature_acts = self.hook_sae_acts_post(act_all)

        # [batch_size, 2, d_in], [batch_size, d_in]
        return feature_acts, hidden_pre

    @override
    def decode(
        self,
        feature_acts: torch.Tensor, #[batch_size, 2, d_in]
    ) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        # Handle sparse tensors using efficient sparse matrix multiplication
        if self.cfg.rescale_acts_by_decoder_norm:
            # need to multiply by the inverse of the norm because division is illegal with sparse tensors
            feature_acts = feature_acts * (1 / self.W_dec.norm(dim=-1))

        # For now, assume activations aren't sparse (this was difficult to implement with SplitTopK)
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head) # May pose an issue? 

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        topk_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in[:, 0, :], # Only calculate on first token residual stream, same for hidden_pre
            sae_out=sae_out[:, 0, :],
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": topk_loss}

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        if not self.cfg.rescale_acts_by_decoder_norm:
            raise NotImplementedError(
                "Folding W_dec_norm is not safe for TopKSAEs when rescale_acts_by_decoder_norm is False, as this may change the topk activations"
            )
        _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SplitTopK(
            [self.cfg.k_context, self.cfg.k_token], 
            [self.n_context_features, 
             self.n_token_features])

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate TopK auxiliary loss.

        This auxiliary loss encourages dead neurons to learn useful features by having
        them reconstruct the residual error from the live neurons. It's a key part of
        preventing neuron death in TopK SAEs.
        """
        # Mostly taken from https://github.com/EleutherAI/sae/blob/main/sae/sae.py, except without variance normalization
        # NOTE: checking the number of dead neurons will force a GPU sync, so performance can likely be improved here
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)
        residual = (sae_in[:, 0, :] - sae_out[:, 0, :]).detach()

        # Heuristic from Appendix B.1 in the paper
        k_aux = sae_in[:, 0, :].shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = _calculate_topk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)
        if self.cfg.rescale_acts_by_decoder_norm:
            _fold_norm_topk(
                W_enc=state_dict["W_enc"],
                b_enc=state_dict["b_enc"],
                W_dec=state_dict["W_dec"],
            )