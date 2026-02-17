"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

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
    TrainStepOutput,
)
from sae_lens.saes.topk_sae import (
    SparseHookPoint,
    _calculate_topk_aux_acts,
    _fold_norm_topk,
    _init_weights_topk,
    _sparse_matmul_nd,
)

KThresh = Union[int, float, torch.Tensor, Callable[[], int | float | torch.Tensor]]  # noqa: UP007


class SplitActivation(nn.Module, ABC):
    """Abstract base class for a split activation function"""

    def __init__(
        self,
        k_thresholds: list[KThresh],
        partition_sizes: list[int],
    ):
        if len(k_thresholds) != len(partition_sizes):
            raise ValueError(
                "Error: Mismatch size between given k budgets and partition sizes (need n sizes for n budgets)"
            )
        super().__init__()

        self.k_thresholds = k_thresholds
        self.partition_sizes = partition_sizes

    @abstractmethod
    def _part_act_fn(
        self, x_part: torch.Tensor, kt: int | float | torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_partitioned = torch.split(x, self.partition_sizes, dim=-1)

        all_acts = []
        for partition, kt in zip(x_partitioned, self.k_thresholds):
            kt_val = kt() if callable(kt) else kt
            all_acts.append(self._part_act_fn(partition, kt_val))

        return torch.cat(all_acts, dim=-1)


class SplitJumpReLU(SplitActivation):
    """Standard JumpReLU activation function split across multiple partitions"""

    def _part_act_fn(self, x_part: torch.Tensor, kt) -> torch.Tensor:  # type: ignore
        return (x_part * (x_part > kt)).to(x_part)


class SplitBatchTopK(SplitActivation):
    """Standard BatchTopK activation function split across multiple partitions"""

    def _part_act_fn(self, x_part: torch.Tensor, kt) -> torch.Tensor:  # type: ignore
        n_samples = x_part.shape[:-1].numel()
        flat_acts = x_part.relu().flatten()

        vals, indices = torch.topk(flat_acts, int(kt * n_samples), dim=-1)

        return (
            torch.zeros_like(flat_acts).scatter(-1, indices, vals).reshape(x_part.shape)
        )


class SplitTopK(SplitActivation):
    def _part_act_fn(self, x_part: torch.Tensor, kt):  # type: ignore
        k = int(kt)
        x_relu = x_part.relu()
        topk_vals, topk_ids = torch.topk(x_relu, k, dim=-1)
        result = torch.zeros_like(x_relu)
        result.scatter_(-1, topk_ids, topk_vals)
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

    pct_context_features: float = 2 / 10  # Use similar ratio to T-SAE paper
    k_context: int = 4  # Probably there are less context related features in general
    k_token: int = 16

    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "context_sae"


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
        self.n_context_features = int(cfg.d_sae * cfg.pct_context_features)
        self.n_token_features = cfg.d_sae - self.n_context_features

        super().__init__(cfg, use_error_term)

        self.register_buffer("threshold_context", torch.tensor(0.0, dtype=torch.double))
        self.register_buffer("threshold_token", torch.tensor(0.0, dtype=torch.double))

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_topk(self)  # type: ignore

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
            "context": enc[..., 0 : self.n_context_features],
            "token": enc[..., self.n_context_features :],
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
        # return SplitJumpReLU(
        #     [
        #         lambda: self.threshold_context,
        #         lambda: self.threshold_token,
        #     ],  # type: ignore
        #     [self.n_context_features, self.n_token_features],
        # )

        return SplitTopK(
            [self.cfg.k_context, self.cfg.k_token],
            [self.n_context_features, self.n_token_features],
        )

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

    pct_context_features: float = 2 / 10
    k_context: int = 4
    k_token: int = 16
    topk_threshold_lr: float = 0.01

    # use_sparse_activations: bool = False # Not implemented for SplitTopK
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "context_sae"


class ContextTrainingSAE(TrainingSAE[ContextTrainingSAEConfig]):
    """
    Context (SplitTopK) variant with training functionality. Calculates a topk-related auxiliary loss, etc.
    For residual stream h_t, we predict context latents and token-specific latents. We reconstruct as normal.
    For residual stream h_{t+k}, we predict only token-specific latents. We reconstruct from its token-specific latents and h_t's context latents.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: ContextTrainingSAEConfig, use_error_term: bool = False):
        # This must go before super.__init__ because getting the activation_fn relies on it
        self.n_context_features = int(cfg.d_sae * cfg.pct_context_features)
        self.n_token_features = cfg.d_sae - self.n_context_features

        super().__init__(cfg, use_error_term)
        self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

        self.register_buffer(
            "threshold_context",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

        self.register_buffer(
            "threshold_token",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)  # type: ignore

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
        enc_conext_t = hidden_pre[:, : self.n_context_features]

        # Get only token specific component via slicing
        enc_ts_token = (
            sae_in_ts @ self.W_enc[:, self.n_context_features :]
            + self.b_enc[self.n_context_features :]
        )
        enc_ts = torch.cat([enc_conext_t, enc_ts_token], dim=-1)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
            enc_ts = enc_ts * self.W_dec.norm(dim=-1)

        act_t = self.activation_fn(hidden_pre)
        act_ts = self.activation_fn(enc_ts)

        act_all = torch.stack([act_t, act_ts], dim=1)

        feature_acts = self.hook_sae_acts_post(act_all)

        # [batch_size, 2, d_in], [batch_size, d_in]
        return feature_acts, hidden_pre

    @override
    def decode(
        self,
        feature_acts: torch.Tensor,  # [batch_size, 2, d_in]
    ) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """

        if self.cfg.rescale_acts_by_decoder_norm:
            # need to multiply by the inverse of the norm because division is illegal with sparse tensors
            feature_acts = feature_acts * (1 / self.W_dec.norm(dim=-1))

        # For now, assume activations aren't sparse (this was difficult to implement with SplitTopK)
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)  # May pose an issue?

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            raise NotImplementedError(
                "Context SAE does not support use_error_term flag"
            )
            # with torch.no_grad():
            #     # Recompute without hooks for true error term
            #     with _disable_hooks(self):
            #         feature_acts_clean = self.encode(x)
            #         x_reconstruct_clean = self.decode(feature_acts_clean)
            #     sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            # sae_out = sae_out + sae_error

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
            sae_in=step_input.sae_in[:, 0, :],
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
        # _fold_norm_topk(W_dec=self.W_dec, b_enc=self.b_enc, W_enc=self.W_enc)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SplitTopK(
            [self.cfg.k_context, self.cfg.k_token],
            [self.n_context_features, self.n_token_features],
        )

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
        residual = (sae_in - sae_out).detach()

        # Heuristic from Appendix B.1 in the paper
        k_aux = sae_in.shape[-1] // 2

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
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        self.update_topk_threshold(feature_acts)

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)

        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts[
                :, 0, :
            ],  # This technically isn't actually used but I slice it to be safe
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        l0_context_thresh = (
            torch.count_nonzero(
                hidden_pre[..., 0 : self.n_context_features] > self.threshold_context
            )
            / hidden_pre.shape[0]
        )

        l0_token_thresh = (
            torch.count_nonzero(
                hidden_pre[..., self.n_context_features :] > self.threshold_token
            )
            / hidden_pre.shape[0]
        )

        metrics = {
            "context_threshold": self.threshold_context,
            "token_threshold": self.threshold_token,
            "l0_context_thresh": l0_context_thresh,
            "l0_token_thresh": l0_token_thresh,
        }

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts.flatten(
                0, 1
            ),  # Flatten for the dead neuron mask to work
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
            metrics=metrics,  # type: ignore
        )

    @torch.no_grad()
    def threshold_helper(
        self, acts_topk_sliced: torch.Tensor, previous_threshold: torch.Tensor
    ):
        positive_mask = acts_topk_sliced > 0
        lr = self.cfg.topk_threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(previous_threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = (
                    acts_topk_sliced[positive_mask].min().to(previous_threshold.dtype)
                )
                return (1 - lr) * previous_threshold + lr * min_positive
            return previous_threshold

    @torch.no_grad()
    def update_topk_threshold(self, acts_topk: torch.Tensor) -> None:
        self.threshold_context = self.threshold_helper(
            acts_topk[..., 0, 0 : self.n_context_features], self.threshold_context
        )
        self.threshold_token = self.threshold_helper(
            acts_topk[..., 0, self.n_context_features :], self.threshold_token
        )

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)

        state_dict["threshold_context"] = self.threshold_context.detach().clone()
        state_dict["threshold_token"] = self.threshold_token.detach().clone()

        # For debugging purposes
        print(state_dict["threshold_context"].mean())
        print(state_dict["threshold_token"].mean())
        # Don't Fold the Norm - weird issues will happen and your dog will be abducted by aliens
        if self.cfg.rescale_acts_by_decoder_norm:
            _fold_norm_topk(
                W_enc=state_dict["W_enc"],
                b_enc=state_dict["b_enc"],
                W_dec=state_dict["W_dec"],
            )
