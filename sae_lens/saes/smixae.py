import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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


@dataclass
class SMIXAEConfig(SAEConfig):
    """
    Configuration class for a SMIXAE.
    """

    n_experts: int = 1024
    d_expert: int = 16
    k_experts: int = 8

    @override
    @classmethod
    def architecture(cls) -> str:
        return "smixae"


class SMIXAE(SAE[SMIXAEConfig]):
    """
    SMIXAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a simple linear encoder and decoder.

    It implements the required abstract methods from BaseSAE:

      - initialize_weights: sets up simple parameter initializations for W_enc, b_enc, W_dec, and b_dec.
      - encode: computes the feature activations from an input.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter
    W_router: nn.Parameter
    b_router: nn.Parameter

    def __init__(self, cfg: SMIXAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_smixae(self)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)

        router_choices = sae_in @ self.W_router + self.b_router
        # Add noise to help avoid dead experts
        # noise = torch.randn_like(router_choices) * self.cfg.router_noise_scale
        # router_choices = router_choices + noise

        # Select top K choices
        topk_values, topk_indices = torch.topk(
            router_choices, self.cfg.k_experts, dim=-1
        )
        # Softmax the values
        topk_values = F.softmax(topk_values, dim=-1)
        # Scatter back to the original size
        router_weights = torch.zeros_like(router_choices)
        router_weights.scatter_(dim=-1, index=topk_indices, src=topk_values)

        router_mask = router_weights.repeat_interleave(self.cfg.d_expert, dim=-1)

        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU, depending on config) and router mask
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre)) * router_mask

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        # 1) linear transform
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        # 2) hook reconstruction
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        # 4) optional out-normalization (e.g. constant_norm_rescale)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 5) if hook_z is enabled, rearrange back to (..., n_heads, d_head).
        return self.reshape_fn_out(sae_out_pre, self.d_head)


@dataclass
class SMIXAETrainingConfig(TrainingSAEConfig):
    """
    Configuration class for training a SMIXAETraining.
    """

    # l1_coefficient: float = 1.0
    # lp_norm: float = 1.0
    # l1_warm_up_steps: int = 0

    n_experts: int = 1024
    d_expert: int = 16
    k_experts: int = 8  # L0 = d_expert * k_experts
    aux_loss_coefficient: float = 0.01
    router_noise_scale: float = 0.5
    expert_threshold_factor: float = 0.2

    @override
    @classmethod
    def architecture(cls) -> str:
        return "smixae"


class SMIXAETraining(TrainingSAE[SMIXAETrainingConfig]):
    """
    SMIXAETraining is a concrete implementation of BaseTrainingSAE using the "standard" SAE architecture.
    It implements:

      - initialize_weights: basic weight initialization for encoder/decoder.
      - encode: inference encoding (invokes encode_with_hidden_pre).
      - decode: a simple linear decoder.
      - encode_with_hidden_pre: computes activations and pre-activations.
      - calculate_aux_loss: computes a sparsity penalty based on the (optionally scaled) p-norm of feature activations.
    """

    b_enc: nn.Parameter
    W_router: nn.Parameter
    b_router: nn.Parameter

    def __init__(self, cfg: SMIXAETrainingConfig):
        # Intercept d_sae
        cfg.d_sae = cfg.d_expert * cfg.n_experts

        super().__init__(cfg)

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_smixae(self)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Process the input (including dtype conversion, hook call, and any activation normalization)
        sae_in = self.process_sae_in(x)

        # Select experts
        self.router_choices = sae_in @ self.W_router + self.b_router
        # Add noise to help avoid dead experts
        # noise = torch.randn_like(router_choices) * self.cfg.router_noise_scale
        # router_choices = router_choices + noise

        # Select top K choices
        topk_values, topk_indices = torch.topk(
            self.router_choices, self.cfg.k_experts, dim=-1
        )
        # Softmax the values
        topk_values = F.softmax(topk_values, dim=-1)
        # Scatter back to the original size
        self.router_weights = torch.zeros_like(self.router_choices)
        self.router_weights.scatter_(dim=-1, index=topk_indices, src=topk_values)

        # Compute the pre-activation (and allow for a hook if desired). Right now this is a dense computation - optimize this later
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)  # type: ignore

        # Get the router mask. We make this an an attribute so that loss can be computed on the router distribution
        router_mask = self.router_weights.repeat_interleave(self.cfg.d_expert, dim=-1)

        # Apply the activation function (and any post-activation hook)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        # Apply the router mask for weighting + masking
        feature_acts = router_mask * feature_acts

        return feature_acts, hidden_pre

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)

        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Compute KL divergence between router distribution and

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
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

        expert_counts = (self.router_weights > 0).float().sum(dim=0)
        avg_mask = self.router_weights.mean(dim=0)

        metrics: dict[str, torch.Tensor | float | int] = {}

        metrics["expert_entropy"] = -(
            avg_mask * (avg_mask + 1e-8).log()
        ).sum() / math.log(self.cfg.n_experts)

        metrics["expert_coverage"] = (expert_counts > 0).float().mean()

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
            metrics=metrics,
        )

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        full_probs = F.softmax(self.router_choices, dim=-1)
        avg_probs = full_probs.mean(dim=0)

        # losses["expert_loss"] = (
        #     -self.cfg.aux_loss_coefficient * (avg_probs + 1e-8).log().mean()
        # )

        selection_freq = (self.router_weights > 0).float().mean(dim=0).detach()

        min_threshold = (
            self.cfg.expert_threshold_factor * self.cfg.k_experts / self.cfg.n_experts
        )

        below_threshold_mask = (selection_freq < min_threshold).float()

        losses["expert_threshold"] = (
            self.cfg.aux_loss_coefficient
            * (below_threshold_mask * (-torch.log(avg_probs + 1e-8))).sum()
        )

        # Compute KL divergence between uniform distribution and average router mask
        # Compute average router mask
        # avg_mask = self.router_weights.mean(dim=0)
        # uniform = torch.full_like(avg_mask, 1.0 / self.cfg.n_experts)

        # losses["load_balancing"] = self.cfg.aux_loss_coefficient * F.kl_div(
        #     (avg_mask + 1e-8).log(), uniform, reduction="sum"
        # )

        return losses

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.GELU()  # Try ReLU, GELU, etc.


def _init_weights_smixae(
    sae: SAE[SMIXAEConfig] | TrainingSAE[SMIXAETrainingConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )

    sae.W_router = nn.Parameter(
        torch.empty(sae.cfg.d_in, sae.cfg.n_experts, dtype=sae.dtype, device=sae.device)
    )

    nn.init.kaiming_uniform_(sae.W_router)

    sae.b_router = nn.Parameter(
        torch.zeros(sae.cfg.n_experts, dtype=sae.dtype, device=sae.device)
    )
