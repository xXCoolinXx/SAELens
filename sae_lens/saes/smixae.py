from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from transformer_lens.hook_points import HookPoint
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


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class GrumpReLU(torch.autograd.Function):
    """Group JumpReLU. Learns an n dimensional ellipsoid threshold for an n dimensional input. Only fires if the input is outside the ellipsoid"""

    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        saved_fractions = (x / threshold) ** 2
        g = saved_fractions.sum(dim=-1, keepdim=True) - 1.0
        return (x * (g > 0)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        saved_fractions = (x / threshold) ** 2
        g = saved_fractions.sum(dim=-1, keepdim=True) - 1.0
        active_mask = g > 0

        # 1. Gradient w.r.t input x
        x_grad = grad_output * active_mask

        # 2. THE FIX: Exact dot product preserves the sign for negative coordinates
        dot_product = (grad_output * x).sum(dim=-1, keepdim=True)

        rect_val = rectangle(g / bandwidth)

        # 3. Exact threshold gradient
        local_threshold_grad = (
            dot_product * (rect_val / bandwidth) * (-2.0 * saved_fractions / threshold)
        )

        threshold_grad = torch.sum(local_threshold_grad, dim=0)

        return x_grad, threshold_grad, None


class GrumpReLULayer(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_dim: int,
        init_val: float = 0.01,
        bandwidth: float = 0.001,
    ):
        super().__init__()
        self.bandwidth = bandwidth
        self.log_threshold = nn.Parameter(
            torch.full((num_groups, group_dim), np.log(init_val))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, num_groups, group_dim]

        # --- INFERENCE PATH ---
        if not self.training:
            # Compute pure math without autograd overhead
            ellipsoid = (x / self.threshold) ** 2
            g = (
                ellipsoid.sum(dim=-1, keepdim=True) - 1.0
            )  # Determine whether it falls outside of the ellipsoid
            return (x * (g > 0)).to(x)

        # --- TRAINING PATH ---
        out: torch.Tensor = GrumpReLU.apply(x, self.threshold, self.bandwidth)  # type: ignore

        return out

    def calculate_pre_act_loss(self, x: torch.Tensor, decoder_norm: torch.Tensor):
        ellipsoid = (x / self.threshold) ** 2
        g = ellipsoid.sum(dim=-1) - 1.0
        summand = torch.relu(-g) * decoder_norm

        return summand.mean(dim=0).sum()

    @property
    def threshold(self):
        return torch.exp(self.log_threshold)


@dataclass
class SMIXAEConfig(SAEConfig):
    """
    Configuration class for a SMIXAE.
    """

    n_experts: int = 1024
    d_expert: int = 16
    # k_experts: int = 8
    d_bottleneck: int = 3
    rescale_acts_by_decoder_norm: bool = True
    grump_init: float = 0.01  # Follow anthropic, may need to scale up a bit
    grump_bandwidth: float = (
        1.0  # Reasonable default for the largher nornms, may also need to scale up
    )

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

    # W_gate: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    log_threshold: nn.Parameter
    b_enc: nn.Parameter
    b_bottleneck: nn.Parameter

    def __init__(self, cfg: SMIXAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        self.grump = GrumpReLULayer(
            self.cfg.n_experts,
            self.cfg.d_bottleneck,
            self.cfg.grump_init,
            self.cfg.grump_bandwidth,
        )

        # Dead expert tracker - remove this later, not used for inference
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        self.cfg.apply_b_dec_to_input = False  # Remove bias term - destroys structure

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_smixae(self)

    # @property
    # def threshold(self) -> torch.Tensor:
    #     return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor into the feature space.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        # Process the input (including dtype conversion, hook call, and any activation normalization)
        (
            _,
            _,
            _,
            h_bottleneck,
        ) = smixae_encode(self, x)  # (batch, n_experts, d_bottleneck)

        return h_bottleneck

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = sae_out_pre @ self.W_dec  # + self.b_dec

        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.ReLU()  # Try ReLU, GELU, etc.

    @property
    def effective_decoder_norm(self) -> torch.Tensor:
        """
        Computes the Frobenius norm of the effective 3D -> Residual projection.
        Returns a tensor of shape (n_experts,)
        """
        W_dec_reshaped = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)

        # W_latent_dec: (n_experts, d_bottleneck, d_expert)
        # W_dec_reshaped: (n_experts, d_expert, d_model)
        W_eff = self.W_latent_dec @ W_dec_reshaped

        return torch.linalg.matrix_norm(W_eff, ord="fro", dim=(-2, -1))


@dataclass
class SMIXAETrainingConfig(TrainingSAEConfig):
    """
    Configuration class for training a SMIXAETraining.
    """

    n_experts: int = 1024
    d_expert: int = 16
    d_bottleneck: int = 3
    # k_experts: int = 8  # L0 = d_expert * k_experts
    rescale_acts_by_decoder_norm: bool = True

    grump_init: float = 0.01  # Follow anthropic, may need to scale up a bit
    grump_bandwidth: float = (
        1.0  # Reasonable default for the largher nornms, may also need to scale up
    )
    pre_act_loss_coefficient: float = 1 / 32
    grump_tanh_coefficient: float = 0.01
    l0_coefficient: float = 1.0

    dead_after_n_passes: int = 1000  # If an expert hasn't fired for 1k passes, it has sadly passed away and we give it emergency aux loss to resucitate

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
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    b_bottleneck: nn.Parameter

    # x : d_model -> SwiGLU : (n_experts, d_expert) -> Bottleneck (n_experts, d_bottleneck) -> Latent Decode : (n_experts, d_expert) -> W_dec : (d_model)

    def __init__(self, cfg: SMIXAETrainingConfig):
        # Intercept d_sae
        cfg.d_sae = cfg.d_expert * cfg.n_experts

        super().__init__(cfg)

        self.hook_l0 = HookPoint()
        self.hook_sae_acts_bottleneck = HookPoint()

        # self.batchtopk = BatchTopK(self.cfg.k_experts)

        self.grump = GrumpReLULayer(
            self.cfg.n_experts,
            self.cfg.d_bottleneck,
            self.cfg.grump_init,
            self.cfg.grump_bandwidth,
        )

        # Dead expert tracker
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        self.cfg.apply_b_dec_to_input = False  # Remove bias term - destroys structure
        self.b_dec.requires_grad_(False)

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_smixae(self)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_latent, hidden_pre_latent, h_bottleneck, hidden_pre_bottleneck = (
            smixae_encode(self, x)
        )

        # Stash activations to avoid SAELens complaining about tensor sizes
        self.h_bottleneck = h_bottleneck
        self.hidden_pre_bottleneck = hidden_pre_bottleneck

        return h_latent, hidden_pre_latent

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = (
            sae_out_pre @ self.W_dec
        )  # + self.b_dec # Bias term destroys manifold structure

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
        sae_out = self.decode(self.h_bottleneck)

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

        metrics = {}

        metrics["experts_above_1e-3_L2"] = (
            (self.h_bottleneck.norm(dim=-1) > 1e-3).float().sum(dim=-1).mean()
        )

        metrics["experts_above_1e-1_L2"] = (
            (self.h_bottleneck.norm(dim=-1) > 1e-1).float().sum(dim=-1).mean()
        )

        metrics["expert_norm_mean"] = self.h_bottleneck.norm(dim=-1).mean()

        metrics["threshold_mean"] = self.grump.threshold.mean()

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

        decoder_norm = self.effective_decoder_norm

        losses["pre_act_loss"] = (
            self.cfg.pre_act_loss_coefficient
            * self.grump.calculate_pre_act_loss(
                self.hidden_pre_bottleneck, decoder_norm
            )
        )

        losses["l0_grump"] = self.cfg.l0_coefficient * (
            torch.tanh(
                self.cfg.grump_tanh_coefficient
                * torch.norm(self.h_bottleneck, dim=-1)
                * decoder_norm
            )
            .sum(dim=-1)
            .mean()
        )

        return losses

    @property
    def effective_decoder_norm(self) -> torch.Tensor:
        """
        Computes the Frobenius norm of the effective 3D -> Residual projection.
        Returns a tensor of shape (n_experts,)
        """
        W_dec_reshaped = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)

        # W_latent_dec: (n_experts, d_bottleneck, d_expert)
        # W_dec_reshaped: (n_experts, d_expert, d_model)
        W_eff = self.W_latent_dec @ W_dec_reshaped

        return torch.linalg.matrix_norm(W_eff, ord="fro", dim=(-2, -1))

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.ReLU()


def _init_weights_smixae(
    sae: SAE[SMIXAEConfig] | TrainingSAE[SMIXAETrainingConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(
            sae.cfg.n_experts * sae.cfg.d_expert,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    sae.b_bottleneck = nn.Parameter(
        torch.zeros(
            sae.cfg.n_experts,
            sae.cfg.d_bottleneck,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    sae.W_bottleneck = nn.Parameter(
        torch.empty(
            sae.cfg.n_experts,
            sae.cfg.d_expert,
            sae.cfg.d_bottleneck,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    sae.W_latent_dec = nn.Parameter(
        torch.empty(
            sae.cfg.n_experts,
            sae.cfg.d_bottleneck,
            sae.cfg.d_expert,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    nn.init.kaiming_uniform_(sae.W_bottleneck)
    nn.init.kaiming_uniform_(sae.W_latent_dec)


def smixae_encode(
    sae: SMIXAE | SMIXAETraining, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sae_in = sae.process_sae_in(x)

    # Standard forward
    hidden_pre_latent = sae.activation_fn(sae_in @ sae.W_enc + sae.b_enc)  # Remove gate
    h_latent = hidden_pre_latent  # * gate  # type: ignore

    h_unflattened = h_latent.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert))

    # Bottleneck
    hidden_pre_bottleneck = (
        torch.einsum("bne,ned->bnd", h_unflattened, sae.W_bottleneck) + sae.b_bottleneck
    )

    # Apply rescale before activation
    if sae.cfg.rescale_acts_by_decoder_norm:
        hidden_pre_bottleneck = (
            hidden_pre_bottleneck * sae.effective_decoder_norm.unsqueeze(-1)
        )

    # Apply grump function
    h_bottleneck = sae.grump(hidden_pre_bottleneck)

    # I don't think hidden_pre_latent is needed but this can always be changed
    return h_latent, hidden_pre_latent, h_bottleneck, hidden_pre_bottleneck
