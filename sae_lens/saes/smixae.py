from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import BatchTopK
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
    # k_experts: int = 8
    d_bottleneck: int = 3
    b_enc_init: float = -0.1
    rescale_acts_by_decoder_norm: bool = True

    # jump_relu_bandwidth: float = 0.05
    # jump_relu_init_threshold = 0.1
    # l0_coefficient: float = 1.0

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

        self.register_buffer(
            "threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

        # Dead expert tracker - remove this later, not used for inference
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        # Let this be set by the loaded class
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
        _, _, z = smixae_encode(self, x)  # (batch, n_experts, d_bottleneck)

        z_norm_mask = z.norm(dim=-1) > self.threshold  # type: ignore # (batch, n_experts) mask

        z = z * z_norm_mask.unsqueeze(-1)  # Apply mask per bottleneck

        return z

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = sae_out_pre @ self.W_dec + self.b_dec

        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.ReLU()  # Try ReLU, GELU, etc.


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
    d_bottleneck: int = 3
    k_experts: int = 8  # L0 = d_expert * k_experts
    aux_loss_coefficient: float = 1 / 32
    b_enc_init: float = -0.1
    rescale_acts_by_decoder_norm: bool = True
    # expert_threshold: float = 0.1

    # jump_relu_bandwidth: float = 0.05
    # jump_relu_init_threshold: float = 2.0
    # l0_coefficient: float = 1.0
    threshold_lr: float = 0.1
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

    # b_enc: nn.Parameter
    # b_enc : nn.Parameter
    # W_gate: nn.Parameter
    b_enc: nn.Parameter
    b_bottleneck: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    # log_threshold: nn.Parameter

    # Flow diagram

    # x : d_model -> SwiGLU : (n_experts, d_expert) -> Bottleneck (n_experts, d_bottleneck) -> Latent Decode : (n_experts, d_expert) -> W_dec : (d_model)

    def __init__(self, cfg: SMIXAETrainingConfig):
        # Intercept d_sae
        cfg.d_sae = cfg.d_expert * cfg.n_experts

        super().__init__(cfg)

        self.hook_l0 = HookPoint()
        self.hook_sae_acts_bottleneck = HookPoint()

        self.batchtopk = BatchTopK(self.cfg.k_experts)

        self.register_buffer(
            "threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

        # Dead expert tracker
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        self.cfg.apply_b_dec_to_input = (
            False  # True  # True  # Remove bias term - destroys structure
        )
        # self.b_dec.requires_grad_(False)

    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_smixae(self)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, hidden_pre, z = smixae_encode(self, x)

        batch_norm_mask = self.batchtopk(z.norm(dim=-1)) > 0  # (batch_size, n_experts)

        self.z_pre = z
        self.z = z * batch_norm_mask.unsqueeze(-1)
        # self.l0 = l0
        self.hook_sae_acts_pre(hidden_pre)
        self.hook_sae_acts_post(h)
        # self.hook_l0(l0)
        self.hook_sae_acts_bottleneck(z)

        did_fire = batch_norm_mask.any(dim=0)
        self.n_passes_since_fired += 1
        self.n_passes_since_fired = (
            self.n_passes_since_fired * ~did_fire
        )  # 0 out if it did fire

        # Process the input (including dtype conversion, hook call, and any activation normalization)
        # sae_in = self.process_sae_in(x)

        # # StepGLU encoder gate

        # # Use step to decouple magnitude from existence
        # gate = Step.apply(
        #     sae_in @ self.W_gate, self.threshold, self.cfg.jump_relu_bandwidth
        # )

        # # Save L0 to apply count loss later
        # self.l0 = torch.sum(gate, dim=-1)  # type: ignore

        # # Standard forward
        # hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc)
        # h = hidden_pre * gate

        # h_unflattened = h.unflatten(-1, (self.cfg.n_experts, self.cfg.d_expert))
        # # expert_mask = (h_unflattened.norm(dim=-1) > self.cfg.expert_threshold).float()
        # # h_unflattened = h_unflattened * expert_mask.unsqueeze(-1)

        # # Bottleneck
        # self.z = torch.einsum(
        #     "bne,ned->bnd", h_unflattened, self.W_bottleneck
        # )  # (batch_size, n_experts, d_bottelneck)

        # # expert_norms = self.z.norm(dim=-1)  # (batch, n_experts)
        # # _, topk_idx = expert_norms.topk(self.cfg.k_experts, dim=-1)
        # # mask = torch.zeros_like(expert_norms).scatter(-1, topk_idx, 1.0)
        # # self.z = self.z * mask.unsqueeze(-1)

        return h, hidden_pre

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = (
            sae_out_pre @ self.W_dec + self.b_dec
        )  # Bias term destroys manifold structure, so only add it to the output

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
        sae_out = self.decode(self.z)

        self.update_threshold(self.z.norm(dim=-1))

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
            (self.z.norm(dim=-1) > 1e-3).float().sum(dim=-1).mean()
        )

        metrics["experts_above_1e-1_L2"] = (
            (self.z.norm(dim=-1) > 1e-1).float().sum(dim=-1).mean()
        )

        metrics["expert_norm_mean"] = self.z.norm(dim=-1).mean()

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

        losses["dead_expert_aux_loss"] = self.calculate_topk_aux_loss(
            step_input.sae_in,
            sae_out,
            self.z_pre,
            self.n_passes_since_fired > self.cfg.dead_after_n_passes,
        )

        # losses["l0"] = (self.cfg.l0_coefficient * self.l0).mean()

        # Apply group lasso on feature acts

        # losses["group_lasso"] = (
        #     self.cfg.aux_loss_coefficient * self.z.norm(dim=-1).sum(dim=-1).mean()
        # )  # L2 norm per expert, 'L1' sum over all experts

        return losses

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        z_pre_mask: torch.Tensor,  # (batch, n_experts, d_bottleneck) before TopK
        dead_expert_mask: torch.Tensor,  # (n_experts,) bool
    ) -> torch.Tensor:
        if dead_expert_mask is None or (num_dead := int(dead_expert_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)

        residual = (sae_in - sae_out).detach()

        # Heuristic: use half of active experts as k_aux
        k_aux = self.cfg.k_experts // 2
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Select top-k_aux dead experts by bottleneck norm
        expert_norms = z_pre_mask.norm(dim=-1)  # (batch, n_experts)
        dead_norms = torch.where(dead_expert_mask[None], expert_norms, -torch.inf)
        topk = dead_norms.topk(k_aux, dim=-1, sorted=False)

        # Build sparse z with only selected dead experts
        aux_mask = torch.zeros_like(expert_norms)
        aux_mask.scatter_(1, topk.indices, 1.0)
        z_aux = z_pre_mask * aux_mask.unsqueeze(-1)  # (batch, n_experts, d_bottleneck)

        # Decode through full expert pipeline
        recons = self.decode(z_aux)

        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @torch.no_grad()
    def update_threshold(self, norms_topk: torch.Tensor) -> None:
        positive_mask = norms_topk > 0
        lr = self.cfg.threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(self.threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = norms_topk[positive_mask].min().to(self.threshold.dtype)
                self.threshold = (1 - lr) * self.threshold + lr * min_positive

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return nn.ReLU()


def _init_weights_smixae(
    sae: SAE[SMIXAEConfig] | TrainingSAE[SMIXAETrainingConfig],
) -> None:
    # sae.W_gate = nn.Parameter(
    #     torch.empty(
    #         sae.cfg.d_in,
    #         sae.cfg.n_experts * sae.cfg.d_expert,
    #         dtype=sae.dtype,
    #         device=sae.device,
    #     )
    # )  # Same dim size but we reshape

    # Add gate bias term to allow more expressivity - pre relu
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

    # sae.log_threshold = nn.Parameter(
    #     torch.ones(sae.cfg.n_experts, dtype=sae.dtype, device=sae.device)
    #     * np.log(sae.cfg.jump_relu_init_threshold)
    # )

    # nn.init.kaiming_uniform_(sae.W_gate)
    nn.init.kaiming_uniform_(sae.W_bottleneck)
    nn.init.kaiming_uniform_(sae.W_latent_dec)


def smixae_encode(
    sae: SMIXAE | SMIXAETraining, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sae_in = sae.process_sae_in(x)

    # StepGLU encoder gate

    # Use step to decouple magnitude from existence
    # gate = sae.activation_fn(
    #     sae_in @ sae.W_gate + sae.b_enc  # , sae.threshold, sae.cfg.jump_relu_bandwidth
    # )

    # Save L0 to apply count loss later
    # l0 = torch.sum(gate, dim=-1)  # type: ignore
    # latent_count = gate.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert)).sum(
    #     dim=-1
    # )  # 0 to 16
    # expert_cost = torch.log1p(
    #     latent_count
    # )  # concave: first latent costs ~0.7, going from 8→9 costs ~0.05
    # l0 = expert_cost.sum(dim=-1)

    # Standard forward
    hidden_pre = sae.activation_fn(sae_in @ sae.W_enc + sae.b_enc)  # Remove gate
    h = hidden_pre  # * gate  # type: ignore

    h_unflattened = h.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert))

    # expert_norms = h_unflattened.norm(dim=-1)
    # expert_mask = Step.apply(expert_norms, sae.threshold, sae.cfg.jump_relu_bandwidth)
    # l0 = expert_mask.sum(dim=-1)

    # expert_mask = (h_unflattened.norm(dim=-1) > self.cfg.expert_threshold).float()

    h_unflattened = h_unflattened  # * expert_mask.unsqueeze(-1)

    # Bottleneck
    z = (
        torch.einsum("bne,ned->bnd", h_unflattened, sae.W_bottleneck) + sae.b_bottleneck
    )  # (batch_size, n_experts, d_bottelneck)

    if sae.cfg.rescale_acts_by_decoder_norm:
        norm = (
            sae.W_latent_dec @ sae.W_dec.view(sae.cfg.n_experts, sae.cfg.d_expert, -1)
        ).norm(dim=-1)
        z = z * norm

    # expert_norms = self.z.norm(dim=-1)  # (batch, n_experts)
    # _, topk_idx = expert_norms.topk(self.cfg.k_experts, dim=-1)
    # mask = torch.zeros_like(expert_norms).scatter(-1, topk_idx, 1.0)
    # self.z = self.z * mask.unsqueeze(-1)

    return h, hidden_pre, z  # , l0
