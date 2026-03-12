from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn            as nn
import torch.nn.functional as F

from liger_kernel.ops.swiglu import LigerSiLUMulFunction

# GENET5_MOE_BACKEND: auto (default), triton, grouped_mm, bmm, pytorch
_MOE_BACKEND = os.environ.get("GENET5_MOE_BACKEND", "auto")


####################
#####  Config  #####
####################


@dataclass
class MoEConfig:
    embed_dim:            int   = 768
    ff_dim:               int   = 1536
    num_experts:          int   = 8
    top_k:                int   = 2
    dropout:              float = 0.0
    load_balance_weight:  float = 0.01
    router_z_loss_weight: float = 0.001


class FusedExpertWeights(nn.Module):
    """Fused expert weights — gate+up merged into single [E, K, 2*ff_dim] parameter"""

    def __init__(self, num_experts, embed_dim, ff_dim):

        super().__init__()

        self.num_experts = num_experts
        self.embed_dim   = embed_dim
        self.ff_dim      = ff_dim

        self.gate_up_weights = nn.Parameter(torch.empty(num_experts, embed_dim, 2 * ff_dim))
        self.down_weights    = nn.Parameter(torch.empty(num_experts, ff_dim, embed_dim))

        self._init_weights()

    def _init_weights(self):

        std = 0.006
        nn.init.normal_(self.gate_up_weights, std=std)
        nn.init.normal_(self.down_weights, std=std)

    @property
    def gate_weights(self):
        """Backward-compat view into fused gate_up_weights[:, :, :ff_dim]"""
        return self.gate_up_weights[:, :, :self.ff_dim]

    @property
    def up_weights(self):
        """Backward-compat view into fused gate_up_weights[:, :, ff_dim:]"""
        return self.gate_up_weights[:, :, self.ff_dim:]


class MoE(nn.Module):
    """Mixture of Experts with bmm, grouped GEMM, or loop fallback

    Dispatch priority: triton > grouped_mm (Hopper/Blackwell) > bmm > loop
    """

    def __init__(self, config):

        super().__init__()

        self.config              = config
        self.embed_dim           = config.embed_dim
        self.ff_dim              = config.ff_dim
        self.num_experts         = config.num_experts
        self.top_k               = config.top_k
        self.load_balance_weight = config.load_balance_weight
        self.router_z_weight     = config.router_z_loss_weight

        self.gate           = nn.Linear(config.embed_dim, config.num_experts, bias=False)
        self.expert_weights = FusedExpertWeights(config.num_experts, config.embed_dim, config.ff_dim)
        self.dropout        = nn.Dropout(config.dropout)

        self._grouped_mm_available = (
            hasattr(torch, '_grouped_mm')
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() in ((9, 0), (10, 0))
        )

        self._triton_available = False
        try:
            from lib.blocks._triton_moe import triton_grouped_gemm
            from lib.blocks._triton_moe import triton_fused_swiglu_gemm
            from lib.blocks._triton_moe import triton_grouped_gemm_scatter
            self._triton_grouped_gemm         = triton_grouped_gemm
            self._triton_fused_swiglu_gemm    = triton_fused_swiglu_gemm
            self._triton_grouped_gemm_scatter = triton_grouped_gemm_scatter
            self._triton_available            = True
        except ImportError:
            pass

    def _prepare_expert_batches(self, top_k_indices, top_k_probs, num_tokens, device):
        """Prepare sorted token batches for fused expert computation"""

        flat_indices = top_k_indices.reshape(-1)
        flat_probs   = top_k_probs.reshape(-1)
        flat_tokens  = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_idx     = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices[sorted_idx]
        sorted_tokens  = flat_tokens[sorted_idx]
        sorted_weights = flat_probs[sorted_idx]

        ones           = torch.ones(sorted_experts.shape[0], dtype=torch.long, device=device)
        expert_counts  = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        expert_counts.scatter_add_(0, sorted_experts.long(), ones)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=device)
        expert_offsets[1:] = expert_counts.cumsum(0)

        return sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts

    def _forward_grouped(self, x_flat, top_k_indices, top_k_probs):
        """Grouped GEMM forward — 3 kernel launches instead of 3×num_experts"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output = torch.zeros_like(x_flat)

        if sorted_tokens.shape[0] == 0:
            return output

        idx_e        = sorted_tokens.unsqueeze(-1).expand(-1, embed_dim)
        sorted_input = torch.gather(x_flat, 0, idx_e)
        offs         = expert_offsets[1:].to(torch.int32)

        gate_w = self.expert_weights.gate_weights
        up_w   = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        gate_out = torch._grouped_mm(sorted_input, gate_w, offs=offs)
        up_out   = torch._grouped_mm(sorted_input, up_w, offs=offs)
        hidden   = LigerSiLUMulFunction.apply(gate_out, up_out)
        out      = torch._grouped_mm(hidden, down_w, offs=offs)

        weighted = (out * sorted_weights.unsqueeze(-1)).to(output.dtype)
        output   = output.scatter_add(0, idx_e, weighted)

        return output

    @torch.compiler.disable
    def _forward_bmm(self, x_flat, top_k_indices, top_k_probs):
        """Batched matmul — 3 bmm calls instead of 3×num_experts matmuls"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output = torch.zeros_like(x_flat)

        if sorted_tokens.shape[0] == 0:
            return output

        max_count = expert_counts.max().item()
        positions = torch.arange(max_count, device=device)
        mask      = positions.unsqueeze(0) < expert_counts.unsqueeze(1)

        flat_idx  = expert_offsets[:-1].unsqueeze(1) + positions.unsqueeze(0)
        flat_idx  = flat_idx.clamp(max=sorted_tokens.shape[0] - 1)

        token_ids    = sorted_tokens[flat_idx.reshape(-1)].reshape(self.num_experts, max_count)
        padded_input = x_flat[token_ids] * mask.unsqueeze(-1).to(x_flat.dtype)

        gate_out   = torch.bmm(padded_input, self.expert_weights.gate_weights)
        up_out     = torch.bmm(padded_input, self.expert_weights.up_weights)
        hidden     = LigerSiLUMulFunction.apply(gate_out, up_out)
        expert_out = torch.bmm(hidden, self.expert_weights.down_weights)

        padded_w = sorted_weights[flat_idx.reshape(-1)].reshape(self.num_experts, max_count, 1)
        weighted = (expert_out * padded_w * mask.unsqueeze(-1).to(expert_out.dtype)).to(output.dtype)

        flat_out  = weighted.reshape(-1, embed_dim)
        flat_ids  = token_ids.reshape(-1)
        flat_mask = mask.reshape(-1)

        output.scatter_add_(
            0,
            flat_ids[flat_mask].unsqueeze(-1).expand(-1, embed_dim),
            flat_out[flat_mask],
        )

        return output

    def _forward_triton(self, x_flat, top_k_indices, top_k_probs):
        """Triton fused MoE — SwiGLU GEMM + scatter GEMM, works on all GPUs"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        if sorted_tokens.shape[0] == 0:
            return torch.zeros_like(x_flat)

        sorted_input = x_flat[sorted_tokens]
        offs         = expert_offsets.to(torch.int32)

        gate_up_w = self.expert_weights.gate_up_weights
        down_w    = self.expert_weights.down_weights

        hidden = self._triton_fused_swiglu_gemm(sorted_input, gate_up_w, offs)
        output = self._triton_grouped_gemm_scatter(
            hidden, down_w, offs, sorted_weights, sorted_tokens, num_tokens)

        return output

    def _forward_pytorch(self, x_flat, top_k_indices, top_k_probs):
        """Loop-based PyTorch fallback"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output = torch.zeros_like(x_flat)

        gate_w = self.expert_weights.gate_weights
        up_w   = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            end   = expert_offsets[expert_id + 1].item()

            if start == end:
                continue

            token_ids = sorted_tokens[start:end]
            weights   = sorted_weights[start:end].unsqueeze(-1)

            expert_input = x_flat[token_ids]

            gate_out = expert_input @ gate_w[expert_id]
            up_out   = expert_input @ up_w[expert_id]
            hidden   = LigerSiLUMulFunction.apply(gate_out, up_out)
            out      = hidden @ down_w[expert_id]

            output.index_add_(0, token_ids, (out * weights).to(output.dtype))

        return output

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape
        num_tokens                     = batch_size * seq_len
        x_flat                         = x.view(num_tokens, embed_dim)

        router_logits              = self.gate(x_flat)
        router_probs               = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs                = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        if _MOE_BACKEND == "triton" and self._triton_available:
            output = self._forward_triton(x_flat, top_k_indices, top_k_probs)
        elif _MOE_BACKEND == "grouped_mm" and self._grouped_mm_available:
            output = self._forward_grouped(x_flat, top_k_indices, top_k_probs)
        elif _MOE_BACKEND == "bmm":
            output = self._forward_bmm(x_flat, top_k_indices, top_k_probs)
        elif _MOE_BACKEND == "pytorch":
            output = self._forward_pytorch(x_flat, top_k_indices, top_k_probs)
        elif self._grouped_mm_available and x_flat.is_cuda:
            output = self._forward_grouped(x_flat, top_k_indices, top_k_probs)
        elif self._triton_available and x_flat.is_cuda:
            output = self._forward_triton(x_flat, top_k_indices, top_k_probs)
        elif x_flat.is_cuda:
            output = self._forward_bmm(x_flat, top_k_indices, top_k_probs)
        else:
            output = self._forward_pytorch(x_flat, top_k_indices, top_k_probs)

        output   = self.dropout(output)
        output   = output.view(batch_size, seq_len, embed_dim)
        aux_loss = self._compute_aux_loss(router_logits, router_probs, top_k_indices)

        return output, aux_loss

    def _compute_aux_loss(self, router_logits, router_probs, top_k_indices):
        """Compute auxiliary losses for load balancing"""

        num_tokens  = router_logits.shape[0]
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)
        f           = expert_mask.sum(dim=0) / (num_tokens * self.top_k)
        P           = router_probs.mean(dim=0)

        # Store per-expert load fraction for logging (no grad, no graph)
        self._last_expert_frac = f.detach()

        load_balance_loss = self.num_experts * (f * P).sum()
        z_loss            = torch.logsumexp(router_logits, dim=-1).square().mean()

        return self.load_balance_weight * load_balance_loss + self.router_z_weight * z_loss

