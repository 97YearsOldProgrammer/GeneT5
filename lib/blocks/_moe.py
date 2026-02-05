from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn            as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from liger_kernel.ops.swiglu import LigerSiLUMulFunction


####################
#####  Config  #####
####################


@dataclass
class MoEConfig:
    embed_dim:            int   = 768
    ff_dim:               int   = 3072
    num_experts:          int   = 8
    top_k:                int   = 2
    dropout:              float = 0.0
    load_balance_weight:  float = 0.01
    router_z_loss_weight: float = 0.001


def _check_triton_support():
    """Check if Triton MoE is supported on current hardware

    Auto-selects backend based on num_experts:
    - num_experts >= 32: Triton (1.1-1.2x faster)
    - num_experts < 32:  PyTorch (faster due to lower kernel overhead)
    Set GENET5_DISABLE_TRITON=1 to force PyTorch fallback.
    """

    import os
    if os.environ.get("GENET5_DISABLE_TRITON", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 7


#######################
#####  Kernels    #####
#######################


@triton.jit
def fused_expert_kernel_v2(
    # Input/output pointers
    X_ptr, Out_ptr,
    # Weight pointers for this expert
    Gate_W_ptr, Up_W_ptr, Down_W_ptr,
    # Intermediate buffer for hidden states
    Hidden_ptr,
    # Token routing info
    token_indices_ptr, token_weights_ptr,
    # Dimensions
    num_tokens, embed_dim, ff_dim,
    # Strides
    stride_x_t, stride_x_d,
    stride_gw_d, stride_gw_f,
    stride_uw_d, stride_uw_f,
    stride_dw_f, stride_dw_d,
    stride_o_t, stride_o_d,
    stride_h_t, stride_h_f,
    # Phase: 0 = gate+up+silu, 1 = down
    phase: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Two-phase fused expert kernel for large ff_dim

    Phase 0: Compute hidden = silu(x @ gate_w) * (x @ up_w), store to Hidden_ptr
    Phase 1: Compute out = hidden @ down_w, atomic add to Out_ptr

    This allows processing large ff_dim in tiles.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < num_tokens

    # Load token indices
    token_indices = tl.load(token_indices_ptr + offs_m, mask=mask_m, other=0)

    if phase == 0:
        # Phase 0: gate + up + silu*mul -> hidden
        mask_n = offs_n < ff_dim

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, embed_dim, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < embed_dim

            # Load x: [BLOCK_M, BLOCK_K]
            x_ptrs = X_ptr + token_indices[:, None] * stride_x_t + k_offs[None, :] * stride_x_d
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

            # Load gate_w: [BLOCK_K, BLOCK_N]
            gw_ptrs = Gate_W_ptr + k_offs[:, None] * stride_gw_d + offs_n[None, :] * stride_gw_f
            gate_w  = tl.load(gw_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

            # Load up_w: [BLOCK_K, BLOCK_N]
            uw_ptrs = Up_W_ptr + k_offs[:, None] * stride_uw_d + offs_n[None, :] * stride_uw_f
            up_w    = tl.load(uw_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

            gate_acc += tl.dot(x_tile, gate_w)
            up_acc   += tl.dot(x_tile, up_w)

        # silu(gate) * up
        hidden = (gate_acc * tl.sigmoid(gate_acc)) * up_acc

        # Store hidden (use sequential indices for intermediate)
        h_ptrs = Hidden_ptr + offs_m[:, None] * stride_h_t + offs_n[None, :] * stride_h_f
        tl.store(h_ptrs, hidden, mask=mask_m[:, None] & mask_n[None, :])

    else:
        # Phase 1: hidden @ down_w -> out (with atomic add)
        mask_n        = offs_n < embed_dim
        token_weights = tl.load(token_weights_ptr + offs_m, mask=mask_m, other=0.0)

        out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, ff_dim, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < ff_dim

            # Load hidden: [BLOCK_M, BLOCK_K]
            h_ptrs = Hidden_ptr + offs_m[:, None] * stride_h_t + k_offs[None, :] * stride_h_f
            h_tile = tl.load(h_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

            # Load down_w: [BLOCK_K, BLOCK_N]
            dw_ptrs = Down_W_ptr + k_offs[:, None] * stride_dw_f + offs_n[None, :] * stride_dw_d
            down_w  = tl.load(dw_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

            out_acc += tl.dot(h_tile, down_w)

        # Apply weights and atomic add
        out_weighted = out_acc * token_weights[:, None]
        out_ptrs     = Out_ptr + token_indices[:, None] * stride_o_t + offs_n[None, :] * stride_o_d
        tl.atomic_add(out_ptrs, out_weighted, mask=mask_m[:, None] & mask_n[None, :])


class FusedExpertWeights(nn.Module):
    """Fused expert weights for batched computation"""

    def __init__(self, num_experts, embed_dim, ff_dim):

        super().__init__()

        self.num_experts = num_experts
        self.embed_dim   = embed_dim
        self.ff_dim      = ff_dim

        self.gate_weights = nn.Parameter(torch.empty(num_experts, embed_dim, ff_dim))
        self.up_weights   = nn.Parameter(torch.empty(num_experts, embed_dim, ff_dim))
        self.down_weights = nn.Parameter(torch.empty(num_experts, ff_dim, embed_dim))

        self._init_weights()

    def _init_weights(self):

        std = 0.02
        nn.init.normal_(self.gate_weights, std=std)
        nn.init.normal_(self.up_weights, std=std)
        nn.init.normal_(self.down_weights, std=std)


class MoE(nn.Module):
    """Mixture of Experts with fused Triton kernels

    Features:
    - Two-phase fused kernel (gate+up+silu -> hidden, hidden @ down -> out)
    - Atomic operations for correct top-k > 1 handling
    - Token sorting by expert for efficient batching
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

        self._triton_supported     = _check_triton_support()
        self._grouped_mm_available = (
            hasattr(torch, '_grouped_mm')
            and os.environ.get("GENET5_GROUPED_MM", "0") == "1"
        )

    def _prepare_expert_batches(self, top_k_indices, top_k_probs, num_tokens, device):
        """Prepare sorted token batches for fused expert computation"""

        flat_indices = top_k_indices.reshape(-1)
        flat_probs   = top_k_probs.reshape(-1)
        flat_tokens  = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)

        sorted_idx     = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices[sorted_idx]
        sorted_tokens  = flat_tokens[sorted_idx]
        sorted_weights = flat_probs[sorted_idx]

        expert_counts  = torch.bincount(sorted_experts, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=device)
        expert_offsets[1:] = expert_counts.cumsum(0)

        return sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts

    def _forward_triton_fused(self, x_flat, top_k_indices, top_k_probs):
        """Fused Triton forward - 2 kernels per expert instead of 3+

        Uses two-phase approach:
        Phase 0: gate + up + silu*mul -> hidden buffer
        Phase 1: hidden @ down -> atomic add to output
        """

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output = torch.zeros_like(x_flat)

        if sorted_tokens.shape[0] == 0:
            return output

        gate_w = self.expert_weights.gate_weights
        up_w   = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        # Block sizes
        BLOCK_M = 32
        BLOCK_K = 64
        BLOCK_N = 64

        # Pre-allocate hidden buffer (reused across experts)
        max_tokens = expert_counts.max().item()
        if max_tokens == 0:
            return output
        hidden_buffer = torch.empty(max_tokens, self.ff_dim, device=device, dtype=x_flat.dtype)

        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            end   = expert_offsets[expert_id + 1].item()
            count = end - start

            if count == 0:
                continue

            expert_token_indices = sorted_tokens[start:end]
            expert_weights_slice = sorted_weights[start:end]
            hidden               = hidden_buffer[:count]

            grid_m       = (count + BLOCK_M - 1) // BLOCK_M
            grid_n_ff    = (self.ff_dim + BLOCK_N - 1) // BLOCK_N
            grid_n_embed = (embed_dim + BLOCK_N - 1) // BLOCK_N

            # Phase 0: gate + up + silu*mul -> hidden
            fused_expert_kernel_v2[(grid_m, grid_n_ff)](
                x_flat, output,
                gate_w[expert_id], up_w[expert_id], down_w[expert_id],
                hidden,
                expert_token_indices, expert_weights_slice,
                count, embed_dim, self.ff_dim,
                x_flat.stride(0), x_flat.stride(1),
                gate_w.stride(1), gate_w.stride(2),
                up_w.stride(1), up_w.stride(2),
                down_w.stride(1), down_w.stride(2),
                output.stride(0), output.stride(1),
                hidden.stride(0), hidden.stride(1),
                phase=0,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
            )

            # Phase 1: hidden @ down -> out
            fused_expert_kernel_v2[(grid_m, grid_n_embed)](
                x_flat, output,
                gate_w[expert_id], up_w[expert_id], down_w[expert_id],
                hidden,
                expert_token_indices, expert_weights_slice,
                count, embed_dim, self.ff_dim,
                x_flat.stride(0), x_flat.stride(1),
                gate_w.stride(1), gate_w.stride(2),
                up_w.stride(1), up_w.stride(2),
                down_w.stride(1), down_w.stride(2),
                output.stride(0), output.stride(1),
                hidden.stride(0), hidden.stride(1),
                phase=1,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
            )

        return output

    def _forward_grouped(self, x_flat, top_k_indices, top_k_probs):
        """Grouped GEMM forward — 3 kernel launches instead of 3×num_experts"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets, expert_counts = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output = torch.zeros_like(x_flat)

        if sorted_tokens.shape[0] == 0:
            return output

        sorted_input = x_flat[sorted_tokens]
        offs         = expert_offsets[1:].to(torch.int32)

        gate_w = self.expert_weights.gate_weights
        up_w   = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        gate_out = torch._grouped_mm(sorted_input, gate_w, offs=offs)
        up_out   = torch._grouped_mm(sorted_input, up_w, offs=offs)
        hidden   = LigerSiLUMulFunction.apply(gate_out, up_out)
        out      = torch._grouped_mm(hidden, down_w, offs=offs)

        output.index_add_(0, sorted_tokens, (out * sorted_weights.unsqueeze(-1)).to(output.dtype))

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

        # Use Triton for many experts (32+), PyTorch for fewer
        # Benchmarks on GB10 show PyTorch is faster for typical configs
        use_triton = (self._triton_supported and
                      x_flat.is_cuda and
                      self.num_experts >= 32)

        if use_triton:
            try:
                output = self._forward_triton_fused(x_flat, top_k_indices, top_k_probs)
            except Exception:
                torch.cuda.synchronize()
                self._triton_supported = False
                output = self._forward_pytorch(x_flat, top_k_indices, top_k_probs)
        elif self._grouped_mm_available and x_flat.is_cuda:
            output = self._forward_grouped(x_flat, top_k_indices, top_k_probs)
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

        load_balance_loss = self.num_experts * (f * P).sum()
        z_loss            = torch.logsumexp(router_logits, dim=-1).square().mean()

        return self.load_balance_weight * load_balance_loss + self.router_z_weight * z_loss

    def get_expert_utilization(self, x):
        """Debug helper: get expert utilization statistics"""

        with torch.no_grad():
            batch_size, seq_len, embed_dim = x.shape
            num_tokens                     = batch_size * seq_len
            x_flat                         = x.view(num_tokens, embed_dim)

            router_logits    = self.gate(x_flat)
            router_probs     = F.softmax(router_logits, dim=-1)
            _, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

            counts = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                for expert_id in range(self.num_experts):
                    counts[expert_id] += (top_k_indices[:, k] == expert_id).sum()

            return {
                "counts":     counts.cpu().tolist(),
                "probs_mean": router_probs.mean(dim=0).cpu().tolist(),
                "probs_std":  router_probs.std(dim=0).cpu().tolist(),
            }
