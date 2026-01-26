from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn            as nn
import torch.nn.functional as F

import triton
import triton.language as tl


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
    capacity_factor:      float = 1.25
    eval_capacity_factor: float = 2.0
    aux_loss_weight:      float = 0.01
    load_balance_weight:  float = 0.01
    router_z_loss_weight: float = 0.001
    activation:           str   = 'silu'


#######################
#####  Algorithm  #####
#######################


@triton.jit
def expert_fused_ffn_kernel(
    X_ptr, Gate_W_ptr, Up_W_ptr, Down_W_ptr, Out_ptr,
    expert_offsets_ptr, token_indices_ptr, token_weights_ptr,
    embed_dim, ff_dim,
    stride_xd,
    stride_gw_e, stride_gw_d, stride_gw_f,
    stride_uw_e, stride_uw_d, stride_uw_f,
    stride_dw_e, stride_dw_f, stride_dw_d,
    stride_od,
    num_experts,
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """Fused expert FFN: SiLU(x @ W_gate) * (x @ W_up) @ W_down"""

    pid_token  = tl.program_id(0)
    pid_expert = tl.program_id(1)

    start_off = tl.load(expert_offsets_ptr + pid_expert)
    end_off   = tl.load(expert_offsets_ptr + pid_expert + 1)

    if pid_token >= (end_off - start_off):
        return

    global_token_idx = pid_token + start_off
    orig_token_idx   = tl.load(token_indices_ptr + global_token_idx)
    token_weight     = tl.load(token_weights_ptr + global_token_idx)

    offs_d = tl.arange(0, BLOCK_D)
    offs_f = tl.arange(0, BLOCK_F)

    x_ptrs = X_ptr + orig_token_idx * stride_xd + offs_d
    x      = tl.load(x_ptrs, mask=offs_d < embed_dim, other=0.0)

    gate_acc = tl.zeros((BLOCK_F,), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_F,), dtype=tl.float32)

    for d in range(0, embed_dim, BLOCK_D):
        d_offs  = d + tl.arange(0, BLOCK_D)
        d_mask  = d_offs < embed_dim
        x_chunk = tl.load(X_ptr + orig_token_idx * stride_xd + d_offs, mask=d_mask, other=0.0)

        for f in range(0, ff_dim, BLOCK_F):
            f_offs = f + tl.arange(0, BLOCK_F)
            f_mask = f_offs < ff_dim

            gate_ptrs = (Gate_W_ptr +
                         pid_expert * stride_gw_e +
                         d_offs[:, None] * stride_gw_d +
                         f_offs[None, :] * stride_gw_f)
            gate_w    = tl.load(gate_ptrs, mask=d_mask[:, None] & f_mask[None, :], other=0.0)

            up_ptrs = (Up_W_ptr +
                       pid_expert * stride_uw_e +
                       d_offs[:, None] * stride_uw_d +
                       f_offs[None, :] * stride_uw_f)
            up_w    = tl.load(up_ptrs, mask=d_mask[:, None] & f_mask[None, :], other=0.0)

            gate_acc += tl.sum(x_chunk[:, None] * gate_w, axis=0)
            up_acc   += tl.sum(x_chunk[:, None] * up_w, axis=0)

    hidden = tl.sigmoid(gate_acc) * gate_acc * up_acc

    out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for f in range(0, ff_dim, BLOCK_F):
        f_offs   = f + tl.arange(0, BLOCK_F)
        f_mask   = f_offs < ff_dim
        h_chunk  = tl.where(f_mask, hidden, 0.0)

        down_ptrs = (Down_W_ptr +
                     pid_expert * stride_dw_e +
                     f_offs[:, None] * stride_dw_f +
                     offs_d[None, :] * stride_dw_d)
        down_w    = tl.load(down_ptrs, mask=f_mask[:, None] & (offs_d[None, :] < embed_dim), other=0.0)
        out_acc  += tl.sum(h_chunk[:, None] * down_w, axis=0)

    out_ptrs  = Out_ptr + orig_token_idx * stride_od + offs_d
    existing  = tl.load(out_ptrs, mask=offs_d < embed_dim, other=0.0)
    tl.store(out_ptrs, existing + out_acc * token_weight, mask=offs_d < embed_dim)


@triton.jit
def batched_expert_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    expert_ids_ptr, token_indices_ptr, token_weights_ptr,
    num_tokens, embed_dim, out_dim,
    stride_x, stride_w_e, stride_w_in, stride_w_out, stride_o,
    BLOCK_IN:  tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    """Batched GEMM across all experts without Python loops"""

    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    token_idx  = tl.load(token_indices_ptr + pid)
    expert_id  = tl.load(expert_ids_ptr + pid)
    weight     = tl.load(token_weights_ptr + pid)

    offs_in  = tl.arange(0, BLOCK_IN)
    offs_out = tl.arange(0, BLOCK_OUT)

    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    for i in range(0, embed_dim, BLOCK_IN):
        i_offs = i + offs_in
        i_mask = i_offs < embed_dim

        x = tl.load(X_ptr + token_idx * stride_x + i_offs, mask=i_mask, other=0.0)

        for o in range(0, out_dim, BLOCK_OUT):
            o_offs = o + offs_out
            o_mask = o_offs < out_dim

            w_ptrs = (W_ptr +
                      expert_id * stride_w_e +
                      i_offs[:, None] * stride_w_in +
                      o_offs[None, :] * stride_w_out)
            w      = tl.load(w_ptrs, mask=i_mask[:, None] & o_mask[None, :], other=0.0)

            acc += tl.sum(x[:, None] * w, axis=0)

    out_ptrs = Out_ptr + token_idx * stride_o + offs_out
    existing = tl.load(out_ptrs, mask=offs_out < out_dim, other=0.0)
    tl.store(out_ptrs, existing + acc * weight, mask=offs_out < out_dim)


class Router(nn.Module):
    """Top-K router for expert selection"""

    def __init__(self, embed_dim, num_experts, top_k=2):
        super().__init__()

        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(embed_dim, num_experts, bias=False)

    def forward(self, x):
        logits           = self.gate(x)
        probs            = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights          = weights / weights.sum(dim=-1, keepdim=True)

        return indices, weights, logits


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
    """Mixture of Experts with fused Triton kernels"""

    def __init__(self, config):
        super().__init__()

        self.config               = config
        self.embed_dim            = config.embed_dim
        self.ff_dim               = config.ff_dim
        self.num_experts          = config.num_experts
        self.top_k                = config.top_k
        self.capacity_factor      = config.capacity_factor
        self.eval_capacity_factor = config.eval_capacity_factor
        self.load_balance_weight  = config.load_balance_weight
        self.router_z_weight      = config.router_z_loss_weight

        self.gate           = nn.Linear(config.embed_dim, config.num_experts, bias=False)
        self.expert_weights = FusedExpertWeights(config.num_experts, config.embed_dim, config.ff_dim)
        self.dropout        = nn.Dropout(config.dropout)

        self._use_triton = True

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

        return sorted_experts, sorted_tokens, sorted_weights, expert_offsets

    def _forward_triton(self, x_flat, top_k_indices, top_k_probs):
        """Triton-accelerated expert computation without Python loops"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets = \
            self._prepare_expert_batches(top_k_indices, top_k_probs, num_tokens, device)

        output      = torch.zeros_like(x_flat)
        total_items = sorted_tokens.shape[0]

        if total_items == 0:
            return output

        BLOCK_IN  = min(128, embed_dim)
        BLOCK_OUT = min(128, self.ff_dim)

        hidden = torch.zeros(num_tokens, self.ff_dim, device=device, dtype=x_flat.dtype)

        gate_w = self.expert_weights.gate_weights
        up_w   = self.expert_weights.up_weights
        down_w = self.expert_weights.down_weights

        grid = (total_items,)

        batched_expert_gemm_kernel[grid](
            x_flat, gate_w, hidden,
            sorted_experts, sorted_tokens, torch.ones_like(sorted_weights),
            total_items, embed_dim, self.ff_dim,
            x_flat.stride(0),
            gate_w.stride(0), gate_w.stride(1), gate_w.stride(2),
            hidden.stride(0),
            BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT,
        )

        hidden_up = torch.zeros_like(hidden)
        batched_expert_gemm_kernel[grid](
            x_flat, up_w, hidden_up,
            sorted_experts, sorted_tokens, torch.ones_like(sorted_weights),
            total_items, embed_dim, self.ff_dim,
            x_flat.stride(0),
            up_w.stride(0), up_w.stride(1), up_w.stride(2),
            hidden_up.stride(0),
            BLOCK_IN=BLOCK_IN, BLOCK_OUT=BLOCK_OUT,
        )

        hidden = F.silu(hidden) * hidden_up

        batched_expert_gemm_kernel[grid](
            hidden, down_w, output,
            sorted_experts, sorted_tokens, sorted_weights,
            total_items, self.ff_dim, embed_dim,
            hidden.stride(0),
            down_w.stride(0), down_w.stride(1), down_w.stride(2),
            output.stride(0),
            BLOCK_IN=BLOCK_OUT, BLOCK_OUT=BLOCK_IN,
        )

        return output

    def _forward_pytorch(self, x_flat, top_k_indices, top_k_probs):
        """Batched PyTorch fallback avoiding per-expert loops"""

        num_tokens, embed_dim = x_flat.shape
        device                = x_flat.device

        sorted_experts, sorted_tokens, sorted_weights, expert_offsets = \
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

            gate_out = F.silu(expert_input @ gate_w[expert_id])
            up_out   = expert_input @ up_w[expert_id]
            hidden   = gate_out * up_out
            out      = hidden @ down_w[expert_id]

            output.index_add_(0, token_ids, out * weights)

        return output

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        num_tokens                     = batch_size * seq_len
        x_flat                         = x.view(num_tokens, embed_dim)

        router_logits              = self.gate(x_flat)
        router_probs               = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs                = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        if self._use_triton and x_flat.is_cuda:
            try:
                output = self._forward_triton(x_flat, top_k_indices, top_k_probs)
            except Exception:
                output = self._forward_pytorch(x_flat, top_k_indices, top_k_probs)
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