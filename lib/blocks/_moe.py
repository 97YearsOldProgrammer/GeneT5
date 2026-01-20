from __future__ import annotations

import math
from dataclasses import dataclass
from typing      import Optional

import torch
import torch.nn            as nn
import torch.nn.functional as F

import triton
import triton.language as tl


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


@triton.jit
def expert_scatter_kernel(
    input_ptr, output_ptr, indices_ptr,
    num_tokens, embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter tokens to expert-specific positions."""
    pid = tl.program_id(0)
    
    if pid < num_tokens:
        src_offset = pid * embed_dim
        dst_idx    = tl.load(indices_ptr + pid)
        dst_offset = dst_idx * embed_dim
        
        for i in range(0, embed_dim, BLOCK_SIZE):
            mask   = i + tl.arange(0, BLOCK_SIZE) < embed_dim
            values = tl.load(input_ptr + src_offset + i + tl.arange(0, BLOCK_SIZE), mask=mask)
            tl.store(output_ptr + dst_offset + i + tl.arange(0, BLOCK_SIZE), values, mask=mask)


@triton.jit
def expert_gather_kernel(
    input_ptr, output_ptr, indices_ptr, weights_ptr,
    num_tokens, embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather expert outputs back to original positions with weighted sum."""
    pid = tl.program_id(0)
    
    if pid < num_tokens:
        dst_offset = pid * embed_dim
        src_idx    = tl.load(indices_ptr + pid)
        src_offset = src_idx * embed_dim
        weight     = tl.load(weights_ptr + pid)
        
        for i in range(0, embed_dim, BLOCK_SIZE):
            mask     = i + tl.arange(0, BLOCK_SIZE) < embed_dim
            values   = tl.load(input_ptr + src_offset + i + tl.arange(0, BLOCK_SIZE), mask=mask)
            values   = values * weight
            existing = tl.load(output_ptr + dst_offset + i + tl.arange(0, BLOCK_SIZE), mask=mask)
            tl.store(output_ptr + dst_offset + i + tl.arange(0, BLOCK_SIZE), existing + values, mask=mask)


class Router(nn.Module):
    """Top-K router for expert selection."""
    
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(embed_dim, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        logits           = self.gate(x)
        probs            = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights          = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights, logits


class GeGLUExpert(nn.Module):
    """Single expert with Gated Linear Unit activation."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.wi_gate = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wi_up   = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wo      = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate   = F.silu(self.wi_gate(x))
        up     = self.wi_up(x)
        hidden = self.dropout(gate * up)
        return self.wo(hidden)


class MoE(nn.Module):
    """Mixture of Experts layer optimized for single GPU."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        self.config               = config
        self.embed_dim            = config.embed_dim
        self.num_experts          = config.num_experts
        self.top_k                = config.top_k
        self.capacity_factor      = config.capacity_factor
        self.eval_capacity_factor = config.eval_capacity_factor
        self.load_balance_weight  = config.load_balance_weight
        self.router_z_weight      = config.router_z_loss_weight
        
        self.gate    = nn.Linear(config.embed_dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            GeGLUExpert(config.embed_dim, config.ff_dim, config.dropout)
            for _ in range(config.num_experts)
        ])
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, embed_dim = x.shape
        num_tokens                     = batch_size * seq_len
        x_flat                         = x.view(num_tokens, embed_dim)
        
        router_logits            = self.gate(x_flat)
        router_probs             = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs              = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        output   = self._forward_experts(x_flat, top_k_indices, top_k_probs)
        output   = output.view(batch_size, seq_len, embed_dim)
        aux_loss = self._compute_aux_loss(router_logits, router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _forward_experts(self, x_flat: torch.Tensor, top_k_indices: torch.Tensor, top_k_probs: torch.Tensor) -> torch.Tensor:
        """Route tokens through experts and combine outputs."""
        num_tokens, embed_dim = x_flat.shape
        output                = torch.zeros_like(x_flat)
        capacity_factor       = self.capacity_factor if self.training else self.eval_capacity_factor
        capacity              = int((num_tokens * self.top_k / self.num_experts) * capacity_factor)
        capacity              = max(capacity, self.top_k)
        
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                
                if not mask.any():
                    continue
                
                token_indices = torch.where(mask)[0]
                
                if len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]
                
                expert_input  = x_flat[token_indices]
                expert_weight = expert_weights[token_indices]
                expert_output = self.experts[expert_id](expert_input)
                
                output[token_indices] += expert_output * expert_weight.unsqueeze(-1)
        
        return output
    
    def _compute_aux_loss(self, router_logits: torch.Tensor, router_probs: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary losses for load balancing."""
        num_tokens  = router_logits.shape[0]
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)
        f           = expert_mask.sum(dim=0) / (num_tokens * self.top_k)
        P           = router_probs.mean(dim=0)
        
        load_balance_loss = self.num_experts * (f * P).sum()
        z_loss            = torch.logsumexp(router_logits, dim=-1).square().mean()
        
        return self.load_balance_weight * load_balance_loss + self.router_z_weight * z_loss
    
    def get_expert_utilization(self, x: torch.Tensor) -> dict:
        """Debug helper: get expert utilization statistics."""
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