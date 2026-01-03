
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class MoEConfig:
    embed_dim:       int   = 768
    ff_dim:          int   = 3072
    num_experts:     int   = 8
    top_k:           int   = 2
    dropout:         float = 0.0
    capacity_factor: float = 1.25
    aux_loss_weight: float = 0.01


class Router(nn.Module):
    
    def __init__(self, embed_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate  = nn.Linear(embed_dim, num_experts, bias=False)
    
    def forward(self, x):   
        logits           = self.gate(x)
        probs            = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights          = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights, logits


class VectorizedExperts(nn.Module):
    """All experts fused into single tensors. Executes via BMM."""
    
    def __init__(self, num_experts, embed_dim, ff_dim, dropout):
        super().__init__()
        self.w_gate  = nn.Parameter(torch.empty(num_experts, embed_dim, ff_dim))
        self.w_up    = nn.Parameter(torch.empty(num_experts, embed_dim, ff_dim))
        self.w_down  = nn.Parameter(torch.empty(num_experts, ff_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for w in [self.w_gate, self.w_up, self.w_down]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    
    def forward(self, x):
        # x: [num_experts, capacity, embed_dim]
        gate   = torch.bmm(x, self.w_gate)
        up     = torch.bmm(x, self.w_up)
        hidden = self.dropout(F.silu(gate) * up)
        return torch.bmm(hidden, self.w_down)


class MoE(nn.Module):
    """
    Mixture of Experts optimized for Apple Silicon.
    Uses BMM for parallel experts + vectorized scatter dispatch.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config          = config
        self.num_experts     = config.num_experts
        self.top_k           = config.top_k
        self.embed_dim       = config.embed_dim
        self.capacity_factor = config.capacity_factor
        
        self.router  = Router(config.embed_dim, config.num_experts, config.top_k)
        self.experts = VectorizedExperts(
            config.num_experts,
            config.embed_dim,
            config.ff_dim,
            config.dropout
        )
        
        # Pre-allocated buffers (resized dynamically if needed)
        self._dispatch_buffer = None
        self._combine_buffer  = None
        self._last_capacity   = 0
    
    def _get_buffers(self, num_tokens, capacity, device, dtype):
        total_slots = self.num_experts * capacity
        total_k     = num_tokens * self.top_k
        
        if self._last_capacity != capacity or self._dispatch_buffer is None:
            self._dispatch_buffer = torch.zeros(total_slots, self.embed_dim, device=device, dtype=dtype)
            self._combine_buffer  = torch.zeros(total_k, self.embed_dim, device=device, dtype=dtype)
            self._last_capacity   = capacity
        else:
            self._dispatch_buffer.zero_()
            self._combine_buffer.zero_()
        
        return self._dispatch_buffer, self._combine_buffer
    
    def forward(self, x):
        B, S, D    = x.shape
        num_tokens = B * S
        x_flat     = x.view(num_tokens, D)
        
        indices, weights, logits = self.router(x_flat)
        
        # Fixed capacity for MPS stability
        capacity = max(int((num_tokens / self.num_experts) * self.capacity_factor), 4)
        
        # Flatten for top-k
        indices_flat = indices.flatten()
        weights_flat = weights.flatten()
        x_repeated   = x_flat.repeat_interleave(self.top_k, dim=0)
        
        # Vectorized slot finding via cumsum
        expert_mask    = F.one_hot(indices_flat, self.num_experts).int()
        token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask
        token_rank     = token_priority.sum(dim=1)
        
        # Filter tokens that fit within capacity
        valid_mask    = (token_rank > 0) & (token_rank <= capacity)
        gather_idx    = (indices_flat * capacity) + (token_rank - 1)
        valid_indices = gather_idx[valid_mask].long()
        valid_x       = x_repeated[valid_mask]
        
        # Get pre-allocated buffers
        dispatched, combined = self._get_buffers(num_tokens, capacity, x.device, x.dtype)
        
        # Scatter all tokens to expert slots in ONE operation
        dispatched.index_copy_(0, valid_indices, valid_x)
        dispatched = dispatched.view(self.num_experts, capacity, D)
        
        # Parallel expert computation via BMM
        expert_out      = self.experts(dispatched)
        expert_out_flat = expert_out.view(-1, D)
        
        # Inverse scatter: gather results back
        valid_positions = valid_mask.nonzero(as_tuple=True)[0]
        combined.index_copy_(0, valid_positions, expert_out_flat[valid_indices])
        
        # Weighted sum
        combined = combined.view(num_tokens, self.top_k, D)
        output   = (combined * weights.unsqueeze(-1)).sum(dim=1)
        
        aux_loss = self._aux_loss(logits, indices)
        return output.view(B, S, D), aux_loss
    
    def _aux_loss(self, logits, indices):
        N     = logits.shape[0]
        probs = F.softmax(logits, dim=-1)
        mask  = F.one_hot(indices, self.num_experts).float().sum(dim=1)
        f     = mask.sum(0) / (N * self.top_k + 1e-9)
        P     = probs.mean(0)
        return self.num_experts * (f * P).sum() * self.config.aux_loss_weight
    
    def export_coreml(self, path='moe.mlpackage', seq_len=128, batch_size=1):
        try:
            import coremltools as ct
        except ImportError:
            print("Install coremltools: pip install coremltools")
            return
        
        self.eval()
        device   = next(self.parameters()).device
        self_cpu = self.to('cpu')
        
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)[0]
        
        example = torch.randn(batch_size, seq_len, self.config.embed_dim)
        traced  = torch.jit.trace(Wrapper(self_cpu), example)
        
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType('input', shape=(batch_size, seq_len, self.config.embed_dim))],
            outputs=[ct.TensorType('output')],
            convert_to='mlprogram',
            compute_units=ct.ComputeUnit.ALL,
        )
        mlmodel.save(path)
        print(f"Saved: {path}")
        self.to(device)