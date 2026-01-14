from __future__ import annotations

import math
from   dataclasses import dataclass
from   typing      import Optional

import torch
import torch.nn             as nn
import torch.nn.functional  as F
import torch.distributed    as dist

import triton
import triton.language as tl

import deepspeed


##################
##### CONFIG #####
##################


@dataclass
class MoEConfig:
    embed_dim:            int            = 768
    ff_dim:               int            = 3072
    num_experts:          int            = 8
    top_k:                int            = 2
    dropout:              float          = 0.0
    capacity_factor:      float          = 1.25
    eval_capacity_factor: float          = 2.0
    aux_loss_weight:      float          = 0.01
    load_balance_weight:  float          = 0.01
    router_z_loss_weight: float          = 0.001
    activation:           str            = 'silu'
    use_deepspeed:        bool           = False
    expert_parallel_size: Optional[int]  = None
    num_local_experts:    Optional[int]  = None


######################
##### TRITON OPS #####
######################


@triton.jit
def grouped_gemm_kernel(
    X_ptr, W_ptr, Y_ptr, expert_ids_ptr,
    total_tokens, in_dim, out_dim, num_experts,
    stride_x_token, stride_x_dim,
    stride_w_expert, stride_w_out, stride_w_in,
    stride_y_token, stride_y_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_n  = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < total_tokens
    mask_n = offs_n < out_dim
    
    expert_ids = tl.load(expert_ids_ptr + offs_m, mask=mask_m, other=0)
    acc        = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, in_dim, BLOCK_K):
        k_offs  = k_start + offs_k
        mask_k  = k_offs < in_dim
        
        x_ptrs  = X_ptr + offs_m[:, None] * stride_x_token + k_offs[None, :] * stride_x_dim
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        for k_idx in range(BLOCK_K):
            if k_start + k_idx < in_dim:
                x_col = tl.load(
                    X_ptr + offs_m * stride_x_token + (k_start + k_idx) * stride_x_dim,
                    mask=mask_m, other=0.0
                )
                w_row = tl.load(
                    W_ptr + expert_ids * stride_w_expert + 
                    offs_n[None, :] * stride_w_out + 
                    (k_start + k_idx) * stride_w_in,
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0.0
                )
                acc += x_col[:, None] * w_row
    
    y_ptrs = Y_ptr + offs_m[:, None] * stride_y_token + offs_n[None, :] * stride_y_dim
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def expert_position_kernel(
    expert_ids_ptr, positions_ptr, expert_counts_ptr,
    num_tokens, num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    pid         = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < num_tokens:
            expert = tl.load(expert_ids_ptr + idx)
            pos    = tl.atomic_add(expert_counts_ptr + expert, 1)
            tl.store(positions_ptr + idx, pos)


###########################
##### TRITON AUTOGRAD #####
###########################


class TritonGroupedGEMM(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, weight, expert_ids):
        num_tokens, in_dim      = x.shape
        num_experts, out_dim, _ = weight.shape
        output                  = torch.empty(num_tokens, out_dim, device=x.device, dtype=x.dtype)
        
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        grid = (
            triton.cdiv(num_tokens, BLOCK_M),
            triton.cdiv(out_dim, BLOCK_N),
        )
        
        grouped_gemm_kernel[grid](
            x, weight, output, expert_ids,
            num_tokens, in_dim, out_dim, num_experts,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1), weight.stride(2),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        ctx.save_for_backward(x, weight, expert_ids)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, expert_ids = ctx.saved_tensors
        num_experts           = weight.shape[0]
        
        selected_weights = weight[expert_ids]
        grad_x           = torch.einsum('no,nod->nd', grad_output, selected_weights)
        
        grad_weight      = torch.zeros_like(weight)
        grad_contribution = torch.einsum('no,nd->nod', grad_output, x)
        
        for i in range(num_experts):
            mask = expert_ids == i
            if mask.any():
                grad_weight[i] = grad_contribution[mask].sum(dim=0)
        
        return grad_x, grad_weight, None


def triton_grouped_gemm(x, weight, expert_ids):
    return TritonGroupedGEMM.apply(x, weight, expert_ids)


#############################
##### TRITON DISPATCHER #####
#############################


class TritonExpertDispatch(nn.Module):
    
    def __init__(self, num_experts, capacity_factor=1.25):
        super().__init__()
        self.num_experts     = num_experts
        self.capacity_factor = capacity_factor
    
    def forward(self, x, expert_ids, expert_weights):
        num_tokens, embed_dim = x.shape
        capacity              = int((num_tokens / self.num_experts) * self.capacity_factor)
        capacity              = max(capacity, 1)
        device                = x.device
        
        positions     = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        BLOCK_SIZE    = 256
        grid          = (triton.cdiv(num_tokens, BLOCK_SIZE),)
        
        expert_position_kernel[grid](
            expert_ids, positions, expert_counts,
            num_tokens, self.num_experts,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        positions      = positions.long()
        expert_counts  = expert_counts.long()
        valid_mask     = positions < capacity
        tokens_dropped = (~valid_mask).sum().item()
        
        dispatched_x = torch.zeros(
            self.num_experts, capacity, embed_dim,
            dtype=x.dtype, device=device
        )
        combine_weights = torch.zeros(
            self.num_experts, capacity,
            dtype=x.dtype, device=device
        )
        token_indices = torch.full(
            (self.num_experts, capacity), -1,
            dtype=torch.long, device=device
        )
        
        valid_tokens    = torch.where(valid_mask)[0]
        valid_experts   = expert_ids[valid_mask]
        valid_positions = positions[valid_mask]
        
        dispatched_x[valid_experts, valid_positions]    = x[valid_tokens]
        combine_weights[valid_experts, valid_positions] = expert_weights[valid_tokens]
        token_indices[valid_experts, valid_positions]   = valid_tokens
        
        return dispatched_x, combine_weights, token_indices, tokens_dropped


##################
##### ROUTER #####
##################


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


##########################
##### EXPERT MODULES #####
##########################


class TritonGeGLUExpert(nn.Module):
    
    def __init__(self, embed_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.wi_gate = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wi_up   = nn.Linear(embed_dim, ff_dim, bias=False)
        self.wo      = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate   = F.silu(self.wi_gate(x))
        up     = self.wi_up(x)
        hidden = self.dropout(gate * up)
        return self.wo(hidden)


################
#####  MOE #####
################


class MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config              = config
        self.embed_dim           = config.embed_dim
        self.num_experts         = config.num_experts
        self.top_k               = config.top_k
        self.use_deepspeed       = config.use_deepspeed and DEEPSPEED_AVAILABLE
        self.load_balance_weight = config.load_balance_weight
        self.router_z_weight     = config.router_z_loss_weight
        
        if dist.is_initialized():
            world_size                = dist.get_world_size()
            self.expert_parallel_size = config.expert_parallel_size or world_size
            self.num_local_experts    = config.num_experts // self.expert_parallel_size
        else:
            self.expert_parallel_size = 1
            self.num_local_experts    = config.num_experts
        
        self.gate = nn.Linear(config.embed_dim, config.num_experts, bias=False)
        
        if self.use_deepspeed and dist.is_initialized():
            self._init_deepspeed_experts(config)
        else:
            self._init_local_experts(config)
        
        self.dispatcher = TritonExpertDispatch(
            config.num_experts, config.capacity_factor
        )
    
    def _init_local_experts(self, config):
        self.expert_wi_gate = nn.Parameter(
            torch.empty(config.num_experts, config.ff_dim, config.embed_dim)
        )
        self.expert_wi_up = nn.Parameter(
            torch.empty(config.num_experts, config.ff_dim, config.embed_dim)
        )
        self.expert_wo = nn.Parameter(
            torch.empty(config.num_experts, config.embed_dim, config.ff_dim)
        )
        self._init_expert_weights()
        self.dropout = nn.Dropout(config.dropout)
    
    def _init_expert_weights(self):
        for param in [self.expert_wi_gate, self.expert_wi_up, self.expert_wo]:
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(param[i], a=math.sqrt(5))
    
    def _init_deepspeed_experts(self, config):
        self.experts = nn.ModuleList([
            TritonGeGLUExpert(config.embed_dim, config.ff_dim, config.dropout)
            for _ in range(self.num_local_experts)
        ])
        
        self.expert_group = None
        if dist.is_initialized():
            ranks             = list(range(dist.get_world_size()))
            self.expert_group = dist.new_group(ranks)
    
    def __del__(self):
        if hasattr(self, 'expert_group') and self.expert_group is not None:
            try:
                dist.destroy_process_group(self.expert_group)
            except:
                pass
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        num_tokens                     = batch_size * seq_len
        x_flat                         = x.view(num_tokens, embed_dim)
        
        router_logits = self.gate(x_flat)
        router_probs  = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        if self.use_deepspeed and dist.is_initialized():
            output, tokens_dropped = self._deepspeed_forward(
                x_flat, top_k_indices, top_k_probs
            )
        else:
            output, tokens_dropped = self._triton_forward(
                x_flat, top_k_indices, top_k_probs
            )
        
        output   = output.view(batch_size, seq_len, embed_dim)
        aux_loss = self._compute_aux_loss(router_logits, router_probs, top_k_indices)
        
        return output, aux_loss
    
    def _triton_forward(self, x_flat, top_k_indices, top_k_probs):
        num_tokens, embed_dim = x_flat.shape
        output                = torch.zeros_like(x_flat)
        total_dropped         = 0
        
        for k in range(self.top_k):
            expert_ids     = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            
            dispatched_x, combine_weights, token_indices, dropped = \
                self.dispatcher(x_flat, expert_ids, expert_weights)
            total_dropped += dropped
            
            gate_out = triton_grouped_gemm(
                dispatched_x.view(-1, embed_dim),
                self.expert_wi_gate,
                torch.arange(self.num_experts, device=x_flat.device).repeat_interleave(
                    dispatched_x.shape[1]
                )
            ).view(self.num_experts, -1, self.config.ff_dim)
            gate_out = F.silu(gate_out)
            
            up_out = triton_grouped_gemm(
                dispatched_x.view(-1, embed_dim),
                self.expert_wi_up,
                torch.arange(self.num_experts, device=x_flat.device).repeat_interleave(
                    dispatched_x.shape[1]
                )
            ).view(self.num_experts, -1, self.config.ff_dim)
            
            hidden = self.dropout(gate_out * up_out)
            
            expert_out = triton_grouped_gemm(
                hidden.view(-1, self.config.ff_dim),
                self.expert_wo,
                torch.arange(self.num_experts, device=x_flat.device).repeat_interleave(
                    hidden.shape[1]
                )
            ).view(self.num_experts, -1, embed_dim)
            
            for e in range(self.num_experts):
                valid_mask    = token_indices[e] >= 0
                valid_tokens  = token_indices[e][valid_mask]
                valid_weights = combine_weights[e][valid_mask].unsqueeze(-1)
                valid_outputs = expert_out[e, :valid_mask.sum()]
                
                output.index_add_(0, valid_tokens, valid_outputs * valid_weights)
        
        return output, total_dropped
    
    def _deepspeed_forward(self, x_flat, top_k_indices, top_k_probs):
        num_tokens, embed_dim = x_flat.shape
        rank                  = dist.get_rank()
        world_size            = dist.get_world_size()
        device                = x_flat.device
        dtype                 = x_flat.dtype
        
        expert_to_rank  = torch.arange(self.num_experts, device=device) % world_size
        primary_experts = top_k_indices[:, 0]
        dest_ranks      = expert_to_rank[primary_experts]
        
        send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        for r in range(world_size):
            send_counts[r] = (dest_ranks == r).sum()
        
        recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)
        dist.all_to_all_single(recv_counts, send_counts, group=self.expert_group)
        
        sorted_indices = torch.argsort(dest_ranks)
        sorted_tokens  = x_flat[sorted_indices]
        sorted_probs   = top_k_probs[sorted_indices, 0]
        sorted_experts = primary_experts[sorted_indices]
        
        total_recv   = recv_counts.sum().item()
        recv_tokens  = torch.zeros(total_recv, embed_dim, dtype=dtype, device=device)
        recv_probs   = torch.zeros(total_recv, dtype=dtype, device=device)
        recv_experts = torch.zeros(total_recv, dtype=torch.long, device=device)
        
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        
        send_token_list = list(sorted_tokens.split(send_splits))
        recv_token_list = list(recv_tokens.split(recv_splits))
        dist.all_to_all(recv_token_list, send_token_list, group=self.expert_group)
        recv_tokens = torch.cat(recv_token_list, dim=0) if total_recv > 0 else recv_tokens
        
        send_prob_list = list(sorted_probs.split(send_splits))
        recv_prob_list = [torch.zeros(s, dtype=dtype, device=device) for s in recv_splits]
        dist.all_to_all(recv_prob_list, send_prob_list, group=self.expert_group)
        recv_probs = torch.cat(recv_prob_list, dim=0) if total_recv > 0 else recv_probs
        
        send_expert_list = list(sorted_experts.split(send_splits))
        recv_expert_list = [torch.zeros(s, dtype=torch.long, device=device) for s in recv_splits]
        dist.all_to_all(recv_expert_list, send_expert_list, group=self.expert_group)
        recv_experts = torch.cat(recv_expert_list, dim=0) if total_recv > 0 else recv_experts
        
        local_expert_start = rank * self.num_local_experts
        local_expert_end   = local_expert_start + self.num_local_experts
        processed_tokens   = torch.zeros_like(recv_tokens)
        
        for local_idx, expert in enumerate(self.experts):
            global_expert_id = local_expert_start + local_idx
            mask = recv_experts == global_expert_id
            if mask.any():
                expert_input           = recv_tokens[mask]
                expert_output          = expert(expert_input)
                processed_tokens[mask] = expert_output * recv_probs[mask].unsqueeze(-1)
        
        send_result_list = list(processed_tokens.split(recv_splits))
        recv_result_list = [
            torch.zeros(s, embed_dim, dtype=dtype, device=device) for s in send_splits
        ]
        dist.all_to_all(recv_result_list, send_result_list, group=self.expert_group)
        result_tokens = torch.cat(recv_result_list, dim=0) if num_tokens > 0 else torch.zeros_like(sorted_tokens)
        
        output = torch.zeros_like(x_flat)
        output[sorted_indices] = result_tokens
        
        return output, 0
    
    def _compute_aux_loss(self, router_logits, router_probs, top_k_indices):
        num_tokens  = router_logits.shape[0]
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)
        f           = expert_mask.sum(dim=0) / (num_tokens * self.top_k)
        P           = router_probs.mean(dim=0)
        
        load_balance_loss = self.num_experts * (f * P).sum()
        z_loss            = torch.logsumexp(router_logits, dim=-1).square().mean()
        
        return self.load_balance_weight * load_balance_loss + self.router_z_weight * z_loss