from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK_M    = 64
BLOCK_N    = 64
BLOCK_K    = 32
DW_BLOCK_M = 64


# ──────────────────────────────────────────────────────────────────────
#  Base grouped GEMM kernels (unchanged)
# ──────────────────────────────────────────────────────────────────────


@triton.jit
def _grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr, offsets_ptr,
    K: tl.constexpr, N: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[e] = A[e] @ B[e] for each expert group, single kernel launch"""

    pid      = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)

    global_m_block = pid // n_blocks
    n_block_id     = pid % n_blocks

    acc_blocks   = 0
    expert_id    = 0
    expert_start = 0
    expert_end   = 0
    local_m      = 0

    for e in tl.static_range(NUM_EXPERTS):
        e_start  = tl.load(offsets_ptr + e)
        e_end    = tl.load(offsets_ptr + e + 1)
        e_count  = e_end - e_start
        e_blocks = tl.cdiv(e_count, BLOCK_M)

        if global_m_block >= acc_blocks and global_m_block < acc_blocks + e_blocks:
            expert_id    = e
            expert_start = e_start
            expert_end   = e_end
            local_m      = global_m_block - acc_blocks

        acc_blocks += e_blocks

    m_offs = expert_start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < expert_end
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(
            A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        b = tl.load(
            B_ptr + expert_id * stride_be + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )

        acc = tl.dot(a, b, acc)

    tl.store(
        C_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _grouped_gemm_dw_kernel(
    A_ptr, dC_ptr, dW_ptr, offsets_ptr,
    K: tl.constexpr, N: tl.constexpr,
    stride_am, stride_ak,
    stride_dcm, stride_dcn,
    stride_dwe, stride_dwk, stride_dwn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """dW[e] = A[e]^T @ dC[e] — weight gradient accumulation per expert"""

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    k_blocks   = tl.cdiv(K, BLOCK_K)
    expert_id  = pid0 // k_blocks
    k_block_id = pid0 % k_blocks
    n_block_id = pid1

    e_start = tl.load(offsets_ptr + expert_id)
    e_end   = tl.load(offsets_ptr + expert_id + 1)
    e_count = e_end - e_start

    k_offs = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offs = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = k_offs < K
    n_mask = n_offs < N

    acc          = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    num_m_blocks = tl.cdiv(e_count, BLOCK_M)

    for m_block in range(num_m_blocks):
        m_offs = e_start + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < e_end

        at = tl.load(
            A_ptr + k_offs[:, None] * stride_ak + m_offs[None, :] * stride_am,
            mask=k_mask[:, None] & m_mask[None, :], other=0.0,
        )
        dc = tl.load(
            dC_ptr + m_offs[:, None] * stride_dcm + n_offs[None, :] * stride_dcn,
            mask=m_mask[:, None] & n_mask[None, :], other=0.0,
        )

        acc = tl.dot(at, dc, acc)

    tl.store(
        dW_ptr + expert_id * stride_dwe + k_offs[:, None] * stride_dwk + n_offs[None, :] * stride_dwn,
        acc.to(dW_ptr.dtype.element_ty),
        mask=k_mask[:, None] & n_mask[None, :],
    )


def _compute_fwd_grid(offsets, num_experts, block_m, block_n, N):
    """Total tiles across all experts for the forward kernel"""

    counts   = offsets[1:] - offsets[:-1]
    m_blocks = ((counts + block_m - 1) // block_m).sum().item()
    n_blocks = math.ceil(N / block_n)
    return (m_blocks * n_blocks,)


class GroupedGEMM(torch.autograd.Function):
    """Triton grouped GEMM with full backward support"""

    @staticmethod
    def forward(ctx, A, B, offsets):

        total_M, K = A.shape
        E, _, N    = B.shape
        C          = torch.empty(total_M, N, device=A.device, dtype=A.dtype)

        grid = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, N)

        _grouped_gemm_kernel[grid](
            A, B, C, offsets,
            K, N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(A, B, offsets)
        ctx.K = K
        ctx.N = N
        ctx.E = E
        return C

    @staticmethod
    def backward(ctx, dC):

        A, B, offsets = ctx.saved_tensors
        K, N, E       = ctx.K, ctx.N, ctx.E
        dC            = dC.to(A.dtype)

        dA = torch.empty_like(A)
        dB = torch.empty_like(B)

        grid_dA = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, K)

        _grouped_gemm_kernel[grid_dA](
            dC, B, dA, offsets,
            N, K,
            dC.stride(0), dC.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            dA.stride(0), dA.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        k_tile_count = math.ceil(K / BLOCK_K)
        n_tile_count = math.ceil(N / BLOCK_N)
        grid_dB      = (E * k_tile_count, n_tile_count)

        _grouped_gemm_dw_kernel[grid_dB](
            A, dC, dB, offsets,
            K, N,
            A.stride(0), A.stride(1),
            dC.stride(0), dC.stride(1),
            dB.stride(0), dB.stride(1), dB.stride(2),
            BLOCK_M=DW_BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        )

        return dA, dB, None


def triton_grouped_gemm(A, B, offsets):
    """Drop-in replacement for torch._grouped_mm with full backward support"""

    return GroupedGEMM.apply(A, B, offsets)


# ──────────────────────────────────────────────────────────────────────
#  Phase 1: Fused SwiGLU GEMM — gate+up matmul with SiLU*mul in epilogue
# ──────────────────────────────────────────────────────────────────────


@triton.jit
def _fused_swiglu_gemm_kernel(
    A_ptr, B_ptr, C_ptr, gate_save_ptr, up_save_ptr, offsets_ptr,
    K: tl.constexpr, FF_DIM: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_gsm, stride_gsn,
    stride_usm, stride_usn,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused gate+up GEMM with SiLU activation: result = silu(gate) * up

    B is [E, K, 2*FF_DIM] with gate in [:FF_DIM] and up in [FF_DIM:]
    C output is [M, FF_DIM], gate_save and up_save are [M, FF_DIM] for backward
    Grid tiles over FF_DIM (not 2*FF_DIM)
    """

    pid      = tl.program_id(0)
    n_blocks = tl.cdiv(FF_DIM, BLOCK_N)

    global_m_block = pid // n_blocks
    n_block_id     = pid % n_blocks

    acc_blocks   = 0
    expert_id    = 0
    expert_start = 0
    expert_end   = 0
    local_m      = 0

    for e in tl.static_range(NUM_EXPERTS):
        e_start  = tl.load(offsets_ptr + e)
        e_end    = tl.load(offsets_ptr + e + 1)
        e_count  = e_end - e_start
        e_blocks = tl.cdiv(e_count, BLOCK_M)

        if global_m_block >= acc_blocks and global_m_block < acc_blocks + e_blocks:
            expert_id    = e
            expert_start = e_start
            expert_end   = e_end
            local_m      = global_m_block - acc_blocks

        acc_blocks += e_blocks

    m_offs = expert_start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = n_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < expert_end
    n_mask = n_offs < FF_DIM

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_offs_up = n_offs + FF_DIM

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(
            A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )

        b_gate = tl.load(
            B_ptr + expert_id * stride_be + k_offs[:, None] * stride_bk + n_offs[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )
        b_up = tl.load(
            B_ptr + expert_id * stride_be + k_offs[:, None] * stride_bk + n_offs_up[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )

        acc_gate = tl.dot(a, b_gate, acc_gate)
        acc_up   = tl.dot(a, b_up, acc_up)

    gate_val = acc_gate.to(tl.float32)
    up_val   = acc_up.to(tl.float32)

    sig      = tl.sigmoid(gate_val)
    result   = sig * gate_val * up_val

    out_dtype = C_ptr.dtype.element_ty

    tl.store(
        C_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn,
        result.to(out_dtype),
        mask=m_mask[:, None] & n_mask[None, :],
    )
    tl.store(
        gate_save_ptr + m_offs[:, None] * stride_gsm + n_offs[None, :] * stride_gsn,
        gate_val.to(out_dtype),
        mask=m_mask[:, None] & n_mask[None, :],
    )
    tl.store(
        up_save_ptr + m_offs[:, None] * stride_usm + n_offs[None, :] * stride_usn,
        up_val.to(out_dtype),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _swiglu_bwd_kernel(
    d_hidden_ptr, gate_save_ptr, up_save_ptr, d_fused_ptr,
    N: tl.constexpr,
    stride_m, stride_n,
    stride_gsm, stride_gsn,
    stride_usm, stride_usn,
    stride_dfm, stride_dfn,
    TOTAL_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Backward through SiLU*mul: d_hidden -> d_gate, d_up (concatenated as d_fused)"""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < TOTAL_M
    n_mask = n_offs < N

    dh = tl.load(
        d_hidden_ptr + m_offs[:, None] * stride_m + n_offs[None, :] * stride_n,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_save_ptr + m_offs[:, None] * stride_gsm + n_offs[None, :] * stride_gsn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_save_ptr + m_offs[:, None] * stride_usm + n_offs[None, :] * stride_usn,
        mask=m_mask[:, None] & n_mask[None, :], other=0.0,
    ).to(tl.float32)

    sig = tl.sigmoid(gate)

    d_gate = dh * up * sig * (1.0 + gate * (1.0 - sig))
    d_up   = dh * gate * sig

    out_dtype = d_fused_ptr.dtype.element_ty

    tl.store(
        d_fused_ptr + m_offs[:, None] * stride_dfm + n_offs[None, :] * stride_dfn,
        d_gate.to(out_dtype),
        mask=m_mask[:, None] & n_mask[None, :],
    )

    n_offs_up = n_offs + N
    tl.store(
        d_fused_ptr + m_offs[:, None] * stride_dfm + n_offs_up[None, :] * stride_dfn,
        d_up.to(out_dtype),
        mask=m_mask[:, None] & n_mask[None, :],
    )


class FusedSwiGLUGEMM(torch.autograd.Function):
    """Fused gate+up GEMM with SiLU activation and full backward"""

    @staticmethod
    def forward(ctx, A, B, offsets):

        total_M, K = A.shape
        E          = B.shape[0]
        FF_DIM     = B.shape[2] // 2
        C          = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)
        gate_save  = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)
        up_save    = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)

        grid = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, FF_DIM)

        _fused_swiglu_gemm_kernel[grid](
            A, B, C, gate_save, up_save, offsets,
            K, FF_DIM,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1),
            gate_save.stride(0), gate_save.stride(1),
            up_save.stride(0), up_save.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(A, B, offsets, gate_save, up_save)
        ctx.K      = K
        ctx.FF_DIM = FF_DIM
        ctx.E      = E
        return C

    @staticmethod
    def backward(ctx, d_hidden):

        A, B, offsets, gate_save, up_save = ctx.saved_tensors
        K, FF_DIM, E                      = ctx.K, ctx.FF_DIM, ctx.E
        total_M                           = A.shape[0]
        d_hidden                          = d_hidden.to(A.dtype)

        d_fused = torch.empty(total_M, 2 * FF_DIM, device=A.device, dtype=A.dtype)

        m_blocks = math.ceil(total_M / BLOCK_M)
        n_blocks = math.ceil(FF_DIM / BLOCK_N)

        _swiglu_bwd_kernel[(m_blocks, n_blocks)](
            d_hidden, gate_save, up_save, d_fused,
            FF_DIM,
            d_hidden.stride(0), d_hidden.stride(1),
            gate_save.stride(0), gate_save.stride(1),
            up_save.stride(0), up_save.stride(1),
            d_fused.stride(0), d_fused.stride(1),
            TOTAL_M=total_M,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        dA = torch.empty_like(A)
        dB = torch.empty_like(B)

        N_full = 2 * FF_DIM
        grid_dA = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, K)

        _grouped_gemm_kernel[grid_dA](
            d_fused, B, dA, offsets,
            N_full, K,
            d_fused.stride(0), d_fused.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            dA.stride(0), dA.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        k_tile_count = math.ceil(K / BLOCK_K)
        n_tile_count = math.ceil(N_full / BLOCK_N)
        grid_dB      = (E * k_tile_count, n_tile_count)

        _grouped_gemm_dw_kernel[grid_dB](
            A, d_fused, dB, offsets,
            K, N_full,
            A.stride(0), A.stride(1),
            d_fused.stride(0), d_fused.stride(1),
            dB.stride(0), dB.stride(1), dB.stride(2),
            BLOCK_M=DW_BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        )

        return dA, dB, None


def triton_fused_swiglu_gemm(A, B, offsets):
    """Fused gate+up GEMM with SiLU: A @ B[:,:,:ff] * silu(A @ B[:,:,ff:])"""

    return FusedSwiGLUGEMM.apply(A, B, offsets)


# ──────────────────────────────────────────────────────────────────────
#  Phase 2: Down GEMM + deterministic weighted scatter (no atomics)
# ──────────────────────────────────────────────────────────────────────


class GroupedGEMMScatter(torch.autograd.Function):
    """Down GEMM + fp32 weighted scatter — deterministic, no atomic contention"""

    @staticmethod
    def forward(ctx, A, B, offsets, sorted_weights, sorted_tokens, num_tokens):

        total_M, K = A.shape
        E, _, N    = B.shape

        gemm_out = torch.empty(total_M, N, device=A.device, dtype=A.dtype)

        grid = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, N)

        _grouped_gemm_kernel[grid](
            A, B, gemm_out, offsets,
            K, N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            gemm_out.stride(0), gemm_out.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        idx_e    = sorted_tokens.unsqueeze(-1).expand(-1, N)
        weighted = gemm_out.float() * sorted_weights.float().unsqueeze(-1)
        output   = torch.zeros(num_tokens, N, device=A.device, dtype=torch.float32)
        output.scatter_add_(0, idx_e, weighted)
        output   = output.to(A.dtype)

        ctx.save_for_backward(A, B, offsets, sorted_weights, sorted_tokens, gemm_out)
        ctx.K          = K
        ctx.N          = N
        ctx.E          = E
        ctx.num_tokens = num_tokens
        return output

    @staticmethod
    def backward(ctx, d_output):

        A, B, offsets, sorted_weights, sorted_tokens, gemm_save = ctx.saved_tensors
        K, N, E, num_tokens = ctx.K, ctx.N, ctx.E, ctx.num_tokens
        total_M             = A.shape[0]
        target_dtype        = A.dtype
        d_output            = d_output.to(target_dtype)

        idx_e     = sorted_tokens.unsqueeze(-1).expand(-1, N)
        d_gather  = torch.gather(d_output, 0, idx_e)
        d_scaled  = (d_gather * sorted_weights.unsqueeze(-1)).to(target_dtype)

        d_weights = (d_gather * gemm_save).sum(dim=-1)

        dA = torch.empty_like(A)
        dB = torch.empty_like(B)

        grid_dA = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, K)

        _grouped_gemm_kernel[grid_dA](
            d_scaled, B, dA, offsets,
            N, K,
            d_scaled.stride(0), d_scaled.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            dA.stride(0), dA.stride(1),
            NUM_EXPERTS=E,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        k_tile_count = math.ceil(K / BLOCK_K)
        n_tile_count = math.ceil(N / BLOCK_N)
        grid_dB      = (E * k_tile_count, n_tile_count)

        _grouped_gemm_dw_kernel[grid_dB](
            A, d_scaled, dB, offsets,
            K, N,
            A.stride(0), A.stride(1),
            d_scaled.stride(0), d_scaled.stride(1),
            dB.stride(0), dB.stride(1), dB.stride(2),
            BLOCK_M=DW_BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        )

        return dA, dB, None, d_weights, None, None


def triton_grouped_gemm_scatter(A, B, offsets, sorted_weights, sorted_tokens, num_tokens):
    """Down GEMM + fp32 weighted scatter — deterministic, no atomics"""

    return GroupedGEMMScatter.apply(A, B, offsets, sorted_weights, sorted_tokens, num_tokens)


# ──────────────────────────────────────────────────────────────────────
#  Phase 3: Counting sort — O(M) radix sort for small expert counts
# ──────────────────────────────────────────────────────────────────────


@triton.jit
def _counting_sort_kernel(
    expert_ids_ptr, token_ids_ptr, weights_ptr,
    out_tokens_ptr, out_weights_ptr, out_offsets_ptr,
    TOTAL_M: tl.constexpr, NUM_EXPERTS: tl.constexpr,
):
    """Single-block counting sort: O(M) for small E, replaces argsort"""

    counts = tl.zeros((NUM_EXPERTS,), dtype=tl.int32)
    for i in range(TOTAL_M):
        eid     = tl.load(expert_ids_ptr + i)
        counts += tl.where(tl.arange(0, NUM_EXPERTS) == eid, 1, 0)

    offsets = tl.zeros((NUM_EXPERTS,), dtype=tl.int32)
    running = 0
    for e in tl.static_range(NUM_EXPERTS):
        e_mask       = tl.arange(0, NUM_EXPERTS) == e
        offsets      = tl.where(e_mask, running, offsets)
        running     += tl.sum(tl.where(e_mask, counts, tl.zeros((NUM_EXPERTS,), dtype=tl.int32)))

    tl.store(out_offsets_ptr + tl.arange(0, NUM_EXPERTS), offsets)
    tl.store(out_offsets_ptr + NUM_EXPERTS, running)

    write_pos = offsets

    for i in range(TOTAL_M):
        eid = tl.load(expert_ids_ptr + i)
        tid = tl.load(token_ids_ptr + i)
        w   = tl.load(weights_ptr + i)

        e_mask = tl.arange(0, NUM_EXPERTS) == eid
        pos    = tl.sum(tl.where(e_mask, write_pos, tl.zeros((NUM_EXPERTS,), dtype=tl.int32)))

        tl.store(out_tokens_ptr + pos, tid)
        tl.store(out_weights_ptr + pos, w)

        write_pos = tl.where(e_mask, write_pos + 1, write_pos)


def triton_counting_sort(flat_indices, flat_tokens, flat_weights, num_experts):
    """Counting sort for expert dispatch — faster than argsort for small E"""

    total_M       = flat_indices.shape[0]
    device        = flat_indices.device
    out_tokens    = torch.empty(total_M, dtype=flat_tokens.dtype, device=device)
    out_weights   = torch.empty(total_M, dtype=flat_weights.dtype, device=device)
    out_offsets   = torch.empty(num_experts + 1, dtype=torch.int32, device=device)

    _counting_sort_kernel[(1,)](
        flat_indices, flat_tokens, flat_weights,
        out_tokens, out_weights, out_offsets,
        TOTAL_M=total_M, NUM_EXPERTS=num_experts,
    )

    return out_tokens, out_weights, out_offsets


# ──────────────────────────────────────────────────────────────────────
#  Phase 4: Persistent kernel (design outline — not implemented)
# ──────────────────────────────────────────────────────────────────────
#
#  Goal: single dispatch with grid=(num_SMs,) that fuses SwiGLU GEMM + down GEMM
#
#  Approach (cooperative grid sync — deadlock-free on sm121):
#    1. Launch cooperative kernel, grid=(48,) on GB10
#    2. All SMs do phase-0 tiles (SwiGLU GEMM) via grid-stride loop
#    3. grid.sync() barrier
#    4. All SMs do phase-1 tiles (down GEMM + scatter) via grid-stride loop
#    5. Scratch: [total_M, ff_dim] intermediate, allocated once
#
#  Expected savings: ~5µs from eliminating 1 kernel launch — marginal for training
#  Not worth pursuing on GB10 (273 GB/s BW makes spin-waits expensive)
#  Revisit for inference latency optimization or H100+ migration
#
#  References:
#    - FlashAttention-3 persistent kernel (Tri Dao) — warp-specialized, Hopper-only
#    - SonicMoE (88% BW on H100 via TMA + persistent) — sm90+ only
#    - cuSync (Microsoft) — two-stream pipeline with global semaphores
#    - Mirage MPK (arxiv 2512.22219) — device-memory event queues, works on Ampere+
#    - Triton persistent matmul tutorial — grid-stride loop, no inter-block sync
