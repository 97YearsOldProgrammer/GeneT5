from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
DW_BLOCK_M = 64


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

    counts         = offsets[1:] - offsets[:-1]
    m_blocks       = ((counts + block_m - 1) // block_m).sum().item()
    n_blocks       = math.ceil(N / block_n)
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
