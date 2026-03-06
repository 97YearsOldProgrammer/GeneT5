from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from torch.library import triton_op, wrap_triton, custom_op


# Tuned for GB10 sm_121 (99KB SMEM/block, 48 SMs, 273 GB/s BW)
# Benchmarked 2026-03-04: 16 experts, 1536 tok/expert, K=768, FF_DIM=768
# Forward GEMM: (128,128,32) warps=8 stages=3 -> 0.45ms (1.17x vs baseline)
BLOCK_M    = 128
BLOCK_N    = 128
BLOCK_K    = 32
FWD_WARPS  = 8
FWD_STAGES = 3
# Fused SwiGLU: (64,128,32) warps=8 stages=3 -> 0.95ms (1.22x vs baseline)
FUSE_M     = 64
FUSE_N     = 128
FUSE_K     = 32
FUSE_WARPS = 8
FUSE_STAGES = 3
# dW kernel: (64,128,128) warps=4 stages=2 -> 1.23ms (1.85x vs baseline)
DW_BLOCK_M = 64
DW_BLOCK_K = 128
DW_BLOCK_N = 128
DW_WARPS   = 4
DW_STAGES  = 2


# ──────────────────────────────────────────────────────────────────────
#  Triton JIT kernels (unchanged)
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


# ──────────────────────────────────────────────────────────────────────
#  Fused SwiGLU GEMM kernel
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


# ──────────────────────────────────────────────────────────────────────
#  Op wrappers: custom_op (opaque to inductor) for fwd, triton_op for bwd
# ──────────────────────────────────────────────────────────────────────


@triton_op("genet5::swiglu_bwd", mutates_args=())
def _swiglu_bwd_op(d_hidden: torch.Tensor, gate_save: torch.Tensor, up_save: torch.Tensor) -> torch.Tensor:
    total_M, FF_DIM = d_hidden.shape
    d_fused  = torch.empty(total_M, 2 * FF_DIM, device=d_hidden.device, dtype=d_hidden.dtype)
    m_blocks = math.ceil(total_M / FUSE_M)
    n_blocks = math.ceil(FF_DIM / FUSE_N)

    wrap_triton(_swiglu_bwd_kernel)[(m_blocks, n_blocks)](
        d_hidden, gate_save, up_save, d_fused,
        FF_DIM,
        d_hidden.stride(0), d_hidden.stride(1),
        gate_save.stride(0), gate_save.stride(1),
        up_save.stride(0), up_save.stride(1),
        d_fused.stride(0), d_fused.stride(1),
        TOTAL_M=total_M, BLOCK_M=FUSE_M, BLOCK_N=FUSE_N,
        num_warps=FUSE_WARPS, num_stages=FUSE_STAGES,
    )
    return d_fused


@_swiglu_bwd_op.register_fake
def _(d_hidden, gate_save, up_save):
    total_M, FF_DIM = d_hidden.shape
    return d_hidden.new_empty(total_M, 2 * FF_DIM)


@triton_op("genet5::grouped_gemm_dw", mutates_args=())
def _grouped_gemm_dw_op(A: torch.Tensor, dC: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    offsets = offsets.int()
    K = A.shape[1]
    N = dC.shape[1]
    E = offsets.shape[0] - 1
    dB = torch.empty(E, K, N, device=A.device, dtype=A.dtype)

    k_tile_count = math.ceil(K / DW_BLOCK_K)
    n_tile_count = math.ceil(N / DW_BLOCK_N)
    grid         = (E * k_tile_count, n_tile_count)

    wrap_triton(_grouped_gemm_dw_kernel)[grid](
        A, dC, dB, offsets,
        K, N,
        A.stride(0), A.stride(1),
        dC.stride(0), dC.stride(1),
        dB.stride(0), dB.stride(1), dB.stride(2),
        BLOCK_M=DW_BLOCK_M, BLOCK_K=DW_BLOCK_K, BLOCK_N=DW_BLOCK_N,
        num_warps=DW_WARPS, num_stages=DW_STAGES,
    )
    return dB


@_grouped_gemm_dw_op.register_fake
def _(A, dC, offsets):
    E = offsets.shape[0] - 1
    K = A.shape[1]
    N = dC.shape[1]
    return A.new_empty(E, K, N)


@custom_op("genet5::grouped_gemm", mutates_args=())
def _grouped_gemm_fwd(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    total_M, K = A.shape
    E, _, N    = B.shape
    offsets    = offsets.int()
    C          = torch.empty(total_M, N, device=A.device, dtype=A.dtype)
    grid       = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, N)

    wrap_triton(_grouped_gemm_kernel)[grid](
        A, B, C, offsets,
        K, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1),
        NUM_EXPERTS=E,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=FWD_WARPS, num_stages=FWD_STAGES,
    )
    return C


@_grouped_gemm_fwd.register_fake
def _(A, B, offsets):
    total_M = A.shape[0]
    N       = B.shape[2]
    return A.new_empty(total_M, N)


def _grouped_gemm_setup_ctx(ctx, inputs, output):
    A, B, offsets = inputs
    ctx.save_for_backward(A, B, offsets)


def _grouped_gemm_bwd(ctx, dC):
    A, B, offsets = ctx.saved_tensors
    dC = dC.to(A.dtype)
    dA = _grouped_gemm_fwd(dC, B.mT, offsets)
    dB = _grouped_gemm_dw_op(A, dC, offsets)
    return dA, dB, None


_grouped_gemm_fwd.register_autograd(_grouped_gemm_bwd, setup_context=_grouped_gemm_setup_ctx)


@custom_op("genet5::fused_swiglu_gemm_fwd", mutates_args=())
def _fused_swiglu_gemm_fwd(A: torch.Tensor, B: torch.Tensor,
                           offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_M, K = A.shape
    E          = B.shape[0]
    FF_DIM     = B.shape[2] // 2
    offsets    = offsets.int()
    C          = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)
    gate_save  = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)
    up_save    = torch.empty(total_M, FF_DIM, device=A.device, dtype=A.dtype)
    grid       = _compute_fwd_grid(offsets, E, FUSE_M, FUSE_N, FF_DIM)

    wrap_triton(_fused_swiglu_gemm_kernel)[grid](
        A, B, C, gate_save, up_save, offsets,
        K, FF_DIM,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1),
        gate_save.stride(0), gate_save.stride(1),
        up_save.stride(0), up_save.stride(1),
        NUM_EXPERTS=E,
        BLOCK_M=FUSE_M, BLOCK_N=FUSE_N, BLOCK_K=FUSE_K,
        num_warps=FUSE_WARPS, num_stages=FUSE_STAGES,
    )
    return C, gate_save, up_save


@_fused_swiglu_gemm_fwd.register_fake
def _(A, B, offsets):
    total_M = A.shape[0]
    FF_DIM  = B.shape[2] // 2
    return (A.new_empty(total_M, FF_DIM),
            A.new_empty(total_M, FF_DIM),
            A.new_empty(total_M, FF_DIM))


def _fused_swiglu_setup_ctx(ctx, inputs, output):
    A, B, offsets          = inputs
    C, gate_save, up_save  = output
    ctx.save_for_backward(A, B, offsets, gate_save, up_save)


def _fused_swiglu_bwd(ctx, dC, _dg, _du):
    A, B, offsets, gate_save, up_save = ctx.saved_tensors
    dC      = dC.to(A.dtype)
    d_fused = _swiglu_bwd_op(dC, gate_save, up_save)
    dA      = _grouped_gemm_fwd(d_fused, B.mT, offsets)
    dB      = _grouped_gemm_dw_op(A, d_fused, offsets)
    return dA, dB, None


_fused_swiglu_gemm_fwd.register_autograd(_fused_swiglu_bwd, setup_context=_fused_swiglu_setup_ctx)


@custom_op("genet5::grouped_gemm_scatter_fwd", mutates_args=())
def _grouped_gemm_scatter_fwd(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
                              sorted_weights: torch.Tensor, sorted_tokens: torch.Tensor,
                              num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_M, K = A.shape
    E, _, N    = B.shape
    offsets    = offsets.int()
    gemm_out   = torch.empty(total_M, N, device=A.device, dtype=A.dtype)
    grid       = _compute_fwd_grid(offsets, E, BLOCK_M, BLOCK_N, N)

    wrap_triton(_grouped_gemm_kernel)[grid](
        A, B, gemm_out, offsets,
        K, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        gemm_out.stride(0), gemm_out.stride(1),
        NUM_EXPERTS=E,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=FWD_WARPS, num_stages=FWD_STAGES,
    )

    idx_e    = sorted_tokens.unsqueeze(-1).expand(-1, N)
    weighted = gemm_out.float() * sorted_weights.float().unsqueeze(-1)
    output   = torch.zeros(num_tokens, N, device=A.device, dtype=torch.float32)
    output.scatter_add_(0, idx_e, weighted)
    output   = output.to(A.dtype)

    return output, gemm_out


@_grouped_gemm_scatter_fwd.register_fake
def _(A, B, offsets, sorted_weights, sorted_tokens, num_tokens):
    N = B.shape[2]
    return (A.new_empty(num_tokens, N), A.new_empty(A.shape[0], N))


def _scatter_setup_ctx(ctx, inputs, output):
    A, B, offsets, sorted_weights, sorted_tokens, num_tokens = inputs
    final_output, gemm_out                                   = output
    ctx.save_for_backward(A, B, offsets, sorted_weights, sorted_tokens, gemm_out)


def _scatter_bwd(ctx, d_output, _d_gemm):
    A, B, offsets, sorted_weights, sorted_tokens, gemm_out = ctx.saved_tensors
    N            = B.shape[2]
    d_output     = d_output.to(A.dtype)

    idx_e     = sorted_tokens.unsqueeze(-1).expand(-1, N)
    d_gather  = torch.gather(d_output, 0, idx_e)
    d_scaled  = (d_gather * sorted_weights.unsqueeze(-1)).to(A.dtype)
    d_weights = (d_gather * gemm_out).sum(dim=-1)

    dA = _grouped_gemm_fwd(d_scaled, B.mT, offsets)
    dB = _grouped_gemm_dw_op(A, d_scaled, offsets)

    return dA, dB, None, d_weights, None, None


_grouped_gemm_scatter_fwd.register_autograd(_scatter_bwd, setup_context=_scatter_setup_ctx)


# ──────────────────────────────────────────────────────────────────────
#  Public API (unchanged signatures)
# ──────────────────────────────────────────────────────────────────────


def triton_grouped_gemm(A, B, offsets):
    """Drop-in replacement for torch._grouped_mm with full backward support"""

    return _grouped_gemm_fwd(A, B, offsets)


def triton_fused_swiglu_gemm(A, B, offsets):
    """Fused gate+up GEMM with SiLU: A @ B[:,:,:ff] * silu(A @ B[:,:,ff:])"""

    C, _, _ = _fused_swiglu_gemm_fwd(A, B, offsets)
    return C


def triton_grouped_gemm_scatter(A, B, offsets, sorted_weights, sorted_tokens, num_tokens):
    """Down GEMM + fp32 weighted scatter — deterministic, no atomics"""

    output, _ = _grouped_gemm_scatter_fwd(A, B, offsets, sorted_weights, sorted_tokens, num_tokens)
    return output


