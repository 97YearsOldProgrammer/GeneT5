import math

import torch


############################
#####  Noise Schedule  #####
############################


def linear_mask_rate(t):
    """Linear schedule: mask_rate = t (Dream 7B)"""

    return t


def linear_loss_weight(t):
    """ELBO weight for linear schedule: w(t) = 1/t, clamped"""

    return (1.0 / t.clamp(min=1e-4)).clamp(max=20.0)


############################
#####  CART Reweight   #####
############################


def cart_weights(masked_ids, mask_token_id, cart_p=0.1):
    """Context-Adaptive Reweighting at Token level (Dream 7B)

    Masked tokens with more unmasked neighbors get higher weight
    Geometric distance weighting: w_i = sum_j 1[j unmasked] * Geo(p, |i-j|-1)
    """

    is_clean = (masked_ids != mask_token_id).float()
    L        = masked_ids.shape[-1]
    device   = masked_ids.device

    # Build geometric kernel: Geo(p, d-1) = p * (1-p)^(d-1) for d >= 1
    dists  = torch.arange(L, device=device, dtype=torch.float32)
    kernel = 0.5 * torch.exp(math.log(cart_p) + (dists.clamp(min=1) - 1) * math.log(1 - cart_p))
    kernel[0] = 0.0

    # Symmetric: context from left and right
    # Convolve clean indicator with kernel
    if is_clean.dim() == 2:
        B = is_clean.shape[0]
        k = torch.cat([kernel.flip(0), kernel[1:]])
        k = k.unsqueeze(0).unsqueeze(0)
        ctx = torch.nn.functional.conv1d(
            is_clean.unsqueeze(1), k, padding=L - 1
        ).squeeze(1)
    else:
        raise ValueError("Expected 2D input [B, L]")

    return ctx.clamp(min=0.01)


############################
#####  Masking         #####
############################


def apply_diffusion_mask(target_ids, prefix_len, mask_token_id, pad_token_id,
                         cart_p=0.0):
    """Apply discrete diffusion masking to target portion of input_ids

    Linear schedule with 1/t ELBO weight (Dream 7B style)
    Optional CART per-token reweighting when cart_p > 0
    """

    B, L   = target_ids.shape
    device = target_ids.device

    # Sample timestep per sample, avoid t=0
    t         = torch.rand(B, device=device).clamp(min=1e-4)
    mask_rate = linear_mask_rate(t)

    # Positions after prefix
    pos          = torch.arange(L, device=device).unsqueeze(0)
    after_prefix = pos >= (prefix_len if isinstance(prefix_len, int)
                           else prefix_len.unsqueeze(1) if torch.is_tensor(prefix_len)
                           else prefix_len)

    # Target region = after prefix AND not padding
    target_mask = after_prefix & (target_ids != pad_token_id)

    # Per-sample random mask at each position, thresholded by mask_rate
    selected = (torch.rand(B, L, device=device) < mask_rate.unsqueeze(1)) & target_mask

    # Apply mask to inputs, set labels at masked positions
    masked_ids = target_ids.clone()
    labels     = torch.full_like(target_ids, -100)

    masked_ids[selected] = mask_token_id
    labels[selected]     = target_ids[selected]

    # Sample-level ELBO weight
    weights = linear_loss_weight(t)

    # Per-token CART reweighting
    if cart_p > 0:
        token_w = cart_weights(masked_ids, mask_token_id, cart_p)
        return masked_ids, labels, weights, token_w

    return masked_ids, labels, weights


def apply_diffusion_mask_packed(input_ids, cu_seqlens, prefix_lens, num_real,
                                mask_token_id, pad_token_id):
    """Diffusion masking for packed sequences — samples t per sub-sequence

    Fully vectorized: no .item() calls, no Python loops over segments
    """

    B, S     = input_ids.shape
    device   = input_ids.device
    N_real   = prefix_lens.shape[0]

    t         = torch.rand(N_real, device=device).clamp(min=1e-4)
    mask_rate = linear_mask_rate(t)
    weights   = linear_loss_weight(t)

    flat      = input_ids.reshape(-1)
    total_len = flat.shape[0]

    # Build real-segment start/end pairs from cu_seqlens and num_real
    cum_nr     = torch.zeros(B + 1, dtype=torch.long, device=device)
    cum_nr[1:] = num_real.cumsum(0)
    row_id       = torch.repeat_interleave(torch.arange(B, device=device), num_real.long())
    local_offset = torch.arange(N_real, device=device) - cum_nr[row_id]
    real_seg_idx = (cum_nr[row_id] + row_id + local_offset).long()

    starts        = cu_seqlens[real_seg_idx]
    ends          = cu_seqlens[real_seg_idx + 1]
    seg_lens      = ends - starts
    target_starts = starts + prefix_lens
    target_lens   = ends - target_starts

    # Build flat position index for all target positions across all segments
    max_tlen   = target_lens.max().item()
    offsets    = torch.arange(max_tlen, device=device).unsqueeze(0)
    abs_pos    = target_starts.unsqueeze(1) + offsets
    valid_mask = offsets < target_lens.unsqueeze(1)

    abs_pos_clamped = abs_pos.clamp(max=total_len - 1)
    tokens_at_pos   = flat[abs_pos_clamped]
    not_pad         = (tokens_at_pos != pad_token_id) & valid_mask

    # Per-segment random mask thresholded by mask_rate
    rand_vals = torch.rand(N_real, max_tlen, device=device)
    selected  = (rand_vals < mask_rate.unsqueeze(1)) & not_pad

    # Apply masking
    masked_ids = flat.clone()
    sel_pos    = abs_pos_clamped[selected]
    labels     = torch.full((total_len,), -100, dtype=torch.long, device=device)
    labels[sel_pos]     = flat[sel_pos]
    masked_ids[sel_pos] = mask_token_id

    return masked_ids.reshape(B, S), labels.reshape(B, S), weights
