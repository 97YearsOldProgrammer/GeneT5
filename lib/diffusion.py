import math

import torch


############################
#####  Noise Schedule  #####
############################


def cosine_mask_rate(t):
    """Cosine schedule: mask_rate = 1 - cos(pi*t/2)"""

    return 1.0 - torch.cos(t * math.pi / 2.0)


def mdlm_loss_weight(t):
    """MDLM loss weight: (pi/2) * tan(pi*t/2), clamped"""

    return (math.pi / 2.0) * torch.tan(t * math.pi / 2.0).clamp(max=10.0)


############################
#####  Masking         #####
############################


def apply_diffusion_mask(target_ids, prefix_len, mask_token_id, pad_token_id):
    """Apply MDLM-style random masking to target portion of input_ids"""

    B, L   = target_ids.shape
    device = target_ids.device

    # Sample timestep per sample
    t         = torch.rand(B, device=device)
    mask_rate = cosine_mask_rate(t)

    # Positions after prefix (broadcast-safe for int or tensor prefix_len)
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

    weights = mdlm_loss_weight(t)

    return masked_ids, labels, weights


def apply_diffusion_mask_packed(input_ids, cu_seqlens, prefix_lens, num_real,
                                mask_token_id, pad_token_id):
    """MDLM masking for packed sequences — samples t per sub-sequence

    Fully vectorized: no .item() calls, no Python loops over segments
    """

    B, S     = input_ids.shape
    device   = input_ids.device
    N_real   = prefix_lens.shape[0]

    t         = torch.rand(N_real, device=device)
    mask_rate = cosine_mask_rate(t)
    weights   = mdlm_loss_weight(t)

    flat      = input_ids.reshape(-1)
    total_len = flat.shape[0]

    # Build real-segment start/end pairs from cu_seqlens and num_real
    # Each row has num_real[b] real segs + 1 padding seg in cu_seqlens
    # Real seg i in row b maps to cu_seqlens index = (cumsum of num_real before b) + b + local_j
    cum_nr     = torch.zeros(B + 1, dtype=torch.long, device=device)
    cum_nr[1:] = num_real.cumsum(0)
    # row_id[i] = which row does real segment i belong to
    row_id       = torch.repeat_interleave(torch.arange(B, device=device), num_real.long())
    local_offset = torch.arange(N_real, device=device) - cum_nr[row_id]
    real_seg_idx = (cum_nr[row_id] + row_id + local_offset).long()

    starts     = cu_seqlens[real_seg_idx]
    ends       = cu_seqlens[real_seg_idx + 1]
    seg_lens   = ends - starts
    target_starts = starts + prefix_lens
    target_lens   = ends - target_starts

    # Build flat position index for all target positions across all segments
    max_tlen   = target_lens.max().item()
    offsets    = torch.arange(max_tlen, device=device).unsqueeze(0)
    abs_pos    = target_starts.unsqueeze(1) + offsets
    valid_mask = offsets < target_lens.unsqueeze(1)

    # Clamp for safe indexing, then mask
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
