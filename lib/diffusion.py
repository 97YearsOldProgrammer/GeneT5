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
