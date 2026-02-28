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
    t = torch.rand(B, device=device)

    # Compute mask rate per sample
    mask_rate = cosine_mask_rate(t)

    # Build mask: only mask target positions (after prefix), not padding
    masked_ids = target_ids.clone()
    labels     = torch.full_like(target_ids, -100)

    for i in range(B):
        p = prefix_len if isinstance(prefix_len, int) else prefix_len[i]

        # Find target region (non-pad after prefix)
        target_region = target_ids[i, p:]
        target_len    = (target_region != pad_token_id).sum().item()

        if target_len == 0:
            continue

        # Random mask within target region
        rate = mask_rate[i].item()
        mask = torch.rand(target_len, device=device) < rate

        # Apply mask to input
        masked_ids[i, p:p + target_len][mask] = mask_token_id

        # Labels: only at masked positions
        labels[i, p:p + target_len][mask] = target_ids[i, p:p + target_len][mask]

    weights = mdlm_loss_weight(t)

    return masked_ids, labels, weights
