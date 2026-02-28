import math

import torch
import torch.nn.functional as F


def cosine_unmask_schedule(T):
    """Compute fraction of tokens to unmask at each step"""

    fractions = []
    for i in range(T):
        t_now  = 1.0 - (i + 1) / T
        t_prev = 1.0 - i / T
        alpha_now  = math.cos(t_now * math.pi / 2.0)
        alpha_prev = math.cos(t_prev * math.pi / 2.0)
        fractions.append(alpha_now - alpha_prev)
    return fractions


@torch.no_grad()
def diffusion_generate(model, prefix_ids, target_len, mask_token_id,
                       pad_token_id, num_steps=32, temperature=1.0):
    """Iterative parallel decoding via MDLM-style progressive unmasking"""

    device = prefix_ids.device
    B      = prefix_ids.size(0)
    P      = prefix_ids.size(1)

    # Start with fully masked target
    target = torch.full((B, target_len), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prefix_ids, target], dim=1)

    # Unmask schedule
    schedule = cosine_unmask_schedule(num_steps)

    # Track which positions are still masked
    is_masked = torch.ones(B, target_len, dtype=torch.bool, device=device)

    for step_idx, unmask_frac in enumerate(schedule):
        # Forward pass (full bidirectional)
        outputs = model(input_ids=input_ids, prefix_len=P, is_diffusion=True)
        logits  = outputs['logits'][:, P:]

        # Suppress mask and pad tokens from predictions
        logits[:, :, mask_token_id] = float('-inf')
        logits[:, :, pad_token_id]  = float('-inf')

        # Sample or argmax at masked positions
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            preds = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, target_len)
        else:
            preds = logits.argmax(dim=-1)

        # Compute confidence at masked positions
        log_probs  = F.log_softmax(logits, dim=-1)
        confidence = log_probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        confidence[~is_masked] = float('-inf')

        # How many to unmask this step
        num_still_masked = is_masked.sum(dim=1)
        num_to_unmask    = (unmask_frac * target_len * torch.ones(B, device=device)).long()
        num_to_unmask    = torch.min(num_to_unmask, num_still_masked)

        # Unmask top-k most confident per sample
        for i in range(B):
            if num_to_unmask[i] == 0:
                continue

            masked_indices = is_masked[i].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue

            masked_conf = confidence[i, masked_indices]
            k           = min(num_to_unmask[i].item(), len(masked_indices))
            _, top_k    = masked_conf.topk(k)

            unmask_pos = masked_indices[top_k]
            input_ids[i, P + unmask_pos] = preds[i, unmask_pos]
            is_masked[i, unmask_pos]     = False

        del outputs, logits

    # Final pass: fill any remaining masks
    if is_masked.any():
        outputs = model(input_ids=input_ids, prefix_len=P, is_diffusion=True)
        logits  = outputs['logits'][:, P:]
        logits[:, :, mask_token_id] = float('-inf')
        final_pred = logits.argmax(dim=-1)
        input_ids[:, P:][is_masked] = final_pred[is_masked]
        del outputs, logits

    return input_ids
