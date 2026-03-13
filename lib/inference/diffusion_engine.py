import torch
import torch.nn.functional as F


def linear_unmask_schedule(T, eps=1e-3):
    """Linear unmask fractions per step (Dream 7B style)

    Timesteps go from 1.0 → eps, unmask fraction = (t_prev - t_now) per step
    """

    timesteps = torch.linspace(1.0, eps, T + 1)
    fractions = []
    for i in range(T):
        fractions.append((timesteps[i] - timesteps[i + 1]).item())
    return fractions


@torch.no_grad()
def diffusion_generate(model, prefix_ids, target_len, mask_token_id,
                       pad_token_id, num_steps=32, temperature=1.0,
                       strategy="confidence"):
    """Iterative parallel decoding via progressive unmasking

    strategy: "confidence" (top-k by log-prob) or "margin" (top1 - top2 gap)
    """

    device = prefix_ids.device
    B      = prefix_ids.size(0)
    P      = prefix_ids.size(1)

    # Start with fully masked target
    target    = torch.full((B, target_len), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prefix_ids, target], dim=1)

    schedule  = linear_unmask_schedule(num_steps)
    is_masked = torch.ones(B, target_len, dtype=torch.bool, device=device)

    for step_idx, unmask_frac in enumerate(schedule):
        outputs = model(input_ids=input_ids)
        logits  = outputs['logits'][:, P:]

        # Suppress mask and pad tokens
        logits[:, :, mask_token_id] = float('-inf')
        logits[:, :, pad_token_id]  = float('-inf')

        # Sample or argmax
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            preds = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, target_len)
        else:
            preds = logits.argmax(dim=-1)

        # Compute per-token confidence
        if strategy == "margin":
            top2     = logits.topk(2, dim=-1).values
            conf_val = (top2[:, :, 0] - top2[:, :, 1])
        else:
            log_probs = F.log_softmax(logits, dim=-1)
            conf_val  = log_probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)

        conf_val[~is_masked] = float('-inf')

        # How many to unmask this step
        num_still_masked = is_masked.sum(dim=1)
        num_to_unmask    = (unmask_frac * target_len * torch.ones(B, device=device)).long()
        num_to_unmask    = torch.min(num_to_unmask, num_still_masked)

        # Vectorized top-k unmask per sample
        for i in range(B):
            if num_to_unmask[i] == 0:
                continue

            masked_indices = is_masked[i].nonzero(as_tuple=True)[0]
            if len(masked_indices) == 0:
                continue

            k          = min(num_to_unmask[i].item(), len(masked_indices))
            _, top_k   = conf_val[i, masked_indices].topk(k)
            unmask_pos = masked_indices[top_k]

            input_ids[i, P + unmask_pos] = preds[i, unmask_pos]
            is_masked[i, unmask_pos]     = False

        del outputs, logits

    # Final pass: fill any remaining masks
    if is_masked.any():
        outputs    = model(input_ids=input_ids)
        logits     = outputs['logits'][:, P:]
        logits[:, :, mask_token_id] = float('-inf')
        final_pred = logits.argmax(dim=-1)
        input_ids[:, P:][is_masked] = final_pred[is_masked]
        del outputs, logits

    return input_ids
