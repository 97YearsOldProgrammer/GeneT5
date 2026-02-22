import json
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


class GRPODataset(data_utils.Dataset):
    """Dataset for GRPO training — loads eval-format JSON with sequence + ref_features"""

    def __init__(self, data_path, tokenizer):

        with open(data_path, 'r') as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample   = self.samples[idx]
        sequence = sample["sequence"]
        encoded  = self.tokenizer.encode(sequence)

        return {
            "input_ids":    encoded,
            "ref_features": sample["ref_features"],
            "sequence":     sequence,
            "sample_idx":   idx,
        }


def grpo_collate(batch, pad_id=0):
    """Collate GRPO batch — pad input_ids, keep ref_features as list"""

    input_ids_list = [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch]
    max_len        = max(ids.size(0) for ids in input_ids_list)

    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded[i, :ids.size(0)] = ids

    ref_features = [b["ref_features"] for b in batch]
    sequences    = [b["sequence"] for b in batch]

    return {
        "input_ids":    padded,
        "ref_features": ref_features,
        "sequences":    sequences,
    }


def compute_log_probs(model, input_ids, labels, vocab_size):
    """Compute per-token log probabilities via teacher-forced forward pass"""

    outputs = model(
        input_ids = input_ids,
        labels    = None,
    )

    logits    = outputs["logits"]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs at label positions
    token_log_probs = log_probs.gather(2, labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    # Mask padding
    mask            = (labels != -100).float()
    token_log_probs = token_log_probs * mask

    return token_log_probs, mask, outputs.get("moe_loss")


def compute_advantages(rewards, group_size):
    """Compute GRPO advantages: per-group normalize rewards"""

    batch_size  = len(rewards) // group_size
    advantages  = torch.zeros_like(rewards)

    for i in range(batch_size):
        start      = i * group_size
        end        = start + group_size
        group      = rewards[start:end]
        group_mean = group.mean()
        group_std  = group.std()

        if group_std > 1e-8:
            advantages[start:end] = (group - group_mean) / group_std
        else:
            advantages[start:end] = 0.0

    return advantages


def grpo_loss(policy_log_probs, ref_log_probs, advantages, mask, beta=0.05):
    """Compute GRPO loss with KL penalty"""

    # Per-sequence sum of log probs
    seq_policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    seq_lengths          = mask.sum(dim=1).clamp(min=1)

    # Per-token KL divergence: policy || reference
    kl_per_token = policy_log_probs - ref_log_probs
    seq_kl       = (kl_per_token * mask).sum(dim=1) / seq_lengths

    # GRPO objective: maximize advantage-weighted log prob, penalize KL
    policy_loss = -(advantages * seq_policy_log_probs).mean()
    kl_loss     = beta * seq_kl.mean()

    total = policy_loss + kl_loss

    return total, policy_loss.item(), kl_loss.item(), seq_kl.mean().item()


def prepare_grpo_inputs(prefix_ids, generated, pad_id, label_pad=-100):
    """Build teacher-forcing inputs from prefix + generated for log prob computation

    prefix_ids: [B*G, prefix_len] — input DNA tokens + bos
    generated:  [B*G, prefix_len + gen_len] — full generated sequence (prefix + output)

    Returns input_ids [B*G, seq_len-1], labels [B*G, seq_len-1] with prefix masked
    """

    seq_len    = generated.size(1)
    prefix_len = prefix_ids.size(1)

    # Model input: all tokens except last (teacher forcing)
    input_ids = generated[:, :-1].clone()

    # Labels: shifted by 1 (next-token prediction)
    labels = generated[:, 1:].clone()

    # Mask prefix region in labels (loss only on generated output)
    labels[:, :prefix_len - 1] = label_pad

    # Mask padding in labels
    labels[labels == pad_id] = label_pad

    return input_ids, labels
