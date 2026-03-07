# Reinforcement Learning Design — GRPO for GeneT5

> **Note (2026-03):** This document describes GRPO for the original autoregressive
> model. The current model uses MDLM diffusion (bidirectional, iterative unmasking).
> GRPO's teacher-forcing log-prob computation and autoregressive generation do not
> apply to diffusion. This document and `lib/grpo/` are retained for reference but
> need rearchitecting before use with the diffusion model.

## Why RL After SFT

SFT trains with teacher forcing — the model sees the correct previous token at every step. At inference, it must generate autoregressively from its own predictions. This train/test mismatch means SFT alone cannot optimize the actual metric we care about: exon/gene F1.

GRPO (Group Relative Policy Optimization) closes this gap by:
1. Generating complete outputs autoregressively
2. Scoring them with verifiable F1 rewards
3. Updating the policy to favor high-reward outputs

## Why GRPO Over PPO

PPO requires a learned value network (~1.5B extra parameters) for advantage estimation. GRPO eliminates this by computing advantages within groups of G outputs per prompt. For genomic annotations, rewards are verifiable (F1 against reference), so no reward model is needed either. The result is simpler code, lower memory, and the same RL signal.

## Architecture

```
For each batch of B prompts:

  Phase 1 — Generate (no_grad, ~80% of wall time)
    Encode once → expand to B*G → autoregressive decode G outputs per prompt

  Phase 2 — Score (no_grad, CPU-friendly)
    Parse each output → extract exon/gene coords → F1 vs reference

  Phase 3 — Advantages (no_grad, trivial)
    Per-group: A_i = (r_i - mean(r)) / std(r)

  Phase 4 — Policy Update (with grad)
    Teacher-forced forward on policy + reference → log probs
    Loss = -A * log_prob + beta * KL(policy || ref) + moe_aux_loss
    DDP gradient sync → optimizer step
```

## Reward Function

| Component | Weight | Source |
|-----------|--------|--------|
| Exon boundary F1 | 0.6 | `lib/util/_reward.py` |
| Gene structure F1 | 0.4 | `lib/util/_reward.py` |

Composite: `reward = 0.6 * exon_f1 + 0.4 * gene_f1`

No format validity reward — SFT already learned the GFF format reliably.

## Key Optimizations

| Optimization | Impact | Implementation |
|---|---|---|
| Encoder caching | ~30% gen speedup | `generate_from_hidden()` in model.py |
| Frozen encoder | Save grad memory | Only decoder gets gradients |
| Large gen batch | GPU utilization | All B*G=64 sequences in one call |
| Short max_length | ~50% gen speedup | 256 tokens (most annotations are 100-300) |
| BF16 autocast | Memory + speed | Same as SFT |

## Parallelism

Standard DDP across both DGX Spark devices:
- Each device: policy (DDP-wrapped) + reference (frozen, no DDP)
- DistributedSampler splits prompts
- DDP syncs gradients at backward()
- BF16 gradient compression hook

Both models fit easily: 2 copies * 3 GB = 6 GB out of 128 GB per device.

## Hyperparameters

| Param | Value | vs SFT |
|---|---|---|
| Learning rate | 1e-6 | 100x smaller |
| KL beta | 0.05 | New (prevents policy drift) |
| Temperature | 0.8 | Needs output diversity |
| Weight decay | 0.01 | Lighter than SFT's 0.1 |
| Grad accumulation | 4 | 16x less than SFT's 64 |
| Max grad norm | 1.0 | Same |
| MoE aux loss weight | 0.01 | Same |
| Group size G | 8 | 8 samples per prompt |
| Max gen length | 256 | Shorter than SFT's 512 |

## Data

- 5K-10K prompts from held-out species (not in SFT training)
- 1 epoch (GRPO overfits fast)
- Effective batch: 64 prompts/update (B=8, G=8, grad_accum=4, 2 devices)

Prepared with `bin/prep_grpo` from raw genome + GFF files.

## File Map

| File | Purpose |
|---|---|
| `lib/util/_reward.py` | exon_f1, gene_f1, composite_reward |
| `lib/util/_grpo.py` | compute_log_probs, compute_advantages, grpo_loss, GRPODataset |
| `lib/model.py` | generate_from_hidden() (encoder caching) |
| `bin/grpo` | CLI entry point and training loop |
| `bin/prep_grpo` | Multi-species data preparation |

## Usage

```bash
# Prepare data from held-out species
PYTHONPATH=. python bin/prep_grpo \
    raw/Species.one raw/Species.two raw/Species.three \
    -o baked/grpo_data.json -n 5000

# Single device
PYTHONPATH=. python bin/grpo \
    baked/grpo_data.json \
    model/exp_grpo/ \
    model/base/ \
    --eval_data baked/eval_data.json

# Distributed (2 nodes)
torchrun --nnodes=2 --nproc_per_node=1 \
    --node_rank=0 --master_addr=IP --master_port=29500 \
    bin/grpo baked/grpo_data.json model/exp_grpo/ model/base/ \
    --compile --eval_data baked/eval_data.json
```

## Monitoring

GRPO logs to `grpo_log.csv` with columns:
- `reward_mean`, `reward_std` — should increase over training
- `kl_mean` — should stay < 1.0 (if > 5.0, reduce beta)
- `policy_loss`, `kl_loss` — loss components
- `gen_sec`, `update_sec` — timing per step

## Expected Timeline

Per epoch on 2 devices with 5K prompts:
- Generation: ~75 min (312 batches * ~15 sec/batch)
- Policy updates: ~15 min
- Total: ~2 hours (compare: SFT is ~34 hours/epoch)
