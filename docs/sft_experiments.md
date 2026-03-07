# GeneT5 SFT Experiment Log

## Model
- GeneT5: 768d, 12L encoder, 12L decoder, 16 MoE experts (top-2)
- 1.53B total parameters (all trainable)
- Vocab: 5113 tokens

## Data
- Training: w20k_ts51_v2 (1.95M chunks, 51 species, 65GB binary)
- Validation: 3K chunks (subset)
- Eval: 20 held-out S.scrofa (pig) samples, exon/gene F1

## Hardware
- 2x DGX Spark (GB10, 96GB unified memory each)
- NCCL over ConnectX-7 InfiniBand
- NGC pytorch:25.12-py3 container


---


## Trial 1: feb10_run1

### Config
| Param | Value |
|-------|-------|
| LR | 1e-4 base, 2e-4 effective (x2 nodes) |
| Batch | 8 x 64 grad_accum x 2 nodes = 1024 effective |
| Warmup | 3% (~57 steps) |
| Label smoothing | 0.1 (Liger fused CE) |
| Optimizer | AdamW 8-bit (betas 0.9/0.95, wd 0.1) |
| Compile | No |
| Epochs | 4 (ran ~1.1) |
| Decoder window | 32 (epochs 1-2), changed to 256 mid-run |

### Results
| Phase | Train Loss | Val Loss | LR |
|-------|-----------|----------|-----|
| Epoch 1 10% | 4.524 | - | 1.50e-4 |
| Epoch 1 50% | 2.490 | - | 1.97e-4 |
| Epoch 1 end | 1.829 | **0.287** | 1.76e-4 |
| Epoch 2 ~10% | 1.378 | - | 1.70e-4 |

- Speed: ~1.01 batch/s, ~34h per epoch
- Optimizer steps/epoch: ~1,908
- Val loss (0.287) much lower than train loss (1.829)
- Eval F1: 0.0 (model outputs empty sequences at this loss)
- Paused at step 2000 to add eval hook and widen decoder window

### Notes
- Large effective batch (1024) acts as regularizer -- val < train
- Slow learning due to few optimizer steps per epoch
- Label smoothing 0.1 applied from start


---


## Trial 2: feb10_run1 continued (label smoothing experiments)

> Logs deleted; reconstructed from memory

### Run 2a: continued with label smoothing 0.1
- Resumed from trial 1 checkpoint (step 2000)
- Continued training with same config
- Val loss reached ~1.38

### Run 2b: label smoothing rerun
- Restarted or resumed with label smoothing
- Epoch 2 val loss improved to ~0.2
- Training ended without compile (no torch.compile)
- Same LR=1e-4, batch=8, grad_accum=64, 2 nodes

### Key takeaway
- With conservative settings (1e-4 LR, 1024 effective batch), val loss tracks or beats train loss
- Label smoothing helped val loss converge
- Without compile, training was ~10% slower


---


## Trial 3: feb12_trail1 (current)

### Config
| Param | Value |
|-------|-------|
| LR | 3e-4 base, 6e-4 effective (x2 nodes) |
| Batch | 8 x 16 grad_accum x 2 nodes = 256 effective |
| Warmup | 6% (~1,831 steps) |
| Label smoothing | 0.0 (disabled) |
| Optimizer | AdamW 8-bit (betas 0.9/0.95, wd 0.1) |
| Compile | Yes (torch.compile encoder + decoder) |
| Epochs | 4 (stopping after 2) |
| Decoder window | default |
| Save steps | 500 |

### Results
| Phase | Train Loss | Val Loss | LR |
|-------|-----------|----------|-----|
| Epoch 1 10% | 2.668 | - | 2.50e-4 |
| Epoch 1 20% | 1.501 | - | 5.00e-4 |
| Epoch 1 30% | 1.073 | - | 6.00e-4 |
| Epoch 1 40% | 0.856 | - | 5.97e-4 |
| Epoch 1 50% | 0.724 | - | 5.93e-4 |
| Epoch 1 60% | 0.636 | - | 5.87e-4 |
| Epoch 1 70% | 0.573 | - | 5.78e-4 |
| Epoch 1 80% | 0.526 | - | 5.68e-4 |
| Epoch 1 90% | 0.489 | - | 5.56e-4 |
| Epoch 1 end | 0.459 | **9.502** | 5.42e-4 |
| Epoch 2 10% | 0.192 | - | 5.26e-4 |
| Epoch 2 30% | 0.192 | - | 4.90e-4 |
| Epoch 2 40% | 0.191 | - | 4.70e-4 |
| Epoch 2 50% | 0.315 | - | 4.49e-4 (spike) |
| Epoch 2 60% | 0.294 | - | 4.26e-4 |
| Epoch 2 70% | 0.280 | - | 4.03e-4 |
| Epoch 2 80% | 0.268 | - | 3.79e-4 |
| Epoch 2 90% | 0.260 | - | 3.55e-4 |
| Epoch 2 end | 0.252 | **6.186** | 3.30e-4 |

- Speed: ~1.15-1.2 batch/s (compile gives ~10% boost)
- Optimizer steps/epoch: ~7,632 (4x more than trial 1)
- Eval F1: 0.0 after both epochs (0 predicted genes)
- **Killed manually after epoch 2** (epoch 3 had just started)

### Observations
- **Severe overfitting**: train 0.252 vs val 6.186 (24x gap); epoch 1 was worse (20x)
- 4x more optimizer steps made train loss drop much faster (0.46 vs 1.83 at epoch 1 end)
- Higher LR + smaller effective batch = faster memorization, worse generalization
- Loss spike at epoch 2 50% (0.191 -> 0.315), recovered to 0.280 by 70%
- Flat plateau at 0.191 in epoch 2 (10-40%) suggests memorization floor
- Val improved epoch 1->2 (9.5 -> 6.2) but still catastrophically overfit
- **Zero eval F1 across all checkpoints** -- model never learned to produce valid gene structures
- torch.compile: ~35min initial compilation, 10% steady-state speedup
- RAM: master ~118GB (compile workers + Claude), worker ~56-70GB


---


## Comparison

| Metric | Trial 1 | Trial 3 |
|--------|---------|---------|
| Effective LR | 2e-4 | 6e-4 |
| Effective batch | 1024 | 256 |
| Steps/epoch | 1,908 | 7,632 |
| Epoch 1 train loss | 1.829 | 0.459 |
| Epoch 1 val loss | **0.287** | 9.502 |
| Epoch 2 train loss | 1.378 (partial) | 0.252 |
| Epoch 2 val loss | - | 6.186 |
| Eval F1 | 0.0 | 0.0 |
| Speed | 1.01 b/s | 1.15 b/s |
| Compile | No | Yes |
| Label smoothing | 0.1 | 0.0 |
| Generalization | Good | Overfit |

### Insights
1. **grad_accum controls update frequency, not just batch size** -- 16 vs 64 means 4x more weight updates from same data, dramatically faster train loss convergence
2. **Larger effective batch regularizes** -- 1024 effective batch averages more samples per update, preventing memorization of individual patterns
3. **Label smoothing helped** in trial 1/2 runs by preventing overconfident predictions
4. **Val loss < train loss is possible** when regularization (large batch + label smoothing + low LR) outweighs model capacity for the val set size
5. **torch.compile gives ~10% throughput boost** after initial 35min compilation; inductor cache persists across restarts
6. **For novel task SFT (no pretraining on target)**, conservative LR with large batch generalizes better; aggressive LR memorizes faster but doesn't transfer


---


## Trial 4: feb19_moe4_t1 (Muon E2E, killed early)

### Config
| Param | Value |
|-------|-------|
| LR | 0.02 base, 0.04 effective (x2 nodes -- LINEAR SCALING BUG) |
| Batch | 8 x 32 grad_accum x 2 nodes = 512 effective |
| Warmup | 3% |
| Label smoothing | 0.1 |
| Optimizer | MuonE2E (Newton-Schulz >=2D, L2-norm 1D) |
| Compile | Yes |
| Epochs | 3 (killed at ~13%) |
| Model | init_moe4 (374M params, 122 trainable tensors) |
| Data | feb17_s51_v1_e1 |

### Results
| Step | Train Loss | LR | Speed |
|------|-----------|-----|-------|
| 100 | 7.649 | ~1.0e-2 | 0.7 b/s |
| 200 | 5.900 | ~1.5e-2 | 0.7 b/s |
| 300 | 7.098 | ~2.0e-2 | 0.7 b/s |
| 400 | 6.686 | ~2.5e-2 | 0.7 b/s |
| 500 | 7.029 | ~3.0e-2 | 0.7 b/s |
| 600 | 6.942 | ~3.5e-2 | 0.7 b/s |
| 700 | 6.876 | ~3.7e-2 | 0.7 b/s |
| 800 | 6.968 | ~3.9e-2 | 0.8 b/s |

- RAM: ~54.8 GB stable (GPU: 3/36 GB)
- Killed manually due to non-convergence (oscillation around 6.9-7.0)

### Diagnosis
1. **Linear LR scaling was wrong for Muon**: `effective_lr = lr * world_size` doubled LR to 0.04. Muon produces unit-norm updates (Newton-Schulz orthogonalization), so step size is purely LR-controlled. Unlike AdamW where gradient magnitude shrinks near minima, Muon's steps are constant-magnitude. Linear scaling just doubles step size with no compensating benefit
2. **Oscillation pattern**: Loss dropped fast to 5.9 (step 200) proving model can learn, but then bounced between 6.7-7.1 as LR climbed through warmup. The unit-norm steps at LR=0.04 overshoot the loss basin, recover, overshoot again
3. **Memory headroom**: 54.8 GB on 128 GB UMA suggests batch_size=16 is feasible (doubles compute utilization)

### Fix Applied
- Removed linear LR scaling from `bin/finet` -- Muon uses raw `args.lr` regardless of world_size
- Nesterov momentum confirmed correct (re-analyzed: `g.lerp_(buf, momentum)` after `buf.lerp_(g, 1-momentum)` is proper Nesterov)
- Next run: lr=0.02 (no scaling), batch_size=16, --no_pin_memory (UMA optimization)

### Optimizer Migration Summary (AdamW -> Muon E2E)
- Removed `WrappedOptimizer` class, `bitsandbytes` dependency, `--optim_8bit` / `--muon` flags
- Default LR changed from 1e-4 (AdamW) to 0.02 (Muon)
- Removed `BNB_CUDA_VERSION` from all container launch scripts
- MuonE2E: 97 tensors (371M params) via Newton-Schulz, 25 tensors (19K params) via L2-norm


---


## Conclusion: Seq-to-GFF approach is fundamentally limited

### The core problem
Across all trials, **eval F1 remained at 0.0** -- the model never produced valid gene predictions on held-out data. Even trial 1 with good val loss (0.287) generated only empty `<bos><eos>` sequences.

The Seq-to-GFF formulation asks the decoder to:
1. Output arbitrary integer coordinates (start/end positions) with no token-level grounding
2. Maintain rigid tab-delimited GFF format with exact column semantics
3. Learn positional arithmetic (e.g., "12847" to "13291") from token sequences where adjacent numbers share no representation similarity
4. Predict phase values (0/1/2) that encode reading frame state across exon-intron junctions

The model can memorize these patterns from training data (train loss 0.25) but cannot generalize because **numerical coordinates have no meaningful relationship in token embedding space**. "12847" and "12848" differ by one base pair but are completely different token sequences.

### Proposed redesign: Seq-to-Seq (DNA in, exon DNA out)
Instead of predicting coordinates, predict the actual exon nucleotide sequences:
- Input: 20kb DNA window (encoder)
- Output: extracted exon sequences separated by delimiter tokens (decoder)
- Post-processing: align predicted exons back to input using minimap2/BLAST to recover coordinates

This approach:
- Keeps encoder and decoder in the same representation space (nucleotide tokens)
- Cross-attention can directly attend to relevant input positions
- No arbitrary integer arithmetic required
- Alignment back to coordinates is a solved classical problem

### Architecture considerations for Seq-to-Seq
- Encoder (~200M params) and decoder share embedding space but need different designs
- Freezing encoder may not work -- encoder was co-trained with GFF decoder, not standalone DNA representation
- May need encoder pretraining phase on masked language modeling (MLM) before SFT
- Decoder output length is much shorter than input (exons are small fraction of genomic window)
