## GeneT5

End-to-end gene prediction using an encoder-decoder transformer, replacing traditional HMM-based tools like Augustus

### Architecture

```
DNA Sequence --> DNABERT-2 Encoder (12L) --> Perceiver Compressor (1024 latents)
                                                    |
                                                    v
Gene Annotation <-- Decoder (12L, MoE x16, GQA) <--+
```

- **Encoder**: DNABERT-2 117M with sparse windowed attention
- **Compressor**: Perceiver with 1024 learned latents (variable-length input -> fixed representation)
- **Decoder**: 12 layers, grouped-query attention (12H/3KV), Mixture-of-Experts (16 experts, top-2)
- **Total**: 1.55B parameters

### Why Deep Learning for Gene Prediction

1. Attention mechanism captures long-range dependencies that Markov models miss
2. Pre-trained encoder (DNABERT-2) provides cross-species transfer learning
3. MoE enables capacity scaling without proportional compute cost
4. Single model handles prokaryotes through vertebrates

### Setup

**Container** (recommended)
```bash
./start-gt5.sh           # single node
./start-gt5-multi.sh     # multi-node (run on both nodes)
```

Requires `nvcr.io/nvidia/pytorch:25.12-py3` (PyTorch 2.10, CUDA 13.1)

**Hardware**: NVIDIA DGX Spark (GB10, 128GB UMA) or any Blackwell GPU

### Quick Start

```bash
# Inside container
cd /workspace/GeneT5

# Train on 5% subset (fast iteration)
PYTHONPATH=. python bin/finet \
    ../baked/w5k_c4.5k_5pct/training.packed \
    ../baked/w5k_c4.5k_5pct/validation.packed \
    ../model/test_run ../model/base \
    --epochs 1 --lr 1e-4 --token_budget 36400 --max_batch_size 8 \
    --grad_accum 64 --compile --log_every_pct 5

# Full training (2x DGX Spark)
bin/distributed.sh ../baked/w5k_c4.5k/training.packed \
    ../baked/w5k_c4.5k/validation.packed \
    ../model/run_001 ../model/base \
    --nnodes 2 --node-rank 0 --master 192.168.100.10 \
    --epochs 4 --lr 1e-4 --token_budget 45500 --max_batch_size 8 \
    --grad_accum 64 --compile --memwatch
```

See [bin/README.md](bin/README.md) for full command recipes

### Project Structure

```
bin/            CLI entry points (init, bake, train, eval)
lib/
  blocks/       Encoder, decoder, perceiver, MoE, sparse attention
  dataset/      Data loading, packing, token budget batching
  util/         Training loop, memory monitoring, logging
  nosing/       Post-processing (exon/intron/protein extraction)
  model.py      Main GeneT5 architecture
  tokenizer.py  Tokenizer wrapper
test/           Evaluation (BUSCO, F1 metrics)
tests/          Development benchmarks and tests
```

### Data Layout

```
/workspace/
  raw/          Raw GFF + FASTA by taxa (prokaryotes, vertebrates, ...)
  baked/        Packed training datasets (tokenized, ready for training)
  model/        Model checkpoints (base/, exp_* experiments)
  logs/         Training logs, memory profiles
```

### Performance

| Config | Batch/s | Notes |
|--------|---------|-------|
| Single node, no compile | 0.55 | Baseline |
| Single node + torch.compile | 0.73 | 33% speedup, dynamic=None |
| 2x Spark + compile (target) | ~1.4 | Linear scaling expected |

### Training Data

5 taxonomic groups, 70,000 genes total, packed binary format

| Taxa | Genes | Avg Size |
|------|-------|----------|
| Prokaryotes | 10,000 | 9 kb |
| Unicellular | 15,000 | 15 kb |
| Invertebrates | 15,000 | 25 kb |
| Vertebrates | 15,000 | 30 kb |
| Plants | 15,000 | 25 kb |
