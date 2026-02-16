## Recipe

Reproducible commands for building and training GeneT5 from scratch

**Container**: `nvcr.io/nvidia/pytorch:25.12-py3` (PyTorch 2.10, CUDA 13.1, Triton 3.6)
**Hardware**: DGX Spark (GB10, 128GB UMA), ConnectX-7 200Gb/s RoCE


---


### Init the Model

```bash
PYTHONPATH=. python bin/init_model.py \
    --save_dir ../model/base \
    --dnabert_path "zhihan1996/DNABERT-2-117M" \
    --tie_weights \
    --decoder_layers 12 \
    --decoder_heads 12 \
    --decoder_kv_heads 3 \
    --decoder_ff_dim 3072 \
    --decoder_dropout 0.1 \
    --use_moe \
    --num_experts 16 \
    --moe_top_k 2 \
    --init_std 0.02 \
    --init_embed_std 0.02 \
    --init_ffn_std 0.02 \
    --init_attn_std 0.02 \
    --init_moe_router_std 0.006 \
    --encoder_window_size 1024 \
    --decoder_window_size 256 \
    2>&1 | tee ../logs/init.log
```


---


### Data Baking

Prepare multi-species training data from raw GFF/FASTA. Species are **auto-discovered** from the raw directory (any subdir containing `fna.gz` + `gff.gz`). No hardcoded species list — the raw dir IS the species list. Canonical transcript filtering and fast tokenizer (Rust backend) are always on.

**Single-node bake**

```bash
PYTHONPATH=. python bin/bake_data ../raw \
    --tokenizer ../model/GeneT5/ \
    --output_dir ../baked/GeneT5/w20k_v3/ \
    --window_size 20000 \
    --species_parallel 4 \
    --val_species B.taurus,S.lycopersicum \
    --val_windows 2000 \
    2>&1 | tee ../logs/baker/w20k_v3.log
```

**Distributed bake (2x DGX Spark)**

Greedy bin-packing distributes species across nodes by genome size. Node 0 handles held-out species and eval generation.

```bash
# Node 0 (master — also handles eval + held-out species)
PYTHONPATH=. python bin/bake_data ../raw \
    --tokenizer ../model/GeneT5/ \
    --output_dir ../baked/GeneT5/w20k_v3/ \
    --window_size 20000 \
    --species_parallel 4 \
    --val_species B.taurus,S.lycopersicum \
    --nnodes 2 --node_rank 0 \
    2>&1 | tee ../logs/baker/w20k_v3_node0.log

# Node 1 (worker)
PYTHONPATH=. python bin/bake_data ../raw \
    --tokenizer ../model/GeneT5/ \
    --output_dir ../baked/GeneT5/w20k_v3/ \
    --window_size 20000 \
    --species_parallel 4 \
    --nnodes 2 --node_rank 1 \
    2>&1 | tee ../logs/baker/w20k_v3_node1.log
```

**4-node bake** (scale to more machines)

```bash
# Each node gets a balanced slice of species by genome size
for RANK in 0 1 2 3; do
    PYTHONPATH=. python bin/bake_data ../raw \
        --tokenizer ../model/GeneT5/ \
        --output_dir ../baked/GeneT5/w20k_v3/ \
        --species_parallel 4 \
        --val_species B.taurus,S.lycopersicum \
        --nnodes 4 --node_rank $RANK \
        2>&1 | tee ../logs/baker/w20k_v3_node${RANK}.log &
done
wait
```

**Output structure** (flat, all species together)

```
../baked/GeneT5/w20k_v3/
  H.sapiens/training.bin        # per-species bin
  H.sapiens/gene_index.json     # sidecar (avoids GFF re-parse for eval)
  M.musculus/training.bin
  B.taurus/training.bin         # held-out species (same dir)
  ...
  training.bin                  # merged training
  validation.bin                # merged validation (held-out only)
  eval.json                     # eval samples from held-out species
  bake_config.json              # records which species are val/eval
```

**CLI flags**

| Flag | Default | Purpose |
| :--- | :------ | :------ |
| `--val_species` | none | Comma-separated held-out species for validation |
| `--val_windows` | 3000 | Max validation windows from held-out species |
| `--window_size` | 20000 | Sliding window size in bp |
| `--species_parallel` | 3 | Species processed in parallel |
| `--n_workers` | auto | Workers per species for chunking |
| `--compress` | none | `zlib` or `zstd` compression |
| `--nnodes` | 1 | Total baking nodes |
| `--node_rank` | 0 | This node's rank (0-indexed) |
| `--eval_samples` | 50 | Eval samples per held-out species |
| `--train` / `--eval` | both | Bake training or eval only |

Validation comes exclusively from held-out species, capped at `--val_windows`. Training species produce zero validation data.

**Subsetting** (for dev/test with smaller data)

```bash
PYTHONPATH=. python bin/subset_packed \
    ../baked/w5k_c4.5k/training.packed \
    ../baked/w5k_c4.5k_5pct/training.packed \
    --fraction 0.05
```

**One-click distributed bake (2x DGX Spark)**

```bash
bin/bake.sh --worker 192.168.100.11 \
    --tokenizer ../model/GeneT5/init \
    --output_dir ../baked/GeneT5/w20k_s51_v2 \
    --window_size 20000 \
    --species_parallel 3 \
    --val_species B.taurus,S.lycopersicum \
    --eval_samples 50 \
    --seed 42 \
    ../raw
```

**Mid-training Eval**

```bash
python bin/prep_eval ../raw/S.scrofa/ -o ../baked/GeneT5/eval/S.scrofa_eval.json
```

---


### Token Management

Init tokenizer with gene-type tokens from GFF

```bash
PYTHONPATH=. python bin/init_tk.py data/new_tokens.txt
```

Append new tokens discovered during baking

```bash
PYTHONPATH=. python bin/append_tk.py data/new_tokens.txt ../model/base/tokenizer.json
```

Resize embeddings after tokenizer changes

```bash
PYTHONPATH=. python bin/resize_model.py ../model/base ../model/base
```


---


### Single-Node Fine-Tuning

```bash
PYTHONPATH=. python bin/finet \
    ../baked/w5k_c4.5k/training.packed \
    ../baked/w5k_c4.5k/validation.packed \
    ../model/run_001 \
    ../model/base \
    --epochs 4 \
    --lr 1e-4 \
    --batch_size 8 \
    --grad_accum 64 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --early_stopping 2 \
    --save_steps 500 \
    --empty_cache_steps 100 \
    --compile \
    --memwatch \
    --log_every_pct 5 \
    2>&1 | tee ../logs/tune/run_001.log
```

**Key Flags**

| Flag               | Purpose                                    |
| :----------------- | :----------------------------------------- |
| `--compile`        | torch.compile on encoder/decoder (~33% speedup) |
| `--memwatch`       | Background memory CSV (5s intervals, then 30s) |
| `--mxfp8`          | MXFP8 quantization (Blackwell only)        |
| `--optim_8bit`     | 8-bit AdamW (saves ~12GB optimizer memory) |
| `--log_every_pct N`| Progress log frequency                     |


---


### Distributed Fine-Tuning (2x DGX Spark)

Uses `torchrun` + NCCL over ConnectX-7 RoCE

**Master** (spark-1089, 192.168.100.10)
```bash
bash bin/sft.sh ../baked/GeneT5/w20k_ts51_v2/training.bin ../baked/GeneT5/w20k_ts51_v2/validation.bin ../model/GeneT5/feb10_run1/ ../model/GeneT5/ --worker 192.168.100.11 --label_smoothing 0.1 --num_workers 3 --batch_size 8 --epochs 4 --save_steps 2000 --optim_8bit --empty_cache_steps 500 --memwatch
```

**RoCE Verification** (before distributed run)

```bash
# Host: set jumbo MTU on both nodes
sudo ip link set enp1s0f1np1 mtu 9216

# Host: verify jumbo frames
ping -M do -s 8972 192.168.100.11

# Container: raw RDMA test
apt-get install -y perftest
ib_write_bw -d rocep1s0f1 -x 3 --report_gbits          # server
ib_write_bw -d rocep1s0f1 -x 3 192.168.100.10 --report_gbits  # client
```

**RoCE Fallback**: If RDMA fails, uncomment Socket transport in `distributed.sh` lines 127-128

**NCCL Logs**: Look for `NET/IB` (RoCE active) vs `NET/Socket` (fallback)


---


### Logging

All commands use `tee` to write to both stdout and a log file. Use this pattern for any new run:

```bash
# Convention: ../logs/<category>/<run_name>.log
# Categories: tune/, baker/, init.log

# Training run with timestamped log
RUN="exp_$(date +%Y%m%d)_desc"
PYTHONPATH=. python bin/finet \
    ../baked/w5k_c4.5k/training.packed \
    ../baked/w5k_c4.5k/validation.packed \
    ../model/$RUN ../model/base \
    --epochs 4 --lr 1e-4 --batch_size 8 \
    --grad_accum 64 --compile --memwatch \
    2>&1 | tee ../logs/tune/${RUN}.log

# Bake run (per-species logs go to baker/ subdir)
PYTHONPATH=. python bin/bake_data \
    --raw_dir ../raw --output_dir ../baked/w5k_c4k \
    --tokenizer ../model/base --window_size 5000 --target 4000 \
    --log_dir ../logs/baker/w5k_c4k \
    2>&1 | tee ../logs/baker/w5k_c4k.log
```

**Log locations**

| Category | Path | Content |
| :------- | :--- | :------ |
| Model init | `../logs/init.log` | Architecture, param counts |
| Baking | `../logs/baker/<dataset>.log` | Per-species stats, packing summary |
| Per-species bake | `../logs/baker/<dataset>/<Species>.log` | Gene extraction, token stats |
| Training | `../logs/tune/<run>.log` | Loss, lr, batch/s, memory |


---


### torch.compile Notes

- **NGC 25.12 bug**: Requires monkeypatch in `triton_heuristics.py` for `cluster_dims` attribute
  (apply in both master and worker containers)
- Use `dynamic=None` (not `True`) to avoid 15+ minute compilation hangs
- First 2 batches recompile (~10s each), then all shapes run from cache (~0.15s)
- Inductor cache at `/tmp/torchinductor_root/` — copy to worker for faster startup
- GB10 shows "Not enough SMs for max_autotune_gemm" — informational, not an error


---


### Evaluation

```bash
PYTHONPATH=. python test/eval_f1 \
    --predictions predictions.gff \
    --reference reference.gff

PYTHONPATH=. python test/eval_busco \
    --predictions predictions.gff \
    --lineage lineage_dataset
```
