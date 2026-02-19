## Recipe

Reproducible commands for building and training GeneT5 from scratch

**Container**: `nvcr.io/nvidia/pytorch:25.12-py3` (PyTorch 2.10, CUDA 13.1, Triton 3.6)
**Hardware**: DGX Spark (GB10, 128GB UMA), ConnectX-7 200Gb/s RoCE


---


### Init the Model

```bash
python -u bin/init_model.py \
    --save_dir ../model/GeneT5/init/init_moe4 \
    --dnabert_path "zhihan1996/DNABERT-2-117M" \
    --tie_weights \
    --layers 12 \
    --heads 12 \
    --ff_dim 3072 \
    --dropout 0.1 \
    --use_moe \
    --num_experts 4 \
    --moe_top_k 1 \
    --init_std 0.02 \
    --init_moe_router_std 0.006 \
    2>&1 | tee ../logs/GeneT5/init.log
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


### Distributed Fine-Tuning (2x DGX Spark)

Uses `torchrun` + NCCL over ConnectX-7 RoCE

**Master** (spark-1089, 192.168.100.10)

```bash
bash -u bin/sft.sh \
    ../baked/GeneT5/feb17_s51_v1_e1/ \
    ../model/GeneT5/feb19_moe4_t1/ \
    ../model/GeneT5/init/init_moe4 \
    --worker 192.168.100.11 \
    --num_workers 2 \
    --batch_size 8 \
    --grad_accum 32 \
    --epochs 3 \
    --lr 0.02 \
    --warmup_ratio 0.03 \
    --label_smoothing 0.1 \
    --save_steps 500 \
    --log_every_pct 2 \
    --memwatch \
    --compile \
    2>&1 | tee ../logs/GeneT5/sft/feb19_moe4_t1.log
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


### Evaluation

```bash
PYTHONPATH=. python test/eval_f1 \
    --predictions predictions.gff \
    --reference reference.gff

PYTHONPATH=. python test/eval_busco \
    --predictions predictions.gff \
    --lineage lineage_dataset
```
