## Recipe

Reproducible commands for building and training GeneT5 from scratch

**Container**: `nvcr.io/nvidia/pytorch:25.12-py3` (PyTorch 2.10, CUDA 13.1, Triton 3.6)
**Hardware**: DGX Spark (GB10, 128GB UMA), ConnectX-7 200Gb/s RoCE


---


### Init the Model

```bash
PYTHONPATH=. python bin/init_model.py \
    --save_dir ../model/init \
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
    --encoder_window_size 512 \
    --decoder_block_size 16 \
    --decoder_window_size 32 \
    --num_latents 1024 \
    2>&1 | tee ../logs/init.log
```


---


### Data Baking

Prepare multi-species training data from raw GFF/FASTA

```bash
PYTHONPATH=. python bin/bake_data \
    --raw_dir ../raw \
    --output_dir ../baked/5k4.5k \
    --n_workers 5 \
    --species_parallel 3 \
    --tokenizer ../model/init \
    --window_size 10000 \
    2>&1 | tee ../logs/baker/5k4.5k.log
```

**Species Limits**

| Taxa           | Gene Limit | Token Est  | Avg Gene Size       |
| :------------- | :--------: | :--------: | :------------------ |
| Prokaryotes    | 10,000     | ~2.2k      | 9,000 bp (no introns) |
| Unicellular    | 15,000     | ~3.3k      | 15,000 bp (~P95+)   |
| Invertebrates  | 15,000     | ~5.5k      | 25,000 bp (~P90)    |
| Vertebrates    | 15,000     | ~6.6k      | 30,000 bp (~P75)    |
| Plants         | 15,000     | ~5.5k      | 25,000 bp (~P85-90) |

**Subsetting** (for dev/test with smaller data)

```bash
PYTHONPATH=. python bin/subset_packed \
    ../baked/5k4.5k/training.packed \
    ../baked/5%5k4.5k/training.packed \
    --fraction 0.05
```


---


### Token Management

Init tokenizer with gene-type tokens from GFF

```bash
PYTHONPATH=. python bin/init_tk.py data/new_tokens.txt
```

Append new tokens discovered during baking

```bash
PYTHONPATH=. python bin/append_tk.py data/new_tokens.txt ../model/init/tokenizer.json
```

Resize embeddings after tokenizer changes

```bash
PYTHONPATH=. python bin/resize_model.py ../model/init ../model/init
```


---


### Single-Node Fine-Tuning

```bash
PYTHONPATH=. python bin/finet \
    ../baked/5k4.5k/training.packed \
    ../baked/5k4.5k/validation.packed \
    ../model/run_001 \
    ../model/init \
    --epochs 4 \
    --lr 1e-4 \
    --token_budget 45500 \
    --max_batch_size 8 \
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
| `--compile`        | torch.compile on encoder/compressor/decoder (~33% speedup) |
| `--memwatch`       | Background memory CSV (5s intervals, then 30s) |
| `--token_budget N` | Variable batch sizes based on label tokens |
| `--mxfp8`          | MXFP8 quantization (Blackwell only)        |
| `--optim_8bit`     | 8-bit AdamW (saves ~12GB optimizer memory) |
| `--log_every_pct N`| Progress log frequency                     |


---


### Distributed Fine-Tuning (2x DGX Spark)

Uses `torchrun` + NCCL over ConnectX-7 RoCE

**Master** (spark-1089, 192.168.100.10)
```bash
bin/distributed.sh \
    ../baked/5k4.5k/training.packed \
    ../baked/5k4.5k/validation.packed \
    ../model/run_002_dist \
    ../model/init \
    --nnodes 2 --node-rank 0 --master 192.168.100.10 \
    --epochs 4 --lr 1e-4 --token_budget 45500 --max_batch_size 8 \
    --grad_accum 64 --compile --log_every_pct 5 --memwatch
```

**Worker** (spark-0b7c, 192.168.100.11)
```bash
bin/distributed.sh \
    ../baked/5k4.5k/training.packed \
    ../baked/5k4.5k/validation.packed \
    ../model/run_002_dist \
    ../model/init \
    --nnodes 2 --node-rank 1 --master 192.168.100.10 \
    --epochs 4 --lr 1e-4 --token_budget 45500 --max_batch_size 8 \
    --grad_accum 64 --compile --log_every_pct 5
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
