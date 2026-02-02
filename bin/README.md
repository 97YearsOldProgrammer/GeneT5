## Recipe

For reproducibility, all the Linux Cmd to build this from scratch is being recorded inside this README. 


---


### Init the Model

```python3
python3 -u bin/init_model.py \
    --save_dir ../model/init \
    --dnabert_path "zhihan1996/DNABERT-2-117M" \
    --tie_weights \
    --decoder_layers 12 \
    --decoder_heads 12 \
    --decoder_kv_heads 3 \
    --decoder_ff_dim 3072 \
    --decoder_dropout 0.1 \
    --use_moe \
    --num_experts 32 \
    --moe_top_k 2 \
    --init_std 0.02 \
    --init_embed_std 0.02 \
    --init_ffn_std 0.02 \
    --init_attn_std 0.02 \
    --init_moe_router_std 0.006 \
    --encoder_block_size 256 \
    --encoder_window_size 34560 \
    --decoder_block_size 16 \
    --decoder_window_size 1600 \
    2>&1 | tee ../logs/init.log
```


---


### Data Baking

To prepare fine-tuning data, using [data_baker](bake_data.py).  

**The General Command Line**
```python3
python3 bin/bake_data.py \
  fasta   \
  gff     \
  out_dir \
  --extract_tokens data/new_tokens.txt \
  2>&1 | tee -a ../logs/baker/[name].txt 
```

For sake of saving life, please use the [bake_data_wrapper](run_bake_data.py) instead for that general command line input. Here is some hyperparameter for how different data is being baked.   

```python3
python3 bin/run_bake_data.py H.archaea E.coli --limit 9000 --threshold 50000
```

| Taxa           | Limit    | Token Est. (@4.5bp) | Avg
| :------------: | :------: | :-----------------: | :------------
| Prokaryotes    | 10000    | ~2.2k               | 9000bps  = ~P99+ (no introns, avg 924 bp)
| Unicellular    | 15000    | ~3.3k               | 15000bps = ~P95+ (yeast genes avg ~1.5 kb)
| Invertebrates  | 30000    | ~6.6k               | 25000bps = ~P90 (Drosophila genes ~2-10 kb)
| Vertebrates    | 45000    | ~10k                | 30000bps = ~P75 (median 23 kb, many >30kb)
| Plants         | 30000    | ~6.6k               | 25000bps = ~P85-90 


---


### Token Manangement

The tokenizer is first build on original DNABert-v2 Model for further usage. To init the tokenizer script, generate the extended tokens first through [new_tokens](init_tk.py).

```python3
python3 bin/init_tk.py data/new_tokens.txt
```

**Further Addition of Tokens**
Since we can't 100% ensure that we can include all weird types from Gff files. We would also have a pipeline for updating tokens into the model toeknizer. To append new tokens into the tokenizer.json, run this:

```python3
python3 bin/append_tk.py data/new_tokens.txt ../model/init/tokenizer.json
```

**Resizing the Embedding Space**
After adding token into the model, the encoder embedding dimension need to be resized before further fine-tuning. Run this script:

```python3
python3 bin/resize_model.py model_path tokenizer_path
```


---

### Bake Data

```python3
python -u bin/bake_data --raw_dir ../raw --output_dir ../baked --n_workers 6 --species_parallel 3 --tokenizer ../model/init 2>&1 | tee ../logs/bake.log 
```


---


### Compacting All Binary File

After parse all datas, remember to run the compacting function.

```python3
 python -u bin/compact.py \
  ../baked/*/training.bin \ 
  -o ../baked/33/train.bin \
  --tokenizer ../model/init/ \
  --compact_target 30000 \
  --hard_limit 32768 \
  --workers 10 \     
  2>&1 | tee ../logs/compact.log
```


---


### Tuning


```python3
python -u bin/finet \
  ../baked/training.bin \
  ../baked/validation.bin \
  ../model/trial1 \
  ../model/init \
  --epochs 4 \
  --lr 1e-4 \
  --batch_size 4 \
  --grad_accum 256 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing \
  --optim_8bit \
  --label_smoothing 0.1 \
  --early_stopping 2 \
  --save_steps 500 \
  --log_memory \
  2>&1 | tee ../logs/tune/1.log
```