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
    --num_latents 1024 \
    2>&1 | tee ../logs/init.log
```


---


### Data Baking

To prepare fine-tuning data, using [data_baker](bake_data.py).  

```python3
python -u bin/bake_data --raw_dir ../raw --output_dir ../baked/5k4.5k --n_workers 5 --species_parallel 3 --tokenizer ../model/init 2>&1 --window_size 10000 | tee ../logs/baker/5k4.5k/bake.log 
```

**Config**

| Taxa           | Limit    | Token Est. (@4.5bp) | Avg
| :------------: | :------: | :-----------------: | :------------
| Prokaryotes    | 10000    | ~2.2k               | 9000bps  = ~P99+ (no introns, avg 924 bp)
| Unicellular    | 15000    | ~3.3k               | 15000bps = ~P95+ (yeast genes avg ~1.5 kb)
| Invertebrates  | 15000    | ~5.5k               | 25000bps = ~P90 (Drosophila genes ~2-10 kb)
| Vertebrates    | 15000    | ~6.6k               | 30000bps = ~P75 (median 23 kb, many >30kb)
| Plants         | 15000    | ~5.5k               | 25000bps = ~P85-90 



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


### Tuning


```python3
python -u bin/finet \
  ../baked/5k4.5k/training.packed \
  ../baked/5k4.5k/validation.packed \
  ../model/feb.3 \
  ../model/init \
  --epochs 4 \
  --lr 1e-4 \
  --token_budget 63700 \
  --max_batch_size 8 \
  --grad_accum 64 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --label_smoothing 0.1 \
  --early_stopping 2 \
  --save_steps 500 \
  --log_memory \
  --empty_cache_steps 100 \
  --memwatch \
  2>&1 | tee ../logs/tune/1.log
```