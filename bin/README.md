## Recipe

For reproducibility, all the Linux Cmd to build this from scratch is being recorded inside this README. 


---


### Init the Model

The idea is to utilize existed DNABert-v2 pre-trained model for giving the model understanding of embedding Genetic Sequences.  

| Component                   | Action          | Reason
| :-------------------------: | :-----------:   | :---- 
| **Encoder Self-Attention**  | **Copy**        | It already knows how to find context in the input.
| **Decoder Self-Attention**  | **Copy**        | It can reuse the Encoder's logic for finding context.
| **Cross-Attention**         | **Random Init** | This is the "new" connection that links input to output.
| **Layer Norms**             | **Copy**        | Keeps the math stable from the start.
| **Output Head**             | **Copy/Init**   | Usually initialized from the Input Embeddings (Shared).

```python3
python3 bin/init_model.py \
    --save_dir "../model/init/" \
    --dnabert_path "zhihan1996/DNABERT-2-117M" \
    --use_moe \
    --num_experts 32 \
    --moe_top_k 2 \
    --tie_weights \
    2>&1 | tee -a ../logs/init.txt
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

| Taxa           | Limit    | Token Est. (@4.5bp) 
| :------------: | :------: | :------------------
| Prokaryotes    | 9,000    | ~2k                 
| Unicellular    | 22,500   | ~5k                 
| Invertebrates  | 45,000   | ~10k                
| Vertebrates    | 90,000   | ~20k                
| Plants         | 45,000   | ~10k                


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

To Tune on distributed device: Run the following Sh script

```sh
# Node 0
MASTER_ADDR=192.168.100.10 NNODES=2 NODE_RANK=0 ./bin/launch_distributed.sh

# Node 1  
MASTER_ADDR=192.168.100.10 NNODES=2 NODE_RANK=1 ./bin/launch_distributed.sh
```