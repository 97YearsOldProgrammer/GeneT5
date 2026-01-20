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
    --save_dir "../Data/model/init/" \
    --dnabert_path "zhihan1996/DNABERT-2-117M" \
    --use_moe \
    --num_experts 8 \
    --moe_top_k 2 \
    --tie_weights
```


---


### Data Baking

To prepare fine-tuning data, using [data_baker](bake_data.py)

```python3
python3 bin/bake_data.py \
  ../Data/genome/raw/C.crescentus/GCF_000022005.1_ASM2200v1_genomic.fna.gz\
  ../Data/genome/raw/C.crescentus/GCF_000022005.1_ASM2200v1_genomic.gff.gz  \
  ../Data/genome/processed/C.crescentus/ \
  --extract_tokens data/new_tokens.txt \
  2>&1 | tee ../Data/logs/baker/C.crescentus.txt
```


---


### Token Manangement

The tokenizer is first build on original DNABert-v2 Model for further usage. To init the tokenizer script, generate the extended tokens first through [new_tokens](init_tk.py).

```python3
python3 bin/init_tk.py data/new_tokens.txt
```

**Further Addition of Tokens**
Since we can't 100% ensure that we can include all weird types from Gff files. We would also have a pipeline for updating tokens into the model toeknizer. To append new tokens into the tokenizer.json, run this:

```python3
python3 bin/append_tk.py data/new_tokens.txt ../Data/model/init/tokenizer.json
```

**Resizing the Embedding Space**
After adding token into the model, the encoder embedding dimension need to be resized before further fine-tuning. Run this script:

```python3
python3 bin/resize_model.py model_path tokenizer_path
```


---


### Tuning

```python3

```