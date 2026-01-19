### End-to-end gene prediction with pure deep learning structure

The idea of this project is using deep learning approach to replace traditional Hidden Markov-model for gene prediction.    
In short human language, for example, the goal is replacing Augustus with cutting-edge deep learning model.     

There are certain pros for considering such action:     

1. Transformer have attention mechanism that reasonablily better than markov process.   
   (no longer solely depend on previous state)     
2. With certain deep learning mechanism, such as: pre-train, fine-tune, MoE.    
   The model no longer need to self-trained again for achieve higher accuary as normal HMM.   

Cons:
1. It's computational hard for running such program. Def need more memory space and time complexity.    
   Both require extremely nice engineering capability to handle them.     
2. It's extremely hard to develop and design such program. That's why no biologists did it so far.  


---


### Expanding the Tokenizer

The tokenizer from the DNABERT model is not enough for our purpose. Since that was only trained for understanding the relationship of raw DNA Sequences. That satisfied the need of a encoder but not the decoder. For our purpose, we need to append more tokens to the original DNABERT-v2 tokenizer for further fine-tuning. A special case of tokenizer is the type of gene that we want the encoder-decoder capable of predicting. As colelcted from gff file of model species from prokaryotics and eukaryotics. The expansion of type tokens would be below:  

```
mobile_genetic_element
origin_of_replication

ncRNA
rRNA
tRNA    
```

Run the following cmd to get new tokens that would be append into the original tokenizer. 
```python3
python3 bin/tokenizer.py --output ./data/new_tokens.txt
```

---


### Dependencies

The first part of getting the nn is copying weights and biases from the pre-trarined model. For achieving that, the CUDA environment is required for accessing the triton packages to get weights and biases from DNABERT-v2 through hugging faces. Here is a full backing receipe for Windows user.  

```powershell
wsl --install -d Ubuntu
```

And then register an account. 

```wsl
pip install torch transformers einops triton
```

The software require pytorch to build building blocks of nn.

```zsh
conda create -n nameyouwannaput
conda install pytorch -c conda-forge
```

Besides, it require the huggingface transformer model to get access to the DNABERT for their weights and tokenizer.

```zsh
conda install -c conda-forge transformers
```

**The packages below require the CUDA for running.**

Furthermore, for optimizing the whole algorithm, such as Spare MoE and Spare Attention Mechanism. We need external packages to get access for creating own CUDA C++ Kernel and parallelism training. For creating custom CUDA Kernel, we need 

```zsh
conda install -c conda-forge triton
```

For parallelism training, we need optimizer packages DeepSpeed.

```zsh
conda install -c conda-forge deepspeed
```

For not CUDA user, the trainning script also support MPS through PyTorch packages. For that purpose, the only required package would be pytorch.   


---


### Fine Tuning

For letting the model have ideal functionality for proper Ab Inito prediction is fine-tuning on datas from multiple species. 

Fine-tuning seem like taking 60GB of RAM through Pytorch packages.   
