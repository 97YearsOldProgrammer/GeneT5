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


### Encoder (BERT)

The current stage of planning is not pre-trainning everything from stracth rather accepte a open-source model from hugging face.        
The choice is DNABert-v2, a pre-trained model that published weights on hugging face.   
That model is already trained for capture local pattern, motif through trainning.   
So, it would be the backbone of encoder block.  
For the first step of design such program is that we need a encoder-decoder transformer model that fine-tuned to *Ab Initio Prediction*.  

*Ab Initio Prediction* is the most basic functionality for such gene finder program. It's require able to generate prediction in form of GFF/GTF in terms of given DNA input sequences.   
Therefore, theoretically, for achieving this, there have to be a decoder part for generating (to transfer what have already learned) the ideal outputs.  



---


### Trainning nn

The last step is trainning the neural network using given dataset.  
All the nn is build up by stacking pytorch. So using pytroch package is necessary.  

This is how you can download all package for Apple Silicon.     

```zsh
conda create -n nameyouwannaput
conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
```

Beside, you also need hugging face transformer package to load the pre-trained model for encoder.   

```zsh
conda install -c conda-forge transformers
```     