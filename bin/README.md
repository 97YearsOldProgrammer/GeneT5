## Recipe

For reproducibility, all the Linux Cmd to build this from scratch is being recorded inside this README. 

--


### Init the Model

The idea is to utilize existed DNABert-v2 pre-trained model for giving the model understanding of embedding Genetic Sequences.  

| Component                   | Action          | Reason
| :-------------------------: | :-----------:   | :---- 
| **Encoder Self-Attention**  | **Copy**        | It already knows how to find context in the input.
| **Decoder Self-Attention**  | **Copy**        | It can reuse the Encoder's logic for finding context.
| **Cross-Attention**         | **Random Init** | This is the "new" connection that links input to output.
| **Layer Norms**             | **Copy**        | Keeps the math stable from the start.
| **Output Head**             | **Copy/Init**   | Usually initialized from the Input Embeddings (Shared).

```zsh
