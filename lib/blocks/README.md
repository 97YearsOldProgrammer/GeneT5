### Architecture

Given the pre-trained DNABERT-v2, we now duplicate the whole encoder-decoder parameter as much as we can for skipping pre-trainning on the span corruption of raw DNA sequences. For time efficiency, we replaced the original FLASH Attention with Spare Attention through either Pytorch FlexAttention Function or Triton custom C++ CUDA Kernel.

Furthermore, we replaced the GEGLU Feedfoward layer with SpareMoE for the decoder to having more diversied information that could be learned throughout the later fine-tuning parts. Since what I believe is biological sequence analysis acquire more diversied sub neural network that are gated for making the correct prediction.   

The current design is using CUDA for fine-tuning. However, once finished trainning, we ought have corresponding pytorch version for running on multiple device. 