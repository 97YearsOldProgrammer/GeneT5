from dataclasses import dataclass
from typing import List, Optional


######################
#### Tokenisation ####
######################


BASE_PAIR = ("A", "C", "G", "T")
BASE2IDX = {base: idx for idx, base in enumerate(BASE_PAIR)}


def apkmer(k: int):
    """Generate all possible k-mers recursively"""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    
    if k == 1:
        return list(BASE_PAIR)
    
    prev_kmers = apkmer(k - 1)
    return [prefix + base for prefix in prev_kmers for base in BASE_PAIR]


@dataclass
class KmerTokenizer:
    """
    DNA sequence to k-mer token IDs using sliding window
    
    Special tokens for span corruption:
        <pad>: 0
        <eos>: 1  
        <mask>: 2
        <extra_id_0> to <extra_id_99>: 3-102
    """
    
    k               : int
    stride          : int = 1
    max_sentinels   : int = 1000
    vocabulary      : Optional[List[str]] = None
    
    def __post_init__(self):
        # Build vocabulary if not provided
        if self.vocabulary is None:
            # Special tokens
            self.vocabulary = ["<pad>", "<eos>", "<mask>"]
            
            # Sentinel tokens for span corruption
            self.vocabulary.extend([f"<extra_id_{i}>" for i in range(self.max_sentinels)])
            
            # Regular k-mers
            self.vocabulary.extend(sorted(apkmer(self.k)))
        
        # Create mapping
        self.token2id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        # Special token IDs
        self.pad_token_id       = self.token2id["<pad>"]
        self.eos_token_id       = self.token2id["<eos>"]
        self.mask_token_id      = self.token2id["<mask>"]
        self.sentinel_start_id  = self.token2id["<extra_id_0>"]
    
    @property
    def vocab_size(self):
        return len(self.vocabulary)
    
    def __call__(self, seq , add_eos=True):
        """seq2token"""
        
        seq     = seq.upper()
        tokens  = []
        
        # Sliding window
        for t in range(0, max(len(seq) - self.k + 1, 0), self.stride):
            kmer = seq[t:t + self.k]
            if kmer in self.token2id:
                tokens.append(self.token2id[kmer])
            else:
                # Handle unknown k-mers (e.g., with 'N')
                tokens.append(self.mask_token_id)
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """token2seq
        """
        tokens = [self.id2token.get(id, 'N' * self.k) for id in token_ids]
        
        # Filter out special tokens
        tokens = [t for t in tokens if not t.startswith('<')]
        
        if not tokens:
            return ""
        
        # Reconstruct from overlapping k-mers
        seq = tokens[0]
        for token in tokens[1:]:
            seq += token[-self.stride:]  # Add last stride characters
        
        return seq