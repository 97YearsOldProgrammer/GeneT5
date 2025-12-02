from dataclasses    import dataclass


######################
#### Tokenisation ####
######################


BASE_PAIR   = ("A", "C", "G", "T")
BASE2IDX    = {base: idx for idx, base in enumerate(BASE_PAIR)}

DEFAULT_FEATURE_LABELS = {
    "exon": 0,
    "intron": 1,
    "five_prime_utr": 2,
    "three_prime_utr": 3,
}

def apkmer(k: int):

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if k == 1:
        return list(BASE_PAIR)

    prev_kmers = apkmer(k - 1)
    return [prefix + base for prefix in prev_kmers for base in BASE_PAIR]

@dataclass
class KmerTokenizer:
    """DNA seq to kmer ids by sliding window algo"""

    k           : int
    stride      : int = 1
    vocabulary  : list = None

    def __post_init__(self):
        # map all kmer with a int
        self.token2id = {token: idx for idx, token in enumerate(self.vocabulary)}

    def __call__(self, seq):
        seq     = seq.upper()
        tokens  = []

        # sliding window algo
        for t in range(0, max(len(seq) - self.k + 1, 0), self.stride):
            token = seq[t:t+self.k]
            if token in self.token2id:
                tokens.append(self.token2id[token])
        
        return tokens
    
class KmerTokenizer:

    NUC_TO_IDX = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    IDX_TO_NUC = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
    
    # Special tokens
    MASK_TOKEN  = '[MASK]'
    SEP_TOKEN   = '[SEP]'
    STATE_TOKEN = '[STATE]'
    PAD_TOKEN   = '[PAD]'
    UNK_TOKEN   = '[UNK]'
    
    def __init__(self, k= 6):
        self.k          = k
        self.vocab_size = 4 ** k
        
        # Special token IDs (after vocab)
        self.mask_token_id      = self.vocab_size
        self.sep_token_id       = self.vocab_size + 1
        self.state_token_id     = self.vocab_size + 2
        self.pad_token_id       = self.vocab_size + 3
        self.unk_token_id       = self.vocab_size + 4
        self.total_vocab_size   = self.vocab_size + 5
        
    def kmer_to_id(self, kmer):

        if len(kmer) != self.k:
            return self.unk_token_id
        
        token_id = 0
        for i, nuc in enumerate(kmer):
            nuc = nuc.upper()
            if nuc not in self.NUC_TO_IDX:
                return self.unk_token_id
            token_id += self.NUC_TO_IDX[nuc] * (4 ** (self.k - 1 - i))
        
        return token_id
    
    def id_to_kmer(self, token_id: int) -> str:
        """Convert token ID to k-mer string"""
        if token_id == self.mask_token_id:
            return self.MASK_TOKEN
        if token_id == self.sep_token_id:
            return self.SEP_TOKEN
        if token_id == self.state_token_id:
            return self.STATE_TOKEN
        if token_id == self.pad_token_id:
            return self.PAD_TOKEN
        if token_id == self.unk_token_id:
            return self.UNK_TOKEN
        if token_id >= self.vocab_size:
            return self.UNK_TOKEN
        
        kmer = ''
        for i in range(self.k):
            idx = (token_id // (4 ** (self.k - 1 - i))) % 4
            kmer += self.IDX_TO_NUC[idx]
        
        return kmer
    
    def tokenize(self, sequence: str, stride: int = None) -> tp.List[int]:
        """
        Tokenize sequence into k-mer token IDs
        
        Args:
            sequence: DNA sequence string
            stride: Step size (default k for non-overlapping)
            
        Returns:
            List of token IDs
        """
        if stride is None:
            stride = self.k
        
        sequence = sequence.upper()
        tokens = []
        
        for i in range(0, len(sequence) - self.k + 1, stride):
            kmer = sequence[i:i + self.k]
            tokens.append(self.kmer_to_id(kmer))
        
        return tokens
    
    def __call__(self, sequence: str, stride: int = None) -> tp.List[int]:
        return self.tokenize(sequence, stride)
    
    def decode(self, token_ids: tp.List[int]) -> str:
        """Decode token IDs back to sequence"""
        return ''.join([self.id_to_kmer(tid) for tid in token_ids])
    
    def to_onehot(self, sequence: str) -> torch.Tensor:
        """
        Convert sequence to one-hot encoding (not k-mer based)
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Tensor of shape (4, L)
        """
        sequence = sequence.upper()
        L = len(sequence)
        onehot = torch.zeros(4, L)
        
        for i, nuc in enumerate(sequence):
            if nuc in self.NUC_TO_IDX:
                onehot[self.NUC_TO_IDX[nuc], i] = 1.0
        
        return onehot