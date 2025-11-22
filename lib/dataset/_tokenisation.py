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