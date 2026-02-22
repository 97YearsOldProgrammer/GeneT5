import torch


BYTE_VOCAB = {
    '<pad>':   0,
    'A':       1,
    'T':       2,
    'G':       3,
    'C':       4,
    'N':       5,
    '<bos>':   6,
    '<eos>':   7,
    '<exon>':  8,
    '<+>':     9,
    '<->':     10,
    '<hints>': 11,
    '<hc>':    12,
    '<lc>':    13,
}

ID_TO_TOKEN = {v: k for k, v in BYTE_VOCAB.items()}

_SPECIALS_SORTED = sorted(
    [t for t in BYTE_VOCAB if len(t) > 1],
    key=len,
    reverse=True,
)


class ByteTokenizer:
    """Byte-level tokenizer for BLT genomic sequences"""

    @property
    def vocab_size(self):
        return len(BYTE_VOCAB)

    @property
    def pad_token_id(self):
        return 0

    @property
    def bos_token_id(self):
        return 6

    @property
    def eos_token_id(self):
        return 7

    def encode(self, text, add_special_tokens=False):
        """Greedy-match special tokens first, then single-char lookup"""

        ids = []
        i   = 0

        while i < len(text):
            matched = False
            for sp in _SPECIALS_SORTED:
                if text[i:i + len(sp)] == sp:
                    ids.append(BYTE_VOCAB[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in BYTE_VOCAB:
                    ids.append(BYTE_VOCAB[ch])
                else:
                    ids.append(BYTE_VOCAB['N'])
                i += 1

        return ids

    def decode(self, ids):
        """Reverse mapping from byte IDs to text"""

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ''.join(ID_TO_TOKEN.get(i, '?') for i in ids)

    def save_pretrained(self, path):
        pass

    def __len__(self):
        return self.vocab_size
