import json
import pathlib as pl

import torch


class SimpleTokenizer:
    """Simple tokenizer for DNA/protein sequences as fallback"""

    def __init__(self, vocab, pad_token="<PAD>", bos_token="<bos>", eos_token="<eos>", unk_token="<UNK>"):
        """Initialize tokenizer with vocabulary"""

        self.vocab        = vocab
        self.id2token     = {v: k for k, v in vocab.items()}
        self.pad_token    = pad_token
        self.bos_token    = bos_token
        self.eos_token    = eos_token
        self.unk_token    = unk_token
        self.pad_token_id = vocab.get(pad_token, 0)
        self.bos_token_id = vocab.get(bos_token, 1)
        self.eos_token_id = vocab.get(eos_token, 2)
        self.unk_token_id = vocab.get(unk_token, 3)

    @classmethod
    def from_vocab(cls, vocab_path):
        """Load from vocab.json file"""

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)

    def encode(self, text, max_length=2048):
        """Encode single text to token IDs"""

        tokens = list(text)
        ids    = [self.vocab.get(t, self.unk_token_id) for t in tokens]

        if len(ids) > max_length - 2:
            ids = ids[:max_length - 2]

        ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def encode_batch(self, texts, max_length=2048):
        """Encode batch of texts"""

        encoded = [self.encode(t, max_length) for t in texts]
        max_len = max(len(e) for e in encoded)

        input_ids      = []
        attention_mask = []

        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=False):
        """Decode token IDs to text"""

        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens      = []

        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            tokens.append(self.id2token.get(id_, self.unk_token))

        return "".join(tokens)

    def decode_batch(self, batch_ids, skip_special_tokens=False):
        """Decode batch of token IDs"""

        return [self.decode(ids.tolist(), skip_special_tokens) for ids in batch_ids]


def read_input(input_path):
    """Read input sequences from FASTA or plain text"""

    path = pl.Path(input_path)

    if path.exists():
        content = path.read_text()

        if content.startswith(">"):
            sequences   = []
            seqids      = []
            current_seq = []
            current_id  = None

            for line in content.strip().split("\n"):
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                        seqids.append(current_id)
                    current_id  = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())

            if current_id is not None:
                sequences.append("".join(current_seq))
                seqids.append(current_id)

            return sequences, seqids
        else:
            sequences = [l.strip() for l in content.strip().split("\n") if l.strip()]
            seqids    = [f"seq_{i}" for i in range(len(sequences))]
            return sequences, seqids
    else:
        return [input_path], ["seq_0"]
