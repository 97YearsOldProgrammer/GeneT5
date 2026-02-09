from pathlib import Path
from transformers import AutoTokenizer


DNABERT_PATH = "zhihan1996/DNABERT-2-117M"


#############################
#####  Token Inventory  #####
#############################


SPECIAL  = ["<bos>", "<eos>"]
FORMAT   = ["[ATT]", "[HIT]", "+", "-", r"[\t]", r"[\n]"]
NUMBERS  = [str(i) for i in range(1000)]
NMASK    = ["N" * i for i in range(1, 7)]
FEATURES = ["exon"]
BIOTYPES = [
    "protein_coding", "lncrna", "pseudogene",
    "trna", "rrna", "snorna", "snrna", "mirna",
]
HINTS    = ["intron_hc", "intron_lc"]

GFF_TOKENS = SPECIAL + FORMAT + NUMBERS + NMASK + FEATURES + BIOTYPES + HINTS


#######################
#####  Tokenizer  #####
#######################


class GeneTokenizer:
    """Tokenizer for GeneT5 gene finder"""

    def __init__(self, path):

        self._tok = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True
        )

    @classmethod
    def from_dnabert(cls, dnabert_path=DNABERT_PATH, save_dir=None):
        """Build gene finder tokenizer from DNABERT-2 base"""

        tok = AutoTokenizer.from_pretrained(dnabert_path, trust_remote_code=True)
        base_size = len(tok)
        tok.add_tokens(GFF_TOKENS)

        print(f"Tokenizer: {base_size} base + {len(GFF_TOKENS)} GFF = {len(tok)}")

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            tok.save_pretrained(save_dir)

        obj      = object.__new__(cls)
        obj._tok = tok
        return obj

    def __len__(self):
        return len(self._tok)

    def __call__(self, text, **kwargs):
        return self._tok(text, **kwargs)

    def encode(self, text, **kwargs):
        return self._tok.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self._tok.decode(token_ids, **kwargs)

    @property
    def vocab_size(self):
        return len(self)

    @property
    def pad_token_id(self):
        return self._tok.pad_token_id or 0

    @property
    def eos_token_id(self):
        return self._tok.convert_tokens_to_ids("<eos>")

    @property
    def bos_token_id(self):
        return self._tok.convert_tokens_to_ids("<bos>")

    def get_vocab(self):
        return self._tok.get_vocab()

    def save_pretrained(self, path):
        self._tok.save_pretrained(path)
