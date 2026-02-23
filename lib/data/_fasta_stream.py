import gzip
import random
import shutil
import pathlib
import bisect

import torch
import torch.utils.data as data_utils

import pyfaidx


DNA_VOCAB      = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 5}
DNA_VOCAB_SIZE = 6
DNA_PAD_ID     = 0

# Codon-level encoding: 64 codons + pad(0) + N-codon(65)
CODON_PAD_ID     = 0
CODON_N_ID       = 65
CODON_VOCAB_SIZE = 66

_BASE_MAP   = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
_CODON_LUT  = {}
for _a in 'ATGC':
    for _b in 'ATGC':
        for _c in 'ATGC':
            _codon          = _a + _b + _c
            _CODON_LUT[_codon] = _BASE_MAP[_a] * 16 + _BASE_MAP[_b] * 4 + _BASE_MAP[_c] + 1


def prepare_fasta_index(raw_dir, skip_species=None):
    """Decompress .fna.gz -> .fna, build .fai indices via pyfaidx

    Idempotent — skips already decompressed species
    """

    raw_dir      = pathlib.Path(raw_dir)
    skip_species = set(skip_species or [])
    manifest     = []

    for sp_dir in sorted(raw_dir.iterdir()):
        if not sp_dir.is_dir():
            continue

        species = sp_dir.name
        if species in skip_species:
            continue

        gz_files  = sorted(sp_dir.glob("*.fna.gz")) or sorted(sp_dir.glob("fna.gz"))
        fna_files = sorted(sp_dir.glob("*.fna")) or sorted(sp_dir.glob("fna"))

        # decompress any .fna.gz that lack a corresponding .fna
        for gz in gz_files:
            fna = gz.with_suffix('')
            if not fna.exists():
                with gzip.open(gz, 'rb') as f_in, open(fna, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                fna_files = sorted(sp_dir.glob("*.fna"))

        if not fna_files:
            continue

        fna_path = fna_files[0]

        # build .fai index if missing
        fai_path = pathlib.Path(str(fna_path) + ".fai")
        if not fai_path.exists():
            fa = pyfaidx.Fasta(str(fna_path), as_raw=True)
            fa.close()

        # read chromosome info from .fai
        chroms = []
        with open(fai_path) as f:
            for line in f:
                parts  = line.strip().split('\t')
                name   = parts[0]
                length = int(parts[1])
                chroms.append((name, length))

        if chroms:
            manifest.append({
                "species":  species,
                "fna_path": str(fna_path),
                "chroms":   chroms,
            })

    return manifest


def _encode_dna(seq):
    """Encode DNA string to integer list using DNA_VOCAB"""

    return [DNA_VOCAB.get(c, 5) for c in seq.upper()]


def _encode_codons(seq, frame=0):
    """Encode DNA string as non-overlapping codons starting at frame offset"""

    seq  = seq.upper()
    seq  = seq[frame:]
    trim = len(seq) - (len(seq) % 3)
    seq  = seq[:trim]

    ids = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        cid   = _CODON_LUT.get(codon, CODON_N_ID)
        ids.append(cid)

    return ids


class FASTAEntropyDataset(data_utils.IterableDataset):
    """Streams random fixed-length DNA windows from indexed FASTA

    Samples species uniformly, chromosomes proportional to length,
    random position within chromosome. Skips windows with >10% N
    """

    def __init__(self, manifest, max_len=512, max_n_frac=0.1, seed=42):

        super().__init__()

        self.manifest   = manifest
        self.max_len    = max_len
        self.max_n_frac = max_n_frac
        self.seed       = seed
        self._handles   = None

    def _open_handles(self):
        """Open pyfaidx handles (called per-worker)"""

        handles = {}
        for entry in self.manifest:
            sp = entry["species"]
            handles[sp] = pyfaidx.Fasta(
                entry["fna_path"], as_raw=True, sequence_always_upper=True,
            )
        return handles

    def _build_chrom_sampler(self, entry):
        """Build cumulative weights for proportional chromosome sampling"""

        filtered = [(name, length) for name, length in entry["chroms"]
                    if length >= self.max_len]
        if not filtered:
            return None, None, None

        names   = [name for name, _ in filtered]
        lengths = [length for _, length in filtered]
        cumwt   = []
        total   = 0
        for ln in lengths:
            total += ln
            cumwt.append(total)

        return names, cumwt, lengths

    def __iter__(self):

        worker_info = data_utils.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed

        rng     = random.Random(worker_seed)
        handles = self._open_handles()

        # pre-build chromosome samplers per species
        samplers = {}
        valid_entries = []
        for entry in self.manifest:
            sp                    = entry["species"]
            names, cumwt, lengths = self._build_chrom_sampler(entry)
            if names is None:
                continue
            samplers[sp] = (names, cumwt, lengths)
            valid_entries.append(entry)

        if not valid_entries:
            return

        max_n = int(self.max_len * self.max_n_frac)

        while True:
            # uniform species sampling
            entry = rng.choice(valid_entries)
            sp    = entry["species"]
            names, cumwt, lengths = samplers[sp]

            # proportional chromosome sampling
            r   = rng.random() * cumwt[-1]
            idx = bisect.bisect_right(cumwt, r)
            idx = min(idx, len(names) - 1)

            chrom_name   = names[idx]
            chrom_length = lengths[idx]

            # random position
            pos = rng.randint(0, chrom_length - self.max_len)
            seq = handles[sp][chrom_name][pos:pos + self.max_len]

            # skip high-N windows
            n_count = seq.count('N') + seq.count('n')
            if n_count > max_n:
                continue

            ids = _encode_dna(seq)
            yield ids

    def __del__(self):

        if self._handles is not None:
            for h in self._handles.values():
                h.close()


class FASTACodonDataset(data_utils.IterableDataset):
    """Streams codon-encoded DNA windows with frame augmentation

    Each base window yields up to 3 codon sequences (frames 0, 1, 2)
    """

    def __init__(self, manifest, max_bases=513, frames=(0, 1, 2),
                 max_n_frac=0.1, seed=42):

        super().__init__()

        self.manifest   = manifest
        self.max_bases  = max_bases
        self.frames     = frames
        self.max_n_frac = max_n_frac
        self.seed       = seed

    def _open_handles(self):
        """Open pyfaidx handles (called per-worker)"""

        handles = {}
        for entry in self.manifest:
            sp         = entry["species"]
            handles[sp] = pyfaidx.Fasta(
                entry["fna_path"], as_raw=True, sequence_always_upper=True,
            )
        return handles

    def _build_chrom_sampler(self, entry):
        """Build cumulative weights for proportional chromosome sampling"""

        filtered = [(name, length) for name, length in entry["chroms"]
                    if length >= self.max_bases]
        if not filtered:
            return None, None, None

        names   = [name for name, _ in filtered]
        lengths = [length for _, length in filtered]
        cumwt   = []
        total   = 0
        for ln in lengths:
            total += ln
            cumwt.append(total)

        return names, cumwt, lengths

    def __iter__(self):

        worker_info = data_utils.get_worker_info()
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed

        rng     = random.Random(worker_seed)
        handles = self._open_handles()

        samplers      = {}
        valid_entries = []
        for entry in self.manifest:
            sp                    = entry["species"]
            names, cumwt, lengths = self._build_chrom_sampler(entry)
            if names is None:
                continue
            samplers[sp] = (names, cumwt, lengths)
            valid_entries.append(entry)

        if not valid_entries:
            return

        max_n = int(self.max_bases * self.max_n_frac)

        while True:
            entry = rng.choice(valid_entries)
            sp    = entry["species"]
            names, cumwt, lengths = samplers[sp]

            r   = rng.random() * cumwt[-1]
            idx = bisect.bisect_right(cumwt, r)
            idx = min(idx, len(names) - 1)

            chrom_length = lengths[idx]
            pos          = rng.randint(0, chrom_length - self.max_bases)
            seq          = handles[sp][names[idx]][pos:pos + self.max_bases]

            n_count = seq.count('N') + seq.count('n')
            if n_count > max_n:
                continue

            frame = rng.choice(self.frames)
            ids   = _encode_codons(seq, frame=frame)
            if len(ids) >= 2:
                yield ids


def codon_collate(batch):
    """Next-codon prediction: input=codons[:-1], label=codons[1:], padded"""

    inputs = []
    labels = []

    for ids in batch:
        if len(ids) < 2:
            continue
        inputs.append(ids[:-1])
        labels.append(ids[1:])

    if not inputs:
        return None

    max_len = max(len(x) for x in inputs)

    input_ids = []
    label_ids = []
    for inp, lab in zip(inputs, labels):
        pad_n = max_len - len(inp)
        input_ids.append(inp + [CODON_PAD_ID] * pad_n)
        label_ids.append(lab + [-100] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(label_ids, dtype=torch.long),
    }


def dna_collate(batch):
    """Next-byte prediction: input=ids[:-1], label=ids[1:]"""

    input_ids = []
    labels    = []

    for ids in batch:
        if len(ids) < 2:
            continue
        input_ids.append(ids[:-1])
        labels.append(ids[1:])

    if not input_ids:
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels":    torch.tensor(labels, dtype=torch.long),
    }
