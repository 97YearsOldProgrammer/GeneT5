import json
import pathlib
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import lib.data._binary as binary


#########################
#####  Consolidate  #####
#########################


def consolidate(binary_paths, output_dir, show_progress=True):
    """Convert per-species GT5B binaries to lean RAM-loadable format

    Streams tokens directly to disk to avoid OOM on large datasets.
    Writes:
      tokens.bin      - flat uint16 concatenated samples
      offsets.npy     - sample boundaries (N+1 int64)
      prefix_lens.npy - input prefix lengths (N int32)
      metadata.json   - dataset info
    """

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / "tokens.bin"
    offsets     = [0]
    prefix_lens = []
    skipped     = 0
    buf         = np.empty(1 << 20, dtype=np.uint16)
    buf_pos     = 0

    with open(tokens_path, "wb") as tok_f:

        def _flush():
            nonlocal buf_pos
            if buf_pos > 0:
                tok_f.write(buf[:buf_pos].tobytes())
                buf_pos = 0

        for path in binary_paths:
            sp_name = pathlib.Path(path).parent.name
            if show_progress:
                print(f"  Reading {sp_name}...", end=" ", flush=True)

            count = 0
            for _, input_ids, target_ids in binary.iter_binary_token_ids(str(path)):
                if input_ids is None:
                    skipped += 1
                    continue

                full = input_ids + target_ids
                n    = len(full)

                # Flush buffer if it would overflow
                if buf_pos + n > len(buf):
                    _flush()
                    # Single sample larger than buffer — write directly
                    if n > len(buf):
                        tok_f.write(np.array(full, dtype=np.uint16).tobytes())
                        offsets.append(offsets[-1] + n)
                        prefix_lens.append(len(input_ids))
                        count += 1
                        continue

                buf[buf_pos:buf_pos + n] = full
                buf_pos += n
                offsets.append(offsets[-1] + n)
                prefix_lens.append(len(input_ids))
                count += 1

            if show_progress:
                print(f"{count:,} samples")

        _flush()

    num_samples  = len(prefix_lens)
    total_tokens = offsets[-1]

    if show_progress:
        print(f"\n  Total: {num_samples:,} samples, {total_tokens:,} tokens")
        if skipped:
            print(f"  Skipped: {skipped} (no pre-tokenized IDs)")

    np.save(output_dir / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(output_dir / "prefix_lens.npy", np.array(prefix_lens, dtype=np.int32))

    tokens_mb = tokens_path.stat().st_size / 1024 / 1024

    meta = {
        "num_samples":  num_samples,
        "total_tokens": total_tokens,
        "skipped":      skipped,
        "format":       "ram_v1",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if show_progress:
        print(f"  tokens.bin: {tokens_mb:.1f} MB")

    return meta


#####################
#####  Dataset  #####
#####################


class RamDataset(Dataset):
    """Pre-tokenized dataset loaded entirely into RAM"""

    def __init__(self, data_dir):

        data_dir         = pathlib.Path(data_dir)
        self.tokens      = np.fromfile(data_dir / "tokens.bin", dtype=np.uint16)
        self.offsets     = np.load(data_dir / "offsets.npy")
        self.prefix_lens = np.load(data_dir / "prefix_lens.npy")
        self._lengths    = (self.offsets[1:] - self.offsets[:-1]).astype(np.int32)

    def __len__(self):

        return len(self.prefix_lens)

    def __getitem__(self, idx):

        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        ids   = self.tokens[start:end].astype(np.int32).tolist()

        return {"input_ids": ids, "prefix_len": int(self.prefix_lens[idx])}

    def get_lengths(self):
        """Sample lengths for batch planning"""

        return self._lengths


###################################
#####  Token Budget Sampler   #####
###################################


PAD_BUCKETS = [2048, 4096, 6144, 8192]


def _bucket_pad(length):
    """Round length up to nearest bucket boundary"""

    for b in PAD_BUCKETS:
        if length <= b:
            return b
    step = PAD_BUCKETS[-1] - PAD_BUCKETS[-2]
    return PAD_BUCKETS[-1] + step * ((length - PAD_BUCKETS[-1] + step - 1) // step)


class TokenBudgetSampler(Sampler):
    """Yields index batches that fit within a token budget

    Sorts by length within random groups for efficiency while
    maintaining epoch-level shuffling for training variance.
    Saves/loads state for resumability.
    """

    def __init__(self, lengths, budget, max_batch=32, sort_pool=256, seed=42,
                 rank=0, world_size=1, drop_last=False):

        self.lengths    = np.asarray(lengths, dtype=np.int32)
        self.budget     = budget
        self.max_batch  = max_batch
        self.sort_pool  = sort_pool
        self.seed       = seed
        self.rank       = rank
        self.world_size = world_size
        self.drop_last  = drop_last
        self.epoch      = 0

        # Pre-compute batches for length estimation
        self._batches = None

    def set_epoch(self, epoch):

        self.epoch = epoch
        self._batches = None

    def _build_batches(self):
        """Plan all batches for current epoch"""

        rng     = np.random.RandomState(self.seed + self.epoch)
        indices = rng.permutation(len(self.lengths))

        # Shard across ranks
        if self.world_size > 1:
            per_rank = len(indices) // self.world_size
            start    = self.rank * per_rank
            end      = start + per_rank if self.rank < self.world_size - 1 else len(indices)
            indices  = indices[start:end]

        # Sort within pools for length grouping
        batches = []
        for pool_start in range(0, len(indices), self.sort_pool):
            pool = indices[pool_start:pool_start + self.sort_pool]
            pool = pool[np.argsort(self.lengths[pool])]

            buf     = []
            buf_max = 0

            for idx in pool:
                seq_len  = int(self.lengths[idx])
                new_max  = max(buf_max, seq_len)
                new_cost = _bucket_pad(new_max) * (len(buf) + 1)

                if buf and (new_cost > self.budget or len(buf) + 1 > self.max_batch):
                    batches.append(buf)
                    buf     = []
                    buf_max = 0

                buf.append(int(idx))
                buf_max = max(buf_max, seq_len)

            if buf:
                if not self.drop_last or len(buf) >= 2:
                    batches.append(buf)

        # Shuffle batch order
        rng.shuffle(batches)
        return batches

    def __iter__(self):

        self._batches = self._build_batches()
        yield from self._batches

    def __len__(self):

        if self._batches is None:
            self._batches = self._build_batches()
        return len(self._batches)

    def state_dict(self):

        return {"epoch": self.epoch, "seed": self.seed}

    def load_state_dict(self, state):

        self.epoch = state["epoch"]
        self.seed  = state.get("seed", self.seed)
        self._batches = None


#########################
#####  Compression  #####
#########################


def compress_dataset(data_dir):
    """Create zstd-compressed tar for upload

    Produces {data_dir}.tar.zst alongside the data directory
    """

    data_dir = pathlib.Path(data_dir)
    archive  = data_dir.parent / f"{data_dir.name}.tar.zst"

    subprocess.run(
        ["tar", "--zstd", "-cf", str(archive),
         "-C", str(data_dir.parent), data_dir.name],
        check=True,
    )

    orig_mb = sum(f.stat().st_size for f in data_dir.iterdir()) / 1024 / 1024
    comp_mb = archive.stat().st_size / 1024 / 1024
    ratio   = comp_mb / orig_mb * 100 if orig_mb > 0 else 0
    print(f"  Compressed: {orig_mb:.1f} MB -> {comp_mb:.1f} MB ({ratio:.0f}%)")

    return archive


def decompress_dataset(archive_path, output_dir=None):
    """Decompress zstd tar archive"""

    archive_path = pathlib.Path(archive_path)
    if output_dir is None:
        output_dir = archive_path.parent

    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive_path),
         "-C", str(output_dir)],
        check=True,
    )

    return output_dir
