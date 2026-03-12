import bisect
import random

import torch
import webdataset as wds

import lib.data._binary as binary


############################
#####  WebDataset I/O  #####
############################


def create_train_pipeline(shard_urls, tokenizer, shuffle_buffer=10000):
    """WebDataset pipeline for sequential tar-based training"""

    def tokenize_sample(sample):

        input_ids  = tokenizer.encode(sample["input.txt"], add_special_tokens=False)
        target_ids = tokenizer.encode(sample["target.txt"], add_special_tokens=False)
        return {"input_ids": input_ids + target_ids, "prefix_len": len(input_ids)}

    return (
        wds.WebDataset(shard_urls, resampled=True, nodesplitter=wds.split_by_node,
                       handler=wds.warn_and_continue)
        .shuffle(shuffle_buffer)
        .decode()
        .map(tokenize_sample)
    )


#################
#####  I/O  #####
#################


#####################
#####  Dataset  #####
#####################


class BinaryTrainDataset:
    """Dataset wrapper for binary training files (single or sharded)"""

    def __init__(self, binary_path, tokenizer, seed=42):
        """
        Accept str (single file, backward compat) or list (sharded mode)
        Init only reads 10-byte headers — no file handles opened here
        """

        self.tokenizer = tokenizer
        self.seed      = seed
        self.epoch     = 0
        self._lengths  = None

        if isinstance(binary_path, list):
            self._paths  = [str(p) for p in binary_path]
            self._multi  = True
        else:
            self._paths  = [str(binary_path)]
            self._multi  = False

        # Build cumulative index from header-only reads
        counts       = [binary.get_chunk_count(p) for p in self._paths]
        self._counts = counts
        self._length = sum(counts)

        # Cumulative sums for bisect: [c0-1, c0+c1-1, ...]
        self._cumsum = []
        running      = 0
        for c in counts:
            running += c
            self._cumsum.append(running)

    def _resolve(self, idx):
        """Map flat index to (file_path, local_index)"""

        if not self._multi:
            return self._paths[0], idx

        fi = bisect.bisect_right(self._cumsum, idx)
        base = self._cumsum[fi - 1] if fi > 0 else 0
        return self._paths[fi], idx - base

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __len__(self):

        return self._length

    def __getitem__(self, idx):

        random.seed(self.seed + self.epoch * self._length + idx)

        path, local = self._resolve(idx)
        chunk      = binary.read_chunk_at_index(path, local)
        input_ids  = self.tokenizer.encode(chunk.get_input_text(), add_special_tokens=False)
        target_ids = self.tokenizer.encode(chunk.get_target_text(), add_special_tokens=False)

        full_ids = input_ids + target_ids

        return {
            "input_ids":  full_ids,
            "prefix_len": len(input_ids),
        }


#####################
#####  Collator #####
#####################


PAD_BUCKETS     = [2048, 4096, 6144, 8192]
DEFAULT_MAX_SEQ = PAD_BUCKETS[-1]
BUDGET_SEQ      = 8192
PACK_SEQ_LEN    = 8192


def _bucket_pad(length):
    """Round length up to nearest bucket boundary for compile shape stability"""

    for b in PAD_BUCKETS:
        if length <= b:
            return b
    step = PAD_BUCKETS[-1] - PAD_BUCKETS[-2]
    return PAD_BUCKETS[-1] + step * ((length - PAD_BUCKETS[-1] + step - 1) // step)


def _length_sorted_chunks(source, sort_size=256):
    """Accumulate samples, sort by length, yield in length order"""

    pool = []
    for sample in source:
        pool.append(sample)
        if len(pool) >= sort_size:
            pool.sort(key=lambda s: len(s["input_ids"]))
            yield from pool
            pool = []

    if pool:
        pool.sort(key=lambda s: len(s["input_ids"]))
        yield from pool


def token_budget_batcher(source, budget, max_batch, collator, sort_size=256):
    """Yield variable-size batches that fit within a token budget"""

    sorted_source = _length_sorted_chunks(source, sort_size)

    buf     = []
    buf_max = 0

    for sample in sorted_source:
        seq_len  = len(sample["input_ids"])
        new_max  = max(buf_max, seq_len)
        new_cost = _bucket_pad(new_max) * (len(buf) + 1)

        if buf and (new_cost > budget or len(buf) + 1 > max_batch):
            result = collator(buf)
            if result is not None:
                yield result
            buf     = []
            buf_max = 0

        buf.append(sample)
        buf_max = max(buf_max, seq_len)

    if buf:
        result = collator(buf)
        if result is not None:
            yield result


###########################
#####  Packed Collate #####
###########################


def packed_collate(batch):
    """Collate packed WebDataset samples into a batch with cu_seqlens

    Always emits one padding segment per row (zero-length if no padding)
    so the masking loop can uniformly skip it with seg_idx += 1
    """

    B = len(batch)
    S = len(batch[0]["input_ids"])

    input_ids = torch.stack([b["input_ids"] for b in batch])

    offsets       = [0]
    prefix_all    = []
    num_real_list = []
    running       = 0

    for b in batch:
        for sl in b["seq_lens"]:
            running += sl
            offsets.append(running)
        # Always add padding segment (zero-length if fully packed)
        pad_len  = S - sum(b["seq_lens"])
        running += pad_len
        offsets.append(running)
        prefix_all.extend(b["prefix_lens"])
        num_real_list.append(len(b["seq_lens"]))

    cu = torch.tensor(offsets, dtype=torch.int32)

    return {
        "input_ids":    input_ids,
        "cu_seqlens":   cu,
        "prefix_lens":  torch.tensor(prefix_all, dtype=torch.int32),
        "num_real":     torch.tensor(num_real_list, dtype=torch.int32),
        "max_seqlen":   int((cu[1:] - cu[:-1]).max().item()),
    }

