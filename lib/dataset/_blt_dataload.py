import torch
import webdataset as wds

import lib.dataset._binary as binary


PATCH_SIZE       = 8
BYTE_PAD_BUCKETS = list(range(PATCH_SIZE * 64, PATCH_SIZE * 9217, PATCH_SIZE * 64))
BYTE_MAX_SEQ     = BYTE_PAD_BUCKETS[-1]
BUDGET_PATCHES   = 8192


def _byte_bucket_pad(length):
    """Round byte length up to nearest bucket boundary (multiple of patch_size)"""

    for b in BYTE_PAD_BUCKETS:
        if length <= b:
            return b
    step = BYTE_PAD_BUCKETS[-1] - BYTE_PAD_BUCKETS[-2]
    return BYTE_PAD_BUCKETS[-1] + step * ((length - BYTE_PAD_BUCKETS[-1] + step - 1) // step)


def _patch_count(byte_len):
    """Number of patches for a given byte length"""

    return (byte_len + PATCH_SIZE - 1) // PATCH_SIZE


def _pad_to_patch(byte_len):
    """Round byte length up to nearest patch_size multiple"""

    return ((byte_len + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE


def _length_sorted_chunks(source, sort_size=256):
    """Accumulate samples, sort by total byte length, yield in order"""

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


def byte_budget_batcher(source, budget, max_batch, collator, sort_size=256):
    """Yield variable-size batches within a patch budget"""

    sorted_source = _length_sorted_chunks(source, sort_size)

    buf     = []
    buf_max = 0

    for sample in sorted_source:
        byte_len  = len(sample["input_ids"])
        new_max   = max(buf_max, byte_len)
        new_cost  = _patch_count(_byte_bucket_pad(new_max)) * (len(buf) + 1)

        if buf and (new_cost > budget or len(buf) + 1 > max_batch):
            result = collator(buf)
            if result is not None:
                yield result
            buf     = []
            buf_max = 0

        buf.append(sample)
        buf_max = max(buf_max, byte_len)

    if buf:
        result = collator(buf)
        if result is not None:
            yield result


def create_blt_train_pipeline(shard_urls, tokenizer, shuffle_buffer=10000):
    """WebDataset pipeline for BLT byte-level training"""

    def tokenize_sample(sample):

        input_ids  = tokenizer.encode(sample["input.txt"], add_special_tokens=False)
        target_ids = tokenizer.encode(sample["target.txt"], add_special_tokens=False)
        return {"input_ids": input_ids + target_ids, "prefix_len": len(input_ids)}

    return (
        wds.WebDataset(shard_urls, resampled=True, nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer)
        .decode()
        .map(tokenize_sample)
    )


class BytePrefixLMCollator:
    """Collator for BLT prefix-LM: separate input/target bytes, pad to patch multiples"""

    def __init__(self, pad_token_id=0, label_pad=-100, max_byte_len=BYTE_MAX_SEQ):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
        self.max_byte_len = max_byte_len
        self._dropped     = 0

    def __call__(self, batch):

        if self.max_byte_len is not None:
            kept = [b for b in batch if len(b["input_ids"]) <= self.max_byte_len]
            if len(kept) < len(batch):
                self._dropped += len(batch) - len(kept)
        else:
            kept = batch
        if not kept:
            return None

        inputs  = []
        targets = []
        for b in kept:
            p = b["prefix_len"]
            inputs.append(b["input_ids"][:p])
            targets.append(b["input_ids"][p:])

        max_in_bytes  = _pad_to_patch(max(len(inp) for inp in inputs))
        max_out_bytes = _pad_to_patch(max(len(tgt) for tgt in targets))

        all_input_bytes  = []
        all_target_bytes = []
        all_labels       = []

        for inp, tgt in zip(inputs, targets):
            in_padded  = inp + [self.pad_token_id] * (max_in_bytes - len(inp))
            out_padded = tgt + [self.pad_token_id] * (max_out_bytes - len(tgt))

            all_input_bytes.append(in_padded)
            all_target_bytes.append(out_padded)

            tgt_len = len(tgt)
            label = [
                out_padded[i + 1] if i + 1 < tgt_len else self.label_pad
                for i in range(max_out_bytes - 1)
            ]
            all_labels.append(label)

        return {
            "input_bytes":  torch.tensor(all_input_bytes, dtype=torch.long),
            "target_bytes": torch.tensor(all_target_bytes, dtype=torch.long),
            "labels":       torch.tensor(all_labels, dtype=torch.long),
        }
