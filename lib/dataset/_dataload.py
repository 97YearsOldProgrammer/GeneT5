import os
import bisect
import random

import torch
import webdataset as wds

import lib.dataset._binary as binary


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
        wds.WebDataset(shard_urls, resampled=True)
        .shuffle(shuffle_buffer)
        .decode()
        .map(tokenize_sample)
    )


#################
#####  I/O  #####
#################


class BinaryDatasetReader:
    """Lazy reader for binary training files with batch tokenization support"""

    def __init__(self, binary_path, tokenizer=None):

        self.binary_path = binary_path
        self.tokenizer   = tokenizer
        self._info       = binary.get_binary_info(binary_path)
        self._num_chunks = self._info["num_chunks"]
        self._lengths    = None

    def __len__(self):

        return self._num_chunks

    @property
    def lengths(self):
        """Get actual token lengths using batch tokenization"""

        if self._lengths is None:
            self._lengths = self._compute_lengths_batched()
        return self._lengths

    def _compute_lengths_batched(self, batch_size=256):
        """Read pre-stored lengths from binary chunks, fallback to tokenization if missing"""

        lengths = []

        for i in range(self._num_chunks):
            chunk = binary.read_chunk_at_index(self.binary_path, i)

            if chunk.input_len is not None:
                lengths.append(chunk.input_len)
            else:
                if self.tokenizer is not None:
                    text    = chunk.get_input_text()
                    encoded = self.tokenizer(text, add_special_tokens=False)
                    lengths.append(len(encoded["input_ids"]))
                else:
                    lengths.append(len(chunk.get_input_text()) // 4)

        return lengths

    def _format_sample(self, chunk):
        """Format chunk as input/target text"""

        return {
            "input_text":  chunk.get_input_text(),
            "target_text": chunk.get_target_text(),
            "gene_ids":    chunk.gene_ids,
            "seqid":       chunk.seqid,
            "start":       chunk.start,
            "end":         chunk.end,
        }

    def get_chunk(self, idx):
        """Get raw chunk at index"""

        return binary.read_chunk_at_index(self.binary_path, idx)

    def get_sample(self, idx):
        """Get formatted sample at index"""

        chunk = self.get_chunk(idx)
        return self._format_sample(chunk)

    def get_samples_batched(self, indices):
        """Get multiple samples with batched tokenization"""

        chunks  = [self.get_chunk(i) for i in indices]
        samples = [self._format_sample(c) for c in chunks]
        return samples


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


PAD_BUCKETS = [2048, 4096, 6144, 8192, 16384]


def _bucket_pad(length):
    """Round length up to nearest bucket boundary for compile shape stability"""

    for b in PAD_BUCKETS:
        if length <= b:
            return b
    return length


class PrefixLMCollator:
    """Collator for prefix-LM training: mask loss on input prefix, compute on output only"""

    def __init__(self, pad_token_id, label_pad=-100):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad

    def __call__(self, batch):

        max_prefix = max(b["prefix_len"] for b in batch)

        # Align prefixes: [prefix | gap_pad | target | tail_pad]
        aligned = []
        for b in batch:
            ids    = b["input_ids"]
            p_len  = b["prefix_len"]
            prefix = ids[:p_len]
            target = ids[p_len:]
            gap    = [self.pad_token_id] * (max_prefix - p_len)
            aligned.append(prefix + gap + target)

        max_len = _bucket_pad(max(len(a) for a in aligned))

        all_input_ids = []
        all_labels    = []

        for i, seq in enumerate(aligned):
            target_len = len(batch[i]["input_ids"]) - batch[i]["prefix_len"]
            pad_len    = max_len - len(seq)
            padded     = seq + [self.pad_token_id] * pad_len

            # Model input: all tokens except last (teacher forcing)
            all_input_ids.append(padded[:-1])

            # Labels: mask prefix, keep target, mask tail pads
            shifted = padded[1:]
            labels  = [self.label_pad] * (max_prefix - 1)
            labels += shifted[max_prefix - 1 : max_prefix - 1 + target_len]
            labels += [self.label_pad] * (max_len - 1 - len(labels))
            all_labels.append(labels)

        return {
            "input_ids":  torch.tensor(all_input_ids, dtype=torch.long),
            "labels":     torch.tensor(all_labels, dtype=torch.long),
            "prefix_len": max_prefix,
        }


class DynamicPaddingCollator:
    """Collator that pads batches dynamically (kept for backward compat)"""

    def __init__(self, pad_token_id, label_pad=-100):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad

    def __call__(self, batch):

        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if "labels" in batch[0] else 0

        input_ids      = []
        attention_mask = []
        labels         = []

        for b in batch:
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len

            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(b.get("attention_mask", [1] * inp_len) + [0] * pad_len)

            if max_target_len > 0:
                lbl_len = len(b["labels"])
                lbl_pad = max_target_len - lbl_len
                labels.append(b["labels"] + [self.label_pad] * lbl_pad)

        result = {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if labels:
            result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result
