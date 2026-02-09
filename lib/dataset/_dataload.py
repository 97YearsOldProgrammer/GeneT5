import os
import random
import math

import torch

import lib.dataset._binary as binary


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
    """Dataset wrapper for binary training files"""

    def __init__(self, binary_path, tokenizer, seed=42):

        self.binary_path = binary_path
        self.tokenizer   = tokenizer
        self.seed        = seed
        self.epoch       = 0

        self._info    = binary.get_binary_info(binary_path)
        self._length  = self._info["num_chunks"]
        self._reader  = BinaryDatasetReader(binary_path, tokenizer)
        self._lengths = None

    @property
    def lengths(self):
        """Get lengths with lazy loading"""
        if self._lengths is None:
            self._lengths = self._reader.lengths
        return self._lengths

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __len__(self):

        return self._length

    def __getitem__(self, idx):

        random.seed(self.seed + self.epoch * len(self) + idx)

        sample     = self._reader.get_sample(idx)
        input_ids  = self.tokenizer.encode(sample["input_text"], add_special_tokens=False)
        target_ids = self.tokenizer.encode(sample["target_text"], add_special_tokens=False)

        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         target_ids,
        }

    def get_batch_tokenized(self, indices):
        """Get batch of samples with batched tokenization"""

        samples      = self._reader.get_samples_batched(indices)
        input_texts  = [s["input_text"] for s in samples]
        target_texts = [s["target_text"] for s in samples]

        input_enc  = self.tokenizer(input_texts, add_special_tokens=False, padding=False, truncation=False)
        target_enc = self.tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=False)

        results = []
        for i in range(len(indices)):
            results.append({
                "input_ids":      input_enc["input_ids"][i],
                "attention_mask": [1] * len(input_enc["input_ids"][i]),
                "labels":         target_enc["input_ids"][i],
            })

        return results


#####################
#####  Collator #####
#####################


class DynamicPaddingCollator:
    """Collator that pads batches dynamically"""

    def __init__(self, pad_token_id, label_pad=-100):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad

    def __call__(self, batch):

        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if batch[0]["labels"] else 0

        input_ids      = []
        attention_mask = []
        labels         = []

        for b in batch:
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len

            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)

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


#####################
#####  Samplers #####
#####################


class SmartBatchSampler:
    """Batch sampler that groups similar-length sequences"""

    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False, shuffle=True):

        self.lengths        = lengths
        self.batch_size     = batch_size
        self.bucket_size    = bucket_size
        self.drop_last      = drop_last
        self.shuffle        = shuffle
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self):

        buckets = []
        bucket  = []

        for idx in self.sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []

        if bucket:
            buckets.append(bucket)

        if self.shuffle:
            random.shuffle(buckets)
            for b in buckets:
                random.shuffle(b)

        all_indices = [idx for bucket in buckets for idx in bucket]
        batches     = []

        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):

        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class TokenBudgetSampler:
    """
    Batch sampler with token budget instead of fixed batch size

    Memory stays constant: large sequences get smaller batches, small sequences get larger batches
    """

    def __init__(self, label_lengths, max_tokens=128000, max_batch_size=32, min_batch_size=1,
                 shuffle=True, drop_last=False, seed=42, indices=None):

        self.label_lengths  = label_lengths
        self.max_tokens     = max_tokens
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.shuffle        = shuffle
        self.drop_last      = drop_last
        self.seed           = seed
        self.epoch          = 0

        if indices is not None:
            self.sorted_indices = sorted(indices,
                                         key=lambda i: label_lengths[i],
                                         reverse=True)
        else:
            self.sorted_indices = sorted(range(len(label_lengths)),
                                         key=lambda i: label_lengths[i],
                                         reverse=True)

        self._batches = self._build_batches()

    def _build_batches(self):
        """Build batches respecting token budget"""

        batches = []
        batch   = []
        max_len = 0

        for idx in self.sorted_indices:
            seq_len = self.label_lengths[idx]

            new_max    = max(max_len, seq_len)
            new_tokens = new_max * (len(batch) + 1)

            would_exceed_budget = new_tokens > self.max_tokens and len(batch) >= self.min_batch_size
            would_exceed_size   = len(batch) >= self.max_batch_size

            if (would_exceed_budget or would_exceed_size) and batch:
                batches.append(batch)
                batch   = []
                max_len = 0

            batch.append(idx)
            max_len = max(max_len, seq_len)

        if batch:
            if len(batch) >= self.min_batch_size or not self.drop_last:
                batches.append(batch)

        return batches

    def set_epoch(self, epoch):
        """Set epoch for shuffling reproducibility"""

        self.epoch = epoch

    def set_max_batches(self, n):
        """Limit iteration to n batches for distributed sync"""

        self._max_batches = n

    def __iter__(self):

        batches = self._batches.copy()

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        limit = getattr(self, '_max_batches', None)
        if limit is not None:
            batches = batches[:limit]

        for batch in batches:
            yield batch

    def __len__(self):

        limit = getattr(self, '_max_batches', None)
        if limit is not None:
            return min(len(self._batches), limit)
        return len(self._batches)

    def get_batch_stats(self):
        """Return statistics about batch sizes"""

        sizes = [len(b) for b in self._batches]
        tokens_per_batch = []

        for batch in self._batches:
            max_len = max(self.label_lengths[i] for i in batch)
            tokens_per_batch.append(max_len * len(batch))

        return {
            "num_batches":    len(self._batches),
            "min_batch_size": min(sizes),
            "max_batch_size": max(sizes),
            "avg_batch_size": sum(sizes) / len(sizes),
            "min_tokens":     min(tokens_per_batch),
            "max_tokens":     max(tokens_per_batch),
            "avg_tokens":     sum(tokens_per_batch) / len(tokens_per_batch),
        }
