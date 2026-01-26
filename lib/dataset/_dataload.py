import json
import random
from concurrent.futures import ThreadPoolExecutor

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
        """Compute lengths using batched tokenization for efficiency"""

        lengths = []

        # Collect all input texts first
        all_texts = []
        for i in range(self._num_chunks):
            chunk = binary.read_chunk_at_index(self.binary_path, i)
            all_texts.append(chunk.get_input_text())

        if self.tokenizer is None:
            # Fallback: rough estimate
            lengths = [len(text) // 4 for text in all_texts]
        else:
            # Batch tokenize for efficiency
            for i in range(0, len(all_texts), batch_size):
                batch       = all_texts[i:i + batch_size]
                encoded     = self.tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
                batch_lens  = [len(ids) for ids in encoded["input_ids"]]
                lengths.extend(batch_lens)

        return lengths

    def _format_sample(self, chunk):
        """Format chunk as input/target text using BinaryChunk methods"""

        input_text  = chunk.get_input_text()
        target_text = chunk.get_target_text()

        return {
            "input_text":  input_text,
            "target_text": target_text,
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
    """Dataset wrapper for binary training files with batch tokenization"""

    def __init__(self, binary_path, tokenizer, max_input_len, max_target_len, seed=42):

        self.binary_path    = binary_path
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.seed           = seed
        self.epoch          = 0

        self._info   = binary.get_binary_info(binary_path)
        self._length = self._info["num_chunks"]
        self._reader = BinaryDatasetReader(binary_path, tokenizer)
        self.lengths = None

        # Cache for batched samples
        self._sample_cache = {}
        self._cache_size   = 1000

    def build_length_index(self):
        """Build length index for smart batching"""

        if self.lengths is None:
            self.lengths = self._reader.lengths
        return self.lengths

    def set_epoch(self, epoch):

        self.epoch = epoch
        self._sample_cache.clear()

    def __len__(self):

        return self._length

    def __getitem__(self, idx):

        random.seed(self.seed + self.epoch * len(self) + idx)

        sample = self._reader.get_sample(idx)

        input_ids  = self.tokenizer.encode(sample["input_text"], add_special_tokens=False)
        target_ids = self.tokenizer.encode(sample["target_text"], add_special_tokens=False)

        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]

        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]

        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         target_ids,
        }

    def get_batch_tokenized(self, indices):
        """Get a batch of samples with batched tokenization (more efficient)"""

        samples     = self._reader.get_samples_batched(indices)
        input_texts = [s["input_text"] for s in samples]
        target_texts = [s["target_text"] for s in samples]

        # Batch tokenize
        input_enc  = self.tokenizer(input_texts, add_special_tokens=False, padding=False, truncation=False)
        target_enc = self.tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=False)

        results = []
        for i in range(len(indices)):
            input_ids  = input_enc["input_ids"][i]
            target_ids = target_enc["input_ids"][i]

            if len(input_ids) > self.max_input_len:
                input_ids = input_ids[:self.max_input_len]

            if len(target_ids) > self.max_target_len:
                target_ids = target_ids[:self.max_target_len]

            results.append({
                "input_ids":      input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels":         target_ids,
            })

        return results



#####################
#####  Utility  #####
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
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

        if labels:
            result["labels"] = labels

        return result


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