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


def build_length_index(binary_path, tokenizer=None, batch_size=256):
    """Build length index for smart batching using batch tokenization"""

    reader = BinaryDatasetReader(binary_path, tokenizer)
    return reader.lengths


#######################
#####  Utilities  #####
#######################


def get_binary_stats(binary_path):
    """Get statistics about a binary dataset file"""

    info       = binary.get_binary_info(binary_path)
    num_chunks = info["num_chunks"]

    raw_count      = 0
    aug_count      = 0
    total_features = 0
    total_hints    = 0

    for i in range(num_chunks):
        chunk = binary.read_chunk_at_index(binary_path, i)
        if chunk.is_augmented:
            aug_count += 1
        else:
            raw_count += 1
        total_features += len(chunk.features)
        total_hints    += len(chunk.hints) if chunk.has_hints else 0

    return {
        "num_chunks":     num_chunks,
        "raw_count":      raw_count,
        "aug_count":      aug_count,
        "total_features": total_features,
        "total_hints":    total_hints,
        "compressed":     info["compressed"],
        "total_size":     info["total_size"],
    }


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


######################
#####  Collator  #####
######################


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


class BatchTokenizingCollator:
    """Collator that uses batch tokenization for efficiency"""

    def __init__(self, tokenizer, dataset, pad_token_id=None, label_pad=-100):

        self.tokenizer    = tokenizer
        self.dataset      = dataset
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.label_pad    = label_pad

    def __call__(self, indices):
        """Collate using batch tokenization"""

        # Get batch with batched tokenization
        batch = self.dataset.get_batch_tokenized(indices)

        # Dynamic padding
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


class CompactingCollator:
    """Collator for compacted samples with block-diagonal decoder attention"""

    def __init__(self, tokenizer, pad_token_id=None, label_pad=-100, sep_token="[SEP]"):

        self.tokenizer    = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.label_pad    = label_pad
        self.sep_token    = sep_token
        self.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    def __call__(self, batch):
        """Collate batch with proper attention masks for compacted samples"""

        groups = {}
        for item in batch:
            group_id = item.get("compact_group", id(item))
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(item)

        all_input_ids       = []
        all_attention_masks = []
        all_labels          = []
        all_decoder_masks   = []

        for group_id, items in groups.items():
            if len(items) == 1:
                item = items[0]
                all_input_ids.append(item["input_ids"])
                all_attention_masks.append([1] * len(item["input_ids"]))
                all_labels.append(item["labels"])
                all_decoder_masks.append(None)
            else:
                packed_input   = []
                packed_labels  = []
                segment_starts = []

                for i, item in enumerate(items):
                    if i > 0:
                        packed_input.append(self.sep_token_id)
                        packed_labels.append(self.label_pad)

                    segment_starts.append(len(packed_input))
                    packed_input.extend(item["input_ids"])
                    packed_labels.extend(item["labels"])

                segment_starts.append(len(packed_input))

                seq_len      = len(packed_labels)
                decoder_mask = [[0] * seq_len for _ in range(seq_len)]

                for seg_idx in range(len(segment_starts) - 1):
                    start = segment_starts[seg_idx]
                    end   = segment_starts[seg_idx + 1]

                    for i in range(start, end):
                        for j in range(start, end):
                            if j <= i:
                                decoder_mask[i][j] = 1

                all_input_ids.append(packed_input)
                all_attention_masks.append([1] * len(packed_input))
                all_labels.append(packed_labels)
                all_decoder_masks.append(decoder_mask)

        max_input_len = max(len(x) for x in all_input_ids)
        max_label_len = max(len(x) for x in all_labels)

        padded_inputs    = []
        padded_masks     = []
        padded_labels    = []
        padded_dec_masks = []

        for i in range(len(all_input_ids)):
            inp_len = len(all_input_ids[i])
            lbl_len = len(all_labels[i])

            inp_pad = max_input_len - inp_len
            lbl_pad = max_label_len - lbl_len

            padded_inputs.append(all_input_ids[i] + [self.pad_token_id] * inp_pad)
            padded_masks.append(all_attention_masks[i] + [0] * inp_pad)
            padded_labels.append(all_labels[i] + [self.label_pad] * lbl_pad)

            if all_decoder_masks[i] is not None:
                dec_mask = all_decoder_masks[i]
                for row in dec_mask:
                    row.extend([0] * lbl_pad)
                for _ in range(lbl_pad):
                    dec_mask.append([0] * max_label_len)
                padded_dec_masks.append(dec_mask)
            else:
                padded_dec_masks.append(None)

        result = {
            "input_ids":      padded_inputs,
            "attention_mask": padded_masks,
            "labels":         padded_labels,
        }

        if any(m is not None for m in padded_dec_masks):
            final_dec_masks = []
            for i, mask in enumerate(padded_dec_masks):
                if mask is None:
                    seq_len = max_label_len
                    causal  = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]
                    final_dec_masks.append(causal)
                else:
                    final_dec_masks.append(mask)

            result["decoder_attention_mask"] = final_dec_masks

        return result


#####################
#####  Sampler  #####
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