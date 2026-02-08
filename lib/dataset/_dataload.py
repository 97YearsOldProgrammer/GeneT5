import os
import struct
import random
import math

import torch

import lib.dataset._binary     as binary
import lib.dataset._compacting as compacting
import lib.dataset._packed     as packed


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
                # Fallback: tokenize this chunk (should rarely happen with proper data prep)
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
            "input_text":    chunk.get_input_text(),
            "target_text":   chunk.get_target_text(),
            "gene_ids":      chunk.gene_ids,
            "seqid":         chunk.seqid,
            "start":         chunk.start,
            "end":           chunk.end,
            "compact_group": getattr(chunk, 'compact_group', None),
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

    def build_length_index(self):
        """Build length index for smart batching (deprecated, use lengths property)"""
        return self.lengths

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
            "compact_group":  sample.get("compact_group"),
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
                "compact_group":  samples[i].get("compact_group"),
            })

        return results


#####################
#####  Collators ####
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


class CompactingCollator:
    """Collator for compacted samples with hybrid isolation and segment masking"""

    def __init__(self, tokenizer, pad_token_id=None, label_pad=-100, sep_token="[SEP]", block_size=64, window_size=256):

        self.tokenizer    = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.label_pad    = label_pad
        self.block_size   = block_size
        self.window_size  = window_size
        self.sep_token    = sep_token
        self.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    def _align_to_block(self, length):
        """Round up to next block boundary"""

        return math.ceil(length / self.block_size) * self.block_size

    def _compute_isolation_start(self, current_pos):
        """Compute next position that ensures window isolation"""

        target = current_pos + self.window_size + 1
        return self._align_to_block(target)

    def _pack_group_inputs(self, items):
        """Pack input sequences with window isolation"""

        packed     = []
        attn_mask  = []
        seg_starts = []
        seg_ends   = []
        seg_ids    = []

        for i, item in enumerate(items):
            seg_starts.append(len(packed))

            for _ in item["input_ids"]:
                seg_ids.append(i)

            packed.extend(item["input_ids"])
            attn_mask.extend([1] * len(item["input_ids"]))
            seg_ends.append(len(packed))

            if i < len(items) - 1:
                packed.append(self.sep_token_id)
                attn_mask.append(1)
                seg_ids.append(-1)

                next_start = self._compute_isolation_start(len(packed))
                pad_len    = next_start - len(packed)

                packed.extend([self.pad_token_id] * pad_len)
                attn_mask.extend([0] * pad_len)
                seg_ids.extend([-1] * pad_len)

        return packed, attn_mask, seg_starts, seg_ends, seg_ids

    def _pack_group_labels(self, items):
        """Pack label sequences with window isolation"""

        packed     = []
        seg_starts = []
        seg_ends   = []

        for i, item in enumerate(items):
            seg_starts.append(len(packed))
            packed.extend(item["labels"])
            seg_ends.append(len(packed))

            if i < len(items) - 1:
                packed.append(self.label_pad)

                next_start = self._compute_isolation_start(len(packed))
                pad_len    = next_start - len(packed)

                packed.extend([self.label_pad] * pad_len)

        return packed, seg_starts, seg_ends

    def _build_segment_mask(self, seg_ids, seq_len):
        """Build 2D segment mask for global/cross attention"""

        mask = [[0] * seq_len for _ in range(seq_len)]

        for i in range(seq_len):
            for j in range(seq_len):
                if seg_ids[i] >= 0 and seg_ids[j] >= 0 and seg_ids[i] == seg_ids[j]:
                    mask[i][j] = 1

        return mask

    def __call__(self, batch):
        """Collate batch with hybrid isolation and segment masking"""

        groups = {}
        for item in batch:
            group_id = item.get("compact_group")
            if group_id is None:
                group_id = id(item)
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(item)

        packed_results = []

        for group_id, items in groups.items():
            if len(items) == 1:
                item    = items[0]
                inp_len = len(item["input_ids"])
                aligned = self._align_to_block(inp_len)
                pad_len = aligned - inp_len

                seg_ids = [0] * inp_len + [-1] * pad_len

                packed_results.append({
                    "input_ids":      item["input_ids"] + [self.pad_token_id] * pad_len,
                    "attention_mask": [1] * inp_len + [0] * pad_len,
                    "labels":         item["labels"],
                    "segment_starts": [0],
                    "segment_ends":   [inp_len],
                    "segment_ids":    seg_ids,
                    "num_segments":   1,
                })
            else:
                inp, attn, inp_starts, inp_ends, seg_ids = self._pack_group_inputs(items)
                lbl, lbl_starts, lbl_ends                = self._pack_group_labels(items)

                packed_results.append({
                    "input_ids":      inp,
                    "attention_mask": attn,
                    "labels":         lbl,
                    "segment_starts": inp_starts,
                    "segment_ends":   inp_ends,
                    "segment_ids":    seg_ids,
                    "num_segments":   len(items),
                })

        max_input_len = max(len(r["input_ids"]) for r in packed_results)
        max_label_len = max(len(r["labels"]) for r in packed_results)

        final_inputs  = []
        final_masks   = []
        final_labels  = []
        final_segs    = []
        final_seg_ids = []

        for r in packed_results:
            inp_pad = max_input_len - len(r["input_ids"])
            lbl_pad = max_label_len - len(r["labels"])

            final_inputs.append(r["input_ids"] + [self.pad_token_id] * inp_pad)
            final_masks.append(r["attention_mask"] + [0] * inp_pad)
            final_labels.append(r["labels"] + [self.label_pad] * lbl_pad)
            final_seg_ids.append(r["segment_ids"] + [-1] * inp_pad)
            final_segs.append({
                "starts":       r["segment_starts"],
                "ends":         r["segment_ends"],
                "num_segments": r["num_segments"],
            })

        return {
            "input_ids":      final_inputs,
            "attention_mask": final_masks,
            "labels":         final_labels,
            "segment_info":   final_segs,
            "segment_ids":    final_seg_ids,
        }


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

    Memory stays constant: large sequences get smaller batches, small sequences get larger batches.
    Budget is max_len_in_batch × num_samples (accounts for padding).
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

        # Sort indices by label length (descending for largest-first)
        if indices is not None:
            self.sorted_indices = sorted(indices,
                                         key=lambda i: label_lengths[i],
                                         reverse=True)
        else:
            self.sorted_indices = sorted(range(len(label_lengths)),
                                         key=lambda i: label_lengths[i],
                                         reverse=True)

        # Pre-compute batches
        self._batches = self._build_batches()

    def _build_batches(self):
        """Build batches respecting token budget"""

        batches = []
        batch   = []
        max_len = 0

        for idx in self.sorted_indices:
            seq_len = self.label_lengths[idx]

            # Calculate new padded total if we add this sample
            new_max    = max(max_len, seq_len)
            new_tokens = new_max * (len(batch) + 1)

            # Check if adding would exceed budget or max batch size
            would_exceed_budget = new_tokens > self.max_tokens and len(batch) >= self.min_batch_size
            would_exceed_size   = len(batch) >= self.max_batch_size

            if (would_exceed_budget or would_exceed_size) and batch:
                batches.append(batch)
                batch   = []
                max_len = 0

            batch.append(idx)
            max_len = max(max_len, seq_len)

        # Don't forget last batch
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
            "num_batches":      len(self._batches),
            "min_batch_size":   min(sizes),
            "max_batch_size":   max(sizes),
            "avg_batch_size":   sum(sizes) / len(sizes),
            "min_tokens":       min(tokens_per_batch),
            "max_tokens":       max(tokens_per_batch),
            "avg_tokens":       sum(tokens_per_batch) / len(tokens_per_batch),
        }


class CompactGroupSampler:
    """Sampler that keeps compact_group members together in same batch"""

    def __init__(self, dataset, batch_size, shuffle=True, seed=42):

        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.epoch      = 0

        self._build_groups()

    def _build_groups(self):
        """Build mapping from compact_group to indices"""

        self.groups = {}

        for idx in range(len(self.dataset)):
            chunk = self.dataset._reader.get_chunk(idx)
            gid   = getattr(chunk, 'compact_group', None)

            if gid is None:
                gid = f"singleton_{idx}"

            if gid not in self.groups:
                self.groups[gid] = []
            self.groups[gid].append(idx)

        self.group_ids = list(self.groups.keys())

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __iter__(self):

        random.seed(self.seed + self.epoch)

        group_ids = self.group_ids.copy()
        if self.shuffle:
            random.shuffle(group_ids)

        current_batch = []

        for gid in group_ids:
            indices = self.groups[gid]

            if len(current_batch) + len(indices) <= self.batch_size:
                current_batch.extend(indices)
            else:
                if current_batch:
                    yield current_batch
                current_batch = indices.copy()

        if current_batch:
            yield current_batch

    def __len__(self):

        return len(self.group_ids)


###############################
#####  Packed Dataset     #####
###############################


class PackedTrainDataset:
    """
    Dataset for pre-packed training data

    Streams pre-packed samples directly - no runtime packing overhead.
    Offset table cached in __init__, file handle opened lazily per-worker.
    """

    def __init__(self, packed_path, seed=42):

        self.packed_path = packed_path
        self.seed        = seed
        self.epoch       = 0
        self._file       = None
        self._file_pid   = None

        # Read header + cache full offset table once
        with open(packed_path, 'rb') as f:
            magic = f.read(4)
            if magic != packed.PACKED_MAGIC:
                raise ValueError(f"Invalid magic header: {magic}")

            _version    = struct.unpack('<B', f.read(1))[0]
            num_samples = struct.unpack('<I', f.read(4))[0]

            raw = f.read(num_samples * 12)

        self._length  = num_samples
        self._offsets = []
        for i in range(num_samples):
            off  = i * 12
            vals = struct.unpack('<QI', raw[off:off + 12])
            self._offsets.append(vals)

    def _get_file(self):
        """Lazy file handle — opens once per process (fork-safe for num_workers>0)"""

        pid = os.getpid()
        if self._file is None or self._file.closed or self._file_pid != pid:
            if self._file is not None and not self._file.closed:
                self._file.close()
            self._file     = open(self.packed_path, 'rb')
            self._file_pid = pid
        return self._file

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __len__(self):

        return self._length

    def __getitem__(self, idx):

        offset, length = self._offsets[idx]
        f              = self._get_file()
        f.seek(offset)
        data           = f.read(length)
        sample         = packed.PackedSample.from_bytes(data)

        return {
            "input_ids":       sample.input_ids,
            "labels":          sample.labels,
            "segment_ids":     sample.segment_ids,
            "segment_starts":  sample.segment_starts,
            "segment_ends":    sample.segment_ends,
            "num_segments":    sample.num_segments,
        }

    def __getstate__(self):
        """Drop file handle for pickling (spawn-based num_workers)"""

        state              = self.__dict__.copy()
        state['_file']     = None
        state['_file_pid'] = None
        return state

    def __del__(self):

        if self._file is not None and not self._file.closed:
            self._file.close()


class PackedCollator:
    """
    Simple collator for pre-packed samples

    Just pads to max length in batch - no complex packing logic needed.
    """

    def __init__(self, pad_token_id, label_pad=-100, align=32):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
        self.align        = align

    def __call__(self, batch):

        align         = self.align
        max_input_len = max(len(b["input_ids"]) for b in batch)
        max_label_len = max(len(b["labels"]) for b in batch)

        if align > 1:
            max_input_len = (max_input_len + align - 1) // align * align
            max_label_len = (max_label_len + align - 1) // align * align

        input_ids      = []
        attention_mask = []
        labels         = []
        segment_ids    = []
        segment_info   = []

        for b in batch:
            inp_len = len(b["input_ids"])
            lbl_len = len(b["labels"])
            inp_pad = max_input_len - inp_len
            lbl_pad = max_label_len - lbl_len

            input_ids.append(b["input_ids"] + [self.pad_token_id] * inp_pad)
            attention_mask.append([1 if s >= 0 else 0 for s in b["segment_ids"]] + [0] * inp_pad)
            labels.append(b["labels"] + [self.label_pad] * lbl_pad)
            segment_ids.append(b["segment_ids"] + [-1] * inp_pad)
            segment_info.append({
                "starts":       b["segment_starts"],
                "ends":         b["segment_ends"],
                "num_segments": b["num_segments"],
            })

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "segment_ids":    segment_ids,
            "segment_info":   segment_info,
        }