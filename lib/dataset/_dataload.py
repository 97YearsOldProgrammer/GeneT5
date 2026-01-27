import random
import math

import lib.dataset._binary     as binary
import lib.dataset._compacting as compacting


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
        """Compute lengths using batched tokenization"""

        lengths   = []
        all_texts = []

        for i in range(self._num_chunks):
            chunk = binary.read_chunk_at_index(self.binary_path, i)
            all_texts.append(chunk.get_input_text())

        if self.tokenizer is None:
            lengths = [len(text) // 4 for text in all_texts]
        else:
            for i in range(0, len(all_texts), batch_size):
                batch      = all_texts[i:i + batch_size]
                encoded    = self.tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
                batch_lens = [len(ids) for ids in encoded["input_ids"]]
                lengths.extend(batch_lens)

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

    def build_length_index(self):
        """Build length index for smart batching"""

        if self.lengths is None:
            self.lengths = self._reader.lengths
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

        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]

        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]

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
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }

        if labels:
            result["labels"] = labels

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