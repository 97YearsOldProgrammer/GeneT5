import json
import random

import torch
from torch.utils.data import Dataset

import lib.dataset._binary as binary


#################
#####  I/O  #####
#################


class BinaryDatasetReader:
    """Lazy reader for binary training files"""

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
        """Get actual token lengths (requires tokenizer)"""

        if self._lengths is None:
            self._lengths = []
            for i in range(self._num_chunks):
                chunk  = binary.read_chunk_at_index(self.binary_path, i)
                sample = self._format_sample(chunk)

                if self.tokenizer:
                    input_ids = self.tokenizer.encode(sample["input_text"], add_special_tokens=False)
                    self._lengths.append(len(input_ids))
                else:
                    # Fallback: rough estimate
                    self._lengths.append(len(sample["input_text"]) // 4)

        return self._lengths

    def _format_sample(self, chunk):
        """Format chunk as input/target text"""

        input_text = chunk.sequence
        if chunk.has_hints and chunk.hints:
            input_text += "\n[HIT]"
            for h in sorted(chunk.hints, key=lambda x: x.get("start", 0)):
                htype   = h.get("type", "exon").lower()
                hstart  = h.get("start", 0)
                hend    = h.get("end", 0)
                hstrand = h.get("strand", "+")
                input_text += f"\n{htype}\t{hstart}\t{hend}\t{hstrand}"

        target_text = "<BOS>"
        for f in sorted(chunk.features, key=lambda x: x.get("start", 0)):
            ftype   = f.get("type", "exon").lower()
            fstart  = f.get("start", 0)
            fend    = f.get("end", 0)
            fstrand = f.get("strand", "+")
            fphase  = f.get("phase", ".")
            target_text += f"\n{ftype}\t{fstart}\t{fend}\t{fstrand}\t{fphase}"
        target_text += "\n<EOS>"

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


def build_length_index(binary_path, tokenizer=None):
    """Build length index for smart batching"""

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


class BinaryTrainDataset(Dataset):
    """PyTorch dataset wrapper for binary training files"""

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
        self.lengths = None  # Lazy load

    def build_length_index(self):
        """Build length index for smart batching (call explicitly if needed)"""

        if self.lengths is None:
            self.lengths = self._reader.lengths
        return self.lengths

    def set_epoch(self, epoch):

        self.epoch = epoch

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
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

        if labels:
            result["labels"] = torch.tensor(labels)

        return result


class CompactingCollator:
    """
    Collator for compacted samples with block-diagonal decoder attention.
    
    When multiple samples are packed into one sequence (separated by [SEP]),
    the decoder should NOT attend across sample boundaries. This collator
    creates a block-diagonal attention mask for the decoder.
    
    Example:
        Sample 1: [BOS] exon 100 200 + . [EOS]
        Sample 2: [BOS] cds  50  150 + 0 [EOS]
        
        Combined: [BOS] exon... [EOS] [SEP] [BOS] cds... [EOS]
        
        Decoder attention mask (1 = attend, 0 = ignore):
        
                  tok1 tok2 tok3 [SEP] tok4 tok5 tok6
        tok1       1    1    1    0     0    0    0
        tok2       1    1    1    0     0    0    0
        tok3       1    1    1    0     0    0    0
        [SEP]      0    0    0    1     0    0    0
        tok4       0    0    0    0     1    1    1
        tok5       0    0    0    0     1    1    1
        tok6       0    0    0    0     1    1    1
    """

    def __init__(self, tokenizer, pad_token_id=None, label_pad=-100, sep_token="[SEP]"):

        self.tokenizer    = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.label_pad    = label_pad
        self.sep_token    = sep_token
        self.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    def __call__(self, batch):
        """
        Collate batch with proper attention masks.
        
        Each item in batch should have:
            - input_ids: list of token ids
            - labels: list of label ids
            - compact_group: group index (samples in same group are packed)
        """

        # Group by compact_group
        groups = {}
        for item in batch:
            group_id = item.get("compact_group", id(item))  # fallback to unique id
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(item)

        # Process each group
        all_input_ids       = []
        all_attention_masks = []
        all_labels          = []
        all_decoder_masks   = []

        for group_id, items in groups.items():
            if len(items) == 1:
                # Single sample - standard processing
                item = items[0]
                all_input_ids.append(item["input_ids"])
                all_attention_masks.append([1] * len(item["input_ids"]))
                all_labels.append(item["labels"])
                all_decoder_masks.append(None)  # Standard causal mask
            else:
                # Multiple samples - pack with separator
                packed_input   = []
                packed_labels  = []
                segment_starts = []

                for i, item in enumerate(items):
                    if i > 0:
                        packed_input.append(self.sep_token_id)
                        packed_labels.append(self.label_pad)  # Don't predict separator

                    segment_starts.append(len(packed_input))
                    packed_input.extend(item["input_ids"])
                    packed_labels.extend(item["labels"])

                segment_starts.append(len(packed_input))  # End marker

                # Build block-diagonal decoder mask
                seq_len      = len(packed_labels)
                decoder_mask = [[0] * seq_len for _ in range(seq_len)]

                for seg_idx in range(len(segment_starts) - 1):
                    start = segment_starts[seg_idx]
                    end   = segment_starts[seg_idx + 1]

                    for i in range(start, end):
                        for j in range(start, end):
                            if j <= i:  # Causal within segment
                                decoder_mask[i][j] = 1

                all_input_ids.append(packed_input)
                all_attention_masks.append([1] * len(packed_input))
                all_labels.append(packed_labels)
                all_decoder_masks.append(decoder_mask)

        # Pad to max length
        max_input_len  = max(len(x) for x in all_input_ids)
        max_label_len  = max(len(x) for x in all_labels)

        padded_inputs  = []
        padded_masks   = []
        padded_labels  = []
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
                # Expand decoder mask with padding
                dec_mask = all_decoder_masks[i]
                for row in dec_mask:
                    row.extend([0] * lbl_pad)
                for _ in range(lbl_pad):
                    dec_mask.append([0] * max_label_len)
                padded_dec_masks.append(dec_mask)
            else:
                padded_dec_masks.append(None)

        result = {
            "input_ids":      torch.tensor(padded_inputs),
            "attention_mask": torch.tensor(padded_masks),
            "labels":         torch.tensor(padded_labels),
        }

        # Add decoder attention mask if any samples are compacted
        if any(m is not None for m in padded_dec_masks):
            # Convert None masks to standard causal
            final_dec_masks = []
            for i, mask in enumerate(padded_dec_masks):
                if mask is None:
                    # Standard causal mask
                    seq_len = max_label_len
                    causal  = [[1 if j <= i else 0 for j in range(seq_len)] for i in range(seq_len)]
                    final_dec_masks.append(causal)
                else:
                    final_dec_masks.append(mask)

            result["decoder_attention_mask"] = torch.tensor(final_dec_masks)

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