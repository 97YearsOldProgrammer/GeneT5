import os
import random

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

        # Concatenate: input + target for causal LM
        full_ids = input_ids + target_ids

        return {
            "input_ids":  full_ids,
            "prefix_len": len(input_ids),
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
            inp_ids = input_enc["input_ids"][i]
            tgt_ids = target_enc["input_ids"][i]
            full    = inp_ids + tgt_ids
            results.append({
                "input_ids":  full,
                "prefix_len": len(inp_ids),
            })

        return results


#####################
#####  Collator #####
#####################


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

        max_len = max(len(a) for a in aligned)

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
