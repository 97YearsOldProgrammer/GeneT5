import json
import random

import lib.dataset._binary as binary


#################
#####  I/O  #####
#################


class BinaryDatasetReader:
    """Lazy reader for binary training files"""

    def __init__(self, binary_path, tokenizer_encode_fn=None):

        self.binary_path      = binary_path
        self.tokenizer_encode = tokenizer_encode_fn
        self._info            = binary.get_binary_info(binary_path)
        self._num_chunks      = self._info["num_chunks"]
        self._lengths         = None

    def __len__(self):

        return self._num_chunks

    @property
    def lengths(self):
        """Get approximate lengths for smart batching"""

        if self._lengths is None:
            self._lengths = []
            for i in range(self._num_chunks):
                chunk = binary.read_chunk_at_index(self.binary_path, i)
                self._lengths.append(chunk.estimate_input_tokens())
        return self._lengths

    def get_chunk(self, idx):
        """Get raw chunk at index"""

        return binary.read_chunk_at_index(self.binary_path, idx)

    def get_sample(self, idx):
        """Get tokenized sample at index"""

        chunk = self.get_chunk(idx)

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


def build_length_index(binary_path, bp_per_token=4.5):
    """Build length index for smart batching without loading all chunks"""

    info    = binary.get_binary_info(binary_path)
    lengths = []

    for i in range(info["num_chunks"]):
        chunk = binary.read_chunk_at_index(binary_path, i)
        lengths.append(chunk.estimate_input_tokens(bp_per_token))

    return lengths


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

        self._info   = ds.get_binary_info(binary_path)
        self._length = self._info["num_chunks"]
        self.lengths = ds.build_length_index(binary_path)

    def set_epoch(self, epoch):

        self.epoch = epoch

    def __len__(self):

        return self._length

    def __getitem__(self, idx):

        random.seed(self.seed + self.epoch * len(self) + idx)

        reader = ds.BinaryDatasetReader(self.binary_path)
        sample = reader.get_sample(idx)

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
