import json
import random
from pathlib import Path

import lib.nosing.nosing as ns


class LazyDataset:
    """Lazy-loading dataset for efficient memory usage"""
    
    def __init__(
        self,
        data_paths,
        tokenizer,
        max_input_len  = 4096,
        max_target_len = 2048,
        noiser         = None,
        hint_token     = "[HIT]",
    ):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.noiser         = noiser
        self.hint_token     = hint_token
        
        self.offsets = []
        self.lengths = []
        
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        print(f"  Indexing {len(data_paths)} file(s)...")
        
        for path in data_paths:
            path         = str(path)
            sample_count = 0
            
            with open(path, "rb") as f:
                offset = 0
                for line in f:
                    if line.strip():
                        self.offsets.append((path, offset))
                        approx_len = min(len(line) // 4, max_input_len)
                        self.lengths.append(approx_len)
                        sample_count += 1
                    offset += len(line)
            
            print(f"    {path}: {sample_count} samples")
        
        print(f"  Total indexed: {len(self.offsets)} samples")
    
    def __len__(self):
        return len(self.offsets)
    
    def _load_sample(self, idx):
        """Load raw sample from file"""
        
        path, offset = self.offsets[idx]
        
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line   = f.readline()
            sample = json.loads(line)
        
        return sample
    
    def __getitem__(self, idx):
        sample      = self._load_sample(idx)
        input_text  = sample.get("input", "")
        target_text = sample.get("target", "")
        
        input_ids  = self.tokenizer.encode(input_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         target_ids,
        }


class NoisedDataset:
    """Dataset with per-epoch noising for SFT"""
    
    def __init__(
        self,
        data_paths,
        tokenizer,
        max_input_len  = 4096,
        max_target_len = 2048,
        noising_config = None,
        hint_token     = "[HIT]",
        seed           = 42,
    ):
        config = noising_config or ns.NoisingConfig()
        noiser = ns.GFFNoiser(config)
        
        self._base_dataset = LazyDataset(
            data_paths,
            tokenizer,
            max_input_len,
            max_target_len,
            noiser,
            hint_token,
        )
        
        self.seed  = seed
        self.epoch = 0
    
    def set_epoch(self, epoch):
        """Set current epoch for noise variation"""
        
        self.epoch = epoch
        random.seed(self.seed + epoch)
    
    def __len__(self):
        return len(self._base_dataset)
    
    @property
    def lengths(self):
        return self._base_dataset.lengths
    
    def __getitem__(self, idx):
        random.seed(self.seed + self.epoch * len(self) + idx)
        return self._base_dataset[idx]


class SmartBatchSampler:
    """Batch sampler that groups similar-length sequences"""
    
    def __init__(
        self,
        lengths,
        batch_size,
        bucket_size = 100,
        drop_last   = False,
        shuffle     = True
    ):
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


class DistributedSmartBatchSampler:
    """Distributed batch sampler with smart batching"""
    
    def __init__(
        self,
        lengths,
        batch_size,
        bucket_size    = 100,
        drop_last      = False,
        shuffle        = True,
        num_replicas   = None,
        rank           = None,
        seed           = 42,
    ):
        self.lengths      = lengths
        self.batch_size   = batch_size
        self.bucket_size  = bucket_size
        self.drop_last    = drop_last
        self.shuffle      = shuffle
        self.num_replicas = num_replicas or 1
        self.rank         = rank or 0
        self.seed         = seed
        self.epoch        = 0
        
        total_batches    = (len(lengths) + batch_size - 1) // batch_size
        self.num_batches = total_batches // self.num_replicas
        if not drop_last and total_batches % self.num_replicas != 0:
            self.num_batches += 1
    
    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling"""
        
        self.epoch = epoch
    
    def __iter__(self):
        random.seed(self.seed + self.epoch)
        
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        buckets = []
        bucket  = []
        for idx in sorted_indices:
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
        
        batches = []
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        
        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]
    
    def __len__(self):
        return self.num_batches


class DistributedSamplerWrapper:
    """Wrapper for distributed sampling"""
    
    def __init__(
        self,
        dataset,
        num_replicas = None,
        rank         = None,
        shuffle      = True,
        seed         = 42,
        drop_last    = False,
    ):
        self.dataset      = dataset
        self.num_replicas = num_replicas or 1
        self.rank         = rank or 0
        self.shuffle      = shuffle
        self.seed         = seed
        self.drop_last    = drop_last
        self.epoch        = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        random.seed(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(self.rank, len(indices), self.num_replicas):
            yield indices[i]
    
    def __len__(self):
        return len(self.dataset) // self.num_replicas


class DynamicPaddingCollator:
    """Collator that pads batches dynamically to longest sequence"""
    
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


def create_distributed_dataloader(
    dataset,
    batch_size,
    lengths            = None,
    bucket_size        = 256,
    shuffle            = True,
    drop_last          = False,
    num_workers        = 0,
    pin_memory         = True,
    collate_fn         = None,
    seed               = 42,
    use_smart_batching = True,
):
    """Create a DataLoader optimized for distributed training"""
    
    if use_smart_batching and lengths is not None:
        batch_sampler = SmartBatchSampler(
            lengths,
            batch_size,
            bucket_size,
            drop_last,
            shuffle,
        )
        
        return {
            "dataset":       dataset,
            "batch_sampler": batch_sampler,
            "collate_fn":    collate_fn,
        }
    else:
        return {
            "dataset":    dataset,
            "batch_size": batch_size,
            "shuffle":    shuffle,
            "collate_fn": collate_fn,
            "drop_last":  drop_last,
        }


def set_dataloader_epoch(dataloader, epoch):
    """Set epoch on dataloader's sampler for proper shuffling"""
    
    if hasattr(dataloader, 'batch_sampler'):
        sampler = dataloader.get('batch_sampler')
    elif hasattr(dataloader, 'sampler'):
        sampler = dataloader.get('sampler')
    else:
        sampler = None
    
    if sampler and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)