import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, DistributedSampler

from ._noising import GFFNoiser, NoisingConfig


class LazyDataset(Dataset):
    
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
        
        self.offsets = []  # (file_path, byte_offset)
        self.lengths = []  # Approximate lengths for smart batching
        
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        # Only print from main process
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        if is_main:
            print(f"  Indexing {len(data_paths)} file(s)...")
        
        for path in data_paths:
            path         = str(path)
            sample_count = 0
            
            with open(path, "rb") as f:
                offset = 0
                for line in f:
                    if line.strip():
                        self.offsets.append((path, offset))
                        
                        # Estimate length from line size
                        approx_len = min(len(line) // 4, max_input_len)
                        self.lengths.append(approx_len)
                        sample_count += 1
                    
                    offset += len(line)
            
            if is_main:
                print(f"    {path}: {sample_count} samples")
        
        if is_main:
            print(f"  Total indexed: {len(self.offsets)} samples")
            if noiser:
                print(f"  Noising: ENABLED")
    
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
    
    def _apply_noising(self, sample):
        """Give Hints to Input"""

        if self.noiser is None:
            return sample["input"]
        
        # Extract original input (contains DNA sequence)
        original_input = sample["input"]
        
        # Parse features from target to create hints
        features = self._parse_features_from_target(sample.get("target", ""))
        
        if not features:
            return original_input
        
        # Extract DNA sequence from input (after task token)
        parts      = original_input.split(" ", 1)
        task_token = parts[0] if parts else "[ATT]"
        dna_seq    = parts[1] if len(parts) > 1 else ""
        
        # Generate noised hints
        hints, scenario, _ = self.noiser.noise_features(features, dna_seq)
        
        # Format hints
        hint_str = self.noiser.format_hints_for_input(hints, self.hint_token)
        
        # Combine: task_token DNA_sequence hint_section
        if hint_str:
            return f"{task_token} {dna_seq}\n{hint_str}"
        else:
            return original_input
    
    def _parse_features_from_target(self, target):

        features = []
        
        for line in target.strip().split("\n"):
            line = line.strip()
            
            # Skip BOS/EOS tokens
            if line.startswith("<") or not line:
                continue
            
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            
            try:
                feat = {
                    "type":   parts[0],
                    "start":  int(parts[1]),
                    "end":    int(parts[2]),
                    "strand": parts[3],
                    "phase":  parts[4] if len(parts) > 4 else ".",
                }
                features.append(feat)
            except (ValueError, IndexError):
                continue
        
        return features
    
    def __getitem__(self, idx):

        sample = self._load_sample(idx)
        
        # Apply noising to input
        input_text  = self._apply_noising(sample)
        target_text = sample.get("target", "")
        
        # Tokenize
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels":         torch.tensor(target_ids, dtype=torch.long),
        }


class NoisedDataset(Dataset):
    """DS For SFT"""
    
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
        """
        Initialize noised dataset.
        
        Args:
            data_paths: Path(s) to JSONL file(s)
            tokenizer: Tokenizer instance
            max_input_len: Maximum input length
            max_target_len: Maximum target length
            noising_config: Configuration for noise parameters
            hint_token: Token for hint section
            seed: Random seed (changes each epoch for variety)
        """
        config = noising_config or NoisingConfig()
        noiser = GFFNoiser(config)
        
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
        """
        Set current epoch for noise variation.
        
        Call this at the start of each epoch to ensure different noise patterns.
        """
        self.epoch = epoch
        random.seed(self.seed + epoch)
    
    def __len__(self):
        return len(self._base_dataset)
    
    @property
    def lengths(self):
        return self._base_dataset.lengths
    
    def __getitem__(self, idx):
        # Seed based on epoch and index for reproducibility
        random.seed(self.seed + self.epoch * len(self) + idx)
        return self._base_dataset[idx]


################################
#####  SMART BATCH SAMPLER #####
################################


class SmartBatchSampler(Sampler):
    """
    Batch sampler that groups similar-length sequences.
    Reduces padding waste by batching sequences of similar lengths together.
    """
    
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
        # Create buckets of similar-length sequences
        buckets = []
        bucket  = []
        
        for idx in self.sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []
        
        if bucket:
            buckets.append(bucket)
        
        # Shuffle within buckets and between buckets
        if self.shuffle:
            random.shuffle(buckets)
            for b in buckets:
                random.shuffle(b)
        
        # Flatten and create batches
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


################################
#####  DISTRIBUTED SAMPLER #####
################################


class DistributedSmartBatchSampler(Sampler):
    """
    Distributed batch sampler with smart batching for DGX Spark multi-node training.
    
    Combines length-based bucketing with distributed sampling to:
    - Reduce padding waste across all processes
    - Ensure each process gets a unique subset of data
    - Support deterministic shuffling with epoch-based seeds
    """
    
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
        if num_replicas is None:
            if not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
        
        self.lengths      = lengths
        self.batch_size   = batch_size
        self.bucket_size  = bucket_size
        self.drop_last    = drop_last
        self.shuffle      = shuffle
        self.num_replicas = num_replicas
        self.rank         = rank
        self.seed         = seed
        self.epoch        = 0
        
        # Calculate batches per replica
        total_batches     = (len(lengths) + batch_size - 1) // batch_size
        self.num_batches  = total_batches // num_replicas
        if not drop_last and total_batches % num_replicas != 0:
            self.num_batches += 1
    
    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
    
    def __iter__(self):
        # Create deterministic random state
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Sort indices by length
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        # Create buckets
        buckets = []
        bucket  = []
        for idx in sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []
        if bucket:
            buckets.append(bucket)
        
        # Shuffle buckets and contents if needed
        if self.shuffle:
            # Shuffle bucket order
            bucket_order = torch.randperm(len(buckets), generator=g).tolist()
            buckets      = [buckets[i] for i in bucket_order]
            
            # Shuffle within each bucket
            for b in buckets:
                indices  = torch.randperm(len(b), generator=g).tolist()
                b[:]     = [b[i] for i in indices]
        
        # Flatten to all indices
        all_indices = [idx for bucket in buckets for idx in bucket]
        
        # Create batches
        batches = []
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        # Shuffle batches
        if self.shuffle:
            batch_order = torch.randperm(len(batches), generator=g).tolist()
            batches     = [batches[i] for i in batch_order]
        
        # Distribute batches across replicas
        # Each replica gets batches[rank::num_replicas]
        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]
    
    def __len__(self):
        return self.num_batches


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper around DistributedSampler that works with any dataset.
    
    Provides proper distribution of samples across DGX Spark nodes.
    """
    
    def __init__(
        self,
        dataset,
        num_replicas = None,
        rank         = None,
        shuffle      = True,
        seed         = 42,
        drop_last    = False,
    ):
        super().__init__(
            dataset,
            num_replicas = num_replicas,
            rank         = rank,
            shuffle      = shuffle,
            seed         = seed,
            drop_last    = drop_last,
        )


################################
#####  DYNAMIC COLLATOR    #####
################################


class DynamicPaddingCollator:
    """
    Collator that pads batches dynamically to the longest sequence.
    More efficient than fixed-length padding.
    """
    
    def __init__(self, pad_token_id, label_pad=-100):

        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
    
    def __call__(self, batch):

        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if batch[0]["labels"].numel() > 0 else 0
        
        input_ids      = []
        attention_mask = []
        labels         = []
        
        for b in batch:
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len
            
            input_ids.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            attention_mask.append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len)])
            )
            
            if max_target_len > 0:
                lbl_len = len(b["labels"])
                lbl_pad = max_target_len - lbl_len
                labels.append(
                    torch.cat([b["labels"], torch.full((lbl_pad,), self.label_pad)])
                )
        
        result = {
            "input_ids":      torch.stack(input_ids).long(),
            "attention_mask": torch.stack(attention_mask).long(),
        }
        
        if labels:
            result["labels"] = torch.stack(labels).long()
        
        return result


################################
#####  DATALOADER FACTORY  #####
################################


def create_distributed_dataloader(
    dataset,
    batch_size,
    lengths        = None,
    bucket_size    = 256,
    shuffle        = True,
    drop_last      = False,
    num_workers    = 0,
    pin_memory     = True,
    collate_fn     = None,
    seed           = 42,
    use_smart_batching = True,
):
    """
    Create a DataLoader optimized for distributed training on DGX Spark.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Per-GPU batch size
        lengths: Sequence lengths for smart batching (optional)
        bucket_size: Size of length-based buckets
        shuffle: Whether to shuffle data
        drop_last: Drop incomplete last batch
        num_workers: Number of data loading workers
        pin_memory: Use pinned memory for faster GPU transfer
        collate_fn: Custom collation function
        seed: Random seed for reproducibility
        use_smart_batching: Use length-based batching
    
    Returns:
        DataLoader configured for distributed training
    """
    from torch.utils.data import DataLoader
    
    is_distributed = dist.is_initialized()
    
    if is_distributed:
        if use_smart_batching and lengths is not None:
            # Use distributed smart batch sampler
            batch_sampler = DistributedSmartBatchSampler(
                lengths      = lengths,
                batch_size   = batch_size,
                bucket_size  = bucket_size,
                drop_last    = drop_last,
                shuffle      = shuffle,
                seed         = seed,
            )
            
            return DataLoader(
                dataset,
                batch_sampler      = batch_sampler,
                collate_fn         = collate_fn,
                num_workers        = num_workers,
                pin_memory         = pin_memory,
                persistent_workers = (num_workers > 0),
            )
        else:
            # Use standard distributed sampler
            sampler = DistributedSamplerWrapper(
                dataset,
                shuffle   = shuffle,
                seed      = seed,
                drop_last = drop_last,
            )
            
            return DataLoader(
                dataset,
                batch_size         = batch_size,
                sampler            = sampler,
                collate_fn         = collate_fn,
                num_workers        = num_workers,
                pin_memory         = pin_memory,
                drop_last          = drop_last,
                persistent_workers = (num_workers > 0),
            )
    else:
        # Non-distributed training
        if use_smart_batching and lengths is not None:
            batch_sampler = SmartBatchSampler(
                lengths,
                batch_size,
                bucket_size,
                drop_last,
                shuffle,
            )
            
            return DataLoader(
                dataset,
                batch_sampler      = batch_sampler,
                collate_fn         = collate_fn,
                num_workers        = num_workers,
                pin_memory         = pin_memory,
                persistent_workers = (num_workers > 0),
            )
        else:
            return DataLoader(
                dataset,
                batch_size         = batch_size,
                shuffle            = shuffle,
                collate_fn         = collate_fn,
                num_workers        = num_workers,
                pin_memory         = pin_memory,
                drop_last          = drop_last,
                persistent_workers = (num_workers > 0),
            )


def set_dataloader_epoch(dataloader, epoch):
    """
    Set epoch on dataloader's sampler for proper shuffling in distributed training.
    
    Should be called at the start of each epoch.
    """
    if hasattr(dataloader, 'batch_sampler'):
        sampler = dataloader.batch_sampler
    elif hasattr(dataloader, 'sampler'):
        sampler = dataloader.sampler
    else:
        return
    
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)