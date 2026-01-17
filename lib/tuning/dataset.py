import torch
import random
from pathlib            import Path
from torch.utils.data   import Dataset, Sampler
from ._parser           import load_dataset


class MixedTaskDataset(Dataset):
    """Unified dataset for gene prediction and classification tasks."""
    
    CLASS_TOKENS = {i: f"<CLASS_{i}>" for i in range(16)}
    
    def __init__(self, data_paths, tokenizer, max_input_len=4096, max_target_len=2048):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples = []
        self.lengths = []
        
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        for path in data_paths:
            raw_samples = load_dataset(path)
            print(f"  Loaded {len(raw_samples)} samples from {path}")
            
            for sample in raw_samples:
                if "target" in sample:
                    self.samples.append({
                        "task": "seq2seq",
                        "input": sample["input"],
                        "target": sample["target"],
                    })
                elif "label" in sample:
                    class_token = self.CLASS_TOKENS.get(sample["label"], "<CLASS_0>")
                    self.samples.append({
                        "task": "classification",
                        "input": sample["input"],
                        "target": class_token,
                        "label": sample["label"],
                    })
                else:
                    raise ValueError(f"Sample must have 'target' or 'label': {sample.keys()}")
                
                inp_len = len(tokenizer.encode(sample["input"], add_special_tokens=False))
                self.lengths.append(min(inp_len, max_input_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_ids = self.tokenizer.encode(sample["input"])
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        target_ids = self.tokenizer.encode(sample["target"])
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
            "task": sample["task"],
        }


class SmartBatchSampler(Sampler):
    """Groups samples by length for efficient batching."""
    
    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    
    def __iter__(self):
        buckets = []
        bucket = []
        
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
        batches = []
        
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


class DynamicPaddingCollator:
    """Collates batches with dynamic padding."""
    
    def __init__(self, pad_token_id, label_pad=-100):
        self.pad_token_id = pad_token_id
        self.label_pad = label_pad
    
    def __call__(self, batch):
        max_input_len = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if "labels" in batch[0] else 0
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for b in batch:
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len
            
            input_ids.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            attention_mask.append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len)])
            )
            
            if "labels" in b:
                lbl_len = len(b["labels"])
                lbl_pad = max_target_len - lbl_len
                labels.append(
                    torch.cat([b["labels"], torch.full((lbl_pad,), self.label_pad)])
                )
        
        result = {
            "input_ids": torch.stack(input_ids).long(),
            "attention_mask": torch.stack(attention_mask).long(),
        }
        
        if labels:
            result["labels"] = torch.stack(labels).long()
        
        return result