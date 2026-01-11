
import torch
import random
from torch.utils.data import Dataset, Sampler

from ._parser import load_dataset


##########################
#####  Smart Batch   #####
##########################


class SmartBatchSampler(Sampler):
    """
    Groups samples by length for efficient batching.
    Minimizes padding by batching similar-length sequences together.
    """
    
    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False, shuffle=True):
        self.lengths     = lengths
        self.batch_size  = batch_size
        self.bucket_size = bucket_size
        self.drop_last   = drop_last
        self.shuffle     = shuffle
        
        # sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    
    def __iter__(self):
        # create buckets of similar lengths
        buckets = []
        bucket  = []
        
        for idx in self.sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []
        
        if bucket:
            buckets.append(bucket)
        
        # shuffle within buckets
        if self.shuffle:
            random.shuffle(buckets)
            for b in buckets:
                random.shuffle(b)
        
        # flatten and create batches
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


class DynamicPaddingCollator:
    """
    Collates batches with dynamic padding to max length in batch.
    """
    
    def __init__(self, pad_token_id, label_pad=-100):
        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
    
    def __call__(self, batch):
        # find max lengths in batch
        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if "labels" in batch[0] else 0
        
        input_ids      = []
        attention_mask = []
        labels         = []
        
        for b in batch:
            # pad input
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len
            
            input_ids.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            attention_mask.append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len)])
            )
            
            # pad labels if present
            if "labels" in b:
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


##########################
#####  Datasets      #####
##########################


class GenePredictionDataset(Dataset):
    """Dataset for gene prediction (seq2seq) task."""
    
    def __init__(self, data_path, tokenizer, max_input_len=2048, max_target_len=1024):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.samples        = load_dataset(data_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_enc = self.tokenizer(
            sample["input"],
            max_length     = self.max_input_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        
        target_enc = self.tokenizer(
            sample["target"],
            max_length     = self.max_target_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids":      input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


class RNAClassificationDataset(Dataset):
    """Dataset for RNA classification task."""
    
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = load_dataset(data_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["input"],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(sample["label"], dtype=torch.long),
        }


class FineTuneDataset(Dataset):
    """
    Dataset for fine-tuning with precomputed lengths for smart batching.
    """
    
    def __init__(self, data_path, tokenizer, max_input_len=4096, max_target_len=2048, task="gene_prediction"):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.task           = task
        self.samples        = load_dataset(data_path)
        
        # precompute lengths for smart batching
        self.lengths = []
        for sample in self.samples:
            inp_len = len(tokenizer.encode(sample["input"], add_special_tokens=False))
            self.lengths.append(min(inp_len, max_input_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # tokenize input
        input_enc = self.tokenizer(
            sample["input"],
            max_length     = self.max_input_len,
            truncation     = True,
            return_tensors = "pt",
            padding        = False,
        )
        
        result = {
            "input_ids":      input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
        }
        
        # handle labels based on task
        if self.task == "gene_prediction":
            target_enc = self.tokenizer(
                sample["target"],
                max_length     = self.max_target_len,
                truncation     = True,
                return_tensors = "pt",
                padding        = False,
            )
            labels = target_enc["input_ids"].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels
        else:
            result["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        
        return result

# In lib/tuning/dataset.py - UPDATED

class MixedTaskDataset(Dataset):
    """
    Unified dataset for both gene_prediction (seq2seq) and classification.
    Classification is converted to seq2seq: input -> "<CLASS_N>"
    """
    
    # Class token mapping
    CLASS_TOKENS = {i: f"<CLASS_{i}>" for i in range(16)}  # adjust range as needed
    
    def __init__(self, data_paths, tokenizer, max_input_len=4096, max_target_len=2048):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples = []
        self.lengths = []
        
        # Handle single path or dict
        if isinstance(data_paths, (str, Path)):
            data_paths = {"auto": data_paths}
        
        for task, path in data_paths.items():
            raw_samples = load_dataset(path)
            
            for sample in raw_samples:
                # Detect task from sample structure
                if "target" in sample:
                    # seq2seq (gene prediction)
                    self.samples.append({
                        "task": "seq2seq",
                        "input": sample["input"],
                        "target": sample["target"],
                    })
                elif "label" in sample:
                    # classification -> convert to seq2seq
                    class_token = self.CLASS_TOKENS.get(sample["label"], "<CLASS_0>")
                    self.samples.append({
                        "task": "classification",
                        "input": sample["input"],
                        "target": class_token,
                        "label": sample["label"],  # keep for metrics
                    })
                
                # Precompute length for smart batching
                inp_len = len(tokenizer.encode(sample["input"], add_special_tokens=False))
                self.lengths.append(min(inp_len, max_input_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Encode input (no padding - collator handles it)
        input_ids = self.tokenizer.encode(sample["input"])
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        # Encode target
        target_ids = self.tokenizer.encode(sample["target"])
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        # Labels: copy of target_ids (pad tokens will be masked in collator)
        labels = target_ids.copy()
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task": sample["task"],
        }