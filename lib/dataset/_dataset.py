"""
Dataset utilities for GeneT5 fine-tuning.
Contains PyTorch Dataset classes and data collators.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Base Dataset
# =============================================================================

class JSONLDataset(Dataset):
    """Base dataset for loading JSONL files."""
    
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Gene Prediction Dataset
# =============================================================================

class GenePredictionDataset(JSONLDataset):
    """
    Dataset for gene prediction (teacher forcing).
    Input:  [GENE] + DNA sequence
    Target: Stripped GFF annotations
    """
    
    def __init__(self, data_path, tokenizer, max_input_len=2048, max_target_len=1024):
        super().__init__(data_path)
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
    
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


# =============================================================================
# RNA Classification Dataset
# =============================================================================

class RNAClassificationDataset(JSONLDataset):
    """
    Dataset for RNA classification.
    Input:  [CLS] + DNA sequence
    Target: RNA type label
    """
    
    def __init__(self, data_path, tokenizer, max_len=512, label_map=None):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.label_map = label_map or {}
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample["input"],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        
        label = sample["label"]
        
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(label, dtype=torch.long),
        }
    
    def get_num_classes(self):
        if self.label_map:
            return len(self.label_map)
        labels = [s["label"] for s in self.samples]
        return max(labels) + 1


# =============================================================================
# Data Collators
# =============================================================================

class Seq2SeqCollator:
    """Collator for gene prediction (seq2seq) task."""
    
    def __init__(self, tokenizer, max_input_len=2048, max_target_len=1024):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
    
    def __call__(self, batch):
        inputs  = [item["input"] for item in batch]
        targets = [item["target"] for item in batch]
        
        input_enc = self.tokenizer(
            inputs,
            max_length     = self.max_input_len,
            padding        = True,
            truncation     = True,
            return_tensors = "pt",
        )
        
        target_enc = self.tokenizer(
            targets,
            max_length     = self.max_target_len,
            padding        = True,
            truncation     = True,
            return_tensors = "pt",
        )
        
        labels = target_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids":      input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels":         labels,
        }


class ClassificationCollator:
    """Collator for RNA classification task."""
    
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
    
    def __call__(self, batch):
        inputs = [item["input"] for item in batch]
        labels = [item["label"] for item in batch]
        
        encoding = self.tokenizer(
            inputs,
            max_length     = self.max_len,
            padding        = True,
            truncation     = True,
            return_tensors = "pt",
        )
        
        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         torch.tensor(labels, dtype=torch.long),
        }


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(task, train_path, val_path, tokenizer, batch_size,
                       max_input_len=2048, max_target_len=1024, max_cls_len=512,
                       num_workers=0, label_map_path=None):
    """
    Create train and validation dataloaders for specified task.
    """
    label_map = None
    if label_map_path and Path(label_map_path).exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
    
    if task == "gene_prediction":
        train_ds = GenePredictionDataset(
            train_path, tokenizer, max_input_len, max_target_len)
        val_ds   = GenePredictionDataset(
            val_path, tokenizer, max_input_len, max_target_len) if val_path else None
    
    elif task == "rna_classification":
        train_ds = RNAClassificationDataset(
            train_path, tokenizer, max_cls_len, label_map)
        val_ds   = RNAClassificationDataset(
            val_path, tokenizer, max_cls_len, label_map) if val_path else None
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )
    
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
        )
    
    return train_loader, val_loader


# =============================================================================
# Train/Val Split Utility
# =============================================================================

def split_dataset(data_path, output_dir, val_ratio=0.1, seed=42):
    """Split JSONL dataset into train and validation sets."""
    import random
    
    with open(data_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    random.seed(seed)
    random.shuffle(samples)
    
    split_idx   = int(len(samples) * (1 - val_ratio))
    train_split = samples[:split_idx]
    val_split   = samples[split_idx:]
    
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    val_path   = output_dir / "val.jsonl"
    
    with open(train_path, "w") as f:
        for s in train_split:
            f.write(json.dumps(s) + "\n")
    
    with open(val_path, "w") as f:
        for s in val_split:
            f.write(json.dumps(s) + "\n")
    
    print(f"Split complete:")
    print(f"  Train: {len(train_split)} samples -> {train_path}")
    print(f"  Val:   {len(val_split)} samples -> {val_path}")
    
    return train_path, val_path


# =============================================================================
# Label Statistics
# =============================================================================

def print_label_distribution(data_path, label_map=None):
    """Print label distribution for classification dataset."""
    from collections import Counter
    
    with open(data_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    if not samples:
        print("No samples found.")
        return
    
    # Check if classification or seq2seq
    if "label" not in samples[0]:
        print("Not a classification dataset.")
        return
    
    labels = [s["label"] for s in samples]
    counts = Counter(labels)
    
    # Reverse label map if provided
    inv_map = {}
    if label_map:
        inv_map = {v: k for k, v in label_map.items()}
    
    print("\nLabel Distribution:")
    print("-" * 40)
    for label, count in sorted(counts.items()):
        name = inv_map.get(label, f"Class {label}")
        pct  = 100 * count / len(labels)
        print(f"  {name:15s}: {count:6d} ({pct:5.1f}%)")
    print("-" * 40)
    print(f"  {'Total':15s}: {len(labels):6d}")