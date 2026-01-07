
import torch
from torch.utils.data import Dataset

from ._parser import load_dataset


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