
import torch
import random

from pathlib            import Path
from torch.utils.data   import Dataset
from ._parser           import load_dataset


class MixedTaskDataset(Dataset):
    
    # class token mapping for classification -> seq2seq conversion
    CLASS_TOKENS = {i: f"<CLASS_{i}>" for i in range(16)}
    
    def __init__(self, data_paths, tokenizer, max_input_len=4096, max_target_len=2048):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.samples = []
        self.lengths = []
        
        # normalize to list
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        # load all files and mix
        for path in data_paths:
            raw_samples = load_dataset(path)
            print(f"  Loaded {len(raw_samples)} samples from {path}")
            
            for sample in raw_samples:
                # detect task from sample structure
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
                        "label": sample["label"],  # keep for potential metrics
                    })
                else:
                    raise ValueError(f"Sample must have 'target' or 'label' field: {sample.keys()}")
                
                # precompute length for smart batching
                inp_len = len(tokenizer.encode(sample["input"], add_special_tokens=False))
                self.lengths.append(min(inp_len, max_input_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # encode input (no padding - collator handles it)
        input_ids = self.tokenizer.encode(sample["input"])
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        # encode target
        target_ids = self.tokenizer.encode(sample["target"])
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        # labels = copy of target_ids (pad tokens masked by collator)
        labels = target_ids.copy()
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task": sample["task"],
        }