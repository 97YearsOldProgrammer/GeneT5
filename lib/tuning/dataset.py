import struct
import numpy as np
import torch
from pathlib          import Path
from torch.utils.data import Dataset


class BinaryDataset(Dataset):

    def __init__(self, data_paths, pad_token_id=0):
        """
        Args:
            data_paths: Path(s) to .bin files or directories containing them
            pad_token_id: Token ID for padding (for length estimation)
        """
        self.pad_token_id = pad_token_id
        self.files        = []  # [(mmap, index_data), ...]
        self.sample_map   = []  # [(file_idx, sample_idx), ...]
        self.lengths      = []  # Approximate lengths for smart batching
        
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        
        print(f"  Loading {len(data_paths)} binary file(s)...")
        
        for path in data_paths:
            path = Path(path)
            
            if path.is_dir():
                bin_files = list(path.glob('**/*.bin'))
            elif path.suffix == '.bin':
                bin_files = [path]
            elif path.suffix == '.jsonl':
                # Auto-convert: look for corresponding .bin
                bin_path = path.with_suffix('.bin')
                if bin_path.exists():
                    bin_files = [bin_path]
                else:
                    print(f"    Warning: No .bin file for {path}, skipping")
                    continue
            else:
                bin_files = list(Path('.').glob(str(path)))
                bin_files = [f for f in bin_files if f.suffix == '.bin']
            
            for bin_file in bin_files:
                self._load_binary_file(bin_file)
        
        print(f"  Total loaded: {len(self.sample_map)} samples from {len(self.files)} file(s)")
    
    def _load_binary_file(self, bin_path):
        """Load a single binary file and its index."""
        bin_path = Path(bin_path)
        idx_path = bin_path.with_suffix('.idx')
        
        if not bin_path.exists():
            print(f"    Warning: Binary file not found: {bin_path}")
            return
        
        if not idx_path.exists():
            print(f"    Warning: Index file not found: {idx_path}")
            return
        
        # Memory-map the binary data
        mmap_data = np.memmap(bin_path, dtype=np.int32, mode='r')
        
        # Load index
        index_data = []
        with open(idx_path, 'rb') as f:
            # Read header
            num_samples = struct.unpack('<Q', f.read(8))[0]
            
            # Read index entries
            for _ in range(num_samples):
                offset, input_len, target_len = struct.unpack('<QII', f.read(16))
                index_data.append((offset, input_len, target_len))
        
        file_idx = len(self.files)
        self.files.append((mmap_data, index_data))
        
        # Add samples to global map
        for sample_idx, (_, input_len, target_len) in enumerate(index_data):
            self.sample_map.append((file_idx, sample_idx))
            self.lengths.append(input_len + target_len)
        
        print(f"    {bin_path.name}: {len(index_data)} samples")
    
    def __len__(self):
        return len(self.sample_map)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_map[idx]
        mmap_data, index_data = self.files[file_idx]
        
        offset, input_len, target_len = index_data[sample_idx]
        
        # Read tokens from memory-mapped file
        input_ids  = torch.from_numpy(
            mmap_data[offset:offset + input_len].astype(np.int64).copy()
        )
        target_ids = torch.from_numpy(
            mmap_data[offset + input_len:offset + input_len + target_len].astype(np.int64).copy()
        )
        
        return {
            "input_ids":      input_ids,
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels":         target_ids,
        }
    
    def get_stats(self):
        """Get dataset statistics."""
        total_tokens = sum(self.lengths)
        avg_length   = total_tokens / len(self) if len(self) > 0 else 0
        max_length   = max(self.lengths) if self.lengths else 0
        min_length   = min(self.lengths) if self.lengths else 0
        
        return {
            "num_samples":  len(self),
            "num_files":    len(self.files),
            "total_tokens": total_tokens,
            "avg_length":   avg_length,
            "max_length":   max_length,
            "min_length":   min_length,
        }