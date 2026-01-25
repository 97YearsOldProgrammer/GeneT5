"""Dataset loading utilities for training (non-PyTorch)"""

import json   as js
import random as rnd

import lib.binary as bi


#####################  Binary Dataset Reader  #####################


class BinaryDatasetReader:
    """Lazy reader for binary training files"""
    
    def __init__(self, binary_path, tokenizer_encode_fn=None):
        
        self.binary_path       = binary_path
        self.tokenizer_encode  = tokenizer_encode_fn
        self._info             = bi.get_binary_info(binary_path)
        self._num_chunks       = self._info["num_chunks"]
        self._lengths          = None
    
    def __len__(self):
        
        return self._num_chunks
    
    @property
    def lengths(self):
        """Get approximate lengths for smart batching"""
        
        if self._lengths is None:
            self._lengths = []
            for i in range(self._num_chunks):
                chunk = bi.read_chunk_at_index(self.binary_path, i)
                self._lengths.append(chunk.estimate_input_tokens())
        return self._lengths
    
    def get_chunk(self, idx):
        """Get raw chunk at index"""
        
        return bi.read_chunk_at_index(self.binary_path, idx)
    
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


#####################  Index Building  #####################


def build_length_index(binary_path, bp_per_token=4.5):
    """Build length index for smart batching without loading all chunks"""
    
    info    = bi.get_binary_info(binary_path)
    lengths = []
    
    for i in range(info["num_chunks"]):
        chunk = bi.read_chunk_at_index(binary_path, i)
        lengths.append(chunk.estimate_input_tokens(bp_per_token))
    
    return lengths


#####################  Utilities  #####################


def get_binary_stats(binary_path):
    """Get statistics about a binary dataset file"""
    
    info       = bi.get_binary_info(binary_path)
    num_chunks = info["num_chunks"]
    
    raw_count      = 0
    aug_count      = 0
    total_features = 0
    total_hints    = 0
    
    for i in range(num_chunks):
        chunk = bi.read_chunk_at_index(binary_path, i)
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
