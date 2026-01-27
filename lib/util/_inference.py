from __future__ import annotations

import os
import json
import torch
import torch.nn.functional as F

from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field

from lib.model import GeneT5
from lib.gff_converter import (
    ModelOutputParser,
    GFFConverter,
    GFFFeature,
    ParsedFeature,
    parse_model_output,
    write_gff3,
    model_output_to_gff3,
)


##########################
#####  Device Utils  #####
##########################


def auto_detect_device() -> torch.device:
    """
    Auto-detect the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device for inference
    """
    if torch.cuda.is_available():
        # Get device with most free memory if multiple GPUs
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free = torch.cuda.mem_get_info()[0]
                free_memory.append((free, i))
            best_device = max(free_memory, key=lambda x: x[0])[1]
            return torch.device(f"cuda:{best_device}")
        return torch.device("cuda:0")
    
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon MPS
        return torch.device("mps")
    
    return torch.device("cpu")


def get_device_info(device: torch.device) -> Dict[str, str]:
    """Get device information for logging"""
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_capability"]  = str(torch.cuda.get_device_capability(device))
        free, total = torch.cuda.mem_get_info(device)
        info["memory_free_gb"]  = f"{free / 1e9:.2f}"
        info["memory_total_gb"] = f"{total / 1e9:.2f}"
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon MPS"
    else:
        info["device_name"] = "CPU"
    
    return info


def select_dtype(device: torch.device, prefer_bf16: bool = True) -> torch.dtype:
    """
    Select optimal dtype for device.
    
    Args:
        device:     Target device
        prefer_bf16: Prefer BF16 if available
        
    Returns:
        Optimal dtype for inference
    """
    if device.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        # MPS has limited dtype support
        return torch.float32
    return torch.float32


#########################
#####  Generation   #####
#########################


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length:   int   = 512
    temperature:  float = 1.0
    top_k:        int   = 50
    top_p:        float = 0.9
    do_sample:    bool  = True
    num_beams:    int   = 1
    bos_token_id: int   = 1
    eos_token_id: int   = 2
    pad_token_id: int   = 0


@dataclass
class InferenceResult:
    """Result from a single inference"""
    input_sequence:  str
    raw_output:      str
    decoded_output:  str
    parsed_features: List[ParsedFeature]
    gff_features:    List[GFFFeature]
    gff_path:        Optional[Path] = None
    metadata:        Dict = field(default_factory=dict)


class GeneT5Inference:
    """
    Inference wrapper for GeneT5 model with automatic GFF conversion.
    
    Handles:
    - Device auto-detection
    - Model loading with optimal dtype
    - Generation with configurable parameters
    - Output parsing and GFF3 conversion
    """
    
    def __init__(
        self,
        model:      GeneT5,
        tokenizer:  any,
        device:     Optional[torch.device] = None,
        dtype:      Optional[torch.dtype] = None,
    ):
        """
        Args:
            model:     GeneT5 model instance
            tokenizer: Tokenizer for encoding/decoding
            device:    Target device (auto-detected if None)
            dtype:     Model dtype (auto-selected if None)
        """
        self.device = device or auto_detect_device()
        self.dtype  = dtype or select_dtype(self.device)
        
        self.model     = model.to(device=self.device, dtype=self.dtype)
        self.tokenizer = tokenizer
        
        self.parser    = ModelOutputParser(strict=False)
        self.converter = GFFConverter()
        
        # Print device info
        info = get_device_info(self.device)
        print(f"GeneT5Inference initialized on {info['device']}")
        if "cuda_device_name" in info:
            print(f"  GPU: {info['cuda_device_name']}")
            print(f"  Memory: {info['memory_free_gb']}GB free / {info['memory_total_gb']}GB total")
        print(f"  Dtype: {self.dtype}")
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path:  Optional[str | Path] = None,
        device:          Optional[torch.device] = None,
        dtype:           Optional[torch.dtype] = None,
    ) -> "GeneT5Inference":
        """
        Load model from checkpoint directory.
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            tokenizer_path:  Path to tokenizer (defaults to checkpoint_path)
            device:          Target device (auto-detected if None)
            dtype:           Model dtype (auto-selected if None)
            
        Returns:
            Initialized GeneT5Inference instance
        """
        checkpoint_path = Path(checkpoint_path)
        tokenizer_path  = Path(tokenizer_path) if tokenizer_path else checkpoint_path
        
        # Auto-detect device first
        target_device = device or auto_detect_device()
        target_dtype  = dtype or select_dtype(target_device)
        
        # Load model
        model = GeneT5.from_pretrained(
            checkpoint_path,
            device = target_device,
            dtype  = target_dtype,
        )
        
        # Load tokenizer
        tokenizer = cls._load_tokenizer(tokenizer_path)
        
        return cls(
            model     = model,
            tokenizer = tokenizer,
            device    = target_device,
            dtype     = target_dtype,
        )
    
    @staticmethod
    def _load_tokenizer(tokenizer_path: Path):
        """Load tokenizer from path - supports multiple tokenizer types"""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            pass
        
        # Fallback: simple vocab-based tokenizer
        vocab_path = tokenizer_path / "vocab.json"
        if vocab_path.exists():
            return SimpleTokenizer.from_vocab(vocab_path)
        
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}")
    
    def encode(
        self,
        sequences:  List[str],
        max_length: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input sequences.
        
        Args:
            sequences:  List of DNA sequences
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            # HuggingFace tokenizer
            encoded = self.tokenizer.batch_encode_plus(
                sequences,
                padding        = "max_length",
                truncation     = True,
                max_length     = max_length,
                return_tensors = "pt",
            )
            return {
                "input_ids":      encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
            }
        else:
            # Simple tokenizer
            batch = self.tokenizer.encode_batch(sequences, max_length=max_length)
            return {
                "input_ids":      batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Tensor of token IDs [batch, seq_len]
            
        Returns:
            List of decoded strings
        """
        if hasattr(self.tokenizer, 'batch_decode'):
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        else:
            return self.tokenizer.decode_batch(token_ids)
    
    @torch.no_grad()
    def generate(
        self,
        encoder_input_ids:      torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        config:                 Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """
        Generate output sequences.
        
        Args:
            encoder_input_ids:      Input token IDs [batch, seq_len]
            encoder_attention_mask: Attention mask [batch, seq_len]
            config:                 Generation configuration
            
        Returns:
            Generated token IDs [batch, gen_len]
        """
        config = config or GenerationConfig()
        self.model.eval()
        
        return self.model.generate(
            encoder_input_ids      = encoder_input_ids,
            encoder_attention_mask = encoder_attention_mask,
            max_length             = config.max_length,
            temperature            = config.temperature,
            top_k                  = config.top_k,
            top_p                  = config.top_p,
            bos_token_id           = config.bos_token_id,
            eos_token_id           = config.eos_token_id,
            pad_token_id           = config.pad_token_id,
        )
    
    def predict(
        self,
        sequences:       List[str],
        seqids:          Optional[List[str]] = None,
        output_dir:      Optional[str | Path] = None,
        source:          str = "GeneT5",
        offsets:         Optional[List[int]] = None,
        gen_config:      Optional[GenerationConfig] = None,
        max_input_length: int = 2048,
        batch_size:      int = 1,
    ) -> List[InferenceResult]:
        """
        Run inference on sequences and optionally save GFF3 output.
        
        Args:
            sequences:        List of input DNA sequences
            seqids:           Sequence IDs for GFF output
            output_dir:       Directory for GFF3 files (None = don't save)
            source:           Source field for GFF3
            offsets:          Position offsets for each sequence
            gen_config:       Generation configuration
            max_input_length: Maximum input sequence length
            batch_size:       Batch size for inference
            
        Returns:
            List of InferenceResult objects
        """
        if seqids is None:
            seqids = [f"seq_{i}" for i in range(len(sequences))]
        if offsets is None:
            offsets = [0] * len(sequences)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        gen_config = gen_config or GenerationConfig()
        results = []
        
        # Process in batches
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids  = seqids[batch_start:batch_end]
            batch_offs = offsets[batch_start:batch_end]
            
            # Encode
            encoded = self.encode(batch_seqs, max_length=max_input_length)
            
            # Generate
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                generated = self.generate(
                    encoder_input_ids      = encoded["input_ids"],
                    encoder_attention_mask = encoded["attention_mask"],
                    config                 = gen_config,
                )
            
            # Decode
            decoded_outputs = self.decode(generated)
            
            # Process each result
            for i, (seq, seqid, offset, raw_output) in enumerate(
                zip(batch_seqs, batch_ids, batch_offs, decoded_outputs)
            ):
                # Clean up output
                decoded = self._clean_output(raw_output)
                
                # Parse features
                parsed_seqs = self.parser.parse_sequence(decoded)
                parsed_features = parsed_seqs[0] if parsed_seqs else []
                
                # Convert to GFF
                self.converter.seqid  = seqid
                self.converter.source = source
                self.converter.offset = offset
                gff_features = self.converter.convert_sequence(parsed_features)
                
                # Save if output_dir specified
                gff_path = None
                if output_dir:
                    gff_path = output_dir / f"{seqid}.gff3"
                    write_gff3(gff_features, gff_path)
                
                results.append(InferenceResult(
                    input_sequence  = seq,
                    raw_output      = raw_output,
                    decoded_output  = decoded,
                    parsed_features = parsed_features,
                    gff_features    = gff_features,
                    gff_path        = gff_path,
                    metadata        = {
                        "seqid":  seqid,
                        "offset": offset,
                        "source": source,
                    },
                ))
        
        return results
    
    def _clean_output(self, raw_output: str) -> str:
        """Clean raw model output for parsing"""
        # Remove special tokens that might be decoded differently
        output = raw_output.strip()
        
        # Normalize BOS/EOS tokens
        output = output.replace("<bos>", "<BOS>")
        output = output.replace("<eos>", "<EOS>")
        output = output.replace("[BOS]", "<BOS>")
        output = output.replace("[EOS]", "<EOS>")
        
        return output
    
    def predict_single(
        self,
        sequence:    str,
        seqid:       str = "seq",
        output_path: Optional[str | Path] = None,
        source:      str = "GeneT5",
        offset:      int = 0,
        gen_config:  Optional[GenerationConfig] = None,
    ) -> InferenceResult:
        """
        Convenience method for single sequence prediction.
        
        Args:
            sequence:    Input DNA sequence
            seqid:       Sequence ID
            output_path: Output GFF3 file path
            source:      Source field for GFF3
            offset:      Position offset
            gen_config:  Generation configuration
            
        Returns:
            InferenceResult
        """
        output_dir = None
        if output_path:
            output_path = Path(output_path)
            output_dir = output_path.parent
        
        results = self.predict(
            sequences  = [sequence],
            seqids     = [seqid],
            output_dir = output_dir,
            source     = source,
            offsets    = [offset],
            gen_config = gen_config,
        )
        
        result = results[0]
        
        # Rename file if specific path given
        if output_path and result.gff_path and result.gff_path != output_path:
            result.gff_path.rename(output_path)
            result.gff_path = output_path
        
        return result


############################
#####  Simple Tokenizer ####
############################


class SimpleTokenizer:
    """
    Simple tokenizer for DNA/protein sequences.
    
    Fallback when HuggingFace tokenizer is not available.
    """
    
    def __init__(
        self,
        vocab:        Dict[str, int],
        pad_token:    str = "<PAD>",
        bos_token:    str = "<BOS>",
        eos_token:    str = "<EOS>",
        unk_token:    str = "<UNK>",
    ):
        self.vocab    = vocab
        self.id2token = {v: k for k, v in vocab.items()}
        
        self.pad_token    = pad_token
        self.bos_token    = bos_token
        self.eos_token    = eos_token
        self.unk_token    = unk_token
        
        self.pad_token_id = vocab.get(pad_token, 0)
        self.bos_token_id = vocab.get(bos_token, 1)
        self.eos_token_id = vocab.get(eos_token, 2)
        self.unk_token_id = vocab.get(unk_token, 3)
    
    @classmethod
    def from_vocab(cls, vocab_path: str | Path) -> "SimpleTokenizer":
        """Load from vocab.json file"""
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)
    
    def encode(self, text: str, max_length: int = 2048) -> List[int]:
        """Encode single text to token IDs"""
        tokens = list(text)  # Character-level for DNA
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        
        # Truncate
        if len(ids) > max_length - 2:
            ids = ids[:max_length - 2]
        
        # Add special tokens
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def encode_batch(
        self,
        texts:      List[str],
        max_length: int = 2048,
    ) -> Dict[str, torch.Tensor]:
        """Encode batch of texts"""
        encoded = [self.encode(t, max_length) for t in texts]
        
        # Pad to max length in batch
        max_len = max(len(e) for e in encoded)
        
        input_ids = []
        attention_mask = []
        
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text"""
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            tokens.append(self.id2token.get(id_, self.unk_token))
        
        return "".join(tokens)
    
    def decode_batch(self, batch_ids: torch.Tensor, skip_special_tokens: bool = False) -> List[str]:
        """Decode batch of token IDs"""
        return [self.decode(ids.tolist(), skip_special_tokens) for ids in batch_ids]


############################
#####  CLI Interface   #####
############################


def main():
    """Command-line interface for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GeneT5 Inference")
    parser.add_argument("--model", "-m", required=True, help="Model checkpoint path")
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file or sequence")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--source", default="GeneT5", help="GFF source field")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device) if args.device else None
    inferencer = GeneT5Inference.from_pretrained(args.model, device=device)
    
    # Read input
    sequences, seqids = read_input(args.input)
    
    # Configure generation
    gen_config = GenerationConfig(
        max_length  = args.max_length,
        temperature = args.temperature,
        top_k       = args.top_k,
        top_p       = args.top_p,
    )
    
    # Run inference
    results = inferencer.predict(
        sequences  = sequences,
        seqids     = seqids,
        output_dir = args.output,
        source     = args.source,
        gen_config = gen_config,
        batch_size = args.batch_size,
    )
    
    # Print summary
    print(f"\nProcessed {len(results)} sequences")
    for r in results:
        print(f"  {r.metadata['seqid']}: {len(r.gff_features)} features -> {r.gff_path}")


def read_input(input_path: str) -> Tuple[List[str], List[str]]:
    """Read input sequences from FASTA or plain text"""
    path = Path(input_path)
    
    if path.exists():
        content = path.read_text()
        
        if content.startswith(">"):
            # FASTA format
            sequences = []
            seqids = []
            current_seq = []
            current_id = None
            
            for line in content.strip().split("\n"):
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                        seqids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            if current_id is not None:
                sequences.append("".join(current_seq))
                seqids.append(current_id)
            
            return sequences, seqids
        else:
            # Plain text - one sequence per line
            sequences = [l.strip() for l in content.strip().split("\n") if l.strip()]
            seqids = [f"seq_{i}" for i in range(len(sequences))]
            return sequences, seqids
    else:
        # Treat as direct sequence
        return [input_path], ["seq_0"]


if __name__ == "__main__":
    main()