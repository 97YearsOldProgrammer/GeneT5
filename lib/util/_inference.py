import os
import json
import pathlib as pl

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import lib.model as model_lib
import lib.util._output as output_lib


##########################
#####  Device Utils  #####
##########################


def auto_detect_device():
    """Auto-detect the best available device"""
    
    if torch.cuda.is_available():
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
        return torch.device("mps")
    
    return torch.device("cpu")


def get_device_info(device):
    """Get device information for logging"""
    
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_capability"]  = str(torch.cuda.get_device_capability(device))
        free, total              = torch.cuda.mem_get_info(device)
        info["memory_free_gb"]   = f"{free / 1e9:.2f}"
        info["memory_total_gb"]  = f"{total / 1e9:.2f}"
    elif device.type == "mps":
        info["device_name"] = "Apple Silicon MPS"
    else:
        info["device_name"] = "CPU"
    
    return info


def select_dtype(device, prefer_bf16=True):
    """Select optimal dtype for device"""
    
    if device.type == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
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
    parsed_features: list
    gff_features:    list
    gff_path:        pl.Path = None
    metadata:        dict    = field(default_factory=dict)


class GeneT5Inference:
    """Inference wrapper for GeneT5 model with automatic GFF conversion"""
    
    def __init__(self, model, tokenizer, device=None, dtype=None):
        """Initialize inference wrapper with model and tokenizer"""
        
        self.device    = device or auto_detect_device()
        self.dtype     = dtype or select_dtype(self.device)
        self.model     = model.to(device=self.device, dtype=self.dtype)
        self.tokenizer = tokenizer
        self.parser    = output_lib.ModelOutputParser(strict=False)
        self.converter = output_lib.GFFConverter()
        
        info = get_device_info(self.device)
        print(f"GeneT5Inference initialized on {info['device']}")
        if "cuda_device_name" in info:
            print(f"  GPU: {info['cuda_device_name']}")
            print(f"  Memory: {info['memory_free_gb']}GB free / {info['memory_total_gb']}GB total")
        print(f"  Dtype: {self.dtype}")
    
    @classmethod
    def from_pretrained(cls, checkpoint_path, tokenizer_path=None, device=None, dtype=None):
        """Load model from checkpoint directory"""
        
        checkpoint_path = pl.Path(checkpoint_path)
        tokenizer_path  = pl.Path(tokenizer_path) if tokenizer_path else checkpoint_path
        target_device   = device or auto_detect_device()
        target_dtype    = dtype or select_dtype(target_device)
        
        model     = model_lib.GeneT5.from_pretrained(checkpoint_path, device=target_device, dtype=target_dtype)
        tokenizer = cls._load_tokenizer(tokenizer_path)
        
        return cls(model=model, tokenizer=tokenizer, device=target_device, dtype=target_dtype)
    
    @staticmethod
    def _load_tokenizer(tokenizer_path):
        """Load tokenizer from path - supports multiple tokenizer types"""
        
        try:
            import transformers as tf
            return tf.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception:
            pass
        
        vocab_path = tokenizer_path / "vocab.json"
        if vocab_path.exists():
            return SimpleTokenizer.from_vocab(vocab_path)
        
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}")
    
    def encode(self, sequences, max_length=2048):
        """Encode input sequences"""
        
        if hasattr(self.tokenizer, 'batch_encode_plus'):
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
            batch = self.tokenizer.encode_batch(sequences, max_length=max_length)
            return {
                "input_ids":      batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        
        if hasattr(self.tokenizer, 'batch_decode'):
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        else:
            return self.tokenizer.decode_batch(token_ids)
    
    @torch.no_grad()
    def generate(self, encoder_input_ids, encoder_attention_mask=None, config=None):
        """Generate output sequences"""
        
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
    
    def predict(self, sequences, seqids=None, output_dir=None, source="GeneT5", 
                offsets=None, gen_config=None, max_input_length=2048, batch_size=1):
        """Run inference on sequences and optionally save GFF3 output"""
        
        if seqids is None:
            seqids = [f"seq_{i}" for i in range(len(sequences))]
        if offsets is None:
            offsets = [0] * len(sequences)
        
        if output_dir:
            output_dir = pl.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        gen_config = gen_config or GenerationConfig()
        results    = []
        
        for batch_start in range(0, len(sequences), batch_size):
            batch_end  = min(batch_start + batch_size, len(sequences))
            batch_seqs = sequences[batch_start:batch_end]
            batch_ids  = seqids[batch_start:batch_end]
            batch_offs = offsets[batch_start:batch_end]
            
            encoded = self.encode(batch_seqs, max_length=max_input_length)
            
            with torch.amp.autocast(self.device.type, dtype=self.dtype):
                generated = self.generate(
                    encoder_input_ids      = encoded["input_ids"],
                    encoder_attention_mask = encoded["attention_mask"],
                    config                 = gen_config,
                )
            
            decoded_outputs = self.decode(generated)
            
            for i, (seq, seqid, offset, raw_output) in enumerate(
                zip(batch_seqs, batch_ids, batch_offs, decoded_outputs)
            ):
                decoded         = self._clean_output(raw_output)
                parsed_seqs     = self.parser.parse_sequence(decoded)
                parsed_features = parsed_seqs[0] if parsed_seqs else []
                
                self.converter.seqid  = seqid
                self.converter.source = source
                self.converter.offset = offset
                gff_features          = self.converter.convert_sequence(parsed_features)
                
                gff_path = None
                if output_dir:
                    gff_path = output_dir / f"{seqid}.gff3"
                    output_lib.write_gff3(gff_features, gff_path)
                
                results.append(InferenceResult(
                    input_sequence  = seq,
                    raw_output      = raw_output,
                    decoded_output  = decoded,
                    parsed_features = parsed_features,
                    gff_features    = gff_features,
                    gff_path        = gff_path,
                    metadata        = {"seqid": seqid, "offset": offset, "source": source},
                ))
        
        return results
    
    def _clean_output(self, raw_output):
        """Clean raw model output for parsing"""
        
        output = raw_output.strip()
        output = output.replace("<bos>", "<BOS>")
        output = output.replace("<eos>", "<EOS>")
        output = output.replace("[BOS]", "<BOS>")
        output = output.replace("[EOS]", "<EOS>")
        
        return output
    
    def predict_single(self, sequence, seqid="seq", output_path=None, source="GeneT5", 
                       offset=0, gen_config=None):
        """Convenience method for single sequence prediction"""
        
        output_dir = None
        if output_path:
            output_path = pl.Path(output_path)
            output_dir  = output_path.parent
        
        results = self.predict(
            sequences  = [sequence],
            seqids     = [seqid],
            output_dir = output_dir,
            source     = source,
            offsets    = [offset],
            gen_config = gen_config,
        )
        
        result = results[0]
        
        if output_path and result.gff_path and result.gff_path != output_path:
            result.gff_path.rename(output_path)
            result.gff_path = output_path
        
        return result


############################
#####  Simple Tokenizer ####
############################


class SimpleTokenizer:
    """Simple tokenizer for DNA/protein sequences as fallback"""
    
    def __init__(self, vocab, pad_token="<PAD>", bos_token="<BOS>", eos_token="<EOS>", unk_token="<UNK>"):
        """Initialize tokenizer with vocabulary"""
        
        self.vocab        = vocab
        self.id2token     = {v: k for k, v in vocab.items()}
        self.pad_token    = pad_token
        self.bos_token    = bos_token
        self.eos_token    = eos_token
        self.unk_token    = unk_token
        self.pad_token_id = vocab.get(pad_token, 0)
        self.bos_token_id = vocab.get(bos_token, 1)
        self.eos_token_id = vocab.get(eos_token, 2)
        self.unk_token_id = vocab.get(unk_token, 3)
    
    @classmethod
    def from_vocab(cls, vocab_path):
        """Load from vocab.json file"""
        
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)
    
    def encode(self, text, max_length=2048):
        """Encode single text to token IDs"""
        
        tokens = list(text)
        ids    = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        
        if len(ids) > max_length - 2:
            ids = ids[:max_length - 2]
        
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def encode_batch(self, texts, max_length=2048):
        """Encode batch of texts"""
        
        encoded = [self.encode(t, max_length) for t in texts]
        max_len = max(len(e) for e in encoded)
        
        input_ids      = []
        attention_mask = []
        
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def decode(self, ids, skip_special_tokens=False):
        """Decode token IDs to text"""
        
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens      = []
        
        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            tokens.append(self.id2token.get(id_, self.unk_token))
        
        return "".join(tokens)
    
    def decode_batch(self, batch_ids, skip_special_tokens=False):
        """Decode batch of token IDs"""
        
        return [self.decode(ids.tolist(), skip_special_tokens) for ids in batch_ids]


############################
#####  CLI Interface   #####
############################


def read_input(input_path):
    """Read input sequences from FASTA or plain text"""
    
    path = pl.Path(input_path)
    
    if path.exists():
        content = path.read_text()
        
        if content.startswith(">"):
            sequences   = []
            seqids      = []
            current_seq = []
            current_id  = None
            
            for line in content.strip().split("\n"):
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                        seqids.append(current_id)
                    current_id  = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            
            if current_id is not None:
                sequences.append("".join(current_seq))
                seqids.append(current_id)
            
            return sequences, seqids
        else:
            sequences = [l.strip() for l in content.strip().split("\n") if l.strip()]
            seqids    = [f"seq_{i}" for i in range(len(sequences))]
            return sequences, seqids
    else:
        return [input_path], ["seq_0"]