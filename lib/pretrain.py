"""
Extended WormBase Dataset with Tokenization and Span Corruption Masking

This module provides:
- K-mer based tokenization for DNA sequences
- T5-style span corruption masking
- Special handling for intron boundary masking
- Train/validation splits for pre-training and fine-tuning
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Dict
import random


###################
##### Masking #####
###################


class SpanCorruptionMasker:
    """
    T5-style span corruption masking
    
    Masks random spans of tokens and replaces them with sentinel tokens.
    The model learns to predict the masked spans in order.
    """
    
    def __init__(
        self,
        tokenizer: KmerTokenizer,
        mask_ratio: float = 0.15,
        mean_span_length: float = 3.0,
        intron_boundary_mask: bool = False,
        boundary_kmer_count: int = 2
    ):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.mean_span_length = mean_span_length
        self.intron_boundary_mask = intron_boundary_mask
        self.boundary_kmer_count = boundary_kmer_count
    
    def _sample_span_lengths(self, total_length: int) -> List[int]:
        """Sample span lengths following Poisson distribution"""
        num_masked = int(total_length * self.mask_ratio)
        
        span_lengths = []
        while sum(span_lengths) < num_masked:
            span_length = np.random.poisson(self.mean_span_length)
            span_length = max(1, min(span_length, total_length))
            span_lengths.append(span_length)
        
        # Trim to exact number
        if sum(span_lengths) > num_masked:
            span_lengths[-1] -= (sum(span_lengths) - num_masked)
        
        return [s for s in span_lengths if s > 0]
    
    def mask_intron_boundaries(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Special masking for intron boundaries:
        Mask first N and last N k-mer tokens for transfer learning
        """
        if len(ids) <= 2 * self.boundary_kmer_count:
            # Sequence too short, use regular masking
            return self.mask_sequence(ids)
        
        # Create masks for boundaries
        masked_input = ids.copy()
        target_tokens = []
        
        # Mask first N k-mers
        first_sentinel_id = self.tokenizer.sentinel_start_id
        first_span = ids[:self.boundary_kmer_count]
        masked_input[:self.boundary_kmer_count] = [first_sentinel_id]
        target_tokens.extend([first_sentinel_id] + first_span)
        
        # Keep middle section
        # Mask last N k-mers
        last_sentinel_id = self.tokenizer.sentinel_start_id + 1
        last_span = ids[-self.boundary_kmer_count:]
        masked_input = masked_input[:self.boundary_kmer_count] + \
                       masked_input[self.boundary_kmer_count:-self.boundary_kmer_count] + \
                       [last_sentinel_id]
        
        target_tokens.extend([last_sentinel_id] + last_span)
        
        # Add final sentinel
        target_tokens.append(self.tokenizer.sentinel_start_id + 2)
        
        return masked_input, target_tokens
    
    def mask_sequence(self, ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Apply span corruption masking to a sequence
        
        Returns:
            masked_input: Input with spans replaced by sentinel tokens
            target: Target sequence with sentinel tokens and masked content
        """
        if self.intron_boundary_mask:
            return self.mask_intron_boundaries(ids)
        
        # Remove EOS if present
        has_eos = False
        if ids and ids[-1] == self.tokenizer.eos_token_id:
            has_eos = True
            ids = ids[:-1]
        
        if len(ids) == 0:
            return [self.tokenizer.eos_token_id], [self.tokenizer.eos_token_id]
        
        # Sample span lengths
        span_lengths = self._sample_span_lengths(len(ids))
        if not span_lengths:
            span_lengths = [1]
        
        # Sample span start positions
        total_masked = sum(span_lengths)
        possible_starts = list(range(len(ids) - total_masked + 1))
        
        if not possible_starts:
            possible_starts = [0]
        
        start_indices = sorted(random.sample(
            possible_starts,
            min(len(span_lengths), len(possible_starts))
        ))
        
        # Create masked input and target
        masked_input = []
        target = []
        sentinel_id = self.tokenizer.sentinel_start_id
        
        current_idx = 0
        for span_idx, start in enumerate(start_indices):
            # Add unmasked tokens before this span
            masked_input.extend(ids[current_idx:start])
            
            # Add sentinel token
            masked_input.append(sentinel_id)
            
            # Add to target: sentinel + masked tokens
            span_end = min(start + span_lengths[span_idx], len(ids))
            target.append(sentinel_id)
            target.extend(ids[start:span_end])
            
            sentinel_id += 1
            current_idx = span_end
        
        # Add remaining unmasked tokens
        masked_input.extend(ids[current_idx:])
        
        # Add final sentinel to target
        target.append(sentinel_id)
        
        # Re-add EOS
        if has_eos:
            masked_input.append(self.tokenizer.eos_token_id)
            target.append(self.tokenizer.eos_token_id)
        
        return masked_input, target


###################
##### Dataset #####
###################


class WormBaseDataset(Dataset):
    """
    Extended WormBase dataset with tokenization and masking
    """
    
    def __init__(
        self,
        gff_file: str,
        fasta_files: List[str],
        k: int = 6,
        mask_ratio: float = 0.15,
        mean_span_length: float = 3.0,
        pretrain_ratio: float = 0.5,
        max_length: int = 512
    ):
        # Parse FASTA
        self.chms = {}
        for fasta in fasta_files:
            self.chms.update(self.parse_fasta(fasta))

        # Parse GFF3 (you'll need to implement these)
        from parse_utils import parse_wormbase_gff, process_introns, process_cds, process_transcripts
        introns_raw, proteins_raw, genes_raw = parse_wormbase_gff(gff_file)
        
        # Process into simple lists
        self.introns = process_introns(introns_raw, self.chms)
        self.proteins = process_cds(proteins_raw, self.chms)
        self.transcripts = process_transcripts(genes_raw, self.chms)
        
        # Initialize tokenizer
        self.tokenizer = KmerTokenizer(k=k)
        self.max_length = max_length
        
        # Initialize maskers
        self.intron_masker = SpanCorruptionMasker(
            self.tokenizer,
            mask_ratio=mask_ratio,
            mean_span_length=mean_span_length,
            intron_boundary_mask=True,
            boundary_kmer_count=2
        )
        
        self.standard_masker = SpanCorruptionMasker(
            self.tokenizer,
            mask_ratio=mask_ratio,
            mean_span_length=mean_span_length,
            intron_boundary_mask=False
        )
        
        # Split transcripts for pre-training and fine-tuning
        self._split_transcripts(pretrain_ratio)
    
    def parse_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Parse FASTA file"""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id:
                sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def _split_transcripts(self, pretrain_ratio: float):
        """Split transcripts into pre-training and fine-tuning sets"""
        n_pretrain = int(len(self.transcripts) * pretrain_ratio)
        
        indices = list(range(len(self.transcripts)))
        random.shuffle(indices)
        
        self.pretrain_indices = set(indices[:n_pretrain])
        self.finetune_indices = set(indices[n_pretrain:])
    
    def tokenize_intron(self, intron_seq: str) -> List[int]:
        """Tokenize intron sequence"""
        return self.tokenizer.encode(intron_seq, add_eos=True)
    
    def tokenize_cds(self, cds_seq: str) -> List[int]:
        """Tokenize CDS sequence"""
        return self.tokenizer.encode(cds_seq, add_eos=True)
    
    def tokenize_transcript(self, transcript_seq: str) -> List[int]:
        """Tokenize transcript sequence"""
        return self.tokenizer.encode(transcript_seq, add_eos=True)
    
    def get_pretrain_transcripts(self):
        """Get transcripts for pre-training"""
        return [self.transcripts[i] for i in sorted(self.pretrain_indices)]
    
    def get_finetune_transcripts(self):
        """Get transcripts for fine-tuning"""
        return [self.transcripts[i] for i in sorted(self.finetune_indices)]
    
    def get_introns(self):
        return self.introns
    
    def get_proteins(self):
        return self.proteins
    
    def get_transcripts(self):
        return self.transcripts


class PretrainingDataset(Dataset):
    """
    Dataset for pre-training phase
    
    Feeds data in order:
    1. All introns (with boundary masking)
    2. All CDS sequences
    3. Pre-training transcripts
    """
    
    def __init__(self, wormbase_dataset: WormBaseDataset, phase: str = 'all'):
        """
        Args:
            phase: 'introns', 'cds', 'transcripts', or 'all'
        """
        self.dataset = wormbase_dataset
        self.phase = phase
        
        # Build indices for each phase
        self.intron_data = self._prepare_introns()
        self.cds_data = self._prepare_cds()
        self.transcript_data = self._prepare_transcripts()
        
        # Combine based on phase
        if phase == 'all':
            self.data = self.intron_data + self.cds_data + self.transcript_data
        elif phase == 'introns':
            self.data = self.intron_data
        elif phase == 'cds':
            self.data = self.cds_data
        elif phase == 'transcripts':
            self.data = self.transcript_data
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def _prepare_introns(self) -> List[Dict]:
        """Prepare intron data with boundary masking"""
        data = []
        for intron in self.dataset.get_introns():
            ids = self.dataset.tokenize_intron(intron)
            if len(ids) > self.dataset.max_length:
                ids = ids[:self.dataset.max_length]
            
            masked_input, target = self.dataset.intron_masker.mask_sequence(ids)
            data.append({
                'input_ids': masked_input,
                'labels': target,
                'type': 'intron'
            })
        return data
    
    def _prepare_cds(self) -> List[Dict]:
        """Prepare CDS data with standard masking"""
        data = []
        for cds in self.dataset.get_proteins():
            ids = self.dataset.tokenize_cds(cds)
            if len(ids) > self.dataset.max_length:
                ids = ids[:self.dataset.max_length]
            
            masked_input, target = self.dataset.standard_masker.mask_sequence(ids)
            data.append({
                'input_ids': masked_input,
                'labels': target,
                'type': 'cds'
            })
        return data
    
    def _prepare_transcripts(self) -> List[Dict]:
        """Prepare transcript data with standard masking"""
        data = []
        for transcript in self.dataset.get_pretrain_transcripts():
            ids = self.dataset.tokenize_transcript(transcript)
            if len(ids) > self.dataset.max_length:
                ids = ids[:self.dataset.max_length]
            
            masked_input, target = self.dataset.standard_masker.mask_sequence(ids)
            data.append({
                'input_ids': masked_input,
                'labels': target,
                'type': 'transcript'
            })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_token_id=0):
    """Collate function for DataLoader"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    max_input_len = max(len(seq) for seq in input_ids)
    max_label_len = max(len(seq) for seq in labels)
    
    input_ids_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    labels_padded = torch.full((len(batch), max_label_len), pad_token_id, dtype=torch.long)
    
    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        input_ids_padded[i, :len(inp)] = torch.tensor(inp)
        labels_padded[i, :len(lab)] = torch.tensor(lab)
    
    return {
        'input_ids': input_ids_padded,
        'decoder_input_ids': labels_padded[:, :-1],  # Shift for teacher forcing
        'labels': labels_padded[:, 1:],  # Shift for loss computation
        'types': [item['type'] for item in batch]
    }
