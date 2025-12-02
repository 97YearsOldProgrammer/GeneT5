import torch
import random
import numpy as np
from torch.utils.data import Dataset


############################
#### Pretraining Data ######
############################


class BoundaryMaskedIntron(Dataset):
    
    """
    Masks only first and last k-mer tokens to learn donor/acceptor site motifs.
    All other positions labeled as intron (0).
    """
    
    def __init__(self, introns, tokenizer, max_length):
        self.introns    = introns
        self.tokenizer  = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.introns)
    
    def __getitem__(self, idx):
        seq = self.introns[idx]
        
        # Tokenize
        tokens          = self.tokenizer(seq)
        original_tokens = tokens.copy()
        
        # Mask donor and acceptor sites
        masked_tokens       = tokens.copy()
        masked_tokens[0]    = self.tokenizer.mask_token_id
        masked_tokens[-1]   = self.tokenizer.mask_token_id
        
        # Create label
        labels      = torch.zeros(len(tokens), dtype=torch.long)
        labels[0]   = original_tokens[0]
        labels[-1]  = original_tokens[-1]
        
        # Pad labels
        if len(labels) < self.max_length:
            labels = torch.cat([labels, torch.zeros(self.max_length - len(labels), dtype=torch.long)])
        else:
            labels = labels[:self.max_length]
        
        # Pad tokens
        if len(masked_tokens) < self.max_length:
            masked_tokens = masked_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(masked_tokens))
        else:
            masked_tokens = masked_tokens[:self.max_length]
        
        return {
            'masked_tokens' : torch.tensor(masked_tokens),
            'labels'        : labels,
            'task_type'     : 'boundary_intron'
        }
        

class SpanMaskedIntron(Dataset):
    
    """
    Random span masking for introns.
    Masks continuous spans of 3-8 tokens to learn internal intron structure.
    """
    
    def __init__(self, introns, tokenizer, max_length=1024, 
                 min_span=3, max_span=8, mask_prob=0.15):
        self.introns    = introns
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.min_span   = min_span
        self.max_span   = max_span
        self.mask_prob  = mask_prob
        
    def __len__(self):
        return len(self.introns)
    
    def _create_span_masks(self, seq_len):

        mask_positions = []
        i = 0
        
        while i < seq_len:
            if random.random() < self.mask_prob:
                # Start a span
                span_len = random.randint(self.min_span, self.max_span)
                span_end = min(i + span_len, seq_len)
                mask_positions.extend(range(i, span_end))
                i = span_end
            else:
                i += 1
        
        return mask_positions
    
    def __getitem__(self, idx):
        seq = self.introns[idx]
        
        # Tokenize
        tokens          = self.tokenizer(seq)
        original_tokens = tokens.copy()
        
        # Create span masks
        mask_positions  = self._create_span_masks(len(tokens))
        
        # Apply masking
        masked_tokens = tokens.copy()
        for pos in mask_positions:
            masked_tokens[pos] = self.tokenizer.mask_token_id
        
        # Create labels: masked = original, unmasked = 0 (intron)
        labels = torch.zeros(len(tokens), dtype=torch.long)
        for pos in mask_positions:
            if pos < len(tokens):
                labels[pos] = original_tokens[pos]
        
        # Pad
        if len(labels) < self.max_length:
            labels = torch.cat([labels, torch.zeros(self.max_length - len(labels), dtype=torch.long)])
        else:
            labels = labels[:self.max_length]
        
        if len(masked_tokens) < self.max_length:
            masked_tokens = masked_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(masked_tokens))
        else:
            masked_tokens = masked_tokens[:self.max_length]
        
        return {
            'masked_tokens': torch.tensor(masked_tokens),
            'labels': labels,
            'task_type': 'span_intron'
        }


class SmartCDSDataset(Dataset):
    """
    Smart CDS/Exon dataset with three types of samples:
    
    1. Original transcripts (25%): Full proteins with NSP tokens at exon junctions
    2. Positive pairs (25%): Adjacent exons from same transcript (has_intron=1)
    3. Negative pairs (50%): Random exon combinations (has_intron=0)
    
    All samples properly labeled for next sentence prediction.
    """
    
    def __init__(self, cds_groups, tokenizer, max_length=1024, mask_prob=0.15):
        self.cds_groups = cds_groups
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        
        # Build all sample types
        self.samples = self._build_samples()
        
    def _build_samples(self):
        """Build balanced dataset with original, positive, and negative samples"""
        samples = []
        
        # Type 1: Original full transcripts (25%)
        for cds_list in self.cds_groups:
            if len(cds_list) >= 2:
                samples.append({
                    'type': 'original',
                    'cds_list': cds_list,
                    'label': 1  # All connections are valid
                })
        
        original_count = len(samples)
        
        # Type 2: Positive pairs (25%) - adjacent exons
        positive_samples = []
        for cds_list in self.cds_groups:
            if len(cds_list) >= 2:
                for i in range(len(cds_list) - 1):
                    positive_samples.append({
                        'type': 'positive',
                        'cds1': cds_list[i],
                        'cds2': cds_list[i + 1],
                        'label': 1
                    })
        
        # Sample to match original count
        if len(positive_samples) > original_count:
            positive_samples = random.sample(positive_samples, original_count)
        samples.extend(positive_samples)
        
        # Type 3: Negative pairs (50%) - random combinations
        negative_count = 2 * original_count
        negative_samples = []
        
        for _ in range(negative_count):
            # Pick two random transcripts
            idx1, idx2 = random.sample(range(len(self.cds_groups)), 2)
            
            if len(self.cds_groups[idx1]) > 0 and len(self.cds_groups[idx2]) > 0:
                cds1 = random.choice(self.cds_groups[idx1])
                cds2 = random.choice(self.cds_groups[idx2])
                
                negative_samples.append({
                    'type': 'negative',
                    'cds1': cds1,
                    'cds2': cds2,
                    'label': 0
                })
        
        samples.extend(negative_samples)
        
        # Shuffle all samples
        random.shuffle(samples)
        
        print(f"Built CDS dataset: {original_count} original, "
              f"{len(positive_samples)} positive, {len(negative_samples)} negative")
        
        return samples
    
    def _process_original(self, cds_list):
        """Process full transcript with NSP tokens at junctions"""
        all_tokens = []
        nsp_positions = []
        
        for i, cds in enumerate(cds_list):
            if i > 0:
                # Insert NSP token at junction
                nsp_positions.append(len(all_tokens))
                all_tokens.append(self.tokenizer.sep_token_id)
            
            tokens = self.tokenizer(cds)
            all_tokens.extend(tokens)
        
        return all_tokens, nsp_positions
    
    def _process_pair(self, cds1, cds2):
        """Process CDS pair with NSP token"""
        tokens1 = self.tokenizer(cds1)
        tokens2 = self.tokenizer(cds2)
        
        nsp_position = len(tokens1)
        all_tokens = tokens1 + [self.tokenizer.sep_token_id] + tokens2
        
        return all_tokens, [nsp_position]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process based on type
        if sample['type'] == 'original':
            tokens, nsp_positions = self._process_original(sample['cds_list'])
        else:
            tokens, nsp_positions = self._process_pair(sample['cds1'], sample['cds2'])
        
        original_tokens = tokens.copy()
        label = sample['label']
        
        # Apply random masking (excluding NSP tokens)
        masked_tokens = tokens.copy()
        mask_positions = []
        
        for i in range(len(tokens)):
            if tokens[i] != self.tokenizer.sep_token_id and random.random() < self.mask_prob:
                mask_positions.append(i)
                masked_tokens[i] = self.tokenizer.mask_token_id
        
        # Truncate if needed
        if len(masked_tokens) > self.max_length:
            masked_tokens = masked_tokens[:self.max_length]
            original_tokens = original_tokens[:self.max_length]
            nsp_positions = [p for p in nsp_positions if p < self.max_length]
        
        # Create MLM labels
        mlm_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        for pos in mask_positions:
            if pos < self.max_length and pos < len(original_tokens):
                mlm_labels[pos] = original_tokens[pos]
        
        # Pad tokens
        if len(masked_tokens) < self.max_length:
            masked_tokens = masked_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(masked_tokens))
        
        # Pad NSP positions
        max_nsp = 20
        if len(nsp_positions) < max_nsp:
            nsp_positions = nsp_positions + [-1] * (max_nsp - len(nsp_positions))
        else:
            nsp_positions = nsp_positions[:max_nsp]
        
        return {
            'masked_tokens': torch.tensor(masked_tokens),
            'mlm_labels': mlm_labels,
            'nsp_positions': torch.tensor(nsp_positions),
            'nsp_label': torch.tensor(label),
            'task_type': f'cds_{sample["type"]}'
        }


class TranscriptPretrainingDataset(Dataset):
    """
    Transcript dataset with state change prediction.
    
    Inserts [STATE] token at transitions:
    - 5UTR -> exon
    - exon -> intron  
    - intron -> exon
    - exon -> 3UTR
    """
    
    STATE_MAP = {
        '5UTR': 0,
        'exon': 1,
        'intron': 2,
        '3UTR': 3,
        'no_change': 4,
    }
    
    def __init__(self, transcripts, tokenizer, max_length=1024, 
                 mask_prob=0.15, mode='pretrain'):
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mode = mode
        
    def __len__(self):
        return len(self.transcripts)
    
    def _process_pretrain(self, parts):
        """Process with state tokens"""
        tokens = []
        state_positions = []
        state_labels = []
        prev_state = None
        
        for part in parts:
            seq = part['seq']
            state = part['typ']
            
            # Insert state token on transition
            if prev_state is not None and prev_state != state:
                state_positions.append(len(tokens))
                tokens.append(self.tokenizer.state_token_id)
                state_labels.append(self.STATE_MAP.get(state, 4))
            
            # Tokenize part
            part_tokens = self.tokenizer(seq)
            tokens.extend(part_tokens)
            prev_state = state
        
        return tokens, state_positions, state_labels
    
    def _process_finetune(self, parts):
        """Process without state tokens, one-to-one mapping"""
        tokens = []
        labels = []
        
        for part in parts:
            seq = part['seq']
            state = part['typ']
            
            part_tokens = self.tokenizer(seq)
            label_value = self.STATE_MAP.get(state, 4)
            
            tokens.extend(part_tokens)
            labels.extend([label_value] * len(part_tokens))
        
        return tokens, labels
    
    def __getitem__(self, idx):
        parts = self.transcripts[idx]
        
        if self.mode == 'pretrain':
            tokens, state_positions, state_labels = self._process_pretrain(parts)
            original_tokens = tokens.copy()
            
            # Mask random tokens (not state tokens)
            masked_tokens = tokens.copy()
            mask_positions = []
            
            for i in range(len(tokens)):
                if tokens[i] != self.tokenizer.state_token_id and random.random() < self.mask_prob:
                    mask_positions.append(i)
                    masked_tokens[i] = self.tokenizer.mask_token_id
            
            # Truncate
            if len(masked_tokens) > self.max_length:
                masked_tokens = masked_tokens[:self.max_length]
                original_tokens = original_tokens[:self.max_length]
                state_positions = [p for p in state_positions if p < self.max_length]
                state_labels = state_labels[:len(state_positions)]
            
            # MLM labels
            mlm_labels = torch.full((self.max_length,), -100, dtype=torch.long)
            for pos in mask_positions:
                if pos < self.max_length and pos < len(original_tokens):
                    mlm_labels[pos] = original_tokens[pos]
            
            # Pad
            if len(masked_tokens) < self.max_length:
                masked_tokens = masked_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(masked_tokens))
            else:
                masked_tokens = masked_tokens[:self.max_length]
            
            # Pad state info
            max_states = 50
            if len(state_positions) < max_states:
                state_positions = state_positions + [-1] * (max_states - len(state_positions))
                state_labels = state_labels + [-100] * (max_states - len(state_labels))
            else:
                state_positions = state_positions[:max_states]
                state_labels = state_labels[:max_states]
            
            return {
                'masked_tokens': torch.tensor(masked_tokens),
                'mlm_labels': mlm_labels,
                'state_positions': torch.tensor(state_positions),
                'state_labels': torch.tensor(state_labels),
                'task_type': 'transcript'
            }
        
        else:  # finetune
            tokens, labels = self._process_finetune(parts)
            
            # Truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                labels = labels[:self.max_length]
            
            # Pad
            if len(tokens) < self.max_length:
                tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
                labels = labels + [-100] * (self.max_length - len(labels))
            else:
                tokens = tokens[:self.max_length]
                labels = labels[:self.max_length]
            
            return {
                'tokens': torch.tensor(tokens),
                'labels': torch.tensor(labels),
                'task_type': 'transcript_finetune'
            }


#############################
#### Combined Dataset #######
#############################


class CombinedPretrainingDataset(Dataset):
    """
    Combined dataset for all pretraining tasks.
    
    Samples from:
    1. Boundary masked introns
    2. Span masked introns
    3. Smart CDS (original/positive/negative)
    4. Transcript state prediction
    """
    
    def __init__(self, boundary_intron_ds, span_intron_ds, cds_ds, 
                 transcript_ds, weights=None):
        self.datasets = {
            'boundary_intron': boundary_intron_ds,
            'span_intron': span_intron_ds,
            'cds': cds_ds,
            'transcript': transcript_ds,
        }
        
        if weights is None:
            weights = [1.0, 1.0, 2.0, 1.0]  # CDS gets double weight
        
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        self.total_length = sum(len(d) for d in self.datasets.values())
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Sample dataset by weight
        r = random.random()
        cumsum = 0
        
        for i, (name, dataset) in enumerate(self.datasets.items()):
            cumsum += self.weights[i]
            if r < cumsum:
                local_idx = idx % len(dataset)
                return dataset[local_idx]
        
        # Fallback
        local_idx = idx % len(self.datasets['boundary_intron'])
        return self.datasets['boundary_intron'][local_idx]


##############################
#### WormBase Dataset ########
##############################


class WormBaseDataset:
    """WormBase dataset for genetic sequence analysis"""
    
    def __init__(self, gff_file, fasta_files):
        # Parse FASTA
        self.chms = {}
        for fasta in fasta_files:
            self.chms.update(parse_fasta(fasta))
        
        # Parse GFF3
        introns_raw, proteins_raw, genes_raw = parse_wormbase_gff(gff_file)
        
        # Process sequences
        self.introns = process_introns(introns_raw, self.chms)
        self.cds_groups = process_cds(proteins_raw, self.chms)
        self.transcripts = process_transcripts(genes_raw, self.chms)
        
        print(f"Loaded {len(self.introns)} introns")
        print(f"Loaded {len(self.cds_groups)} CDS groups")
        print(f"Loaded {len(self.transcripts)} transcripts")
    
    def get_pretraining_datasets(self, tokenizer, mask_prob=0.15, max_length=1024):
        """Get combined pretraining dataset"""
        
        # Split introns for two strategies
        n = len(self.introns)
        boundary_introns = self.introns[:n//2]
        span_introns = self.introns[n//2:]
        
        boundary_ds = BoundaryMaskedIntron(boundary_introns, tokenizer, max_length)
        span_ds = SpanMaskedIntron(span_introns, tokenizer, max_length, mask_prob=mask_prob)
        cds_ds = SmartCDSDataset(self.cds_groups, tokenizer, max_length, mask_prob)
        transcript_ds = TranscriptPretrainingDataset(
            self.transcripts, tokenizer, max_length, mask_prob, mode='pretrain'
        )
        
        return CombinedPretrainingDataset(boundary_ds, span_ds, cds_ds, transcript_ds)
    
    def get_finetune_dataset(self, tokenizer, max_length=1024):
        """Get fine-tuning dataset"""
        return TranscriptPretrainingDataset(
            self.transcripts, tokenizer, max_length, mask_prob=0.0, mode='finetune'
        )


######################
#### Collate Fn ######
######################


def pretrain_collate_fn(batch):
    """Collate function for pretraining batches"""
    
    # Group by task type
    task_types = [item['task_type'] for item in batch]
    
    # Stack tensors
    input_onehot = torch.stack([item['input_onehot'] for item in batch])
    masked_tokens = torch.stack([item['masked_tokens'] for item in batch])
    
    # MLM labels (different key names per dataset)
    if 'labels' in batch[0]:
        mlm_labels = torch.stack([item['labels'] for item in batch])
    elif 'mlm_labels' in batch[0]:
        mlm_labels = torch.stack([item['mlm_labels'] for item in batch])
    else:
        mlm_labels = None
    
    result = {
        'input_onehot': input_onehot,
        'masked_tokens': masked_tokens,
        'mlm_labels': mlm_labels,
        'task_types': task_types,
    }
    
    # NSP labels (CDS only)
    if 'nsp_label' in batch[0]:
        result['nsp_labels'] = torch.stack([item['nsp_label'] for item in batch])
        result['sep_positions'] = torch.stack([item['sep_position'] for item in batch])
    
    # State labels (transcript only)
    if 'state_labels' in batch[0]:
        result['state_labels'] = torch.stack([item['state_labels'] for item in batch])
        result['state_positions'] = torch.stack([item['state_positions'] for item in batch])
    
    return result


def finetune_collate_fn(batch):
    """Collate function for fine-tuning batches"""
    
    input_onehot = torch.stack([item['input_onehot'] for item in batch])
    tokens = torch.stack([item['tokens'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_onehot': input_onehot,
        'tokens': tokens,
        'labels': labels,
    }


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = KmerTokenizer(k=6)
    
    test_seq = "ATCGATCGATCGATCGATCGATCG"
    tokens = tokenizer(test_seq)
    print(f"Sequence: {test_seq}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens)}")
    print(f"One-hot shape: {tokenizer.to_onehot(test_seq).shape}")
    
    # Test special tokens
    print(f"\nMask token ID: {tokenizer.mask_token_id}")
    print(f"Sep token ID: {tokenizer.sep_token_id}")
    print(f"State token ID: {tokenizer.state_token_id}")
    print(f"Total vocab size: {tokenizer.total_vocab_size}")
