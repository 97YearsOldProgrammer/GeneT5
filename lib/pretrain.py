import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from typing import List, Tuple, Dict
import random
from transunet import TransUNet


class MaskedSequenceDataset(Dataset):
    """
    Dataset for self-supervised pre-training on intron sequences
    Masks donor/acceptor sites and other k-mer positions
    """
    
    def __init__(
        self,
        sequences: List[str],              # List of DNA sequences
        seq_length: int = 1000,            # Fixed sequence length
        mask_prob: float = 0.15,           # Probability of masking a position
        mask_donor_acceptor: bool = True,  # Specifically mask splice sites
        donor_motif: str = "GTAAGT",       # Donor site motif (can be regex-like)
        acceptor_motif: str = "YYYAG",     # Acceptor site motif (Y = pyrimidine)
        vocab: Dict[str, int] = None,      # Nucleotide vocabulary
        augment: bool = True,              # Data augmentation
    ):
        super().__init__()
        self.sequences = sequences
        self.seq_length = seq_length
        self.mask_prob = mask_prob
        self.mask_donor_acceptor = mask_donor_acceptor
        self.donor_motif = donor_motif
        self.acceptor_motif = acceptor_motif
        self.augment = augment
        
        # Default DNA vocabulary
        if vocab is None:
            self.vocab = {
                'A': 0, 'C': 1, 'G': 2, 'T': 3,
                'N': 4,     # Unknown
                '<PAD>': 5, # Padding
                '<MASK>': 6 # Mask token
            }
        else:
            self.vocab = vocab
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.mask_token_id = self.vocab['<MASK>']
        self.pad_token_id = self.vocab['<PAD>']
        
    def __len__(self):
        return len(self.sequences)
    
    def _find_motif_positions(self, seq: str, motif: str) -> List[int]:
        """Find all positions where motif occurs (allowing Y for C/T)"""
        positions = []
        motif_len = len(motif)
        
        for i in range(len(seq) - motif_len + 1):
            subseq = seq[i:i + motif_len]
            if self._match_motif(subseq, motif):
                positions.append(i)
        
        return positions
    
    @staticmethod
    def _match_motif(subseq: str, motif: str) -> bool:
        """Check if subseq matches motif (with IUPAC codes)"""
        if len(subseq) != len(motif):
            return False
        
        for s, m in zip(subseq, motif):
            if m == 'N':  # Any nucleotide
                continue
            elif m == 'Y':  # Pyrimidine (C or T)
                if s not in ['C', 'T']:
                    return False
            elif m == 'R':  # Purine (A or G)
                if s not in ['A', 'G']:
                    return False
            elif s != m:
                return False
        
        return True
    
    def _sequence_to_indices(self, seq: str) -> List[int]:
        """Convert DNA sequence to indices"""
        return [self.vocab.get(nt, self.vocab['N']) for nt in seq.upper()]
    
    def _augment_sequence(self, seq: str) -> str:
        """Apply data augmentation to sequence"""
        if not self.augment:
            return seq
        
        # Random reverse complement (50% chance)
        if random.random() < 0.5:
            seq = self._reverse_complement(seq)
        
        # Random crop (if longer than seq_length)
        if len(seq) > self.seq_length:
            start = random.randint(0, len(seq) - self.seq_length)
            seq = seq[start:start + self.seq_length]
        
        return seq
    
    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join([complement.get(nt, 'N') for nt in seq[::-1]])
    
    def _create_masked_sequence(
        self, 
        seq_indices: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Create masked sequence for pre-training
        
        Returns:
            masked_seq: Sequence with some positions masked
            labels: Original tokens at masked positions
            mask_positions: Boolean mask of which positions were masked
        """
        seq_len = len(seq_indices)
        masked_seq = seq_indices.copy()
        labels = [-100] * seq_len  # -100 = ignore in loss
        mask_positions = [0] * seq_len
        
        # Get original sequence string for motif finding
        seq_str = ''.join([self.inv_vocab.get(idx, 'N') for idx in seq_indices])
        
        # Find donor and acceptor sites
        splice_sites = []
        if self.mask_donor_acceptor:
            donor_pos = self._find_motif_positions(seq_str, self.donor_motif)
            acceptor_pos = self._find_motif_positions(seq_str, self.acceptor_motif)
            
            # Add all positions in the motifs
            for pos in donor_pos:
                splice_sites.extend(range(pos, pos + len(self.donor_motif)))
            for pos in acceptor_pos:
                splice_sites.extend(range(pos, pos + len(self.acceptor_motif)))
        
        # Mask positions
        for i in range(seq_len):
            # Skip padding tokens
            if seq_indices[i] == self.pad_token_id:
                continue
            
            # Higher probability to mask splice sites
            if i in splice_sites:
                prob = min(0.8, self.mask_prob * 3)  # 3x more likely
            else:
                prob = self.mask_prob
            
            if random.random() < prob:
                mask_positions[i] = 1
                labels[i] = seq_indices[i]  # Store original token
                
                # 80% replace with <MASK>, 10% random, 10% keep
                rand = random.random()
                if rand < 0.8:
                    masked_seq[i] = self.mask_token_id
                elif rand < 0.9:
                    # Random token (not PAD or MASK)
                    masked_seq[i] = random.randint(0, 3)  # A, C, G, T
                # else: keep original (10%)
        
        return masked_seq, labels, mask_positions
    
    def __getitem__(self, idx):
        # Get sequence
        seq = self.sequences[idx]
        
        # Augmentation
        seq = self._augment_sequence(seq)
        
        # Pad or truncate to fixed length
        if len(seq) < self.seq_length:
            seq = seq + 'N' * (self.seq_length - len(seq))
        else:
            seq = seq[:self.seq_length]
        
        # Convert to indices
        seq_indices = self._sequence_to_indices(seq)
        
        # Create masked version
        masked_seq, labels, mask_positions = self._create_masked_sequence(seq_indices)
        
        # Convert to tensors
        # Shape: (seq_length,)
        masked_seq = torch.LongTensor(masked_seq)
        labels = torch.LongTensor(labels)
        mask_positions = torch.BoolTensor(mask_positions)
        
        return {
            'input_ids': masked_seq,      # Masked sequence
            'labels': labels,              # Original tokens (with -100 for non-masked)
            'mask_positions': mask_positions  # Boolean mask
        }


class TransUNetForPretraining(nn.Module):
    """
    TransUNet with a prediction head for masked token prediction
    Used for self-supervised pre-training
    """
    
    def __init__(
        self,
        transunet: TransUNet,
        vocab_size: int = 7,  # A, C, G, T, N, <PAD>, <MASK>
        embed_dim: int = 768,
    ):
        super().__init__()
        
        # Base TransUNet encoder (we only use encoder during pre-training)
        self.transunet = transunet
        
        # Token prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: (B, seq_length) masked sequence indices
            labels: (B, seq_length) original tokens (-100 for non-masked)
        
        Returns:
            loss (if labels provided), logits
        """
        B, L = input_ids.shape
        
        # Convert indices to one-hot encoding
        # input_ids: (B, L) -> (B, 1, L, vocab_size)
        vocab_size = 7
        x = F.one_hot(input_ids, num_classes=vocab_size).float()  # (B, L, vocab_size)
        x = x.unsqueeze(1)  # (B, 1, L, vocab_size)
        
        # ===== ENCODER: Get features from TransUNet =====
        # Use encoder + transformer parts
        x_enc, skips = self.transunet.encoder(x)  # (B, 256, L/16, vocab_size/16)
        
        # Patch embedding
        x_patch, (h, w) = self.transunet.patch_embed(x_enc)  # (B, N, embed_dim)
        x_patch = x_patch + self.transunet.pos_embed
        x_patch = self.transunet.pos_drop(x_patch)
        
        # Transformer blocks
        for blk in self.transunet.transformer_blocks:
            x_patch = blk(x_patch)
        x_patch = self.transunet.norm(x_patch)  # (B, N, embed_dim)
        
        # ===== PREDICTION HEAD =====
        # Predict tokens for each position
        logits = self.prediction_head(x_patch)  # (B, N, vocab_size)
        
        # Need to interpolate back to original sequence length
        # x_patch is (B, N, vocab_size) where N = (L/16) * (vocab_size/16)
        # We want (B, L, vocab_size)
        
        # Reshape and interpolate
        logits = logits.transpose(1, 2)  # (B, vocab_size, N)
        logits = F.interpolate(
            logits, 
            size=L, 
            mode='linear', 
            align_corners=False
        )  # (B, vocab_size, L)
        logits = logits.transpose(1, 2)  # (B, L, vocab_size)
        
        # ===== LOSS CALCULATION =====
        loss = None
        if labels is not None:
            # Only calculate loss on masked positions (labels != -100)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        return {
            'loss': loss,
            'logits': logits,
        }


def create_pretraining_model(
    seq_length: int = 1000,
    vocab_size: int = 7,
    **transunet_kwargs
):
    """
    Create TransUNet model for pre-training
    
    Args:
        seq_length: Length of input sequences
        vocab_size: Size of vocabulary (nucleotides + special tokens)
        **transunet_kwargs: Arguments for TransUNet
    """
    # Create base TransUNet
    # For sequences: H=seq_length, W=vocab_size (one-hot dimension)
    base_model = TransUNet(
        img_size=seq_length,
        in_channels=1,  # Single channel (will be one-hot)
        num_classes=vocab_size,  # Not used in pre-training
        **transunet_kwargs
    )
    
    # Wrap with pre-training head
    model = TransUNetForPretraining(
        transunet=base_model,
        vocab_size=vocab_size,
        embed_dim=transunet_kwargs.get('embed_dim', 768)
    )
    
    return model


def pretrain(
    model: TransUNetForPretraining,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = 'cuda',
    save_dir: str = './pretrained_models',
    save_every: int = 10,
):
    """
    Pre-train TransUNet with masked token prediction
    """
    import os
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Calculate accuracy on masked positions
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)
            mask = (labels != -100)
            acc = (preds[mask] == labels[mask]).float().mean()
            
            train_loss += loss.item()
            train_acc += acc.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}"
            })
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # ===== VALIDATION =====
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, labels)
                    loss = outputs['loss']
                    
                    logits = outputs['logits']
                    preds = logits.argmax(dim=-1)
                    mask = (labels != -100)
                    acc = (preds[mask] == labels[mask]).float().mean()
                    
                    val_loss += loss.item()
                    val_acc += acc.item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"✓ Saved best model (val_loss={val_loss:.4f})")
        
        else:
            print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        scheduler.step()
    
    print(f"\n{'='*60}")
    print("Pre-training completed!")
    print(f"{'='*60}")


def load_pretrained_weights(
    model: TransUNet,
    pretrained_path: str,
    strict: bool = False
):
    """
    Load pre-trained encoder weights into TransUNet for fine-tuning
    
    Args:
        model: TransUNet model for fine-tuning
        pretrained_path: Path to pre-trained checkpoint
        strict: Whether to strictly match all parameters
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract encoder weights from pre-training model
    pretrained_dict = checkpoint['model_state_dict']
    
    # Filter out prediction head and only keep encoder/transformer
    encoder_dict = {}
    for k, v in pretrained_dict.items():
        # Skip prediction head
        if 'prediction_head' in k:
            continue
        # Remove 'transunet.' prefix if present
        if k.startswith('transunet.'):
            k = k.replace('transunet.', '')
        encoder_dict[k] = v
    
    # Load into model
    model.load_state_dict(encoder_dict, strict=strict)
    
    print(f"✓ Loaded pre-trained weights from {pretrained_path}")
    print(f"  - Loaded {len(encoder_dict)} parameter tensors")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("TransUNet Self-Supervised Pre-training")
    print("="*60)
    
    # Dummy intron sequences
    sequences = [
        "GTAAGTATGCATGCTAGCTAGCTAGCTAGCTGACCTAG" * 25,  # ~1000 bp
        "ACCTTAGCGATCGATCGATCGATCGATCGATCGGTAAGT" * 25,
    ] * 100
    
    # Create dataset
    dataset = MaskedSequenceDataset(
        sequences=sequences,
        seq_length=1000,
        mask_prob=0.15,
        mask_donor_acceptor=True,
        donor_motif="GTAAGT",
        acceptor_motif="ACCTAG",
        augment=True
    )
    
    print(f"\nDataset: {len(dataset)} sequences")
    print(f"Sequence length: {dataset.seq_length}")
    print(f"Masking probability: {dataset.mask_prob}")
    
    # Test dataset
    sample = dataset[0]
    print(f"\nSample batch keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Masked positions: {sample['mask_positions'].sum().item()} / {dataset.seq_length}")
    
    print("\n" + "="*60)
    print("✓ Pre-training setup ready!")
    print("="*60)