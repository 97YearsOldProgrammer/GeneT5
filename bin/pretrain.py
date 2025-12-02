"""
Pretraining Script for 1D GeneTransUNet

Pretraining objectives:
1. Masked Language Modeling (MLM) - predict masked k-mer tokens
2. Next Sentence Prediction (NSP) - predict intron between CDS segments
3. State Change Prediction - predict state transitions in transcripts

Usage:
    python pretrain.py --gff_file data/wormbase.gff3 --fasta_files data/*.fa \
                       --output_dir checkpoints/pretrain --epochs 100
"""

import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

from model_1d import GeneTransUNet1DPretraining, count_parameters
from data_utils import (
    KmerTokenizer, WormBaseDataset, CombinedPretrainingDataset,
    pretrain_collate_fn
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


######################
#### Loss Functions ##
######################


class PretrainingLoss(nn.Module):
    """
    Combined loss for pretraining:
    - MLM: Cross-entropy for masked token prediction
    - NSP: Binary cross-entropy for intron prediction
    - State: Cross-entropy for state change prediction
    """
    
    def __init__(
        self,
        mlm_weight: float = 1.0,
        nsp_weight: float = 0.5,
        state_weight: float = 0.5,
        vocab_size: int = 4096,
    ):
        super().__init__()
        
        self.mlm_weight = mlm_weight
        self.nsp_weight = nsp_weight
        self.state_weight = state_weight
        
        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss = nn.CrossEntropyLoss()
        self.state_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(
        self,
        mlm_logits: torch.Tensor,
        mlm_labels: torch.Tensor,
        nsp_logits: Optional[torch.Tensor] = None,
        nsp_labels: Optional[torch.Tensor] = None,
        state_logits: Optional[torch.Tensor] = None,
        state_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Returns:
            Dictionary with total_loss and individual losses
        """
        losses = {}
        total_loss = 0.0
        
        # MLM loss
        if mlm_logits is not None and mlm_labels is not None:
            # mlm_logits: (B, L, vocab_size)
            # mlm_labels: (B, L)
            B, L, V = mlm_logits.shape
            mlm_loss = self.mlm_loss(
                mlm_logits.view(B * L, V),
                mlm_labels.view(B * L)
            )
            losses['mlm_loss'] = mlm_loss
            total_loss = total_loss + self.mlm_weight * mlm_loss
        
        # NSP loss
        if nsp_logits is not None and nsp_labels is not None:
            nsp_loss = self.nsp_loss(nsp_logits, nsp_labels)
            losses['nsp_loss'] = nsp_loss
            total_loss = total_loss + self.nsp_weight * nsp_loss
        
        # State change loss
        if state_logits is not None and state_labels is not None:
            # state_logits: (B, N_states, num_states)
            # state_labels: (B, N_states)
            B, N, C = state_logits.shape
            state_loss = self.state_loss(
                state_logits.view(B * N, C),
                state_labels.view(B * N)
            )
            losses['state_loss'] = state_loss
            total_loss = total_loss + self.state_weight * state_loss
        
        losses['total_loss'] = total_loss
        return losses


######################
#### Metrics #########
######################


def compute_mlm_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute MLM accuracy (only on masked positions)"""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_nsp_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute NSP accuracy"""
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    return accuracy.item()


def compute_state_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute state prediction accuracy"""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


######################
#### Training Loop ###
######################


class Trainer:
    """Trainer class for pretraining"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: PretrainingLoss,
        device: torch.device,
        output_dir: str,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.scaler = GradScaler() if mixed_precision else None
        
        # Metrics tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'mlm_loss': 0.0,
            'nsp_loss': 0.0,
            'state_loss': 0.0,
        }
        epoch_metrics = {
            'mlm_acc': 0.0,
            'nsp_acc': 0.0,
            'state_acc': 0.0,
        }
        
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_onehot = batch['input_onehot'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device) if batch['mlm_labels'] is not None else None
            
            # Optional NSP data
            sep_positions = batch.get('sep_positions')
            nsp_labels = batch.get('nsp_labels')
            if sep_positions is not None:
                sep_positions = sep_positions.to(self.device)
                nsp_labels = nsp_labels.to(self.device)
            
            # Optional state data
            state_positions = batch.get('state_positions')
            state_labels = batch.get('state_labels')
            if state_positions is not None:
                state_positions = state_positions.to(self.device)
                state_labels = state_labels.to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_onehot,
                        sep_positions=sep_positions,
                        state_positions=state_positions,
                    )
                    
                    losses = self.loss_fn(
                        mlm_logits=outputs['mlm_logits'],
                        mlm_labels=mlm_labels,
                        nsp_logits=outputs.get('nsp_logits'),
                        nsp_labels=nsp_labels,
                        state_logits=outputs.get('state_logits'),
                        state_labels=state_labels,
                    )
                
                # Backward pass
                loss = losses['total_loss'] / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_onehot,
                    sep_positions=sep_positions,
                    state_positions=state_positions,
                )
                
                losses = self.loss_fn(
                    mlm_logits=outputs['mlm_logits'],
                    mlm_labels=mlm_labels,
                    nsp_logits=outputs.get('nsp_logits'),
                    nsp_labels=nsp_labels,
                    state_logits=outputs.get('state_logits'),
                    state_labels=state_labels,
                )
                
                loss = losses['total_loss'] / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Compute metrics
            with torch.no_grad():
                if mlm_labels is not None:
                    epoch_metrics['mlm_acc'] += compute_mlm_accuracy(
                        outputs['mlm_logits'], mlm_labels
                    )
                if nsp_labels is not None and 'nsp_logits' in outputs:
                    epoch_metrics['nsp_acc'] += compute_nsp_accuracy(
                        outputs['nsp_logits'], nsp_labels
                    )
                if state_labels is not None and 'state_logits' in outputs:
                    epoch_metrics['state_acc'] += compute_state_accuracy(
                        outputs['state_logits'], state_labels
                    )
            
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {losses['total_loss'].item():.4f}"
                )
        
        # Average metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return {**epoch_losses, **epoch_metrics}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation loop"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'mlm_loss': 0.0,
            'nsp_loss': 0.0,
            'state_loss': 0.0,
        }
        val_metrics = {
            'mlm_acc': 0.0,
            'nsp_acc': 0.0,
            'state_acc': 0.0,
        }
        
        num_batches = 0
        
        for batch in self.val_loader:
            input_onehot = batch['input_onehot'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device) if batch['mlm_labels'] is not None else None
            
            sep_positions = batch.get('sep_positions')
            nsp_labels = batch.get('nsp_labels')
            if sep_positions is not None:
                sep_positions = sep_positions.to(self.device)
                nsp_labels = nsp_labels.to(self.device)
            
            state_positions = batch.get('state_positions')
            state_labels = batch.get('state_labels')
            if state_positions is not None:
                state_positions = state_positions.to(self.device)
                state_labels = state_labels.to(self.device)
            
            outputs = self.model(
                input_onehot,
                sep_positions=sep_positions,
                state_positions=state_positions,
            )
            
            losses = self.loss_fn(
                mlm_logits=outputs['mlm_logits'],
                mlm_labels=mlm_labels,
                nsp_logits=outputs.get('nsp_logits'),
                nsp_labels=nsp_labels,
                state_logits=outputs.get('state_logits'),
                state_labels=state_labels,
            )
            
            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            
            if mlm_labels is not None:
                val_metrics['mlm_acc'] += compute_mlm_accuracy(
                    outputs['mlm_logits'], mlm_labels
                )
            if nsp_labels is not None and 'nsp_logits' in outputs:
                val_metrics['nsp_acc'] += compute_nsp_accuracy(
                    outputs['nsp_logits'], nsp_labels
                )
            if state_labels is not None and 'state_logits' in outputs:
                val_metrics['state_acc'] += compute_state_accuracy(
                    outputs['state_logits'], state_labels
                )
            
            num_batches += 1
        
        for key in val_losses:
            val_losses[key] /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return {**val_losses, **val_metrics}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Full training loop"""
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Resumed from epoch {start_epoch}")
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            logger.info(f"\nEpoch {epoch} Summary (took {epoch_time:.1f}s):")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Train MLM Acc: {train_metrics['mlm_acc']:.4f}")
            
            if val_metrics:
                logger.info(f"  Val Loss: {val_metrics['total_loss']:.4f}")
                logger.info(f"  Val MLM Acc: {val_metrics['mlm_acc']:.4f}")
                
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
            else:
                is_best = train_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = train_metrics['total_loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Save metrics
            metrics_path = self.output_dir / 'metrics.jsonl'
            with open(metrics_path, 'a') as f:
                metrics = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                }
                f.write(json.dumps(metrics) + '\n')
        
        logger.info("Training complete!")
        return self.best_val_loss


######################
#### Main ############
######################


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain 1D GeneTransUNet')
    
    # Data
    parser.add_argument('--gff_file', type=str, required=True,
                        help='Path to GFF3 annotation file')
    parser.add_argument('--fasta_files', type=str, nargs='+', required=True,
                        help='Paths to FASTA genome files')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pretrain',
                        help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='Transformer embedding dimension')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--conv4_depth', type=int, default=22,
                        help='Depth of Conv4_x stage (22 or 23)')
    parser.add_argument('--kmer_size', type=int, default=6,
                        help='K-mer size for tokenization')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='Masking probability')
    
    # Loss weights
    parser.add_argument('--mlm_weight', type=float, default=1.0,
                        help='MLM loss weight')
    parser.add_argument('--nsp_weight', type=float, default=0.5,
                        help='NSP loss weight')
    parser.add_argument('--state_weight', type=float, default=0.5,
                        help='State prediction loss weight')
    
    # Optimization
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = KmerTokenizer(k=args.kmer_size)
    vocab_size = tokenizer.vocab_size
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Data
    logger.info("Loading data...")
    dataset = WormBaseDataset(args.gff_file, args.fasta_files)
    
    # Get pretraining dataset
    pretrain_dataset = dataset.get_pretraining_datasets(
        tokenizer, mask_prob=args.mask_prob, max_length=args.seq_length
    )
    
    # Split into train/val
    total_size = len(pretrain_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        pretrain_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
    )
    
    # Model
    model = GeneTransUNet1DPretraining(
        seq_length=args.seq_length,
        in_channels=4,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        conv4_depth=args.conv4_depth,
    )
    model = model.to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = len(train_loader) * args.warmup_epochs // args.gradient_accumulation_steps
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=1e-6,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps],
    )
    
    # Loss
    loss_fn = PretrainingLoss(
        mlm_weight=args.mlm_weight,
        nsp_weight=args.nsp_weight,
        state_weight=args.state_weight,
        vocab_size=vocab_size,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )
    
    # Save config
    config_path = Path(args.output_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train
    trainer.train(args.epochs, resume_from=args.resume)


if __name__ == '__main__':
    main()