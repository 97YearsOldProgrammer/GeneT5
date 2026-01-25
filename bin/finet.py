#!/usr/bin/env python3


import argparse as ap
import json     as js
import os
import random   as rnd
import pathlib  as pl

import torch
from torch.utils.data import Dataset, DataLoader
from transformers     import get_cosine_schedule_with_warmup

import lib.binary   as bi
import lib.dataload as dl


#####################  Constants  #####################


DEFAULTS = {
    "max_input_len":  4096,
    "max_target_len": 2048,
    "batch_size":     4,
    "lr":             1e-4,
    "epochs":         3,
    "weight_decay":   0.1,
    "warmup_ratio":   0.1,
    "grad_accum":     64,
    "max_grad_norm":  1.0,
    "bucket_size":    256,
    "num_workers":    0,
}


#####################  Dataset Classes  #####################


class BinaryTrainDataset(Dataset):
    """PyTorch dataset wrapper for binary training files"""
    
    def __init__(self, binary_path, tokenizer, max_input_len, max_target_len, seed=42):
        
        self.binary_path    = binary_path
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.seed           = seed
        self.epoch          = 0
        
        self._info   = bi.get_binary_info(binary_path)
        self._length = self._info["num_chunks"]
        
        self.lengths = dl.build_length_index(binary_path)
    
    def set_epoch(self, epoch):
        """Set epoch for noise variation"""
        
        self.epoch = epoch
    
    def __len__(self):
        
        return self._length
    
    def __getitem__(self, idx):
        
        rnd.seed(self.seed + self.epoch * len(self) + idx)
        
        reader = dl.BinaryDatasetReader(self.binary_path)
        sample = reader.get_sample(idx)
        
        input_ids  = self.tokenizer.encode(sample["input_text"], add_special_tokens=False)
        target_ids = self.tokenizer.encode(sample["target_text"], add_special_tokens=False)
        
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
        
        if len(target_ids) > self.max_target_len:
            target_ids = target_ids[:self.max_target_len]
        
        return {
            "input_ids":      input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels":         target_ids,
        }


class DynamicPaddingCollator:
    """Collator that pads batches dynamically"""
    
    def __init__(self, pad_token_id, label_pad=-100):
        
        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
    
    def __call__(self, batch):
        
        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if batch[0]["labels"] else 0
        
        input_ids      = []
        attention_mask = []
        labels         = []
        
        for b in batch:
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len
            
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            
            if max_target_len > 0:
                lbl_len = len(b["labels"])
                lbl_pad = max_target_len - lbl_len
                labels.append(b["labels"] + [self.label_pad] * lbl_pad)
        
        result = {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
        
        if labels:
            result["labels"] = torch.tensor(labels)
        
        return result


class SmartBatchSampler:
    """Batch sampler that groups similar-length sequences"""
    
    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False, shuffle=True):
        
        self.lengths        = lengths
        self.batch_size     = batch_size
        self.bucket_size    = bucket_size
        self.drop_last      = drop_last
        self.shuffle        = shuffle
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    
    def __iter__(self):
        
        buckets = []
        bucket  = []
        
        for idx in self.sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []
        
        if bucket:
            buckets.append(bucket)
        
        if self.shuffle:
            rnd.shuffle(buckets)
            for b in buckets:
                rnd.shuffle(b)
        
        all_indices = [idx for bucket in buckets for idx in bucket]
        batches     = []
        
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle:
            rnd.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


#####################  Training Functions  #####################


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum, max_grad_norm):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(
                encoder_input_ids = batch["input_ids"],
                decoder_input_ids = batch["labels"][:, :-1],
                labels            = batch["labels"][:, 1:],
            )
            
            loss = outputs["loss"] / grad_accum
        
        loss.backward()
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    
    model.eval()
    total_loss  = 0
    num_batches = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(
                    encoder_input_ids = batch["input_ids"],
                    decoder_input_ids = batch["labels"][:, :-1],
                    labels            = batch["labels"][:, 1:],
                )
            
            total_loss  += outputs["loss"].item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


#####################  Checkpoint Functions  #####################


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None):
    """Save training checkpoint"""
    
    save_path = pl.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    
    if config:
        checkpoint["config"] = config
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch         = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("config", {}).get("best_val_loss", float('inf'))
    
    print(f"  Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return {"epoch": epoch, "best_val_loss": best_val_loss}


#####################  Main Entry  #####################


def main():
    """Execute fine-tuning pipeline"""
    
    args = parse_args()
    
    print(f"\n{' GeneT5 Fine-Tuning (Binary Input) ':=^60}")
    
    device = get_device()
    print(f"Device: {device}")
    
    set_seeds(args.seed)
    
    output_dir = pl.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = load_tokenizer(args.model_path)
    model     = load_model(args.model_path, device)
    
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    train_loader, val_loader   = create_dataloaders(train_dataset, val_dataset, args, tokenizer)
    
    optimizer, scheduler = setup_optimizer(model, train_loader, args)
    
    start_epoch, best_val_loss = handle_checkpoint(model, optimizer, scheduler, args, device)
    
    save_config(args, tokenizer, train_dataset, val_dataset, output_dir)
    
    run_training(
        model, train_loader, val_loader, optimizer, scheduler,
        device, args, output_dir, start_epoch, best_val_loss, train_dataset
    )
    
    save_final(model, tokenizer, args, output_dir, best_val_loss)


#####################  Argument Parsing  #####################


def parse_args():
    """Parse command line arguments"""
    
    parser = ap.ArgumentParser(description="Fine-tune GeneT5 with binary data")
    
    parser.add_argument("train_bin",  help="Training binary file path")
    parser.add_argument("val_bin",    help="Validation binary file path")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("model_path", help="Path to pretrained model")
    
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max_input_len", type=int, default=DEFAULTS["max_input_len"])
    parser.add_argument("--max_target_len", type=int, default=DEFAULTS["max_target_len"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"])
    parser.add_argument("--bucket_size", type=int, default=DEFAULTS["bucket_size"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--early_stopping", type=int, default=None)
    
    return parser.parse_args()


#####################  Setup Functions  #####################


def get_device():
    """Get appropriate compute device"""
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    
    torch.manual_seed(seed)
    rnd.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_path):
    """Load tokenizer from model path"""
    
    # Import here to avoid top-level dependency
    from lib.tokenizer import GeneTokenizer
    
    print(f"\nLoading tokenizer...")
    tokenizer = GeneTokenizer(pl.Path(model_path))
    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer


def load_model(model_path, device):
    """Load model from path"""
    
    # Import here to avoid top-level dependency
    from lib.model import GeneT5
    
    print(f"\nLoading model...")
    model = GeneT5.from_pretrained(pl.Path(model_path), device="cpu", dtype=torch.float32)
    model = model.to(device)
    
    stats = model.get_param_stats()
    print(f"  Trainable: {stats['total_trainable']:,}")
    print(f"  Frozen:    {stats['total_frozen']:,}")
    
    return model


def load_datasets(args, tokenizer):
    """Load training and validation datasets"""
    
    print(f"\nLoading datasets...")
    print(f"  Training:   {args.train_bin}")
    print(f"  Validation: {args.val_bin}")
    
    train_dataset = BinaryTrainDataset(
        args.train_bin,
        tokenizer,
        args.max_input_len,
        args.max_target_len,
        args.seed,
    )
    
    val_dataset = BinaryTrainDataset(
        args.val_bin,
        tokenizer,
        args.max_input_len,
        args.max_target_len,
        args.seed,
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, args, tokenizer):
    """Create data loaders with smart batching"""
    
    print(f"\nSetting up dataloaders...")
    
    collator = DynamicPaddingCollator(tokenizer.pad_token_id, -100)
    
    train_sampler = SmartBatchSampler(
        train_dataset.lengths,
        args.batch_size,
        args.bucket_size,
        drop_last = True,
        shuffle   = True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler = train_sampler,
        collate_fn    = collator,
        num_workers   = args.num_workers,
        pin_memory    = True,
    )
    
    val_sampler = SmartBatchSampler(
        val_dataset.lengths,
        args.batch_size,
        args.bucket_size,
        drop_last = False,
        shuffle   = False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler = val_sampler,
        collate_fn    = collator,
        num_workers   = args.num_workers,
        pin_memory    = True,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    return train_loader, val_loader


def setup_optimizer(model, train_loader, args):
    """Setup optimizer and scheduler"""
    
    print(f"\nSetting up optimizer...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        betas        = (0.9, 0.95),
        weight_decay = args.weight_decay,
    )
    
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    return optimizer, scheduler


def handle_checkpoint(model, optimizer, scheduler, args, device):
    """Handle checkpoint resumption"""
    
    start_epoch   = 0
    best_val_loss = float('inf')
    
    if args.checkpoint:
        data          = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)
        start_epoch   = data["epoch"]
        best_val_loss = data.get("best_val_loss", float('inf'))
    
    return start_epoch, best_val_loss


def save_config(args, tokenizer, train_dataset, val_dataset, output_dir):
    """Save training configuration"""
    
    config = {
        **vars(args),
        "vocab_size":    len(tokenizer),
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
    }
    
    with open(output_dir / "finetune_config.json", "w") as f:
        js.dump(config, f, indent=2)


#####################  Training Loop  #####################


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 device, args, output_dir, start_epoch, best_val_loss, train_dataset):
    """Execute training loop"""
    
    print(f"\n{'=' * 60}")
    print("Training...")
    print(f"{'=' * 60}")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        train_dataset.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, args.max_grad_norm
        )
        
        val_loss = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            
            print(f"  ✓ New best validation loss! Saving best_model.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                output_dir / "best_model.pt",
                {"best_val_loss": best_val_loss, "best_epoch": epoch + 1}
            )
        else:
            patience_counter += 1
            es_str = args.early_stopping if args.early_stopping else '∞'
            print(f"  No improvement ({patience_counter}/{es_str})")
            
            if args.early_stopping and patience_counter >= args.early_stopping:
                print(f"\n  Early stopping triggered!")
                break
        
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                output_dir / "checkpoint_latest.pt",
                {"best_val_loss": best_val_loss}
            )
            print(f"  Saved checkpoint_latest.pt")


def save_final(model, tokenizer, args, output_dir, best_val_loss):
    """Save final model and artifacts"""
    
    print(f"\n{'=' * 60}")
    print(f"Saving final model...")
    
    model.save(output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    
    model_path = pl.Path(args.model_path)
    with open(model_path / "config.json") as f:
        model_config = js.load(f)
    with open(output_dir / "config.json", "w") as f:
        js.dump(model_config, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Fine-tuning complete!")
    print(f"  Output: {output_dir}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}")


#####################  Script Entry  #####################


if __name__ == "__main__":
    main()
