#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import json
import math
import random

from pathlib            import Path
from torch.utils.data   import DataLoader, Dataset, Sampler
from transformers       import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from lib.model  import GeneT5
from lib.tuning import load_dataset


##########################
#####  Smart Batch   #####
##########################


class SmartBatchSampler(Sampler):
    """
    Groups samples by length for efficient batching.
    Minimizes padding by batching similar-length sequences together.
    """
    
    def __init__(self, lengths, batch_size, bucket_size=100, drop_last=False, shuffle=True):
        self.lengths     = lengths
        self.batch_size  = batch_size
        self.bucket_size = bucket_size
        self.drop_last   = drop_last
        self.shuffle     = shuffle
        
        # sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    
    def __iter__(self):
        # create buckets of similar lengths
        buckets = []
        bucket  = []
        
        for idx in self.sorted_indices:
            bucket.append(idx)
            if len(bucket) >= self.bucket_size:
                buckets.append(bucket)
                bucket = []
        
        if bucket:
            buckets.append(bucket)
        
        # shuffle within buckets
        if self.shuffle:
            random.shuffle(buckets)
            for b in buckets:
                random.shuffle(b)
        
        # flatten and create batches
        all_indices = [idx for bucket in buckets for idx in bucket]
        batches     = []
        
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class DynamicPaddingCollator:
    """
    Collates batches with dynamic padding to max length in batch.
    """
    
    def __init__(self, pad_token_id, label_pad=-100):
        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
    
    def __call__(self, batch):
        # find max lengths in batch
        max_input_len  = max(len(b["input_ids"]) for b in batch)
        max_target_len = max(len(b["labels"]) for b in batch) if "labels" in batch[0] else 0
        
        input_ids      = []
        attention_mask = []
        labels         = []
        
        for b in batch:
            # pad input
            inp_len = len(b["input_ids"])
            pad_len = max_input_len - inp_len
            
            input_ids.append(
                torch.cat([b["input_ids"], torch.full((pad_len,), self.pad_token_id)])
            )
            attention_mask.append(
                torch.cat([b["attention_mask"], torch.zeros(pad_len)])
            )
            
            # pad labels if present
            if "labels" in b:
                lbl_len = len(b["labels"])
                lbl_pad = max_target_len - lbl_len
                labels.append(
                    torch.cat([b["labels"], torch.full((lbl_pad,), self.label_pad)])
                )
        
        result = {
            "input_ids":      torch.stack(input_ids).long(),
            "attention_mask": torch.stack(attention_mask).long(),
        }
        
        if labels:
            result["labels"] = torch.stack(labels).long()
        
        return result


##########################
#####  Dataset       #####
##########################


class FineTuneDataset(Dataset):
    """
    Dataset for fine-tuning with precomputed lengths for smart batching.
    """
    
    def __init__(self, data_path, tokenizer, max_input_len=4096, max_target_len=2048, task="gene_prediction"):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.task           = task
        self.samples        = load_dataset(data_path)
        
        # precompute lengths for smart batching
        self.lengths = []
        for sample in self.samples:
            inp_len = len(tokenizer.encode(sample["input"], add_special_tokens=False))
            self.lengths.append(min(inp_len, max_input_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # tokenize input
        input_enc = self.tokenizer(
            sample["input"],
            max_length     = self.max_input_len,
            truncation     = True,
            return_tensors = "pt",
            padding        = False,
        )
        
        result = {
            "input_ids":      input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
        }
        
        # handle labels based on task
        if self.task == "gene_prediction":
            target_enc = self.tokenizer(
                sample["target"],
                max_length     = self.max_target_len,
                truncation     = True,
                return_tensors = "pt",
                padding        = False,
            )
            labels = target_enc["input_ids"].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels
        else:
            result["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        
        return result


##########################
#####  Training      #####
##########################


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    num_steps  = 0
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # forward
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
            num_steps += 1
        
        total_loss += loss.item() * grad_accum
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                encoder_input_ids = batch["input_ids"],
                decoder_input_ids = batch["labels"][:, :-1],
                labels            = batch["labels"][:, 1:],
            )
            
            total_loss += outputs["loss"].item()
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None):
    """Save training checkpoint."""
    save_path = Path(save_path)
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
    print(f"Saved checkpoint: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch


##########################
#####  Main          #####
##########################


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GeneT5 model")
    
    # data
    parser.add_argument("--train_data",     type=str, required=True,
        help="Path to training data (jsonl).")
    parser.add_argument("--val_data",       type=str, default=None,
        help="Path to validation data (jsonl).")
    parser.add_argument("--output_dir",     type=str, default="outputs/finetune",
        help="Output directory for checkpoints.")
    
    # model
    parser.add_argument("--model_path",     type=str, required=True,
        help="Path to pretrained GeneT5 checkpoint directory.")
    parser.add_argument("--checkpoint",     type=str, default=None,
        help="Resume from checkpoint.")
    parser.add_argument("--task",           type=str, default="gene_prediction",
        choices=["gene_prediction", "classification"],
        help="Fine-tuning task.")
    
    # GPT-3 style fine-tuning params
    parser.add_argument("--epochs",         type=int,   default=3,
        help="Number of epochs (GPT-3: 2-3).")
    parser.add_argument("--batch_size",     type=int,   default=64,
        help="Batch size (GPT-3: 64-128).")
    parser.add_argument("--lr",             type=float, default=1e-5,
        help="Learning rate (GPT-3: 1e-5 to 2e-5).")
    parser.add_argument("--weight_decay",   type=float, default=0.1,
        help="Weight decay (GPT-3: 0.1).")
    parser.add_argument("--beta1",          type=float, default=0.9,
        help="AdamW beta1 (GPT-3: 0.9).")
    parser.add_argument("--beta2",          type=float, default=0.95,
        help="AdamW beta2 (GPT-3: 0.95).")
    parser.add_argument("--warmup_ratio",   type=float, default=0.1,
        help="Warmup ratio.")
    parser.add_argument("--scheduler",      type=str,   default="cosine",
        choices=["cosine", "linear"],
        help="LR scheduler (GPT-3: cosine or linear).")
    parser.add_argument("--grad_accum",     type=int,   default=1,
        help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm",  type=float, default=1.0,
        help="Max gradient norm for clipping.")
    parser.add_argument("--dropout",        type=float, default=0.1,
        help="Dropout rate (GPT-3: 0.1).")
    
    # sequence lengths
    parser.add_argument("--max_input_len",  type=int,   default=4096,
        help="Max input length (GPT-3: 4096).")
    parser.add_argument("--max_target_len", type=int,   default=2048,
        help="Max target length.")
    
    # batching
    parser.add_argument("--bucket_size",    type=int,   default=256,
        help="Bucket size for smart batching.")
    parser.add_argument("--num_workers",    type=int,   default=4,
        help="DataLoader workers.")
    
    # misc
    parser.add_argument("--seed",           type=int,   default=42,
        help="Random seed.")
    parser.add_argument("--save_every",     type=int,   default=1,
        help="Save checkpoint every N epochs.")
    parser.add_argument("--freeze_encoder", action="store_true",
        help="Freeze encoder weights.")
    
    # wandb
    parser.add_argument("--wandb",          action="store_true",
        help="Enable wandb logging.")
    parser.add_argument("--wandb_project",  type=str,   default="genet5-finetune",
        help="Wandb project name.")
    parser.add_argument("--wandb_run",      type=str,   default=None,
        help="Wandb run name.")
    
    return parser.parse_args()


def setup_wandb(args):
    """Initialize weights & biases."""
    try:
        import wandb
        
        wandb.init(
            project = args.wandb_project,
            name    = args.wandb_run,
            config  = vars(args),
        )
        return wandb
    except ImportError:
        print("wandb not installed, skipping logging")
        return None


def main():
    args   = parse_args()
    device = get_device()
    
    print(f"\n{'=' * 60}")
    print(f"GeneT5 Fine-Tuning")
    print(f"{'=' * 60}")
    print(f"Device:     {device}")
    print(f"Task:       {args.task}")
    print(f"Model:      {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print(f"Epochs:     {args.epochs}")
    
    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb
    wandb = None
    if args.wandb:
        wandb = setup_wandb(args)
    
    # load tokenizer
    print(f"\n[1] Loading tokenizer...")
    model_path = Path(args.model_path)
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    print(f"    Vocab size: {len(tokenizer)}")
    print(f"    Pad token:  {tokenizer.pad_token}")
    
    # load model
    print(f"\n[2] Loading model...")
    model = GeneT5.from_pretrained(model_path, device="cpu")
    model = model.to(device)
    
    if args.freeze_encoder:
        model.freeze_encoder()
        print("    Encoder frozen")
    
    stats = model.get_param_stats()
    print(f"    Trainable params: {stats['total_trainable']:,}")
    print(f"    Frozen params:    {stats['total_frozen']:,}")
    
    # datasets
    print(f"\n[3] Loading datasets...")
    train_dataset = FineTuneDataset(
        args.train_data, tokenizer,
        args.max_input_len, args.max_target_len, args.task
    )
    print(f"    Train samples: {len(train_dataset)}")
    
    val_dataset = None
    if args.val_data:
        val_dataset = FineTuneDataset(
            args.val_data, tokenizer,
            args.max_input_len, args.max_target_len, args.task
        )
        print(f"    Val samples:   {len(val_dataset)}")
    
    # smart batch sampler
    print(f"\n[4] Setting up dataloaders...")
    train_sampler = SmartBatchSampler(
        lengths     = train_dataset.lengths,
        batch_size  = args.batch_size,
        bucket_size = args.bucket_size,
        drop_last   = True,
        shuffle     = True,
    )
    
    collator = DynamicPaddingCollator(
        pad_token_id = tokenizer.pad_token_id,
        label_pad    = -100,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler = train_sampler,
        collate_fn    = collator,
        num_workers   = args.num_workers,
        pin_memory    = True,
    )
    print(f"    Train batches: {len(train_loader)}")
    
    val_loader = None
    if val_dataset:
        val_sampler = SmartBatchSampler(
            lengths     = val_dataset.lengths,
            batch_size  = args.batch_size,
            bucket_size = args.bucket_size,
            drop_last   = False,
            shuffle     = False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler = val_sampler,
            collate_fn    = collator,
            num_workers   = args.num_workers,
            pin_memory    = True,
        )
        print(f"    Val batches:   {len(val_loader)}")
    
    # optimizer (GPT-3 style: AdamW with β1=0.9, β2=0.95)
    print(f"\n[5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        betas        = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
    )
    
    # scheduler
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    print(f"    Total steps:  {total_steps}")
    print(f"    Warmup steps: {warmup_steps}")
    print(f"    Scheduler:    {args.scheduler}")
    
    # resume from checkpoint
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)
    
    # save config
    config = vars(args)
    config["vocab_size"]  = len(tokenizer)
    config["total_steps"] = total_steps
    
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # training loop
    print(f"\n[6] Training...")
    print(f"{'=' * 60}")
    
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, args.max_grad_norm
        )
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.2e}")
        
        # eval
        val_loss = None
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            print(f"  Val loss:   {val_loss:.4f}")
        
        # wandb logging
        if wandb:
            log_dict = {
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "lr":         scheduler.get_last_lr()[0],
            }
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)
        
        # save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, ckpt_path, config)
        
        # save best
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path     = output_dir / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_path, config)
            print(f"  New best model (val_loss: {best_val_loss:.4f})")
    
    # save final
    print(f"\n[7] Saving final model...")
    final_path = output_dir / "final_model.pt"
    save_checkpoint(model, optimizer, scheduler, args.epochs, final_path, config)
    
    # save model weights separately for easy loading
    model.save(output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    
    # copy config
    with open(model_path / "config.json", "r") as f:
        model_config = json.load(f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    if wandb:
        wandb.finish()
    
    print(f"\n{'=' * 60}")
    print(f"Fine-tuning complete!")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()