#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lib.train as train
from lib import tuning
from lib.model import GeneT5
from lib.tokenizer import GeneTokenizer


DEFAULTS = {
    "max_input_len":  4096,
    "max_target_len": 2048,
    "batch_size":     64,
    "lr":             1e-5,
    "epochs":         3,
    "weight_decay":   0.1,
    "warmup_ratio":   0.1,
    "grad_accum":     1,
    "max_grad_norm":  1.0,
    "dropout":        0.1,
    "bucket_size":    256,
    "num_workers":    4,
}


parser = argparse.ArgumentParser(description="Fine-tune GeneT5")

# data - support multiple paths
parser.add_argument("--train_data", type=str, required=True, nargs="+",
    help="Training data paths (jsonl). Can pass multiple for mixed tasks.")
parser.add_argument("--val_data", type=str, default=None, nargs="*",
    help="Validation data paths.")
parser.add_argument("--output_dir", type=str, default="outputs/finetune")

# model
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--freeze_encoder", action="store_true")

# training params (override defaults)
parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
parser.add_argument("--max_input_len", type=int, default=DEFAULTS["max_input_len"])
parser.add_argument("--max_target_len", type=int, default=DEFAULTS["max_target_len"])
parser.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_every", type=int, default=1)
args = parser.parse_args()


# setup
print(f"\n{' GeneT5 Fine-Tuning ':=^60}")
device = train.get_device()
print(f"Device: {device}")

# seed
torch.manual_seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# load tokenizer
print(f"\nLoading tokenizer...")
model_path = Path(args.model_path)
tokenizer = GeneTokenizer(model_path)
print(f"  Vocab size: {len(tokenizer)}")


# load model
print(f"\nLoading model...")
model = GeneT5.from_pretrained(model_path, device="cpu").to(device)

if args.freeze_encoder:
    model.freeze_encoder()
    print("  Encoder frozen")

stats = model.get_param_stats()
print(f"  Trainable: {stats['total_trainable']:,}")
print(f"  Frozen:    {stats['total_frozen']:,}")


# load datasets
print(f"\nLoading datasets...")

train_dataset = tuning.MixedTaskDataset(
    data_paths     = args.train_data,
    tokenizer      = tokenizer,
    max_input_len  = args.max_input_len,
    max_target_len = args.max_target_len,
)
print(f"  Train samples: {len(train_dataset)}")

val_dataset = None
if args.val_data:
    val_dataset = tuning.MixedTaskDataset(
        data_paths     = args.val_data,
        tokenizer      = tokenizer,
        max_input_len  = args.max_input_len,
        max_target_len = args.max_target_len,
    )
    print(f"  Val samples: {len(val_dataset)}")


# dataloaders with smart batching
print(f"\nSetting up dataloaders...")

collator = tuning.DynamicPaddingCollator(
    pad_token_id = tokenizer.pad_token_id,
    label_pad    = -100,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler = tuning.SmartBatchSampler(
        lengths     = train_dataset.lengths,
        batch_size  = args.batch_size,
        bucket_size = DEFAULTS["bucket_size"],
        drop_last   = True,
        shuffle     = True,
    ),
    collate_fn  = collator,
    num_workers = DEFAULTS["num_workers"],
    pin_memory  = True,
)
print(f"  Train batches: {len(train_loader)}")

val_loader = None
if val_dataset:
    val_loader = DataLoader(
        val_dataset,
        batch_sampler = tuning.SmartBatchSampler(
            lengths     = val_dataset.lengths,
            batch_size  = args.batch_size,
            bucket_size = DEFAULTS["bucket_size"],
            drop_last   = False,
            shuffle     = False,
        ),
        collate_fn  = collator,
        num_workers = DEFAULTS["num_workers"],
        pin_memory  = True,
    )
    print(f"  Val batches: {len(val_loader)}")


# optimizer & scheduler
print(f"\nSetting up optimizer...")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = args.lr,
    betas        = (0.9, 0.95),
    weight_decay = DEFAULTS["weight_decay"],
)

total_steps  = len(train_loader) * args.epochs // args.grad_accum
warmup_steps = int(total_steps * DEFAULTS["warmup_ratio"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps, eta_min=args.lr * 0.1
)

print(f"  Total steps:  {total_steps}")
print(f"  Warmup steps: {warmup_steps}")

# resume checkpoint
start_epoch = 0
if args.checkpoint:
    start_epoch = train.load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)

# save config
config = {**vars(args), "vocab_size": len(tokenizer), "total_steps": total_steps}
with open(output_dir / "finetune_config.json", "w") as f:
    json.dump(config, f, indent=2)


# training loop
print(f"\n{'=' * 60}")
print("Training...")
best_val_loss = float("inf")

for epoch in range(start_epoch, args.epochs):
    print(f"\nEpoch {epoch + 1}/{args.epochs}")
    print("-" * 40)
    
    train_loss = train.train_epoch_seq2seq(
        model, train_loader, optimizer, scheduler,
        device, args.grad_accum, DEFAULTS["max_grad_norm"]
    )
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
    
    val_loss = None
    if val_loader:
        val_loss = train.evaluate_seq2seq(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}")
    
    # save checkpoint
    if (epoch + 1) % args.save_every == 0:
        train.save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            output_dir / f"checkpoint_epoch{epoch + 1}.pt", config
        )
    
    # save best
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        train.save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            output_dir / "best_model.pt", config
        )
        print(f"  ✓ New best (val_loss: {best_val_loss:.4f})")


# save final
print(f"\nSaving final model...")
train.save_checkpoint(model, optimizer, scheduler, args.epochs, output_dir / "final_model.pt", config)
model.save(output_dir / "pytorch_model.bin")
tokenizer.save_pretrained(output_dir)

# copy model config
with open(model_path / "config.json") as f:
    model_config = json.load(f)
with open(output_dir / "config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"\n{'=' * 60}")
print(f"✓ Fine-tuning complete! Output: {output_dir}")
print(f"{'=' * 60}")