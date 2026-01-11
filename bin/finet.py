#!/usr/bin/env python3

import argparse
import torch
import json
import random

from pathlib          import Path
from torch.utils.data import DataLoader
from transformers     import AutoTokenizer

from lib.model  import GeneT5
import lib.train as train
from lib import tuning


parser = argparse.ArgumentParser(
    description="Fine-tune GeneT5 model on gene prediction task.")

# data
parser.add_argument("--train_data", type=str, required=True,
    help="Path to training data (jsonl).")
parser.add_argument("--val_data", type=str, default=None,
    help="Path to validation data (jsonl).")
parser.add_argument("--output_dir", type=str, default="outputs/finetune",
    help="Output directory for checkpoints.")

# model
parser.add_argument("--model_path", type=str, required=True,
    help="Path to pretrained GeneT5 checkpoint directory.")
parser.add_argument("--checkpoint", type=str, default=None,
    help="Resume from checkpoint.")
parser.add_argument("--task", type=str, default="gene_prediction",
    choices=["gene_prediction", "classification"],
    help="Fine-tuning task.")

# GPT-3 style fine-tuning params
parser.add_argument("--epochs", type=int,   default=3,
    help="Number of epochs (GPT-3: 2-3).")
parser.add_argument("--batch_size", type=int,   default=64,
    help="Batch size (GPT-3: 64-128).")
parser.add_argument("--lr", type=float, default=1e-5,
    help="Learning rate (GPT-3: 1e-5 to 2e-5).")
parser.add_argument("--weight_decay", type=float, default=0.1,
    help="Weight decay (GPT-3: 0.1).")
parser.add_argument("--beta1", type=float, default=0.9,
    help="AdamW beta1 (GPT-3: 0.9).")
parser.add_argument("--beta2", type=float, default=0.95,
    help="AdamW beta2 (GPT-3: 0.95).")
parser.add_argument("--warmup_ratio", type=float, default=0.1,
    help="Warmup ratio.")
parser.add_argument("--scheduler", type=str,   default="cosine",
    choices=["cosine", "linear"],
    help="LR scheduler (GPT-3: cosine or linear).")
parser.add_argument("--grad_accum", type=int,   default=1,
    help="Gradient accumulation steps.")
parser.add_argument("--max_grad_norm", type=float, default=1.0,
    help="Max gradient norm for clipping.")
parser.add_argument("--dropout", type=float, default=0.1,
    help="Dropout rate (GPT-3: 0.1).")

# sequence lengths
parser.add_argument("--max_input_len",  type=int, default=4096,
    help="Max input length (GPT-3: 4096).")
parser.add_argument("--max_target_len", type=int, default=2048,
    help="Max target length.")

# batching
parser.add_argument("--bucket_size", type=int, default=256,
    help="Bucket size for smart batching.")
parser.add_argument("--num_workers", type=int, default=4,
    help="DataLoader workers.")

# misc
parser.add_argument("--seed",           type=int,   default=42,
    help="Random seed.")
parser.add_argument("--save_every",     type=int,   default=1,
    help="Save checkpoint every N epochs.")
parser.add_argument("--freeze_encoder", action="store_true",
    help="Freeze encoder weights.")

args = parser.parse_args()


# Visual separator
print(f"\n{' GeneT5 Fine-Tuning ':=^60}")

device = train.get_device()
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
train_dataset = tuning.FineTuneDataset(
    args.train_data, tokenizer,
    args.max_input_len, args.max_target_len, args.task
)
print(f"    Train samples: {len(train_dataset)}")

val_dataset = None
if args.val_data:
    val_dataset = tuning.FineTuneDataset(
        args.val_data, tokenizer,
        args.max_input_len, args.max_target_len, args.task
    )
    print(f"    Val samples:   {len(val_dataset)}")


# smart batch sampler
print(f"\n[4] Setting up dataloaders...")
train_sampler = tuning.SmartBatchSampler(
    lengths     = train_dataset.lengths,
    batch_size  = args.batch_size,
    bucket_size = args.bucket_size,
    drop_last   = True,
    shuffle     = True,
)

collator = tuning.DynamicPaddingCollator(
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
    val_sampler = tuning.SmartBatchSampler(
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
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
else:
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"    Total steps:  {total_steps}")
print(f"    Warmup steps: {warmup_steps}")
print(f"    Scheduler:    {args.scheduler}")


# resume from checkpoint
start_epoch = 0
if args.checkpoint:
    start_epoch = train.load_checkpoint(model, optimizer, scheduler, args.checkpoint, device)


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
    train_loss = train.train_epoch_seq2seq(
        model, train_loader, optimizer, scheduler,
        device, args.grad_accum, args.max_grad_norm
    )
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  LR:         {scheduler.get_last_lr()[0]:.2e}")
    
    # eval
    val_loss = None
    if val_loader:
        val_loss = train.evaluate_seq2seq(model, val_loader, device)
        print(f"  Val loss:   {val_loss:.4f}")
    
    # save checkpoint
    if (epoch + 1) % args.save_every == 0:
        ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        train.save_checkpoint(model, optimizer, scheduler, epoch + 1, ckpt_path, config)
    
    # save best
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path     = output_dir / "best_model.pt"
        train.save_checkpoint(model, optimizer, scheduler, epoch + 1, best_path, config)
        print(f"  New best model (val_loss: {best_val_loss:.4f})")


# save final
print(f"\n[7] Saving final model...")
final_path = output_dir / "final_model.pt"
train.save_checkpoint(model, optimizer, scheduler, args.epochs, final_path, config)

# save model weights separately for easy loading
model.save(output_dir / "pytorch_model.bin")
tokenizer.save_pretrained(output_dir)

# copy config
with open(model_path / "config.json", "r") as f:
    model_config = json.load(f)
with open(output_dir / "config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Fine-tuning complete!")
print(f"  Output: {output_dir}")
print(f"{'=' * 60}")