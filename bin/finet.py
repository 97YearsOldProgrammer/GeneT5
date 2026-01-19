import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data   import DataLoader, random_split
from transformers       import get_cosine_schedule_with_warmup


from lib import train as lib_train
from lib import tuning
from lib.model      import GeneT5
from lib.tokenizer  import GeneTokenizer


# Training Parameters
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
    "val_split":      0.2,
}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GeneT5")

    # data - multiple jsonl files that get mixed
    parser.add_argument("train_data", type=str, nargs="+",
        help="Training data paths (jsonl). Multiple files get randomly mixed.")
    parser.add_argument("output_dir", type=str,
        help="Output directory for checkpoints and final model.")

    # model
    parser.add_argument("model_path", type=str,
        help="Path to pretrained GeneT5 model directory.")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Resume from checkpoint path.")

    # training params (override defaults)
    parser.add_argument("--epochs", type=int, 
        default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, 
        default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, 
        default=DEFAULTS["lr"])
    parser.add_argument("--max_input_len", type=int, 
        default=DEFAULTS["max_input_len"])
    parser.add_argument("--max_target_len", type=int, 
        default=DEFAULTS["max_target_len"])
    parser.add_argument("--grad_accum", type=int, 
        default=DEFAULTS["grad_accum"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"],
        help="Weight decay for AdamW optimizer. Default: 0.1")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"],
        help="Ratio of total steps for learning rate warmup. Default: 0.1 (10%)")
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"],
        help="Maximum gradient norm for clipping. Default: 1.0")
    parser.add_argument("--bucket_size", type=int, default=DEFAULTS["bucket_size"],
        help="Bucket size for smart batch sampling. Default: 256")
    parser.add_argument("--val_split", type=float, 
        default=DEFAULTS["val_split"],
        help="Fraction of data to use for validation (0.0-1.0). Default: 0.2 (20%)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility (shuffling, dropout, etc).")
    parser.add_argument("--save_every", type=int, default=1,
        help="Save checkpoint every N epochs.")
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"],
        help="Number of dataloader workers. Use 0 for large datasets.")
    parser.add_argument("--early_stopping", type=int, default=None,
        help="Stop training if validation loss doesn't improve for N epochs.")
    args = parser.parse_args()

    # Validate val_split
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {args.val_split}")

    # setup
    print(f"\n{' GeneT5 Fine-Tuning ':=^60}")
    device = lib_train.get_device()
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
    model_path  = Path(args.model_path)
    tokenizer   = GeneTokenizer(model_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # load model in FP32 (training uses BF16 autocast)
    print(f"\nLoading model in FP32...")
    model = GeneT5.from_pretrained(model_path, device=device, dtype=torch.float32)
    stats = model.get_param_stats()
    print(f"  Trainable: {stats['total_trainable']:,}")
    print(f"  Frozen:    {stats['total_frozen']:,}")

    # load dataset with lazy loading
    print(f"\nLoading training data (lazy mode)...")
    print(f"  Files: {args.train_data}")

    full_dataset = tuning.LazyDataset(
        args.train_data,
        tokenizer,
        args.max_input_len,
        args.max_target_len,
    )

    # Split into train/val
    if args.val_split > 0:
        val_size   = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        print(f"\n  Split:")
        print(f"    Train samples: {len(train_dataset)} ({100*(1-args.val_split):.1f}%)")
        print(f"    Val samples:   {len(val_dataset)} ({100*args.val_split:.1f}%)")
        
        train_lengths = [full_dataset.lengths[i] for i in train_dataset.indices]
        val_lengths   = [full_dataset.lengths[i] for i in val_dataset.indices]
    else:
        train_dataset = full_dataset
        val_dataset   = None
        train_lengths = full_dataset.lengths
        val_lengths   = None
        print(f"  No validation split (val_split=0)")

    # dataloader with smart batching
    print(f"\nSetting up dataloaders...")

    collator = tuning.DynamicPaddingCollator(
        tokenizer.pad_token_id,
        -100,
    )

    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=tuning.SmartBatchSampler(
            train_lengths,
            args.batch_size,
            DEFAULTS["bucket_size"],
            True,
            True,
        ),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0), 
    )
    print(f"  Train batches: {len(train_loader)}")

    # Validation loader
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=tuning.SmartBatchSampler(
                val_lengths,
                args.batch_size,
                DEFAULTS["bucket_size"],
                False,
                False,
            ),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"  Val batches:   {len(val_loader)}")

    # optimizer & scheduler
    print(f"\nSetting up optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=DEFAULTS["weight_decay"],
    )

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * DEFAULTS["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps, 
        total_steps
    )

    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # resume checkpoint
    start_epoch    = 0
    best_val_loss  = float('inf')
    patience_counter = 0
    
    if args.checkpoint:
        checkpoint_data = lib_train.load_checkpoint(
            model, optimizer, scheduler, args.checkpoint, device
        )
        start_epoch   = checkpoint_data["epoch"]
        best_val_loss = checkpoint_data.get("best_val_loss", float('inf'))
        print(f"  Resumed from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.4f}")

    # save config
    config = {
        **vars(args), 
        "vocab_size":     len(tokenizer), 
        "total_steps":    total_steps,
        "train_samples":  len(train_dataset),
        "val_samples":    len(val_dataset) if val_dataset else 0,
    }
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # training loop
    print(f"\n{'=' * 60}")
    print("Training...")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = lib_train.train_epoch_seq2seq(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, DEFAULTS["max_grad_norm"]
        )
        
        # Validate
        val_loss = None
        if val_loader is not None:
            val_loss = lib_train.evaluate_seq2seq(model, val_loader, device)
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                
                print(f"  ✓ New best validation loss! Saving best_model.pt")
                lib_train.save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    output_dir / "best_model.pt", 
                    {**config, "best_val_loss": best_val_loss, "best_epoch": epoch + 1}
                )
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.early_stopping if args.early_stopping else '∞'})")
                
                if args.early_stopping and patience_counter >= args.early_stopping:
                    print(f"\n  Early stopping triggered! No improvement for {args.early_stopping} epochs.")
                    break
        else:
            print(f"  Train Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            lib_train.save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                output_dir / "checkpoint_latest.pt", 
                {**config, "best_val_loss": best_val_loss}
            )
            print(f"  Saved checkpoint_latest.pt")

    # save final
    print(f"\n{'=' * 60}")
    print(f"Saving final model...")
    lib_train.save_checkpoint(
        model, optimizer, scheduler, args.epochs, 
        output_dir / "final_model.pt", 
        {**config, "best_val_loss": best_val_loss}
    )
    model.save(output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)

    # copy model config
    with open(model_path / "config.json") as f:
        model_config = json.load(f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Fine-tuning complete!")
    print(f"  Output: {output_dir}")
    if val_loader is not None:
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best model saved as: best_model.pt")
    print(f"  Final model saved as: final_model.pt")
    print(f"  Latest checkpoint: checkpoint_latest.pt")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()