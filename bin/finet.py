#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data   import DataLoader, random_split
from transformers       import get_cosine_schedule_with_warmup

from lib import train as lib_train
from lib import dataset
from lib.model     import GeneT5
from lib.tokenizer import GeneTokenizer


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
    parser = argparse.ArgumentParser(description="Fine-tune GeneT5 with hint-based noising")

    # Data
    parser.add_argument("train_data", type=str, nargs="+",
        help="Training data paths (JSONL files).")
    parser.add_argument("output_dir", type=str,
        help="Output directory for checkpoints and final model.")
    parser.add_argument("model_path", type=str,
        help="Path to pretrained GeneT5 model directory.")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Resume from checkpoint path.")

    # Noising
    parser.add_argument("--enable_noising", action="store_true", default=True,
        help="Enable hint-based noising (default: True).")
    parser.add_argument("--no_noising", action="store_false", dest="enable_noising",
        help="Disable noising (train on raw data).")
    parser.add_argument("--hint_token", type=str, default="[HIT]",
        help="Token to mark hint section in input.")
    
    # Noising scenario weights (sum should be ~1.0)
    parser.add_argument("--scenario_full_mix", type=float, default=0.40,
        help="Weight for full_mix scenario (intron + CDS hints).")
    parser.add_argument("--scenario_intron_only", type=float, default=0.25,
        help="Weight for intron_only scenario.")
    parser.add_argument("--scenario_cds_only", type=float, default=0.20,
        help="Weight for cds_only scenario.")
    parser.add_argument("--scenario_degraded", type=float, default=0.10,
        help="Weight for degraded scenario (heavy noise).")
    parser.add_argument("--scenario_ab_initio", type=float, default=0.05,
        help="Weight for ab_initio scenario (no hints).")

    # Training params
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
    parser.add_argument("--val_split", type=float, default=DEFAULTS["val_split"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--early_stopping", type=int, default=None)

    args = parser.parse_args()

    # Validate
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {args.val_split}")

    # Setup
    print(f"\n{' GeneT5 Fine-Tuning ':=^60}")
    device = lib_train.get_device()
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    model_path = Path(args.model_path)
    tokenizer  = GeneTokenizer(model_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # Load model
    print(f"\nLoading model in FP32...")
    model = GeneT5.from_pretrained(model_path, device=device, dtype=torch.float32)
    stats = model.get_param_stats()
    print(f"  Trainable: {stats['total_trainable']:,}")
    print(f"  Frozen:    {stats['total_frozen']:,}")

    # Setup noising config
    noising_config = None
    if args.enable_noising:
        noising_config = dataset.NoisingConfig(
            scenario_weights={
                'full_mix':    args.scenario_full_mix,
                'intron_only': args.scenario_intron_only,
                'cds_only':    args.scenario_cds_only,
                'degraded':    args.scenario_degraded,
                'ab_initio':   args.scenario_ab_initio,
            }
        )
        print(f"\nNoising enabled:")
        print(f"  Scenarios: {noising_config.scenario_weights}")

    # Load dataset
    print(f"\nLoading training data (lazy mode with per-epoch noising)...")
    print(f"  Files: {args.train_data}")

    full_dataset = dataset.NoisedDataset(
        args.train_data,
        tokenizer,
        args.max_input_len,
        args.max_target_len,
        noising_config,
        args.hint_token,
        args.seed,
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

    # Dataloaders
    print(f"\nSetting up dataloaders...")

    collator = dataset.DynamicPaddingCollator(
        tokenizer.pad_token_id,
        -100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=dataset.SmartBatchSampler(
            train_lengths,
            args.batch_size,
            args.bucket_size,
            True,
            True,
        ),
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"  Train batches: {len(train_loader)}")

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=dataset.SmartBatchSampler(
                val_lengths,
                args.batch_size,
                args.bucket_size,
                False,
                False,
            ),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"  Val batches:   {len(val_loader)}")

    # Optimizer & scheduler
    print(f"\nSetting up optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    print(f"  Total steps:  {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Resume checkpoint
    start_epoch      = 0
    best_val_loss    = float('inf')
    patience_counter = 0

    if args.checkpoint:
        checkpoint_data = lib_train.load_checkpoint(
            model, optimizer, scheduler, args.checkpoint, device
        )
        start_epoch   = checkpoint_data["epoch"]
        best_val_loss = checkpoint_data.get("best_val_loss", float('inf'))
        print(f"  Resumed from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.4f}")

    # Save config
    config = {
        **vars(args),
        "vocab_size":    len(tokenizer),
        "total_steps":   total_steps,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset) if val_dataset else 0,
        "noising":       args.enable_noising,
    }
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n{'=' * 60}")
    print("Training...")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Set epoch for noising variation
        if hasattr(full_dataset, 'set_epoch'):
            full_dataset.set_epoch(epoch)
            print(f"  Noise seed: {args.seed + epoch}")

        # Train
        train_loss = lib_train.train_epoch_seq2seq(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, args.max_grad_norm
        )

        # Validate
        val_loss = None
        if val_loader is not None:
            val_loss = lib_train.evaluate_seq2seq(model, val_loader, device)
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

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
                    print(f"\n  Early stopping triggered!")
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

    # Save final
    print(f"\n{'=' * 60}")
    print(f"Saving final model...")
    lib_train.save_checkpoint(
        model, optimizer, scheduler, args.epochs,
        output_dir / "final_model.pt",
        {**config, "best_val_loss": best_val_loss}
    )
    model.save(output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)

    # Copy model config
    with open(model_path / "config.json") as f:
        model_config = json.load(f)
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Fine-tuning complete!")
    print(f"  Output: {output_dir}")
    if val_loader is not None:
        print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()