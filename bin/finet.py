#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers     import get_cosine_schedule_with_warmup

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
    parser = argparse.ArgumentParser(description="Fine-tune GeneT5 with hint-based noising (DGX Spark distributed support)")

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
    
    # Noising scenario weights
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

    # Distributed training params (DGX Spark)
    parser.add_argument("--distributed", action="store_true", default=False,
        help="Enable distributed training across multiple DGX Spark systems.")
    parser.add_argument("--backend", type=str, default="nccl",
        help="Distributed backend (nccl for GPU, gloo for CPU).")
    parser.add_argument("--find_unused_params", action="store_true", default=False,
        help="Enable find_unused_parameters in DDP (needed for MoE in some cases).")
    parser.add_argument("--no_lr_scaling", action="store_true", default=False,
        help="Disable automatic learning rate scaling with world size.")

    args = parser.parse_args()

    # Validate
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError(f"val_split must be between 0.0 and 1.0, got {args.val_split}")

    # Setup distributed training
    dist_info = None
    if args.distributed or "RANK" in os.environ:
        dist_info = lib_train.setup_distributed(backend=args.backend)
        if dist_info:
            args.distributed = True
    
    is_main = lib_train.is_main_process()
    
    # Setup
    if is_main:
        print(f"\n{' GeneT5 Fine-Tuning ':=^60}")
    
    device = lib_train.get_device()
    
    if is_main:
        print(f"Device: {device}")
        if dist_info:
            print(f"Distributed Training:")
            print(f"  World size:  {dist_info['world_size']}")
            print(f"  Rank:        {dist_info['rank']}")
            print(f"  Local rank:  {dist_info['local_rank']}")
            print(f"  Backend:     {args.backend}")

    # Seed (different per rank for data augmentation diversity)
    base_seed  = args.seed
    rank_seed  = base_seed + (dist_info['rank'] if dist_info else 0)
    torch.manual_seed(rank_seed)
    random.seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize before proceeding
    lib_train.barrier()

    # Load tokenizer
    if is_main:
        print(f"\nLoading tokenizer...")
    model_path = Path(args.model_path)
    tokenizer  = GeneTokenizer(model_path)
    if is_main:
        print(f"  Vocab size: {len(tokenizer)}")

    # Load model
    if is_main:
        print(f"\nLoading model in FP32...")
    model = GeneT5.from_pretrained(model_path, device="cpu", dtype=torch.float32)
    
    if is_main:
        stats = model.get_param_stats()
        print(f"  Trainable: {stats['total_trainable']:,}")
        print(f"  Frozen:    {stats['total_frozen']:,}")
    
    # Wrap model for distributed training
    if args.distributed:
        if is_main:
            print(f"\nWrapping model with DistributedDataParallel...")
        model = lib_train.wrap_model_distributed(
            model,
            device,
            find_unused_params = args.find_unused_params,
        )
    else:
        model = model.to(device)

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
        if is_main:
            print(f"\nNoising enabled:")
            print(f"  Scenarios: {noising_config.scenario_weights}")

    # Load dataset
    if is_main:
        print(f"\nLoading training data (lazy mode with per-epoch noising)...")
        print(f"  Files: {args.train_data}")

    full_dataset = dataset.NoisedDataset(
        args.train_data,
        tokenizer,
        args.max_input_len,
        args.max_target_len,
        noising_config,
        args.hint_token,
        base_seed,
    )

    # Split into train/val
    if args.val_split > 0:
        val_size   = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(base_seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        if is_main:
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
        if is_main:
            print(f"  No validation split (val_split=0)")

    # Dataloaders with distributed support
    if is_main:
        print(f"\nSetting up dataloaders...")

    collator = dataset.DynamicPaddingCollator(
        tokenizer.pad_token_id,
        -100,
    )

    # Create distributed-aware dataloaders
    train_loader = dataset.create_distributed_dataloader(
        train_dataset,
        args.batch_size,
        lengths            = train_lengths,
        bucket_size        = args.bucket_size,
        shuffle            = True,
        drop_last          = True,
        num_workers        = args.num_workers,
        pin_memory         = True,
        collate_fn         = collator,
        seed               = base_seed,
        use_smart_batching = True,
    )
    
    if is_main:
        print(f"  Train batches (per GPU): {len(train_loader)}")

    val_loader = None
    if val_dataset is not None:
        val_loader = dataset.create_distributed_dataloader(
            val_dataset,
            args.batch_size,
            lengths            = val_lengths,
            bucket_size        = args.bucket_size,
            shuffle            = False,
            drop_last          = False,
            num_workers        = args.num_workers,
            pin_memory         = True,
            collate_fn         = collator,
            seed               = base_seed,
            use_smart_batching = True,
        )
        if is_main:
            print(f"  Val batches (per GPU):   {len(val_loader)}")

    # Optimizer & scheduler
    if is_main:
        print(f"\nSetting up optimizer...")

    world_size = lib_train.get_world_size()
    
    # Effective batch size accounting for gradient accumulation and world size
    effective_batch_size = args.batch_size * args.grad_accum * world_size
    
    # Learning rate scaling
    if args.distributed and not args.no_lr_scaling:
        # Linear scaling rule: lr_scaled = lr_base * world_size
        scaled_lr = args.lr * world_size
        if is_main:
            print(f"  LR scaling: {args.lr} -> {scaled_lr} (world_size={world_size})")
    else:
        scaled_lr = args.lr
    
    if is_main:
        print(f"  Effective batch size: {effective_batch_size}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = scaled_lr,
        betas        = (0.9, 0.95),
        weight_decay = args.weight_decay
    )

    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    if is_main:
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
        if is_main:
            print(f"  Resumed from epoch {start_epoch}")
            print(f"  Best val loss so far: {best_val_loss:.4f}")

    # Save config (main process only)
    config = {
        **vars(args),
        "vocab_size":           len(tokenizer),
        "total_steps":          total_steps,
        "train_samples":        len(train_dataset),
        "val_samples":          len(val_dataset) if val_dataset else 0,
        "noising":              args.enable_noising,
        "world_size":           world_size,
        "effective_batch_size": effective_batch_size,
        "scaled_lr":            scaled_lr,
    }
    
    if is_main:
        with open(output_dir / "finetune_config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Synchronize before training
    lib_train.barrier()

    # Training loop
    if is_main:
        print(f"\n{'=' * 60}")
        print("Training...")
        print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs):
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 40)

        # Set epoch for noising variation and distributed sampler
        if hasattr(full_dataset, 'set_epoch'):
            full_dataset.set_epoch(epoch)
            if is_main:
                print(f"  Noise seed: {base_seed + epoch}")
        
        # Set epoch on dataloader sampler
        dataset.set_dataloader_epoch(train_loader, epoch)

        # Train
        if args.distributed:
            train_loss, moe_loss = lib_train.train_epoch_seq2seq_distributed(
                model, train_loader, optimizer, scheduler,
                device, args.grad_accum, args.max_grad_norm
            )
        else:
            train_loss = lib_train.train_epoch_seq2seq(
                model, train_loader, optimizer, scheduler,
                device, args.grad_accum, args.max_grad_norm
            )
            moe_loss = 0.0

        # Validate
        val_loss = None
        if val_loader is not None:
            if args.distributed:
                val_loss, val_moe_loss = lib_train.evaluate_seq2seq_distributed(model, val_loader, device)
            else:
                val_loss = lib_train.evaluate_seq2seq(model, val_loader, device)
            
            if is_main:
                if args.distributed:
                    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MoE Loss: {moe_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                else:
                    print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0

                    print(f"  ✓ New best validation loss! Saving best_model.pt")
                    lib_train.save_checkpoint_distributed(
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
            if is_main:
                if args.distributed:
                    print(f"  Train Loss: {train_loss:.4f} | MoE Loss: {moe_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                else:
                    print(f"  Train Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save periodic checkpoint
        if is_main and (epoch + 1) % args.save_every == 0:
            lib_train.save_checkpoint_distributed(
                model, optimizer, scheduler, epoch + 1,
                output_dir / "checkpoint_latest.pt",
                {**config, "best_val_loss": best_val_loss}
            )
            print(f"  Saved checkpoint_latest.pt")
        
        # Synchronize at end of epoch
        lib_train.barrier()

    # Save final model (main process only)
    if is_main:
        print(f"\n{'=' * 60}")
        print(f"Saving final model...")
    
    lib_train.save_checkpoint_distributed(
        model, optimizer, scheduler, args.epochs,
        output_dir / "final_model.pt",
        {**config, "best_val_loss": best_val_loss}
    )
    
    if is_main:
        # Unwrap model for saving
        unwrapped_model = lib_train.unwrap_model(model)
        unwrapped_model.save(output_dir / "pytorch_model.bin")
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
        if args.distributed:
            print(f"  World Size: {world_size}")
        print(f"{'=' * 60}")

    # Cleanup distributed training
    if args.distributed:
        lib_train.cleanup_distributed()


if __name__ == "__main__":
    main()