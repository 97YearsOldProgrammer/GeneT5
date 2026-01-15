
import argparse
import json
import random
import sys
from pathlib import Path

import  torch
from    torch.utils.data import DataLoader


import  lib.train       as     train
from    lib             import tuning
from    lib.model       import GeneT5
from    lib.tokenizer   import GeneTokenizer


# Trainning Parameter for Saving Space
DEFAULTS = {
    "max_input_len":  4096,
    "max_target_len": 2048,
    "batch_size":     4,
    "lr":             1e-5,
    "epochs":         3,
    "weight_decay":   0.1,
    "warmup_ratio":   0.1,
    "grad_accum":     128,
    "max_grad_norm":  1.0,
    "bucket_size":    256,
    "num_workers":    8,
}


# Require a Wrapper for multi parallel working
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
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--max_input_len", type=int, default=DEFAULTS["max_input_len"])
    parser.add_argument("--max_target_len", type=int, default=DEFAULTS["max_target_len"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility (shuffling, dropout, etc).")
    parser.add_argument("--save_every", type=int, default=1,
        help="Save checkpoint every N epochs.")
    # Add num_workers arg so you can tune it easily
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"],
        help="Number of dataloader workers.")

    args = parser.parse_args()

    # setup
    print(f"\n{' GeneT5 Fine-Tuning ':=^60}")
    device = train.get_device()
    print(f"Device: {device}")

    # seed - ensures reproducible shuffling and initialization
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

    # load model
    print(f"\nLoading model...")
    # Ensure model loads on CPU first to save GPU memory during init
    model = GeneT5.from_pretrained(model_path, device="cpu").to(device)

    stats = model.get_param_stats()
    print(f"  Trainable: {stats['total_trainable']:,}")
    print(f"  Frozen:    {stats['total_frozen']:,}")

    # load dataset - multiple files get mixed
    print(f"\nLoading training data...")
    print(f"  Files: {args.train_data}")

    train_dataset = tuning.MixedTaskDataset(
        data_paths     = args.train_data,
        tokenizer      = tokenizer,
        max_input_len  = args.max_input_len,
        max_target_len = args.max_target_len,
    )
    print(f"  Total samples: {len(train_dataset)}")

    # show task distribution
    task_counts = {}
    for sample in train_dataset.samples:
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    print(f"  Task distribution: {task_counts}")

    # dataloader with smart batching
    print(f"\nSetting up dataloader...")

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
        num_workers = args.num_workers, # Use the argument
        pin_memory  = True,
        # persistent_workers=True can help speed up epochs on Mac if RAM allows
        persistent_workers = (args.num_workers > 0), 
    )
    print(f"  Total batches: {len(train_loader)}")

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
        print(f"  Resumed from epoch {start_epoch}")

    # save config
    config = {**vars(args), "vocab_size": len(tokenizer), "total_steps": total_steps}
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # training loop
    print(f"\n{'=' * 60}")
    print("Training...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        train_loss = train.train_epoch_seq2seq(
            model, train_loader, optimizer, scheduler,
            device, args.grad_accum, DEFAULTS["max_grad_norm"]
        )
        print(f"  Loss: {train_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # save checkpoint
        if (epoch + 1) % args.save_every == 0:
            train.save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                output_dir / f"checkpoint_epoch{epoch + 1}.pt", config
            )

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

if __name__ == "__main__":
    main()
