#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from lib.train import (
    train_epoch,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    setup_gene_prediction_model,
    setup_rna_classification_model,
    prepare_tokenizer,
    prepare_optimizer_scheduler,
)

from lib.tuning.dataset     import GenePredictionDataset, RNAClassificationDataset
from lib.tuning._chunking   import preprocess_and_chunk
from lib.tuning._parser     import RNA_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeneT5 model")
    
    # data
    parser.add_argument("--train_data",     type=str, required=True)
    parser.add_argument("--val_data",       type=str, default=None)
    parser.add_argument("--output_dir",     type=str, default="outputs")
    
    # model
    parser.add_argument("--model_path",     type=str, default="google/t5-base")
    parser.add_argument("--checkpoint",     type=str, default=None)
    parser.add_argument("--task",           type=str, default="gene_prediction",
                        choices=["gene_prediction", "classification"])
    
    # training
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=5e-5)
    parser.add_argument("--weight_decay",   type=float, default=0.01)
    parser.add_argument("--grad_accum",     type=int,   default=1)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--max_input_len",  type=int,   default=2048)
    parser.add_argument("--max_target_len", type=int,   default=1024)
    parser.add_argument("--dropout",        type=float, default=0.1)
    
    # chunking
    parser.add_argument("--chunk",          action="store_true")
    parser.add_argument("--chunked_dir",    type=str, default=None)
    
    # wandb
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--wandb_project",  type=str, default="genet5")
    parser.add_argument("--wandb_run",      type=str, default=None)
    
    # misc
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--save_every",     type=int, default=1)
    
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


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args   = parse_args()
    device = get_device()
    
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    
    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # wandb
    wandb = None
    if args.wandb:
        wandb = setup_wandb(args)
    
    # tokenizer
    special_tokens = ["[GENE]", "[CLS]", "<BOS>", "<EOS>"]
    tokenizer      = prepare_tokenizer(args.model_path, special_tokens)
    
    # chunking if requested
    train_path = args.train_data
    val_path   = args.val_data
    
    if args.chunk:
        chunked_dir = Path(args.chunked_dir or output_dir / "chunked")
        chunked_dir.mkdir(parents=True, exist_ok=True)
        
        train_chunked = chunked_dir / "train_chunked.jsonl"
        preprocess_and_chunk(
            args.train_data, tokenizer, train_chunked,
            args.max_input_len, args.max_target_len, args.task
        )
        train_path = train_chunked
        
        if args.val_data:
            val_chunked = chunked_dir / "val_chunked.jsonl"
            preprocess_and_chunk(
                args.val_data, tokenizer, val_chunked,
                args.max_input_len, args.max_target_len, args.task
            )
            val_path = val_chunked
    
    # datasets
    if args.task == "gene_prediction":
        train_dataset = GenePredictionDataset(
            train_path, tokenizer, args.max_input_len, args.max_target_len
        )
        val_dataset = None
        if val_path:
            val_dataset = GenePredictionDataset(
                val_path, tokenizer, args.max_input_len, args.max_target_len
            )
    else:
        train_dataset = RNAClassificationDataset(
            train_path, tokenizer, args.max_input_len
        )
        val_dataset = None
        if val_path:
            val_dataset = RNAClassificationDataset(
                val_path, tokenizer, args.max_input_len
            )
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")
    
    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
    
    # model
    if args.task == "gene_prediction":
        model = setup_gene_prediction_model(args.model_path, tokenizer, device)
    else:
        num_classes = len(RNA_CLASSES)
        model = setup_rna_classification_model(
            args.model_path, tokenizer, num_classes, device, args.dropout
        )
    
    # optimizer & scheduler
    optimizer, scheduler = prepare_optimizer_scheduler(
        model, train_loader,
        args.lr, args.weight_decay,
        args.epochs, args.grad_accum, args.warmup_ratio
    )
    
    # load checkpoint
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint)
    
    # training loop
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_accum
        )
        print(f"Train loss: {train_loss:.4f}")
        
        # eval
        val_metrics = None
        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            print(f"Val loss: {val_metrics['loss']:.4f}")
            if val_metrics["accuracy"] is not None:
                print(f"Val acc: {val_metrics['accuracy']:.4f}")
        
        # wandb logging
        if wandb:
            log_dict = {
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "lr":         scheduler.get_last_lr()[0],
            }
            if val_metrics:
                log_dict["val_loss"] = val_metrics["loss"]
                if val_metrics["accuracy"] is not None:
                    log_dict["val_acc"] = val_metrics["accuracy"]
            wandb.log(log_dict)
        
        # save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, ckpt_path)
        
        # save best
        if val_metrics and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path     = output_dir / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_path)
            print(f"New best model saved (val_loss: {best_val_loss:.4f})")
    
    # save final
    final_path = output_dir / "final_model.pt"
    save_checkpoint(model, optimizer, scheduler, args.epochs, final_path)
    
    if wandb:
        wandb.finish()
    
    print("\nTraining complete!")