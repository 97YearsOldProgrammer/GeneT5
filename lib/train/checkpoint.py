import json
from pathlib import Path

import torch

from lib.train.distributed import is_main_process, unwrap_model


##################################
#####  Checkpoint Functions  #####
##################################


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cpu"):
    """Load model, optimizer, and scheduler states from checkpoint"""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    target_model = unwrap_model(model)
    target_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch         = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("config", {}).get("best_val_loss", float('inf'))

    if is_main_process():
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

    return {
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
        "config":        checkpoint.get("config", {})
    }


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None, global_step=None):
    """Save model, optimizer, and scheduler states (only on main process)"""

    if not is_main_process():
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    target_model = unwrap_model(model)

    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if global_step is not None:
        checkpoint["global_step"] = global_step

    if config:
        checkpoint["config"] = config

    torch.save(checkpoint, save_path)


def save_final_model(model, tokenizer, model_path, output_dir):
    """Save final model weights, tokenizer, and config"""

    final_model = unwrap_model(model)
    final_model.save(output_dir / 'pytorch_model.bin')
    tokenizer.save_pretrained(output_dir)

    model_path = Path(model_path)
    with open(model_path / 'config.json') as f:
        model_config = json.load(f)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
