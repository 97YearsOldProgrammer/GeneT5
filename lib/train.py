import os
from pathlib import Path

import torch
import torch.nn             as nn
import torch.distributed    as dist

from torch.optim  import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, EncoderDecoderModel



################################
#####  Distributed Setup   #####
################################


def setup_distributed(backend="nccl"):
    """
    Initialize distributed training environment for DGX Spark multi-node setup.
    
    Uses NCCL backend for GPU-to-GPU communication over ConnectX-7 200Gb/s network.
    Environment variables should be set by torchrun/torch.distributed.launch.
    """
    if not dist.is_initialized():
        # Check if running in distributed mode
        if "RANK" not in os.environ:
            return None
        
        rank       = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device before init
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend    = backend,
            rank       = rank,
            world_size = world_size,
        )
        
        return {
            "rank":       rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_main":    rank == 0,
        }
    
    return {
        "rank":       dist.get_rank(),
        "world_size": dist.get_world_size(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "is_main":    dist.get_rank() == 0,
    }


def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():

    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():

    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronization barrier for distributed training"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor):
    """Average tensor across all processes"""
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def broadcast_object(obj, src=0):
    """Broadcast Python object from src to all processes"""
    if not dist.is_initialized():
        return obj
    
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


############################
#####  Model Wrapping  #####
############################


def wrap_model_distributed(model, device, find_unused_params=False, gradient_as_bucket_view=True):
    """Wrap model with DistributedDataParallel for multi-GPU training"""
    
    if not dist.is_initialized():
        return model.to(device)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids             = [local_rank],
        output_device          = local_rank,
        find_unused_parameters = find_unused_params,
        gradient_as_bucket_view = gradient_as_bucket_view,
    )
    
    return model


def unwrap_model(model):
    """Unwrap model from DistributedDataParallel."""
    if hasattr(model, 'module'):
        return model.module
    return model


################################
#####  Training Functions  #####
################################


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch with BF16 mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    num_steps  = 0
    
    # BF16 for training (model stays FP32, compute in BF16)
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(**batch) if not hasattr(model, 'forward_finetune') else model(**batch)
            
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs.loss
            
            loss = loss / grad_accum
        
        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1
        
        total_loss += loss.item() * grad_accum
    
    avg_loss = total_loss / len(dataloader)
    
    # Average loss across processes in distributed training
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def train_epoch_seq2seq(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_grad_norm=1.0):
    """Train for one epoch - seq2seq with BF16 mixed precision and distributed support"""
    model.train()
    total_loss = 0
    
    # BF16 for training (model stays FP32, compute in BF16)
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
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
        
        if dtype == torch.float16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % grad_accum == 0:
            if dtype == torch.float16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
    
    avg_loss = total_loss / len(dataloader)
    
    # Average loss across processes in distributed training
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def train_epoch_seq2seq_distributed(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_accum    = 1,
    max_grad_norm = 1.0,
    scaler        = None,
    use_amp       = True,
):
    """
    Train for one epoch with full distributed support for DGX Spark.
    
    Optimized for multi-node training:
    - Gradient synchronization via NCCL
    - Proper handling of MoE load balancing loss
    - Mixed precision with configurable scaler
    """
    
    model.train()
    total_loss     = 0.0
    total_moe_loss = 0.0
    num_batches    = 0
    
    # Determine dtype and create scaler if needed
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and use_amp) else torch.float32
    if scaler is None and dtype == torch.float16:
        scaler = torch.amp.GradScaler('cuda')
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
            outputs = model(
                encoder_input_ids = batch["input_ids"],
                decoder_input_ids = batch["labels"][:, :-1],
                labels            = batch["labels"][:, 1:],
            )
            
            # Handle both dict and object outputs
            if isinstance(outputs, dict):
                loss     = outputs["loss"]
                moe_loss = outputs.get("moe_loss", 0.0)
            else:
                loss     = outputs.loss
                moe_loss = getattr(outputs, 'moe_loss', 0.0)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
        
        # Backward pass
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Optimizer step with gradient accumulation
        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Accumulate losses
        total_loss     += loss.detach().item()
        total_moe_loss += moe_loss if isinstance(moe_loss, (int, float)) else moe_loss.item()
        num_batches    += 1
    
    # Compute averages
    avg_loss     = total_loss / max(num_batches, 1)
    avg_moe_loss = total_moe_loss / max(num_batches, 1)
    
    # Synchronize metrics across all processes
    if dist.is_initialized():
        metrics      = torch.tensor([avg_loss, avg_moe_loss, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size   = get_world_size()
        avg_loss     = metrics[0].item() / world_size
        avg_moe_loss = metrics[1].item() / world_size
    
    return avg_loss, avg_moe_loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation set with BF16."""
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**batch)
                
                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs.loss
            
            total_loss += loss.item()
            
            if isinstance(outputs, dict) and "logits" in outputs:
                logits   = outputs["logits"]
                preds    = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += batch["labels"].size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else None
    
    # Synchronize metrics in distributed training
    if dist.is_initialized():
        metrics = torch.tensor([avg_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss   = metrics[0].item() / world_size
        if metrics[2].item() > 0:
            accuracy = metrics[1].item() / metrics[2].item()
    
    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate_seq2seq(model, dataloader, device):
    """Evaluate seq2seq model with BF16 and distributed support."""
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
            
            if isinstance(outputs, dict):
                total_loss += outputs["loss"].item()
            else:
                total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Synchronize in distributed training
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss   = loss_tensor[0].item() / world_size
    
    model.train()
    return avg_loss


def evaluate_seq2seq_distributed(model, dataloader, device, use_amp=True):
    """Evaluate seq2seq model with full distributed support for DGX Spark"""
    
    model.eval()
    total_loss     = 0.0
    total_moe_loss = 0.0
    num_batches    = 0
    
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and use_amp) else torch.float32
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', dtype=dtype, enabled=use_amp):
                outputs = model(
                    encoder_input_ids = batch["input_ids"],
                    decoder_input_ids = batch["labels"][:, :-1],
                    labels            = batch["labels"][:, 1:],
                )
            
            if isinstance(outputs, dict):
                total_loss     += outputs["loss"].item()
                total_moe_loss += outputs.get("moe_loss", 0.0)
                if isinstance(total_moe_loss, torch.Tensor):
                    total_moe_loss = total_moe_loss.item()
            else:
                total_loss += outputs.loss.item()
            
            num_batches += 1
    
    avg_loss     = total_loss / max(num_batches, 1)
    avg_moe_loss = total_moe_loss / max(num_batches, 1)
    
    # Synchronize across all processes
    if dist.is_initialized():
        metrics    = torch.tensor([avg_loss, avg_moe_loss, num_batches], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = get_world_size()
        avg_loss     = metrics[0].item() / world_size
        avg_moe_loss = metrics[1].item() / world_size
    
    model.train()
    return avg_loss, avg_moe_loss


##################################
#####  Checkpoint Functions  #####
##################################


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device="cpu"):
    """Load model, optimizer, and scheduler states from checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP wrapped models
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


def save_checkpoint(model, optimizer, scheduler, epoch, save_path, config=None):
    """Save model, optimizer, and scheduler states (only on main process)"""
    
    if not is_main_process():
        return
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP model for saving
    target_model = unwrap_model(model)
    
    checkpoint = {
        "epoch":                epoch,
        "model_state_dict":     target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    
    if config:
        checkpoint["config"] = config
    
    torch.save(checkpoint, save_path)


def save_checkpoint_distributed(model, optimizer, scheduler, epoch, save_path, config=None):
    """
    Save checkpoint with distributed training support.
    Only main process saves, others wait at barrier.
    """
    barrier()  # Ensure all processes are synced
    
    if is_main_process():
        save_checkpoint(model, optimizer, scheduler, epoch, save_path, config)
    
    barrier()  # Wait for save to complete


###################################
#####  Model Setup Functions  #####
###################################


def setup_gene_prediction_model(model_path, tokenizer, device):
    """Setup model for gene prediction task."""
    
    base_model = EncoderDecoderModel.from_pretrained(model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask, labels=None):
            return self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )
    
    model = ModelWrapper(base_model)
    model.to(device)
    return model


def prepare_tokenizer(model_path, special_tokens=None):
    """Load and prepare tokenizer with special tokens"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if special_tokens is None:
        special_tokens = ["[GENE]", "[CLS]"]
    
    new_tokens = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        if is_main_process():
            print(f"Added tokens: {new_tokens}")
    
    return tokenizer


def prepare_optimizer_scheduler(model, train_loader, lr, weight_decay, 
                                epochs, grad_accum, warmup_ratio, scheduler_type="linear"):
    """Prepare optimizer and scheduler."""

    total_steps  = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    return optimizer, scheduler


def prepare_optimizer_scheduler_distributed(
    model,
    train_loader,
    lr,
    weight_decay,
    epochs,
    grad_accum,
    warmup_ratio,
    scheduler_type = "cosine",
    betas          = (0.9, 0.95),
):
    """
    Prepare optimizer and scheduler for distributed training.
    
    Accounts for world size in learning rate scaling (linear scaling rule).
    """
    from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
    
    world_size   = get_world_size()
    total_steps  = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Linear scaling rule for distributed training
    scaled_lr = lr * world_size
    
    optimizer = AdamW(
        model.parameters(),
        lr           = scaled_lr,
        betas        = betas,
        weight_decay = weight_decay,
    )
    
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    if is_main_process():
        print(f"  Base LR:      {lr}")
        print(f"  Scaled LR:    {scaled_lr} (world_size={world_size})")
        print(f"  Total steps:  {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
    
    return optimizer, scheduler


###############################
#####  Utility Functions  #####
###############################


def get_device():

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_rank0(*args, **kwargs):
    """Print only from rank 0 process"""
    if is_main_process():
        print(*args, **kwargs)


def log_metrics(metrics, epoch, step=None, prefix=""):
    """Log metrics only from main process"""
    if not is_main_process():
        return
    
    if step is not None:
        header = f"[Epoch {epoch}, Step {step}]"
    else:
        header = f"[Epoch {epoch}]"
    
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                             for k, v in metrics.items()])
    print(f"{prefix}{header} {metric_str}")